"""Bounded host-to-device prefetch, preserving DataLoader order and CPU bookkeeping."""
from collections import deque
from queue import Empty, Full, Queue
from threading import Event, Thread
import time

import torch

from ..torch_util import move_to_device


def _record_stream(value, stream):
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record_stream(item, stream)
    elif hasattr(value, 'record_stream'):
        value.record_stream(stream)


class CUDABatchPrefetcher:
    """One ordered producer, at most two queued batches, explicit stream lifetimes.

    Construct once per epoch and close before changing dataset iteration state.
    DataLoader workers perform collation; this thread only drains pinned batches
    and submits copies. Index/metadata stay on CPU for logging. Disabled mode is
    a transparent iterator and also works without CUDA.
    """
    def __init__(self, loader, device, enabled=True):
        self.device = torch.device(device)
        self.enabled = enabled and self.device.type == 'cuda'
        self.iterator = iter(loader)  # Start/fork CPU workers on the calling thread.
        self.wait_seconds = 0.0
        self.last_wait_seconds = 0.0
        self.closed = False
        self.exhausted = False
        if self.enabled:
            self.stream = torch.cuda.Stream(device=self.device)
            self.queue = Queue(maxsize=2)
            self.stop = Event()
            self.retired = deque()
            self.thread = Thread(target=self._produce, name='olmo-h2d-prefetch', daemon=True)
            self.thread.start()

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.queue.put(value, timeout=0.1)
                return True
            except Full:
                pass
        return False

    def _produce(self):
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                while not self.stop.is_set():
                    try:
                        cpu = next(self.iterator)
                    except StopIteration:
                        self._put(None)
                        return
                    gpu = {k: v if k in ('index', 'metadata', 'instance_mask')
                           else move_to_device(v, self.device, non_blocking=True) for k, v in cpu.items()}
                    event = torch.cuda.Event()
                    event.record(self.stream)
                    if not self._put((gpu, event, cpu)):
                        # Keep pinned sources alive until the last submitted copy finishes.
                        event.synchronize()
                        return
        except BaseException as exc:
            self._put(exc)

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed or self.exhausted:
            raise StopIteration
        start = time.perf_counter()
        if not self.enabled:
            try:
                return next(self.iterator)
            finally:
                self.last_wait_seconds = time.perf_counter() - start
                self.wait_seconds += self.last_wait_seconds
        item = self.queue.get()
        if item is None:
            self.exhausted = True
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        gpu, event, cpu = item
        current = torch.cuda.current_stream(self.device)
        current.wait_event(event)
        _record_stream(gpu, current)
        self.retired.append((event, cpu))
        while self.retired and (len(self.retired) > 2 or self.retired[0][0].query()):
            retired_event, retired_cpu = self.retired.popleft()
            retired_event.synchronize()
            del retired_cpu
        self.last_wait_seconds = time.perf_counter() - start
        self.wait_seconds += self.last_wait_seconds
        return gpu

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.enabled:
            self.stop.set()
            self.thread.join()
            self.stream.synchronize()
            self.retired.clear()
            while True:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break
        # Do not shut down persistent DataLoader workers; the next epoch reuses them.
        self.iterator = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
