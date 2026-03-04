import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import parse_input
import time
import re
import multiprocessing as mp
import benepar
import sys
import itertools
import torch
from tqdm import tqdm
from queue import Empty
import logging
from logging.handlers import QueueHandler, QueueListener
from parse_input import split_text_into_sents, process_doc_into_maxlen, prepare_dataset, count_lines_linux_style

def setup_worker_logger(log_queue, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        qh = QueueHandler(log_queue)
        logger.addHandler(qh)

    return logger


def setup_main_logger(log_queue, log_file):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '[%(asctime)s][%(process)d][%(threadName)s]'
        '[%(levelname)s][%(name)s] %(message)s'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    listener = QueueListener(log_queue, file_handler)
    listener.start()
    return listener


# ---------- 配置 ----------
# 28 threads per GPU
# 1 MAIN
# 1 GPUs, 6 CPU preprocessor = 8
# 1 GPUTreer
# 16 CPUtreer, 
# 1 writer, 1 feeder, 1 batcher = 3
MAX_TOKENS_PER_BATCH = 8000       # 每个 batch 最大token数（可调，64*250=16000）
MAX_SAMPLES_PER_BATCH = 4        # 每个 batch 最多样本数（可调）
NUM_CPU_WORKERS = min(max(1, mp.cpu_count() - 2), 4)
NUM_GPU_TREEWORKERS = 2
NUM_CPU_TREEWORKERS = 0
# GPU_IDS = list(range(torch.cuda.device_count() if 'torch' in globals() and torch.cuda.is_available() else 16))  # 默认 16 或替换成 [0,1,...]
GPU_IDS = [0] * 2
SAMPLE_QUEUE_MAXSIZE = 4096       # 预处理样本队列长度上限
TASK_QUEUE_MAXSIZE = 4096 
RESULT_QUEUE_MAXSIZE = 4096 
SPLIT_MAX_LEN = 450

def batcher_proc(
    sample_queue, task_queue, log_queue, stop_event,
    max_samples_per_batch=MAX_SAMPLES_PER_BATCH,
    max_tokens_per_batch=MAX_TOKENS_PER_BATCH,
):
    logger = setup_worker_logger(
        log_queue,
        name=f"batcher"
    )
    logger.info("batcher started")
    pending = []
    sub_idxs = []
    endingchars = []
    indices = []

    def flush_batch():
        nonlocal pending, indices, endingchars, sub_idxs
        if not pending:
            return
        task_queue.put((indices, sub_idxs, endingchars, pending))
        pending = []
        endingchars = []
        indices = []
        sub_idxs = []
    while True:
        if stop_event.is_set() and sample_queue.empty():
            break
        try:
            item = sample_queue.get(timeout=0.5)
        except Empty:
            flush_batch()
            continue

        idx, sub_idx, endchar, sent = item
        indices.append(idx)
        sub_idxs.append(sub_idx)
        endingchars.append(endchar)
        pending.append(sent)
        if len(pending) >= max_samples_per_batch:
            flush_batch()

    flush_batch()
    for _ in range(len(GPU_IDS)):
        task_queue.put(None)
    logger.info(f"batcher finished")


def custom_parse_encoded(
    self, examples, encoded, return_compressed=False, return_scores=False
):
    with torch.no_grad():
        batch = self.pad_encoded(encoded)
        span_scores, tag_scores = self.forward(batch)
        if return_scores:
            raise NotImplementedError
        else:
            # Start/stop tokens don't count, so subtract 2
            lengths = batch["valid_token_mask"].sum(-1) - 2
            charts_np = self.decoder.charts_from_pytorch_scores_batched(
                span_scores, lengths.to(span_scores.device)
            )
        if tag_scores is not None:
            tag_ids_np = tag_scores.argmax(-1).cpu().numpy()
        else:
            tag_ids_np = None
    
    return charts_np, tag_ids_np

def custom_parse(
        self,
        examples,
        return_compressed=False,
        return_scores=False,
    ):
        training = self.training
        self.eval()
        encoded = [self.encode(example) for example in examples]
        charts_np, tag_ids_np = self.custom_parse_encoded(
            self,
            examples,
            encoded,
            return_compressed=return_compressed,
            return_scores=return_scores,
        )
        return charts_np, tag_ids_np

def custom_parse_sents(self, sents):
    end_sentinel = object()
    for batch_sents in itertools.zip_longest(
        *([iter(sents)] * self.batch_size), fillvalue=end_sentinel
    ):
        batch_inputs = []
        for sent in batch_sents:
            if sent is end_sentinel:
                break
            elif not isinstance(sent,  benepar.InputSentence):
                raise ValueError(
                    "Sentences must be one of: InputSentence"
                )
            batch_inputs.append(self._with_missing_fields_filled(sent))
    charts_np, tag_ids_np = self._parser.custom_parse(self._parser, batch_inputs, return_compressed=True)
    return batch_inputs, charts_np, tag_ids_np

def load_model_to_device(device):
    beneparser = benepar.Parser("benepar_en3", batch_size=MAX_SAMPLES_PER_BATCH)
    beneparser._parser.to(device)
    beneparser.custom_parse_sents = custom_parse_sents
    beneparser._parser.custom_parse = custom_parse
    beneparser._parser.custom_parse_encoded = custom_parse_encoded
    return beneparser

def cpu_tree_proc(task_queue, result_queue, log_queue):
    logger = setup_worker_logger(
        log_queue,
        name=f"cpu_parser"
    )
    logger.info("cpu_parser started")
    import torch
    device = torch.device(f"cpu")
    parser = load_model_to_device(device)
    logger.info(f"pid={os.getpid()} using device {device}")
    while True:
        item = task_queue.get()
        if item is None:
            logger.info(f"cpu parser received sentinel, exiting")
            break
        idx, subidxs, endingchars, batch = item
        try:
            TreeGen = parser.parse_sents(batch)
            results = []
            for id, sub_idx, endchar, tree in zip(idx, subidxs, endingchars, TreeGen):
                tree = tree[0]
                leaves = tree.leaves()
                if len(leaves) == 1 and re.match(r"\n+", leaves[0]):
                    parsed_string = leaves[0].replace("\n", "(Ċ Ċ) ").rstrip()
                else:
                    parsed_string = tree.pformat(margin=100000) if tree.leaves() != ['\n'] else "(Ċ Ċ)"
                parsed_string += endchar
                results.append((id, sub_idx, parsed_string))
        except Exception as e:
            logger.info(f"inference error on cpu parser")
            raise e
        result_queue.put(results)

def gpu_worker_proc(gpu_id, task_queue, result_queue, log_queue):
    logger = setup_worker_logger(
        log_queue,
        name=f"gpu_worker_{gpu_id}"
    )
    logger.info("GPU worker started")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    parser = load_model_to_device(device)
    logger.info(f"pid={os.getpid()} gpu={gpu_id} using device {device}")
    while True:
        item = task_queue.get()
        if item is None:
            logger.info(f"gpu{gpu_id} received sentinel, exiting")
            break
        idx, subidxs, endingchars, batch = item
        try:
            examples, charts_np, tag_ids_np = parser.custom_parse_sents(parser, batch)
            result_queue.put((idx, subidxs, endingchars, examples, charts_np, tag_ids_np))
        except Exception as e:
            logger.info(f"inference error on gpu{gpu_id}: {e}")
            raise e
        
def gpu_treer_proc(result_queue, writer_queue, log_queue):
    logger = setup_worker_logger(
        log_queue,
        name=f"gpu_treer"
    )
    logger.info("Treer started")
    local_parser = benepar.Parser("benepar_en3")
    while True:
        item = result_queue.get()
        if item is None:
            logger.info(f"cpu parser received sentinel, exiting")
            break
        idx, subidxs, endingchars, examples, charts_np, tag_ids_np = item
        try:
            results = []
            for id, sub_idx, endchar, inp, output in zip(
                idx, subidxs, endingchars, 
                examples, scores_totree(local_parser._parser, examples, charts_np, tag_ids_np)
            ):
                if inp.tags is not None:
                    output = output.without_predicted_tags()
                tree = output.to_tree(
                    inp.pos(),
                    local_parser._parser.decoder.label_from_index,
                    local_parser._parser.tag_from_index,
                )
                tree = tree[0]
                leaves = tree.leaves()
                if len(leaves) == 1 and re.match(r"\n+", leaves[0]):
                    parsed_string = leaves[0].replace("\n", "(Ċ Ċ) ").rstrip()
                else:
                    parsed_string = tree.pformat(margin=100000) if tree.leaves() != ['\n'] else "(Ċ Ċ)"
                parsed_string += endchar
                results.append((id, sub_idx, parsed_string))
        except Exception as e:
            logger.info(f"inference error on gpu treer")
            raise e
        writer_queue.put(results)


def scores_totree(self, examples, charts_np, tag_ids_np):
    for i in range(len(examples)):
        example_len = len(examples[i].words)
        output = self.decoder.compressed_output_from_chart(charts_np[i])
        if tag_ids_np is not None:
            output = output.with_tags(tag_ids_np[i, 1 : example_len + 1])
        yield output

def writer_proc(result_queue, log_queue, feedback_q, out_path, total_docs, next_expected=0):
    logger = setup_worker_logger(
        log_queue,
        name=f"writer"
    )
    logger.info("writer started")
    buffer = {}  # {doc_idx: out}
    fh = open(out_path, "a+", encoding="utf-8")
    finished_gpu_workers = 0

    while next_expected < total_docs:
        item = result_queue.get()

        if item is None:
            finished_gpu_workers += 1
            if finished_gpu_workers==len(GPU_IDS):
                break
            continue

        # item: list[(idx, sub_idx, out)]
        for idx, sub_idx, out in item:
            if idx not in buffer:
                buffer[idx] = {}
            buffer[idx][sub_idx] = out

        while next_expected in buffer:
            start = 0
            flag = False
            while start in buffer[next_expected]:
                if buffer[next_expected][start][-1]=="\n":
                    flag=True
                    break
                start += 1
            if not flag:
                break
            start = 0
            parse_string = ""
            while True:
                parse_string += buffer[next_expected][start]
                if parse_string[-1]=="\n":
                    break
                start += 1
            fh.write(parse_string)
            del buffer[next_expected]
            next_expected += 1
            feedback_q.put(1)

        if next_expected % 100 == 0:
            fh.flush()

    fh.flush()
    fh.close()
    logger.info("exit")

def main_driver(config,  log_queue, total_docs=None):
    logger = setup_worker_logger(
        log_queue,
        name=f"MAIN"
    )
    logger.info("main started")
    sample_queue = mp.Queue(maxsize=SAMPLE_QUEUE_MAXSIZE)
    task_queue = mp.Queue(maxsize=TASK_QUEUE_MAXSIZE)
    result_queue = mp.Queue(maxsize=RESULT_QUEUE_MAXSIZE)
    writer_queue = mp.Queue(maxsize=RESULT_QUEUE_MAXSIZE)
    ds_feed_q = mp.Queue(maxsize=NUM_CPU_WORKERS * 2)
    feedback_q = mp.Queue(maxsize=100000)
    OUTPUT_FILE, ds = prepare_dataset(config)
    start_index = count_lines_linux_style(OUTPUT_FILE)
    total_docs = len(ds) if total_docs is None else total_docs
    processes = []
    cpu_workers = []
    for _ in range(NUM_CPU_WORKERS):
        p = mp.Process(target=_cpu_worker_process_entry, args=(ds_feed_q, sample_queue, log_queue))
        processes.append(p)
        cpu_workers.append(p)

    feeder_proc = mp.Process(target=_feeder_proc, args=(ds, start_index, ds_feed_q, log_queue, total_docs))
    processes.append(feeder_proc)

    stop_event = mp.Event()
    batcher_p = mp.Process(
        target=batcher_proc,
        args=(sample_queue, task_queue, log_queue, stop_event),
    )
    processes.append(batcher_p)

    gpu_procs = []
    for gid in GPU_IDS:
        p = mp.Process(target=gpu_worker_proc, args=(gid, task_queue, result_queue, log_queue))
        gpu_procs.append(p)
        processes.append(p)

    gpu_tree_workers = []
    for _ in range(NUM_GPU_TREEWORKERS):
        p = mp.Process(target=gpu_treer_proc, args=(result_queue, writer_queue, log_queue))
        gpu_tree_workers.append(p)
        processes.append(p)

    cpu_tree_workers = []
    for _ in range(NUM_CPU_TREEWORKERS):
        p = mp.Process(target=cpu_tree_proc, args=(task_queue, writer_queue, log_queue))
        cpu_tree_workers.append(p)
        processes.append(p)

    writer_p = mp.Process(
        target=writer_proc,
        args=(writer_queue, log_queue, feedback_q, OUTPUT_FILE, total_docs, start_index),
    )
    processes.append(writer_p)
    for p in processes:
        p.start()

    pbar = tqdm(total=total_docs)
    pbar.update(start_index)
    try:
        while True:
            all_finished = True
            for p in processes:
                if p.is_alive():
                    all_finished = False
                elif p.exitcode is not None and p.exitcode != 0:
                    logger.info(f"主进程监控：发现子进程 {p.name} 异常退出 (code: {p.exitcode})")
                    raise Exception("子进程崩溃，触发主进程退出")
            
            if all_finished:
                break
            cpu_finished = True
            for p in cpu_workers:
                if p.is_alive():
                    cpu_finished = False
            if cpu_finished:
                stop_event.set()
            gpu_finished = True
            for p in gpu_procs:
                if p.is_alive():
                    gpu_finished = False
            if gpu_finished:
                for i in range(NUM_GPU_TREEWORKERS):
                    result_queue.put(None)
                
            while True:
                try:
                    id = feedback_q.get_nowait()
                    pbar.update(id)
                except Empty:
                    break
            time.sleep(1)

    except (Exception, KeyboardInterrupt) as e:
        logger.info(e)
        for p in processes:
            if p.is_alive():
                p.terminate() # 强制杀掉还没死的子进程
        raise e

    for p in cpu_workers:
        p.join()
    batcher_p.join()
    # 等待 GPU workers 完成（batcher 会放 sentinel None 给每个 GPU）
    for p in gpu_procs:
        p.join()

    for p in gpu_tree_workers:
        p.join()

    feeder_proc.join()
    while not writer_queue.empty():
        time.sleep(0.1)
    writer_p.join()
    logger.info("All done.")


# 小辅助 wrapper：把 ds_feed_q -> cpu_worker_loop
def _cpu_worker_process_entry(ds_feed_q, sample_queue, log_queue):
    logger = setup_worker_logger(
        log_queue,
        name=f"cpu_worker"
    )
    logger.info("CPU worker started")
    while True:
        item = ds_feed_q.get()
        if item is None:
            break
        index, document = item
        try:
            doc = split_text_into_sents(document)
            split_sents = process_doc_into_maxlen(doc, max_len=SPLIT_MAX_LEN)
            for i, sent in enumerate(split_sents):
                sample_queue.put((index, i, "\n" if i==len(split_sents)-1 else " ", benepar.InputSentence(words=sent)))
        except Exception as e:
            logger.info(f"[cpu_worker] error idx={index}: {e}")
    logger.info("CPU worker finished")

def _feeder_proc(ds_iterator, start_index, ds_feed_q, log_queue, total_docs=None):
    logger = setup_worker_logger(
        log_queue,
        name=f"feeder worker"
    )
    logger.info("feeder worker started")
    max_len = len(ds_iterator)
    if total_docs is not None:
        max_len = min(max_len, total_docs)
    for idx in range(start_index, max_len):
        ds_feed_q.put((idx, ds_iterator[idx]))
    for _ in range(NUM_CPU_WORKERS):
        ds_feed_q.put(None)
    logger.info("feeder worker done")


def main(config):
    mp.set_start_method("spawn", force=True)
    log_queue = mp.Queue(maxsize=10000)
    listener = setup_main_logger(
        log_queue,
        log_file=f"logs/{config}.log"
    )
    main_driver(config, log_queue=log_queue)
    log_queue.put_nowait(None)
    listener.stop()


if __name__ == "__main__":
    main("finewebedu.*-00(172|173|174|175).*arrow")
    # main("AX-g1")