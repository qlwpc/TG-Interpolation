import os
import json
import time
import logging
import traceback
from filelock import FileLock

# --- 任务管理类 (保持逻辑一致) ---
class TaskManager:
    def __init__(self, task_list, file_dir, status_file="task_status.json"):
        self.status_file = os.path.join(file_dir, status_file)
        self.lock_file = status_file + ".lock"
        self.lock = FileLock(self.lock_file)
        
        if not os.path.exists(self.status_file):
            self._save_state({"todo": task_list, "doing": [], "done": [], "error": []})

    def _load_state(self):
        with open(self.status_file, "r") as f:
            return json.load(f)

    def _save_state(self, state):
        # for key in state.keys():
        #     if key in ["todo", "doing", "done"]:
        #         state[key] = sorted(state[key])
        with open(self.status_file, "w") as f:
            json.dump(state, f, indent=4)

    def get_next_task(self, reverse=False):
        with self.lock:
            state = self._load_state()
            if not state["todo"]: return None
            task = state["todo"].pop(-1 if reverse else 0)
            state["doing"].append(task)
            self._save_state(state)
            return task

    def mark_done(self, task):
        with self.lock:
            state = self._load_state()
            if task in state["doing"]:
                state["doing"].remove(task)
                state["done"].append(task)
                self._save_state(state)

    def mark_error(self, task, error_msg=""):
        with self.lock:
            state = self._load_state()
            if task in state["doing"]:
                state["doing"].remove(task)
                state["error"].append({"task": task, "time": time.ctime(), "msg": str(error_msg)})
                self._save_state(state)


def setup_task_logger(task_name, log_dir="logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    task_name = task_name[1:]
    log_file = os.path.join(log_dir, f"{task_name}.log")
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    handler.setFormatter(formatter)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(handler)
    return logger, handler

def close_task_logger(logger, handler):
    handler.flush()
    handler.close()
    logger.removeHandler(handler)

from worker import main
def worker_process(task_files, file_dir, reverse_order=False, rank=None):
    manager = TaskManager(task_files, file_dir)
    rank = rank if rank is not None else os.getpid()
    print(f"进程 {rank} 启动...")

    while True:
        task = manager.get_next_task(reverse=reverse_order)
        if task is None:
            print(f"进程 {rank}: 任务已全部领完。")
            break
            
        task_id = os.path.splitext(task)[0]
        logger, handler = setup_task_logger(task_id)
        
        logger.info(f"进程 {rank} 开始处理分片: {task}")
        
        try:
            # --- 模拟实际处理逻辑 ---
            # 在这里，你的所有业务代码都应该使用 logger.info/error 记录
            logger.info("正在执行数据加载...") 
            main(f"finewebedu{task}")

            logger.info("数据处理完成，准备保存结果。")
            manager.mark_done(task)
            logger.info("任务执行成功。")
            
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"任务执行崩溃: \n{error_detail}")
            manager.mark_error(task, error_msg=str(e))
            print(f"进程 {rank}: 分片 {task} 失败，详情见日志 {task_id}.log")
            
        finally:
            close_task_logger(logger, handler)

if __name__ == "__main__":
    shards = [f".*-00({i:03d}|{i+1:03d}|{i+2:03d}|{i+3:03d}).*arrow" for i in range(0, 984, 4)]
    file_dir = "./"
    import multiprocessing
    worker_process(shards, file_dir, False)