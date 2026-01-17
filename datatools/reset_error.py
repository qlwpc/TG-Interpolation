import os
import json
from filelock import FileLock

def reset_error_tasks(status_file="task_status.json", file_dir=""):
    status_file = os.path.join(file_dir, status_file)
    lock_file = status_file + ".lock"
    lock = FileLock(lock_file)
    
    if not os.path.exists(status_file):
        print(f"错误: 找不到管理文件 {status_file}")
        return

    with lock:
        # 1. 读取当前状态
        with open(status_file, "r") as f:
            state = json.load(f)
        
        error_tasks = state.get("error", [])
        
        if not error_tasks:
            print("没有发现处于 error 状态的任务，无需重置。")
            return
        
        print(f"发现 {len(error_tasks)} 个失败的任务。正在重置...")

        # 2. 提取任务名并放回 todo
        # 注意：之前的脚本中 error 存储的是字典 {"task": "...", "msg": "..."}
        # 我们只需要把其中的任务名取出来
        tasks_to_retry = []
        for item in error_tasks:
            if isinstance(item, dict):
                tasks_to_retry.append(item["task"])
            else:
                tasks_to_retry.append(item) # 兼容性处理

        # 将失败任务添加到 todo 列表的最前面（优先重试）
        state["todo"] = tasks_to_retry + state["todo"]
        
        # 3. 清空 error 列表
        state["error"] = []
        
        # 4. 保存更新后的状态
        with open(status_file, "w") as f:
            json.dump(state, f, indent=4)
            
        print(f"成功！已将 {len(tasks_to_retry)} 个任务重新移至 todo 队列。")
        print("现在你可以重新启动你的主处理脚本了。")

if __name__ == "__main__":
    # 运行重置
    reset_error_tasks(file_dir="/2024233198")