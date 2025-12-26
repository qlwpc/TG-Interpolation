"""
Run this to initialize a new training config to a file.
"""
import logging
import sys
import os
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime

sys.path.append(os.path.expanduser("~/TG-Interpolation"))

from olmo.config import TrainConfig, EvaluatorConfig, EvaluatorType, TGConfig
from olmo.exceptions import OLMoCliError
from olmo.util import clean_opt, prepare_cli_environment

log = logging.getLogger(__name__)

class BashCL:
    def __init__(self):
        self.Commands = ["#!/bin/bash"]

    def add_commands(self, command):
        self.Commands.append(command)

    def __str__(self):
        return "\n".join([str(obj) for obj in self.Commands])

class SBATCH:
    def __init__(self, **kwargs):
        self.configs = kwargs
    
    def set_param(self, key, value) -> str:
        split = "=" if key[:2]=="--" else " "
        return f"#SBATCH {split.join([key, str(value)])}"

    def __str__(self):
        return "\n".join([self.set_param(key, value) for key,value in self.configs.items()])

class TORCHRUN:
    def __init__(self, config_path, scripts="scripts/train.py", **kwargs):
        self.torchrun = ["torchrun  \\", self.set_param("--master_port", kwargs["--master_port"]), 
                                        self.set_param("--nproc-per-node", kwargs["--nproc-per-node"]),
                                        "    " + scripts + " \\",
                                        "    " + str(config_path.resolve())+ " \\"]
        self.configs = kwargs
        self.configs.pop("--master_port")
        self.configs.pop("--nproc-per-node")
        self.config_path = config_path
    
    def set_param(self, key, value) -> str:
        split = "=" if key[:2]=="--" else " "
        return f"    {split.join([key, str(value)])} \\"

    def __str__(self):
        return "\n".join(self.torchrun + [self.set_param(key, value) for key,value in self.configs.items()])

Device_args = {
    "SIST_A40" :      {"-c": 3, "--mem-per-cpu": 32768, "--partition": "critical", "-A": "tukw-critical", "--exclude": "ai_gpu[26-33]"}, 
    "SIST_TITAN" :    {"-c": 2, "--mem-per-cpu": 16384, "--partition": "critical", "-A": "tukw-critical"},
    "SIST_shanghai" : {"-c": 4, "--mem-per-cpu": 32768, "--partition": "ShangHAI", "-A": "tukw-ShangHAI"},
    "SIST_normal":    {},
    "RTX3090":        {"-c": 1, "--mem-per-cpu": 1, },
    "A6000" :         {"-c": 1, "--mem-per-cpu": 1, },
    "H800" :          {"-c": 8,},
    "RTX5090":        {"-c": 8, "--partition": "gpu"}
}

INPUTFORMAT = {
    "terminal": ["terminal", "pause1/2", "pause1/2_label"],
    "tree": ["tree", "tree_shuffle", "tree_shuffle_mask"], 
    "tg": ["tg", "mixing", "tgnomask", "tgnomask_aug", "tgtree"]
}

Models = {
    "tg": {"model.transformer_grammar_type": "tg", },
    "tree": {"model.transformer_grammar_type": "tree"},
    "tree-500M": {"model.transformer_grammar_type": "tree"},
    "tree-100M-early" : {"model.transformer_grammar_type": "tree"},
    "tree-1B" : {"model.transformer_grammar_type": "tree"},
    "terminal": {"model.transformer_grammar_type": "terminal"},
    "terminal-1B": {"model.transformer_grammar_type": "terminal"},
    "terminal-500M": {"model.transformer_grammar_type": "terminal"},
    "terminal-100M-early" : {"model.transformer_grammar_type": "terminal"},
    "tgtree": {"model.transformer_grammar_type": "tgtree"},
    "tgtree-500M": {"model.transformer_grammar_type": "tgtree"},
    "tgtree-100M-early": {"model.transformer_grammar_type": "tgtree"},
    "tgnomask_aug-500M": {"model.transformer_grammar_type": "tgnomask_aug"},
    "tgnomask_aug-100M-early": {"model.transformer_grammar_type": "tgnomask_aug"},
    "tgnomask": {"model.transformer_grammar_type": "tgnomask"},
    "tgnomask_aug": {"model.transformer_grammar_type": "tgnomask_aug"},
    "tree_shuffle": {"model.transformer_grammar_type": "tree_shuffle"},
    "tree_shuffle_mask": {"model.transformer_grammar_type": "tree_shuffle_mask"},
    "tree_mix_tg" : {"model.transformer_grammar_type": "mixing"},
    "nomask_mix_tg" : {"model.transformer_grammar_type": "mixing"},
    "pause2048": {"model.transformer_grammar_type": "pause1/2"},
    "pause4096": {"model.transformer_grammar_type": "pause1/2", "model.max_sequence_length": 4096},
    "pauselabel2048": {"model.transformer_grammar_type": "pause1/2_label"},
    "pauselabel4096": {"model.transformer_grammar_type": "pause1/2_label", "model.max_sequence_length": 4096},
    "terminal1024" : {"model.transformer_grammar_type": "terminal", "model.max_sequence_length": 1024},
}
mixing = {
    "tree_mix_tg" : [TGConfig(grammar_type="tgtree", n_heads=6), TGConfig(grammar_type="tg", n_heads=6)],
    "nomask_mix_tg" : [TGConfig(grammar_type="tg", n_heads=6), TGConfig(grammar_type="tgnomask", n_heads=6)]
}

test_only_params = {"eval_on_load": True, "eval_no_save": True}
finetune_params = {"reset_optimizer_state": True, "reset_trainer_state": True, "eval_interval": 1000000}

train_params = {
    "pretrain_tg": {"global_train_batch_size": 280, "device_train_microbatch_size": 30, "optimizer.learning_rate": 0.0076}, 
    "pretrain_tree": {"global_train_batch_size": 244, "device_train_microbatch_size": 28, "optimizer.learning_rate": 0.007}, 
    "pretrain_terminal": {},
    "pretrain_mix": {"global_train_batch_size": 224, "device_train_microbatch_size": 28, "optimizer.learning_rate": 0.0076}, 
    "xsum_finetune": {"finetune_task": "xsum", "max_duration":"3ep", "global_train_batch_size": 40, "device_train_microbatch_size":10, "optimizer.learning_rate": 6e-5,
                      "scheduler.t_warmup": 100, "scheduler.min_lr": 1e-6, "device_eval_batch_size": 1, "eval_interval": 1000000, **finetune_params},
    "boolq": {"finetune_task": "boolq", "optimizer.learning_rate": 3.0e-4,  "scheduler.t_warmup": 100, "scheduler.min_lr": 1e-6,  "max_duration": "5ep",
               "global_train_batch_size": 40, "device_train_microbatch_size":10,  **finetune_params},
    "cb": {"finetune_task": "cb", "optimizer.learning_rate": 3.0e-4,  "scheduler.t_warmup": 100, "scheduler.min_lr": 1e-5,  "max_duration": "3ep",
               "global_train_batch_size": 40, "device_train_microbatch_size":10,  **finetune_params},
    "copa": {"finetune_task": "copa", "optimizer.learning_rate": 5.0e-4, "optimizer.weight_decay": 0.1,  "scheduler.t_warmup": 50, "scheduler.min_lr": 2e-5,  "max_duration": "10ep",
               "global_train_batch_size": 40, "device_train_microbatch_size":10,  **finetune_params},
    "multirc": {"finetune_task": "multirc", "optimizer.learning_rate": 3.0e-4,  "scheduler.t_warmup": 100, "scheduler.min_lr": 1e-5,  "max_duration": "3ep",
               "global_train_batch_size": 40, "device_train_microbatch_size":10,  **finetune_params},
    "record": {"finetune_task": "record", "optimizer.learning_rate": 3.0e-4,  "scheduler.t_warmup": 100, "scheduler.min_lr": 1e-5,  "max_duration": "3ep",
               "global_train_batch_size": 40, "device_train_microbatch_size":10,  **finetune_params},
    "rte": {"finetune_task": "rte", "optimizer.learning_rate": 3.0e-4,  "scheduler.t_warmup": 100, "scheduler.min_lr": 1e-5,  "max_duration": "3ep",
               "global_train_batch_size": 40, "device_train_microbatch_size":10,  **finetune_params},
    "wic": {"finetune_task": "wic", "optimizer.learning_rate": 3.0e-4,  "scheduler.t_warmup": 100, "scheduler.min_lr": 1e-5,  "max_duration": "1ep",
               "global_train_batch_size": 40, "device_train_microbatch_size":10,  **finetune_params},
    "wsc": {"finetune_task": "wsc", "optimizer.learning_rate": 3.0e-4,  "scheduler.t_warmup": 100, "scheduler.min_lr": 1e-5,  "max_duration": "5ep",
               "global_train_batch_size": 40, "device_train_microbatch_size":10,  **finetune_params},
    "wic": {"finetune_task": "wic", "optimizer.learning_rate": 3.0e-4,  "scheduler.t_warmup": 100, "scheduler.min_lr": 1e-5,  "max_duration": "1ep",
               "global_train_batch_size": 40, "device_train_microbatch_size":10,  **finetune_params},
    "wsc": {"finetune_task": "wsc", "optimizer.learning_rate": 3.0e-4,  "scheduler.t_warmup": 100, "scheduler.min_lr": 1e-5,  "max_duration": "5ep",
               "global_train_batch_size": 40, "device_train_microbatch_size":10,  **finetune_params},
    "hellaswag": {**test_only_params},
    "winogrande": {**test_only_params},
    "docppl": {**test_only_params,},
    "xsum_test": {**test_only_params,},
    "blimp": {**test_only_params, },
    "SG": {**test_only_params, },
}

GPU_tasks = {
    "pretrain_tg": 8,
    "pretrain_mix": 8,
    "pretrain_tree": 4,
    "pretrain_terminal": 4,
    "docppl": 1,
    "xsum_finetune": 4,
    "xsum_test": 4,
    "blimp": 2,
    "SG": 2,
    "boolq": 4,
    "cb": 2,
    "copa": 2,
    "multirc": 2,
    "record": 2,
    "rte": 2,
    "wic": 2,
    "wsc": 2,
    "hellaswag": 4,
    "winogrande": 2,
}

Evaltasks = {
    "pretrain": [EvaluatorConfig(label="TG-ppl-validation")],
    "docppl": [EvaluatorConfig(label="tg_approx_doc", type=EvaluatorType.tg_doc, device_eval_batch_size=60)],
    "xsum_test": [EvaluatorConfig(label="xsum", type=EvaluatorType.rouge)],
    "xsum_finetune": [EvaluatorConfig(label="xsum", type=EvaluatorType.rouge)],
    "SG": [EvaluatorConfig(label="syntactic_generalization", type=EvaluatorType.downstream)],
    "blimp": [EvaluatorConfig(label="BLiMP", type=EvaluatorType.downstream, device_eval_batch_size=100)],
    "boolq": [EvaluatorConfig(label="boolq", type=EvaluatorType.downstream)],
    "cb": [EvaluatorConfig(label="cb", type=EvaluatorType.downstream)],
    "copa": [EvaluatorConfig(label="copa", type=EvaluatorType.downstream)],
    "multirc": [EvaluatorConfig(label="multirc", type=EvaluatorType.downstream)],
    "record": [EvaluatorConfig(label="record", type=EvaluatorType.downstream)],
    "rte": [EvaluatorConfig(label="rte", type=EvaluatorType.downstream)],
    "wic": [EvaluatorConfig(label="wic", type=EvaluatorType.downstream)],
    "wsc": [EvaluatorConfig(label="wsc", type=EvaluatorType.downstream)],
    "hellaswag": [EvaluatorConfig(label="hellaswag", type=EvaluatorType.downstream, device_eval_batch_size=5)],
    "winogrande": [EvaluatorConfig(label="winogrande", type=EvaluatorType.downstream, device_eval_batch_size=5)],
}

def generate_sbatch_content(config_path:Path, Device:str, modelname:str, task:str, run_name:str, load_path:Optional[str]=None, DEBUG=None):
    timestamp : datetime = datetime.now()
    sbatch_args = {"-N" : 1, **Device_args[Device]}
    run_args = {"--run_name": "${run_name}", 
                "--workspace" : "${workspace}"}

    input_format = None
    for form, grammar in INPUTFORMAT.items():
        if modelname in grammar:
            input_format = form
            break
    n_tasks = GPU_tasks[task]

    sbatch_args["-n"] = n_tasks
    sbatch_args["-t"] = "120:00:00"
    sbatch_args["--gres"] = f"gpu:{n_tasks}"
    run_args["--nproc-per-node"] = n_tasks
    run_args["--master_port"] = f"{timestamp.microsecond % 10000 + 10000}"
    run_args["--save_folder"] = "${workspace}/saved_models/${run_name}" if task[:8]=="pretrain" else "${workspace}/saved_models/test_models/${run_name}"
    if load_path is not None:
        run_args["--load_path"] = load_path

    MainContent = BashCL()
    MainContent.add_commands(SBATCH(**sbatch_args))
    
    if DEBUG is not None:
        debug_configs = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
        MainContent.add_commands(debug_configs)

    default_commands = f"""
workspace=${{HOME}}/TG-Interpolation
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=${{PYTHONPATH}}:${{workspace}}

nvidia-smi
wandb offline
cd ${{workspace}}
run_name={run_name}
"""
    
    MainContent.add_commands(default_commands)
    if Device in ["H800"]:
        tar_data = f"""
date
tar -xvf dataset.tar -C /dev/shm
date
"""
        MainContent.add_commands(tar_data)


    MainContent.add_commands(TORCHRUN(config_path=config_path, **run_args))
    script_filename = f"{run_name}.sh"
    script_filename = os.path.join(os.getcwd(), "run_scripts", modelname, script_filename)
    with open(script_filename, 'w+') as f:
        f.write(str(MainContent))
    
    print(f"已生成sbatch脚本: {script_filename}")

def generate_config(save_path: Path, args_list: List[str], Device:str, modelname: str, task: str) -> None:
    default_yaml_path = os.path.expanduser("~/TG-Interpolation/train_configs/terminal.yaml")
    if modelname[-2:] == "1B":
        default_yaml_path = os.path.expanduser("~/TG-Interpolation/train_configs/terminal-1B.yaml")
    elif modelname[-4:] == "500M":
        default_yaml_path = os.path.expanduser("~/TG-Interpolation/train_configs/terminal-500M.yaml")
    override_args = {
        **Models[modelname],
        **train_params[task],
    }
    override_args = [f"{key}={value}" for key, value in override_args.items()]
    args_list = args_list + override_args
    cfg = TrainConfig.load(default_yaml_path, args_list)
    
    workspace = "${workspace}" if Device not in ["H800"] else "/dev/shm"

    cfg.tokenizer.vocabulary = cfg.tokenizer.identifier = f"{workspace}/dataset/bbc-news/TG_GPT2_tokenizer.json"
    modelConfig = Models[modelname]
    input_format = None
    for form, grammar in INPUTFORMAT.items():
        if modelConfig["model.transformer_grammar_type"] in grammar:
            input_format = form
            break
    train_path = f"{workspace}/dataset/bbc-news/" + input_format + "/train.npy"
    cfg.data.paths[0] = train_path

    if task[:8]=="pretrain" or task=="docppl" and input_format == "terminal":
        Evallist = [EvaluatorConfig(label="TG-ppl-validation"), EvaluatorConfig(label="TG-ppl-validation-test")]
        Evallist[0].data.paths = [f"{workspace}/dataset/bbc-news/" + input_format + "/dev.npy"]
        Evallist[1].data.paths = [f"{workspace}/dataset/bbc-news/" + input_format + "/test.npy"]
        Evallist[0].data.pin_memory = Evallist[1].data.pin_memory = True
        Evallist[0].data.generate_doc_lengths = Evallist[1].data.generate_doc_lengths = (input_format!="tg")
        Evaltasks[task] = Evallist
        cfg.model.flex_attention = True
    elif task=="docppl":
        Evaltasks["docppl"][0].label = "tg_approx_doc" if input_format=="tg" else "txl_approx_doc"
    elif task=="blimp":
        if cfg.model.transformer_grammar_type[:8] in ["terminal", "pause1/2"]:
            Evaltasks["blimp"][0].device_eval_batch_size = 100
        else:
            Evaltasks["blimp"][0].device_eval_batch_size = 150

    if Device == "SIST_TITAN":
        cfg.model.flash_attention = False

    if cfg.model.transformer_grammar_type=="mixing":
        cfg.model.mix_head_type = mixing[modelname]

    cfg.evaluators = Evaltasks[task]
    cfg.device_eval_batch_size = "${device_train_microbatch_size}"
    cfg.wandb.name = "${run_name}"
    log.info("Configuration:")
    log.info(cfg)
    cfg.save(save_path)
    log.info(f"Config saved to {save_path}")

model_paths = {
    "tree_mix_tg" : "/saved_models/tgtree_mix_tg_pretrain/step69817-unsharded",
    "terminal": "/saved_models/Terminal-lr005-bs144/step34115-unsharded", 
    "tgtree": "/saved_models/TGtree/step69817-unsharded",
    "tgnomask_aug": "/saved_models/TGnomask_aug_pretrain/step55853-unsharded",
    "tgnomask": "/saved_models/nomask_test/step55853-unsharded",
    "nomask_mix_tg": "/saved_models/TG_mix_nomask_bs240_lr0076/step69817-unsharded",
    "tg": "/saved_models/TG_test/step55457-unsharded",
    "tree": "/saved_models/Tree_test/step49440-unsharded",
    "tree_shuffle": "/saved_models/Tree_shuffle_pretrain/step49440-unsharded",
    "tree_shuffle_mask": "/saved_models/treeshufflemask_pretrain/step49440-unsharded",
    "terminal-1B": "/saved_models/terminal_1B/step34115-unsharded",
    "tree-1B": "/saved_models/Tree_1B/step49440-unsharded",
    "pause2048" : "/saved_models/pause_pretrain/step40267-unsharded",
    "pauselabel2048": "/saved_models/pause_2labels_2048/step40267-unsharded",
    "pause4096" : "/saved_models/pause_pretrain_4096/step40938-unsharded",
    "pauselabel4096": "/saved_models/pause_2labels_4096/step40938-unsharded",
    "terminal1024": "/saved_models/terminal_100M_1024/step34115-unsharded",
    "terminal-500M" : "/saved_models/terminal_500M/step34115-unsharded",
    "terminal-100M-early": "/saved_models/terminal_100M_early/step14425-unsharded",
    "tree-500M" : "/saved_models/Tree_500M/step49440-unsharded",
    "tree-100M-early" : "/saved_models/Tree_100M_early/step19233-unsharded",
    "tgtree-500M" : "/saved_models/TGTree_500M/step55853-unsharded",
    "tgnomask_aug-500M" : "/saved_models/TGnomaskaug_500M/step55853-unsharded",
    "tgnomask_aug-100M-early": "/saved_models/TGnomaskaug_100M_early/step21637-unsharded",
    "tgtree-100M-early": "/saved_models/TGTree_100M_early/step21637-unsharded",
}



def robust_directory_check(directory_path, script_path="~/TG-Interpolation/datatools/rsync.sh"):
    directory_path = os.path.abspath(directory_path)
    if os.path.isdir(directory_path):
        return True
    print("✗ 目标目录不存在，准备rsync...")
    script_path = os.path.expanduser(script_path)
    dirs = directory_path.split("/")
    command = ["bash", script_path] + dirs[-2:]
    try:
        env = os.environ.copy()
        env['TARGET_DIR'] = directory_path
        result = subprocess.Popen(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )
        result.wait()
        print("✓ 执行成功")
        if result.stdout:
            print(f"脚本输出:\n{result.stdout}")
        if os.path.isdir(directory_path):
            print("✓ 目录创建成功")
            return True
        else:
            print("✗ 脚本执行成功但目录仍未创建")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ 脚本执行失败 (返回码: {e.returncode})")
        print(f"错误信息: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ 脚本文件不存在: {script_path}")
        return False
    except subprocess.TimeoutExpired:
        print("✗ 脚本执行超时")
        return False


class TaskInfo:
    def __init__(self, modelname, run_name, load_path):
        self.modelname = modelname
        self.run_name = run_name
        self.load_path = load_path
    
    def check_model_path(self):
        robust_directory_check(self.load_path)
    


if __name__ == "__main__":
    prepare_cli_environment()
    args_list = []
    # try:
    #     save_path, args_list = sys.argv[1], sys.argv[2:]
    # except IndexError:
    #     raise OLMoCliError(f"Usage: {sys.argv[0]} [SAVE_PATH] [OPTIONS]")
    Device = "RTX3090"
    modelname = "terminal-1B"
    # task = ["xsum_finetune", "boolq", "rte"]
    # task += ["docppl"]
    # task += ["hellaswag"]
    task = ["winogrande"]
    load_path = True
    if load_path is not None and load_path!=False:
        load_path = os.path.expanduser("~/TG-Interpolation" + model_paths[modelname])
        robust_directory_check(load_path)
    
    save_dir = os.path.join(os.getcwd(), "run_scripts", modelname)
    os.makedirs(save_dir, exist_ok=True)
    for pertask in task:
        run_name = f"{pertask}_test"
        save_path = os.path.join(save_dir, f"config_{run_name}.yaml")
        generate_config(Path(save_path), [clean_opt(s) for s in args_list], Device=Device, modelname=modelname, task=pertask)
        generate_sbatch_content(config_path=Path(save_path), Device=Device, modelname=modelname, task=pertask, run_name=run_name, load_path=load_path)
