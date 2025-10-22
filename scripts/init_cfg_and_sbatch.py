"""
Run this to initialize a new training config to a file.
"""
import logging
import sys
import os
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
                                        scripts + "   \\",
                                        str(config_path.resolve())+ "   \\"]
        self.configs = kwargs
        self.configs.pop("--master_port")
        self.configs.pop("--nproc-per-node")
        self.config_path = config_path
    
    def set_param(self, key, value) -> str:
        split = "=" if key[:2]=="--" else " "
        return f"      {split.join([key, str(value)])} \\"

    def __str__(self):
        return "\n".join(self.torchrun + [self.set_param(key, value) for key,value in self.configs.items()])

Device_args = {
    "SIST_A40" :      {"-c": 3, "--mem-per-cpu": 32768, "--partition": "critical", "-A": "tukw-critical", "--exclude": "ai_gpu[26-35]"}, 
    "SIST_TITAN" :    {"-c": 2, "--mem-per-cpu": 16384, "--partition": "critical", "-A": "tukw-critical"},
    "SIST_shanghai" : {"-c": 4, "--mem-per-cpu": 32768, "--partition": "shangHAI", "-A": "tukw-shangHAI"},
    "SIST_normal":    {},
    "RTX3090":        {"-c": 1, "--mem-per-cpu": 1, },
    "A6000" :         {"-c": 1, "--mem-per-cpu": 1, },
    "H800" :          {"-c": 8,},
    "RTX5090":        {}
}

INPUTFORMAT = {
    "terminal": ["terminal"],
    "tree": ["tree", "tree_shuffle", "tree_shuffle_mask"], 
    "tg": ["tg", "mixing", "tgnomask", "tgnomask_aug"]
}

Models = {
    "tg": {"model.transformer_grammar_type": "tg", },
    "tree": {"model.transformer_grammar_type": "tree"},
    "terminal": {"model.transformer_grammar_type": "terminal"},
    "tgnomask": {"model.transformer_grammar_type": "tgnomask"},
    "tgnomask_aug": {"model.transformer_grammar_type": "tgnomask_aug"},
    "tree_shuffle": {"model.transformer_grammar_type": "tree_shuffle"},
    "tree_shuffle_mask": {"model.transformer_grammar_type": "tree_shuffle_mask"},
    "tree_mix_tg" : {"model.transformer_grammar_type": "mixing"},
    "nomask_mix_tg" : {"model.transformer_grammar_type": "mixing"},
}
mixing = {
    "tree_mix_tg" : [TGConfig(grammar_type="tgtree", n_heads=6), TGConfig(grammar_type="tg", n_heads=6)],
    "nomask_mix_tg" : [TGConfig(grammar_type="tg", n_heads=6), TGConfig(grammar_type="tgnomask", n_heads=6)]
}

test_only_params = {"eval_on_load": True, "eval_no_save": True}
finetune_params = {"reset_optimizer_state": True, "reset_trainer_state": True}

train_params = {
    "pretrain_tg": {"global_train_batch_size": 280, "device_train_microbatch_size": 30, "optimizer.learning_rate": 0.0076}, 
    "pretrain_tree": {"global_train_batch_size": 244, "device_train_microbatch_size": 28, "optimizer.learning_rate": 0.007}, 
    "pretrain_terminal": {},
    "pretrain_mix": {"global_train_batch_size": 224, "device_train_microbatch_size": 28, "optimizer.learning_rate": 0.0076}, 
    "xsum_finetune": {"finetune_task": "xsum", "max_duration":"3ep", "global_train_batch_size": 40, "device_train_batch_size":10, "optimizer.learning_rate": 6e-5,
                      "scheduler.t_warmup": 100, "scheduler.min_lr": 1e-6, "device_eval_batch_size": 1, "eval_interval": 1000000, **finetune_params},
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
    "blimp": 4,
    "SG": 2,
}

Evaltasks = {
    "pretrain": [EvaluatorConfig(label="TG-ppl-validation")],
    "docppl": [EvaluatorConfig(label="tg_approx_doc", type=EvaluatorType.tg_doc, device_eval_batch_size=60)],
    "xsum_test": [EvaluatorConfig(label="xsum", type=EvaluatorType.rouge)],
    "xsum_finetune": [EvaluatorConfig(label="xsum", type=EvaluatorType.rouge)],
    "SG": [EvaluatorConfig(label="syntactic_generalization", type=EvaluatorType.downstream)],
    "blimp": [EvaluatorConfig(label="BLiMP", type=EvaluatorType.downstream, device_eval_batch_size=100)],
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
    run_args["--master_port"] = f"1{timestamp.minute:02d}{timestamp.second:02d}"
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
    
    with open(script_filename, 'w+') as f:
        f.write(str(MainContent))
    
    print(f"已生成sbatch脚本: {script_filename}")

def generate_config(save_path: Path, args_list: List[str], Device:str, modelname: str, task: str) -> None:
    default_yaml_path = os.path.expanduser("~/TG-Interpolation/train_configs/terminal.yaml")
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

    if task[:8]=="pretrain":
        Evallist = [EvaluatorConfig(label="TG-ppl-validation"), EvaluatorConfig(label="TG-ppl-validation-test")]
        Evallist[0].data.paths = [f"{workspace}/dataset/bbc-news/" + input_format + "/dev.npy"]
        Evallist[1].data.paths = [f"{workspace}/dataset/bbc-news/" + input_format + "/test.npy"]
        Evallist[0].data.pin_memory = Evallist[1].data.pin_memory = True
        Evallist[0].data.generate_doc_lengths = Evallist[1].data.generate_doc_lengths = (input_format!="tg")
        Evaltasks[task] = Evallist
        cfg.model.flex_attention = True
    elif task=="docppl":
        Evaltasks["docppl"][0].label = "tg_approx_doc" if input_format=="tg" else "txl_approx_doc"
    
    if cfg.model.transformer_grammar_type=="mixing":
        cfg.model.mix_head_type = mixing[modelname]

    cfg.evaluators = Evaltasks[task]
    cfg.device_eval_batch_size = "${device_train_microbatch_size}"
    cfg.wandb.name = "${run_name}"
    log.info("Configuration:")
    log.info(cfg)
    cfg.save(save_path)
    log.info(f"Config saved to {save_path}")



if __name__ == "__main__":
    prepare_cli_environment()

    try:
        save_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [SAVE_PATH] [OPTIONS]")
    Device = "RTX3090"
    modelname = "tree_mix_tg"
    task = "xsum_finetune"
    run_name = "tgtree_mix_tg_xsum"
    load_path = "/home/wangpch/TG-Interpolation/saved_models/tgtree_mix_tg_pretrain/step69817-unsharded"
    generate_config(Path(save_path), [clean_opt(s) for s in args_list], Device=Device, modelname=modelname, task=task)
    generate_sbatch_content(config_path=Path(save_path), Device=Device, modelname=modelname, task=task, run_name=run_name, load_path=load_path)
