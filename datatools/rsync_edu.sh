#!/bin/bash

#/home/wangpch/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu/sample-100BT/0.0.0/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
#rsync -avzP ~/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu/sample-100BT/0.0.0/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/ SIST:/public/home/wangpch/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu/sample-100BT/0.0.0/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
sftp -P 22112 2024233198@10.15.171.204

#rsync -e 'ssh -p 22112 2024233198@10.15.171.204' ~/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu/sample-100BT/0.0.0/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/ /mnt/inaisfs/user-fs/2024233198/fineweb-edu/