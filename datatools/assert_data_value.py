import numpy as np
import os
from joblib import Parallel, delayed

dir = ["terminal", "tree", "tg"]
base = "/storage/wangpch"

def check_file(name):
    arr = np.load(name)
    mx = np.max(arr)
    mi = np.min(arr)
    print(f"infile {form} {file}: max={mx} min={mi}")
    assert (mi >=0 and mx <151732)

process_list = []
for form in dir:
    cwd = os.path.join(base, form)
    for file in sorted(os.listdir(cwd)):
        filename = os.path.join(cwd, file)
        process_list.append(filename)



Parallel(n_jobs=16)(delayed(check_file)(name) for name in process_list)