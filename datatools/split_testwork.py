import json

if __name__ == "__main__":

    path = "../dataset/bbc-news/test_index.json"
    with open(path, 'r') as file:
        indices = json.load(file)

    split_sz = 0
    cnt = 0
    for prefix, index in indices.items():
        split_sz += len(index)
        cnt += 1
        if cnt >= 6:
            break

    splits = []
    current = []
    cur_sz = 0
    for prefix, index in indices.items():
        current.append(prefix)
        cur_sz += len(index)
        if cur_sz >= split_sz:
            splits.append(current)
            current = []
            cur_sz=0
    
    if current:
        splits.append(current)
        current = []
    
    for split in splits:
        print(split)