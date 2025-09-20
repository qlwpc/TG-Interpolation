from datasets import load_dataset
import json
import re

if __name__ == "__main__":
    ds = load_dataset("lighteval/summarization", "xsum")
    ds = ds['train']
    ori_xsum = load_dataset("EdinburghNLP/xsum")
    ori_xsum = ori_xsum["train"]
    doc_to_id = {}
    for case in ori_xsum:
        doc = re.sub(r'[\s\n]+', ' ', case["document"])
        if case["document"][0:len("The warning begins at 22:00")]=="The warning begins at 22:00":
            print(doc)
        doc_to_id[doc.strip()] = case["id"]
    
    train_ids = []
    for doc in ds:
        train_ids.append(doc_to_id[doc["article"].strip()])
    
    with open('save_ids.json', "w+") as file:
        json.dump(train_ids, file)