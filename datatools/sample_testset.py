from datasets import load_dataset
import random
from collections import defaultdict
import numpy as np
import json

data = {
'CC-MAIN-2013-20': 179829, 
'CC-MAIN-2013-48': 394978, 
'CC-MAIN-2014-10': 405967,
'CC-MAIN-2014-15': 295272,
'CC-MAIN-2014-23': 409959,
'CC-MAIN-2014-35': 410024,
'CC-MAIN-2014-41': 434595,
'CC-MAIN-2014-42': 266235,
'CC-MAIN-2014-49': 208198,
'CC-MAIN-2014-52': 426622,
'CC-MAIN-2015-06': 371702,
'CC-MAIN-2015-11': 339287,
'CC-MAIN-2015-14': 277087,
'CC-MAIN-2015-18': 413268,
'CC-MAIN-2015-22': 436181,
'CC-MAIN-2015-27': 358885,
'CC-MAIN-2015-32': 422596,
'CC-MAIN-2015-35': 419640,
'CC-MAIN-2015-40': 265078,
'CC-MAIN-2015-48': 398074,
'CC-MAIN-2016-07': 385458,
'CC-MAIN-2016-18': 300127,
'CC-MAIN-2016-22': 304515,
'CC-MAIN-2016-26': 346842,
'CC-MAIN-2016-30': 404091,
'CC-MAIN-2016-36': 400199,
'CC-MAIN-2016-40': 405529,
'CC-MAIN-2016-44': 380138,
'CC-MAIN-2016-50': 357533,
'CC-MAIN-2017-04': 346540,
'CC-MAIN-2017-09': 342052,
'CC-MAIN-2017-13': 321266,
'CC-MAIN-2017-17': 297048,
'CC-MAIN-2017-22': 226857,
'CC-MAIN-2017-26': 187329,
'CC-MAIN-2017-30': 176278,
'CC-MAIN-2017-34': 155458,
'CC-MAIN-2017-39': 141214,
'CC-MAIN-2017-43': 112349,
'CC-MAIN-2017-47': 99589,
'CC-MAIN-2017-51': 43473,
'CC-MAIN-2018-05': 56428,
'CC-MAIN-2018-09': 77457,
'CC-MAIN-2018-13': 71439,
'CC-MAIN-2018-17': 39604,
'CC-MAIN-2018-22': 34768,
'CC-MAIN-2018-26': 64213,
'CC-MAIN-2018-30': 72027,
'CC-MAIN-2018-34': 64859,
'CC-MAIN-2018-39': 128624,
'CC-MAIN-2018-43': 112320,
'CC-MAIN-2018-47': 104023,
'CC-MAIN-2018-51': 93310,
'CC-MAIN-2019-04': 71865,
'CC-MAIN-2019-09': 73431,
'CC-MAIN-2019-13': 46914,
'CC-MAIN-2019-18': 55300,
'CC-MAIN-2019-22': 65535,
'CC-MAIN-2019-26': 61755,
'CC-MAIN-2019-30': 55531,
'CC-MAIN-2019-35': 56957,
'CC-MAIN-2019-39': 57581,
'CC-MAIN-2019-43': 59253,
'CC-MAIN-2019-47': 38378,
'CC-MAIN-2019-51': 47298,
'CC-MAIN-2020-05': 35457,
'CC-MAIN-2020-10': 34828,
'CC-MAIN-2020-16': 27325,
'CC-MAIN-2020-24': 23966,
'CC-MAIN-2020-29': 24457,
'CC-MAIN-2020-34': 45084,
'CC-MAIN-2020-40': 27196,
'CC-MAIN-2020-45': 38394,
'CC-MAIN-2020-50': 40962,
'CC-MAIN-2021-04': 38669,
'CC-MAIN-2021-10': 46915,
'CC-MAIN-2021-17': 45753,
'CC-MAIN-2021-21': 26805,
'CC-MAIN-2021-25': 27329,
'CC-MAIN-2021-31': 19897,
'CC-MAIN-2021-39': 38794,
'CC-MAIN-2021-43': 35957,
'CC-MAIN-2021-49': 44092,
'CC-MAIN-2022-05': 46145,
'CC-MAIN-2022-21': 42912,
'CC-MAIN-2022-27': 48358,
'CC-MAIN-2022-33': 59444,
'CC-MAIN-2022-40': 12565,
'CC-MAIN-2022-49': 16197,
'CC-MAIN-2023-06': 11402,
'CC-MAIN-2023-14': 10650,
'CC-MAIN-2023-23': 11918,
'CC-MAIN-2023-40': 11210,
'CC-MAIN-2023-50': 21448
}
def extract_article(file_path, article_index):
    """
    高效提取单篇文章（使用内存映射避免全文件加载）
    """
    # 内存映射方式加载大文件
    tokens = np.load(file_path, mmap_mode='r')
    bos_positions = np.where(tokens == 50257)[0]
    ret_list = []
    for index in article_index:
        start = bos_positions[index]
        next_bos = bos_positions[index+1] if index+1 < len(bos_positions) else len(tokens)
        segment = tokens[start:next_bos]
        ret_list.append(segment)
        
    return ret_list

if __name__ == "__main__":
    
    # 计算总文章数 (3,892,610篇)
    total_articles = sum(data.values())
    print(f"总训练集: {total_articles}篇")
    # 设置测试集比例 (10%)
    size_split = [6000, 10000, 20000, 30000, 40000, 100000, 200000]
    ratio_split = [x/total_articles for x in size_split]
    test_labels = [defaultdict(list) for x in size_split]
    random.seed(42)
    test_ratio = size_split[-1] / total_articles
    test_size = int(total_articles * test_ratio)  # 目标测试集大小: 389,261篇
    # 生成测试集标签


    for prefix, num_articles in data.items():
        # 计算当前文件应抽取的文章数 (至少1篇)
        n_test = max(1, round(num_articles * test_ratio))
        
        # 随机选择文章索引 (避免连续样本)
        indices = random.sample(range(num_articles), n_test)
        for i in range(len(size_split)):
            split_n_test = max(1, round(num_articles * ratio_split[i]))
            test_labels[i][prefix] = indices[:split_n_test]

    # 实际测试集大小
        # actual_test_size = sum(len(v) for v in test_labels.values())
        # print(f"目标测试集: {test_size}篇, 实际测试集: {actual_test_size}篇")
    test_list = [[] for x in size_split]
    for prefix, num_articles in data.items():
        ret = extract_article("bbc_tokenized/terminal/" + prefix + ".npy", test_labels[-1][prefix])
        for i in range(len(size_split)):
            split_n_test = max(1, round(num_articles * ratio_split[i]))
            test_list[i].extend(ret[:split_n_test])
    
    for i in range(len(size_split)):
        test_data = np.concatenate(test_list[i], axis=0)
        np.save(f"bbc_tokenized/terminal/test_{size_split[i]}.npy", test_data)
        with open(f"bbc_tokenized/terminal/test_{size_split[i]}_index.json", "w+") as output:
            json.dump(test_labels[i], output, indent=None)