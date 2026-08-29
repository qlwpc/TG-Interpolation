import itertools

def cartesian_product_with_underscore(sets):
    """
    生成多个字符串集合的笛卡尔积，并将结果用下划线连接
    
    参数:
    sets: 包含多个字符串集合的列表
    
    返回:
    一个生成器，产生所有可能的组合（用下划线连接）
    """
    # 使用itertools.product计算笛卡尔积
    for combination in itertools.product(*sets):
        # 将元组中的字符串用下划线连接
        yield '_'.join(combination)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "parse_pretrain_data"))
from convert_TG_and_tokenize import convert_TG_format

# 示例用法
if __name__ == "__main__":

    with open("tmp_out.txt", "r") as file:
        input = "".join(file.readlines())
    print(convert_TG_format(input))
    # 示例字符串集合
    # sets = [
    #     ['ReCoRD'],
    #     ['test', 'val', 'train'],
    #     ['query', 'text'],
    # ]
    
    # # 生成所有组合
    # results = list(cartesian_product_with_underscore(sets))
    
    # # 打印结果
    # for i, result in enumerate(results, 1):
    #     print(f"{result}", end=",")
    # print("")
    # # 输出数量
    # # print(f"\n总共生成了 {len(results)} 种组合")