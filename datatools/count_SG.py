import re

test_suite_dict = {
    "Agreement" : ["number_orc", "number_prep", "number_src"], 
    "Center_Embedding" : ["center_embed", "center_embed_mod"],
    "Garden_Path_Effects" : ["mvrr", "mvrr_mod", "npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod"],
    "Gross_Syntactic_Expectation" : ["subordination", "subordination_orc-orc", "subordination_pp-pp", "subordination_src-src"],
    "Licensing" : ["npi_orc_any", "npi_orc_ever", "npi_src_any", "npi_src_ever", \
            "reflexive_orc_fem", "reflexive_orc_masc", "reflexive_prep_fem", "reflexive_prep_masc", "reflexive_src_fem", "reflexive_src_masc"],
    "Long_Distance_Dependencies" : ["fgd_subject", "fgd_object", "fgd_pp", "fgd-embed3", "fgd-embed4", "fgd_hierarchy", "cleft", "cleft_modifier"],
    # "nn-nv-rpl" : ["nn-nv-rpl"] # extra test in SG but not in test suites
}

subtask_to_category = {}
for category, subtasks in test_suite_dict.items():
    for subtask in subtasks:
        subtask_to_category[subtask] = category

# 示例日志文本（替换为您的完整日志内容）
log_text = """
2025-11-25 01:08:03.509	rtx3090:0	olmo.train:1196	INFO	[eval_step=35/-1]
task is cleft score is {'np_mismatch': 19.251331329345703, 'np_match': 15.797313690185547, 'vp_match': 23.626144409179688, 'vp_mismatch': 28.57933807373047}
result is True
task is cleft score is {'np_mismatch': 12.306087493896484, 'np_match': 10.57159423828125, 'vp_match': 14.541145324707031, 'vp_mismatch': 19.319538116455078}
result is True
2025-11-25 01:08:37.323	rtx3090:0	olmo.train:1196	INFO	[eval_step=36/-1]
task is cleft score is {'np_mismatch': 16.61956024169922, 'np_match': 14.129440307617188, 'vp_match': 20.68157958984375, 'vp_mismatch': 25.966663360595703}
result is True
"""

with open("./run_folder/tgnomask_aug-100M-early/SGresult.txt", 'r') as file:
    log_text = "".join(file.readlines())
    print(log_text)

# 解析日志文本，提取 (task, result) 对
pattern = r"task is (\w+) score is \{.*?\}\nresult is (True|False)"
matches = re.findall(pattern, log_text)

instances = []
for match in matches:
    task_name = match[0]
    result_str = match[1]
    result_bool = result_str == 'True'
    instances.append((task_name, result_bool))

# 按分类分组结果
category_results = {}
for category in test_suite_dict:
    category_results[category] = []

for task_name, result in instances:
    if task_name in subtask_to_category:
        category = subtask_to_category[task_name]
        category_results[category].append(result)
    else:
        print(f"Warning: task '{task_name}' not found in test_suite_dict")

# 计算每个分类的accuracy
category_accuracy = {}
for category, results in category_results.items():
    if len(results) > 0:
        accuracy = sum(results) / len(results)
        category_accuracy[category] = accuracy
    else:
        category_accuracy[category] = None

SG = 0.0
cnt = 0
# 打印结果
for category, acc in category_accuracy.items():
    if acc is not None:
        print(f"Category {category}: accuracy = {acc*100:.2f}")
        SG += acc
        cnt += 1
    else:
        print(f"Category {category}: No data")

SG /= cnt
print(f"overall SG = {SG*100:.2f}")