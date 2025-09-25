import os
from tokenizers import Tokenizer

# 配置
folder_path = "../dataset/BLiMP/test300"   # 替换为你的txt文件夹路径

BLiMP_TASK_ANAPHOR_AGR = ["anaphor_gender_agreement", "anaphor_number_agreement"]
BLiMP_TASK_ARG_STRUCTURE = ["animate_subject_passive", "animate_subject_trans", "causative",
                            "drop_argument", "inchoative", "intransitive", "passive_1", "passive_2", "transitive"]
BLiMP_TASK_BINDING = ["principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
                      "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
                      "principle_A_reconstruction"]
BLiMP_TASK_CONTROL_RAISING = ["existential_there_object_raising", "existential_there_subject_raising",
                              "expletive_it_object_raising", "tough_vs_raising_1", "tough_vs_raising_2"]
BLiMP_TASK_DET_NOUN_AGR = ["determiner_noun_agreement_1", "determiner_noun_agreement_2",
                           "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
                           "determiner_noun_agreement_with_adj_2", "determiner_noun_agreement_with_adj_irregular_1",
                           "determiner_noun_agreement_with_adj_irregular_2", "determiner_noun_agreement_with_adjective_1"]
BLiMP_TASK_ELLIPSIS = ["ellipsis_n_bar_1", "ellipsis_n_bar_2"]
BLiMP_TASK_FILLER_GAP = ["wh_questions_object_gap", "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance", 
                         "wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap", 
                         "wh_vs_that_with_gap_long_distance"]
BLiMP_TASK_IRREGULAR_FORMS = ["irregular_past_participle_adjectives", "irregular_past_participle_verbs"]
BLiMP_TASK_ISLAND_EFFECTS = ["adjunct_island", "complex_NP_island", "coordinate_structure_constraint_complex_left_branch",
                             "coordinate_structure_constraint_object_extraction", "left_branch_island_echo_question",
                             "left_branch_island_simple_question", "sentential_subject_island", "wh_island"]
BLiMP_TASK_NPI_LICENSING = ["matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
                            "only_npi_licensor_present", "only_npi_scope", "sentential_negation_npi_licensor_present",
                            "sentential_negation_npi_scope"]
BLiMP_TASK_QUANTIFIERS = ["existential_there_quantifiers_1", "existential_there_quantifiers_2",
                          "superlative_quantifiers_1", "superlative_quantifiers_2"]
BLiMP_TASK_SUBJECT_VERB_AGR = ["distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
                               "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
                               "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2"]
BLiMP_TASK_DICT = {
    "anaphor_agreement" : BLiMP_TASK_ANAPHOR_AGR,
    "argument_structure" : BLiMP_TASK_ARG_STRUCTURE,
    "binding" : BLiMP_TASK_BINDING,
    "control_raising" : BLiMP_TASK_CONTROL_RAISING,
    "determiner_noun_agreement" : BLiMP_TASK_DET_NOUN_AGR,
    "ellipsis" : BLiMP_TASK_ELLIPSIS,
    "filler_gap_dependency" : BLiMP_TASK_FILLER_GAP,
    "irregular_forms" : BLiMP_TASK_IRREGULAR_FORMS,
    "island_effects" : BLiMP_TASK_ISLAND_EFFECTS,
    "npi_licensing" : BLiMP_TASK_NPI_LICENSING,
    "quantifiers" : BLiMP_TASK_QUANTIFIERS,
    "subject_verb_agreement" : BLiMP_TASK_SUBJECT_VERB_AGR, 
}
BLiMP_TASK_LIST = [x for v in BLiMP_TASK_DICT.values() for x in v]

# 初始化tokenizer
tokenizer = Tokenizer.from_file("TG_GPT2_tokenizer.json")

def process_file(file_path):
    print(f"正在处理文件: {os.path.basename(file_path)}")
    max_len = 0
    overall_max = 0
    batch_size = 300
    buffer = []
    result = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            tokens = tokenizer.encode(line).ids
            token_count = len(tokens)
            buffer.append(token_count)

            if i % batch_size == 0:
                batch_max = max(buffer)
                result.append(batch_max)
                # print(f"第 {i // 300} 句最大token数: {batch_max}")
                max_len = max(max_len, batch_max)
                buffer = []

        # 最后一批不足300行的统计
        if buffer:
            batch_max = max(buffer)
            result.append(batch_max)
            # print(f"第 {i-len(buffer)+1}-{i} 行最大token数: {batch_max}")
            max_len = max(max_len, batch_max)

    print("每句最大Token数：", result)
    print(f"文件 {os.path.basename(file_path)} 总体最大token数: {max_len}\n")
    return max_len

def main():
    overall_max = 0
    for filename in BLiMP_TASK_LIST:
        file_path = os.path.join(folder_path, filename + ".txt")
        file_max = process_file(file_path)
        overall_max = max(overall_max, file_max)

    print("="*50)
    print(f"所有文件总体最大token数: {overall_max}")

if __name__ == "__main__":
    main()
