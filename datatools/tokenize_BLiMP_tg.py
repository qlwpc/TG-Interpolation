from tokenizers import Tokenizer
from tqdm import tqdm
import numpy as np
import os
import sys
from joblib import Parallel, delayed
import argparse
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from olmo.data.tg_mask import SentencepieceVocab

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
# BLiMP_TASK_LIST = ["aaa"]

def convert_treenpy_to_TG(tree, vocab):
    T = tree.shape[0]
    for token in tree:
        if vocab.is_closing_non_terminal(token):
            T += 1
    TG = np.zeros((T, ), tree.dtype)
    i = 0
    for token in tree:
        TG[i] = token
        i += 1
        if vocab.is_closing_non_terminal(token):
            TG[i] = token
            i += 1
    assert i==T
    return TG

def padding_tokens(tokens, max_len=136):
        # guarantee len(tokens) < max_len
        result = np.full(max_len, args.pad, dtype=np.uint16)
        result[:len(tokens)] = tokens
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_OutDomain', action='store_true')  # 接收单个字符串
    parser.add_argument("--input_dir", type=str, default="../dataset/BLiMP/tree300/")
    parser.add_argument("--output_dir", type=str, default="../dataset/BLiMP/tg300/")
    parser.add_argument("--eos", type=int, default=50256)
    parser.add_argument("--bos", type=int, default=50257)
    parser.add_argument("--pad", type=int, default=50258)
    args = parser.parse_args()
    vocab = SentencepieceVocab.from_vocab_file("TG_GPT2_tokenizer.json")
    tokenizer = Tokenizer.from_file("TG_GPT2_tokenizer.json")

    exclude_list = os.listdir(args.input_dir)
    exclude_list = sorted(exclude_list)
    # print(exclude_list)
    tokenize_list = []
    for filename in BLiMP_TASK_LIST:
        full_path = "tg_300_" + filename + ".npy"
        if full_path not in exclude_list:
            tokenize_list.append(filename)

    def tokenize_blimp_file(input):
        if input not in BLiMP_TASK_LIST:
            return None
        task_id = BLiMP_TASK_LIST.index(input)
        print(f"start tokenizing {input} as task {task_id}.")

        token_lines = []
        # Always 1000 * 2 * 300 lines
        tree_data = np.load(os.path.join(args.input_dir, f"tree_300_{input}.npy"))
        for tree in tqdm(tree_data):
            tg = convert_treenpy_to_TG(tree, vocab)
            token_lines.append(padding_tokens(tg))
        
        final_ids = np.stack(token_lines, dtype=np.uint16)
        np.save(os.path.join(args.output_dir, f"tg_300_{input}.npy"), final_ids)
    

    # Parallel(n_jobs=5)(delayed(tokenize_testppl_file)(name) for name in tokenize_list)
    for name in tokenize_list:
        tokenize_blimp_file(name)
    
    data = []
    for name in BLiMP_TASK_LIST:
        data.append(np.load(os.path.join(args.output_dir, f"tg_300_{name}.npy")))
    
    data = np.stack(data, dtype=np.uint16)
    np.save(os.path.join(args.output_dir, f"blimp_tg_300.npy"), data)
