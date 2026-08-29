from types import MethodType

from olmo.eval.downstream import MMLU


def _record(subject, index):
    return {
        "subject": subject,
        "question": f"question-{subject}-{index}",
        "choices": [f"choice-{index}-{j}" for j in range(4)],
        "answer": index % 4,
    }


def test_mmlu_validation_holdout_removes_fixed_per_subject_shots():
    records = [
        _record(subject, index)
        for subject in ("subject_a", "subject_b")
        for index in range(6)
    ]
    dataset = MMLU.__new__(MMLU)
    dataset._subcategories = {"subject_a": [], "subject_b": []}
    dataset.shots_num = 2
    dataset.split = "validation"
    dataset.exclude_shots_from_eval = True
    dataset.mc_labels = False
    dataset.metric_type = "len_norm"
    dataset.dataset = list(records)

    def load_local_datasets(self, split=None, ret=False, dataset_path=None):
        assert split == "validation"
        assert dataset_path == "cais/mmlu"
        if ret:
            return list(records)
        self.dataset = list(records)

    dataset.load_local_datasets = MethodType(load_local_datasets, dataset)
    dataset.prepare_shots()

    shot_keys = {
        dataset._record_key(doc)
        for subject_docs in dataset.shots.values()
        for doc in subject_docs[: dataset.shots_num]
    }
    eval_keys = {dataset._record_key(doc) for doc in dataset.dataset}
    assert shot_keys.isdisjoint(eval_keys)
    assert len(shot_keys) == 4
    assert len(eval_keys) == 8
    assert dataset.validation_holdout_metadata == {
        "source_split": "validation",
        "shots": 4,
        "eval_examples": 8,
    }
