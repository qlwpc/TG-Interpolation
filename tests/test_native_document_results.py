"""Native DocPPL document-result recovery and strict aggregation tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from olmo.gpst.eval import document_ppl as gpst_ppl
from olmo.gpst.eval.document_ppl import GoldSegment
from scripts.native_document_results import (
    DocumentResultStore,
    aggregate_rows,
    atomic_json,
)


def pushdown_contract(tmp_path: Path, normalization: str = "stack_legal") -> dict:
    return {
        "evaluation_contract_version": 1,
        "model": "pushdown",
        "checkpoint": {"path": str(tmp_path / "checkpoint")},
        "native_data": {"path": str(tmp_path / "corpus")},
        "tokenizer": {"path": str(tmp_path / "tokenizer.json")},
        "settings": {"attachment_normalization": normalization},
    }


def pushdown_row(document_id: int, fingerprint: str | None = None) -> dict:
    row = {
        "document_id": document_id,
        "terminal_count": 10,
        "sentence_count": 2,
        "document_count": 1,
        "candidate_slots": 600,
        "model_candidate_forwards": 500,
        "samples_per_sentence": 300,
        "legacy_log_likelihood": -20.0 - document_id,
        "uniform_mixture_log_likelihood": -22.0 - document_id,
        "token_only_log_likelihood": -18.0 - document_id,
        "attachment_normalization": "stack_legal",
        "protocol_version": 1,
        "structure_source": "native_pushdown_nary_topk",
        "candidate_aggregation": "truncated_joint_sum",
        "ppl_denominator": "terminal_count",
        "prefix_policy": "candidate0",
        "max_sequence_length": 2048,
        "deduplicated_trees": False,
    }
    if fingerprint is not None:
        row["run_fingerprint"] = fingerprint
    return row


def test_store_resumes_only_an_identical_run_and_aggregates_exact_coverage(tmp_path):
    directory = tmp_path / "documents"
    store = DocumentResultStore(str(directory), pushdown_contract(tmp_path))
    store.write(0, pushdown_row(0))
    store.write(1, pushdown_row(1))

    resumed = DocumentResultStore(
        str(directory), pushdown_contract(tmp_path), resume=True
    )
    assert resumed.completed_ids == {0, 1}
    result = resumed.aggregate({0, 1})
    assert result["complete"] is True
    assert result["validated_document_count"] == 2
    assert result["terminal_count"] == 20

    changed = pushdown_contract(tmp_path, normalization="sentence_causal")
    with pytest.raises(ValueError, match="different model/data/protocol run"):
        DocumentResultStore(str(directory), changed, resume=True)


def test_aggregate_rejects_missing_duplicate_and_mixed_protocol_rows():
    rows = [pushdown_row(0), pushdown_row(1)]
    with pytest.raises(ValueError, match="incomplete document results"):
        aggregate_rows("pushdown", rows[:1], {0, 1}, 300)
    with pytest.raises(ValueError, match="duplicate document results"):
        aggregate_rows("pushdown", [rows[0], rows[0]], None, 300)
    rows[1]["attachment_normalization"] = "sentence_causal"
    rows[1]["protocol_version"] = 2
    with pytest.raises(ValueError, match="mismatch"):
        aggregate_rows("pushdown", rows, None, 300)


def test_atomic_json_never_emits_nan_and_leaves_no_temporary_file(tmp_path):
    destination = tmp_path / "result.json"
    atomic_json(destination, {"ok": 1})
    assert json.loads(destination.read_text()) == {"ok": 1}
    with pytest.raises(ValueError):
        atomic_json(destination, {"bad": float("nan")})
    assert json.loads(destination.read_text()) == {"ok": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_store_rejects_existing_documents_without_a_manifest(tmp_path):
    directory = tmp_path / "documents"
    directory.mkdir()
    atomic_json(directory / "document_00000.json", pushdown_row(0, "unknown"))
    with pytest.raises(ValueError, match="no run manifest"):
        DocumentResultStore(str(directory), pushdown_contract(tmp_path), resume=True)


def test_merge_cli_validates_and_atomically_writes_pushdown_result(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    for doc_id in (0, 1):
        atomic_json(documents / f"document_{doc_id:05d}.json", pushdown_row(doc_id))
    output = tmp_path / "aggregate.json"
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/merge_native_document_ppl.py"),
            "pushdown",
            str(documents),
            "--expected-documents",
            "2",
            "--expected-samples-per-sentence",
            "300",
            "--output",
            str(output),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["complete"] is True
    assert result["validated_document_count"] == 2
    assert result["legacy_log_likelihood"] == pytest.approx(-41.0)


def test_gpst_evaluator_emits_complete_documents_and_skips_resumed_ids(monkeypatch):
    candidate = (GoldSegment((10, 11), (0,)),)

    class Corpus:
        samples_per_sentence = 2
        rows = [(0, (candidate, candidate)), (0, (candidate, candidate)),
                (1, (candidate, candidate))]

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

    class Model:
        def eval(self):
            return self

    monkeypatch.setattr(
        gpst_ppl,
        "_score_items",
        lambda _model, items, *_args: torch.ones(len(items), dtype=torch.float64),
    )
    documents = []
    full = gpst_ppl.evaluate_gold_tree_document_ppl(
        Model(), Corpus(), "cpu", document_complete=lambda _, row: documents.append(row)
    )
    assert [row["document_id"] for row in documents] == [0, 1]
    assert sum(row["sentence_count"] for row in documents) == full.sentence_count == 3
    assert sum(row["terminal_count"] for row in documents) == full.terminal_count
    assert sum(row["log_likelihood"] for row in documents) == pytest.approx(
        full.log_likelihood
    )
    aggregate = aggregate_rows("gpst", documents, {0, 1}, 2)
    assert aggregate["complete"] is True
    assert aggregate["candidate_aggregation"] == "truncated_joint_sum"

    resumed_documents = []
    resumed = gpst_ppl.evaluate_gold_tree_document_ppl(
        Model(), Corpus(), "cpu", completed_document_ids={0},
        document_complete=lambda _, row: resumed_documents.append(row),
    )
    assert [row["document_id"] for row in resumed_documents] == [1]
    assert resumed.document_count == 1
    assert resumed.sentence_count == 1
    assert resumed.log_likelihood == pytest.approx(documents[1]["log_likelihood"])
