"""Atomic document results and compatible, complete native DocPPL aggregation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

COUNTS = ("terminal_count", "sentence_count", "document_count", "candidate_slots", "model_candidate_forwards")
LIKELIHOODS = {
    "gpst": ("log_likelihood",),
    "pushdown": ("legacy_log_likelihood", "uniform_mixture_log_likelihood", "token_only_log_likelihood"),
}
PROTOCOL_FIELDS = {
    "gpst": (
        "normalized_mixture", "deduplicated_trees", "structure_source",
        "candidate_aggregation", "ppl_denominator", "prefix_policy",
        "max_action_nodes", "max_terminals",
    ),
    "pushdown": (
        "attachment_normalization", "protocol_version", "structure_source",
        "candidate_aggregation", "ppl_denominator", "prefix_policy",
        "max_sequence_length", "deduplicated_trees",
    ),
}
OPTIONAL_CONTRACT_FIELDS = ("run_fingerprint",)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def input_identity(path: str) -> dict:
    """Pin location and file metadata; hash small metadata/config/tokenizer files.

    Large model weights and corpus arrays are identified by size and mtime,
    avoiding a multi-GB reread by every GPU worker. This is not a content audit.
    """
    root = Path(path).resolve(strict=True)
    paths = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
    records = []
    for file in paths:
        stat = file.stat()
        record = {"name": str(file.relative_to(root)) if root.is_dir() else file.name,
                  "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if file.suffix in {".json", ".yaml", ".yml"}:
            record["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        records.append(record)
    return {"path": str(root), "files": records}


def validate_row(model: str, row: dict) -> None:
    for key in (*COUNTS, "samples_per_sentence"):
        value = row.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if row["model_candidate_forwards"] > row["candidate_slots"]:
        raise ValueError("model_candidate_forwards exceeds candidate_slots")
    for key in LIKELIHOODS[model]:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"missing or non-finite {key}")
    for key in PROTOCOL_FIELDS[model]:
        if row.get(key) is None:
            raise ValueError(f"missing required protocol metadata {key}")
    if row["prefix_policy"] != "candidate0" or row["ppl_denominator"] != "terminal_count":
        raise ValueError("unsupported prefix policy or PPL denominator")
    if type(row["deduplicated_trees"]) is not bool:
        raise ValueError("deduplicated_trees must be boolean")
    if model == "pushdown":
        norm = row["attachment_normalization"]
        if type(row["protocol_version"]) is not int or row["protocol_version"] not in (1, 2) or norm not in ("stack_legal", "sentence_causal", "none"):
            raise ValueError("unsupported Pushdown protocol")
        if norm != "none" and row["protocol_version"] != (1 if norm == "stack_legal" else 2):
            raise ValueError("Pushdown normalization and protocol version disagree")
        if row["candidate_aggregation"] != "truncated_joint_sum" or type(row["max_sequence_length"]) is not int or row["max_sequence_length"] <= 0:
            raise ValueError("unsupported Pushdown aggregation or context limit")
    else:
        if type(row["normalized_mixture"]) is not bool:
            raise ValueError("normalized_mixture must be boolean")
        expected_aggregation = "uniform_mixture" if row["normalized_mixture"] else "truncated_joint_sum"
        if row["candidate_aggregation"] != expected_aggregation:
            raise ValueError("GPST mixture flag and candidate aggregation disagree")
        if any(type(row[key]) is not int or row[key] <= 0 for key in ("max_action_nodes", "max_terminals")):
            raise ValueError("GPST context limits must be positive integers")


def aggregate_rows(model: str, rows: list[dict], expected_ids: set[int] | None = None,
                   expected_samples: int | None = None) -> dict:
    if not rows:
        raise ValueError("no result rows")
    reference = rows[0]
    fields = (*PROTOCOL_FIELDS[model], *OPTIONAL_CONTRACT_FIELDS, "samples_per_sentence")
    ids = []
    atomic = ["document_id" in row for row in rows]
    if any(atomic) and not all(atomic):
        raise ValueError("cannot mix atomic document rows and shard totals")
    for row in rows:
        validate_row(model, row)
        for field in fields:
            if row.get(field) != reference.get(field):
                raise ValueError(f"result protocol/identity mismatch: {field}")
        if expected_samples is not None and row["samples_per_sentence"] != expected_samples:
            raise ValueError("unexpected samples_per_sentence")
        if all(atomic):
            doc_id = row["document_id"]
            if type(doc_id) is not int or doc_id < 0 or row["document_count"] != 1:
                raise ValueError("invalid atomic document ID/count")
            ids.append(doc_id)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate document results")
    if expected_ids is not None:
        if not all(atomic):
            raise ValueError("exact document coverage requires atomic document results")
        if set(ids) != expected_ids:
            raise ValueError(f"incomplete document results: missing={sorted(expected_ids-set(ids))[:10]} unexpected={sorted(set(ids)-expected_ids)[:10]}")
    result = {key: sum(row[key] for row in rows) for key in COUNTS}
    result.update({field: reference[field] for field in fields if field in reference})
    for field in LIKELIHOODS[model]:
        ll = math.fsum(row[field] for row in rows)
        result[field] = ll
        result[field.replace("log_likelihood", "perplexity")] = math.exp(-ll / result["terminal_count"])
    result["candidate_compression_ratio"] = result["candidate_slots"] / result["model_candidate_forwards"]
    if expected_ids is not None:
        result.update(complete=True, validated_document_count=len(expected_ids),
                      validated_document_id_min=min(expected_ids), validated_document_id_max=max(expected_ids))
    return result


class DocumentResultStore:
    def __init__(self, directory: str, contract: dict, resume: bool = False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = contract["model"]
        payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        self.fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        manifest = {"run_fingerprint": self.fingerprint, "contract": contract}
        path = self.directory / "run_manifest.json"
        if not path.exists():
            if any(self.directory.glob("document_*.json")):
                raise ValueError("existing document results have no run manifest")
            # Link a completed file exclusively, so concurrent workers cannot
            # overwrite an incompatible manifest during startup.
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as stream:
                tmp = Path(stream.name)
                stream.write((json.dumps(manifest, sort_keys=True) + "\n").encode())
            try:
                os.link(tmp, path)
            except FileExistsError:
                pass
            finally:
                tmp.unlink()
        if json.loads(path.read_text()) != manifest:
            raise ValueError("result directory belongs to a different model/data/protocol run")
        self.rows = self.read_rows()
        if self.rows and not resume:
            raise ValueError("document results already exist; use --resume-document-results")

    def read_rows(self) -> list[dict]:
        rows = []
        for path in sorted(self.directory.glob("document_*.json")):
            row = json.loads(path.read_text())
            if row.get("run_fingerprint") != self.fingerprint:
                raise ValueError(f"incompatible document result: {path}")
            validate_row(self.model, row)
            doc_id = row.get("document_id")
            if type(doc_id) is not int or doc_id < 0 or row["document_count"] != 1 or path.name != f"document_{doc_id:05d}.json":
                raise ValueError(f"document filename/ID/count mismatch: {path}")
            rows.append(row)
        return rows

    @property
    def completed_ids(self) -> set[int]:
        return {row["document_id"] for row in self.rows}

    def write(self, doc_id: int, row: dict) -> None:
        validate_row(self.model, row)
        if row.get("document_id") != doc_id or row["document_count"] != 1:
            raise ValueError("callback must contain exactly its completed document")
        atomic_json(self.directory / f"document_{doc_id:05d}.json",
                    {**row, "run_fingerprint": self.fingerprint})

    def aggregate(self, expected_ids: set[int]) -> dict:
        rows = [row for row in self.read_rows() if row["document_id"] in expected_ids]
        return aggregate_rows(self.model, rows, expected_ids)


def prepare_result_store(args, model: str, settings: dict) -> DocumentResultStore | None:
    if args.resume_document_results and not args.document_result_dir:
        raise ValueError("--resume-document-results requires --document-result-dir")
    if not args.document_result_dir:
        return None
    if args.max_sentences is not None:
        raise ValueError("--max-sentences cannot write resumable results: it may truncate a document")
    contract = {
        "evaluation_contract_version": 1,
        "model": model,
        "checkpoint": input_identity(args.checkpoint),
        "native_data": input_identity(args.native_data),
        "tokenizer": input_identity(args.tokenizer_path),
        "settings": settings,
    }
    return DocumentResultStore(args.document_result_dir, contract, args.resume_document_results)


def selected_document_ids(corpus) -> set[int]:
    return set(map(int, corpus.native.document_ids[corpus.start_sentence:corpus.start_sentence + len(corpus)]))
