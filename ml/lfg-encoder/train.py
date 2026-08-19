#!/usr/bin/env python3
"""Validate, plan, or GPU-train the experimental LFG prompt encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "synthetic.v1.jsonl"
POLICY = ROOT / "promotion-policy.json"
BASE_MODEL = "answerdotai/ModernBERT-base"
BASE_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
DEFAULT_OUTPUT = ROOT / "artifacts" / "experimental"
REQUIRED_FIELDS = {
    "id",
    "text",
    "intents",
    "capabilities",
    "risks",
    "verified",
    "family",
}
LABEL_FIELDS = ("intents", "capabilities", "risks")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_policy(path: Path = POLICY) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    for index, record in enumerate(records, 1):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"record {index} missing {sorted(missing)}")
        record_id = str(record["id"]).strip()
        if not record_id:
            raise ValueError(f"record {index} has an empty id")
        if record_id in seen:
            raise ValueError(f"duplicate id {record_id}")
        if not isinstance(record["verified"], bool):
            raise TypeError(f"record {record_id} verified must be true or false")
        if not str(record["text"]).strip() or not str(record["family"]).strip():
            raise ValueError(f"record {record_id} has empty text or family")
        for field in LABEL_FIELDS:
            values = record[field]
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"record {record_id} {field} must be a list of labels")
        if not record["intents"]:
            raise ValueError(f"record {record_id} has no intent label")
        seen.add(record_id)
    if not records:
        raise ValueError("dataset is empty")
    return records


def dataset_report(path: Path, policy_path: Path = POLICY) -> dict[str, Any]:
    records = load_records(path)
    policy = load_policy(policy_path)
    verified = sum(record["verified"] for record in records)
    labels = sorted(
        {
            label
            for record in records
            for field in LABEL_FIELDS
            for label in record[field]
        }
    )
    minimum = int(policy["minimum_human_verified_examples"])
    return {
        "schema_version": "lfg.encoder-dataset-report.v1",
        "records": len(records),
        "verified": verified,
        "unverified": len(records) - verified,
        "required_verified": minimum,
        "families": sorted({str(record["family"]) for record in records}),
        "labels": labels,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "promotion_ready": verified >= minimum,
    }


def training_plan(args: argparse.Namespace) -> dict[str, Any]:
    report = dataset_report(args.dataset)
    development = bool(args.development_run)
    allowed = bool(report["promotion_ready"] or development)
    reason = (
        "human-verified sample gate passed"
        if report["promotion_ready"]
        else (
            "development run explicitly allowed; every artifact remains ineligible"
            if development
            else (
                f"blocked: {report['verified']} verified examples; "
                f"{report['required_verified']} required"
            )
        )
    )
    return {
        "schema_version": "lfg.encoder-training-plan.v1",
        "task": "multi_label_prompt_preflight",
        "engine": args.engine,
        "training_device": "cuda-required",
        "cpu_role": "post-export benchmark only",
        "base_model": args.base_model,
        "base_revision": args.revision,
        "dataset": str(args.dataset.resolve()),
        "dataset_report": report,
        "output": str(args.output.resolve()),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "threshold": args.threshold,
        "development_run": development,
        "training_allowed": allowed,
        "reason": reason,
        "promotion_eligible_after_training": False,
        "remaining_gates": [
            "held-out quality policy",
            "ONNX export parity",
            "quantization quality loss",
            "Rust target-CPU p95 latency",
        ],
    }


def validate(args: argparse.Namespace) -> None:
    print(json.dumps(dataset_report(args.dataset), indent=2))


def plan(args: argparse.Namespace) -> None:
    print(json.dumps(training_plan(args), indent=2))


def _expected_calibration_error(
    probabilities: Any,
    labels: Any,
    *,
    bins: int = 15,
) -> float:
    import numpy as np

    predictions = probabilities >= 0.5
    confidences = np.where(predictions, probabilities, 1.0 - probabilities).reshape(-1)
    correctness = (predictions == labels).astype(np.float32).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in pairwise(edges):
        included = (confidences > lower) & (confidences <= upper)
        if not included.any():
            continue
        result += float(included.mean()) * abs(
            float(correctness[included].mean()) - float(confidences[included].mean())
        )
    return result


def _prepare_output(path: Path, *, resume: bool) -> None:
    if path.exists() and any(path.iterdir()) and not resume:
        raise SystemExit(
            f"output is not empty: {path}. Choose a fresh folder or pass --resume."
        )
    path.mkdir(parents=True, exist_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(args: argparse.Namespace) -> None:
    plan_payload = training_plan(args)
    if not plan_payload["training_allowed"]:
        raise SystemExit(plan_payload["reason"])

    _prepare_output(args.output, resume=args.resume)
    os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")
    fast_model_loader: Any | None = None
    try:
        import numpy as np
        import torch
        if args.engine == "unsloth":
            try:
                from unsloth import FastModel as fast_model_loader
            except ImportError as error:
                raise SystemExit(
                    "Unsloth is the default LFG engine but is not installed. Install "
                    "requirements.txt; use --engine transformers only for an explicit "
                    "compatibility diagnostic."
                ) from error
        from datasets import Dataset
        from sklearn.metrics import f1_score, recall_score
        from sklearn.preprocessing import MultiLabelBinarizer
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise SystemExit(
            "Install the locked LFG training dependencies before training."
        ) from error

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not detected. LFG training never falls back to CPU; CPU is "
            "reserved for the post-export proof."
        )

    records = load_records(args.dataset)
    combined = [
        sorted(
            {
                label
                for field in LABEL_FIELDS
                for label in record[field]
            }
        )
        for record in records
    ]
    binarizer = MultiLabelBinarizer().fit(combined)
    labels = binarizer.classes_.tolist()
    id2label = dict(enumerate(labels))
    label2id = {label: index for index, label in id2label.items()}

    if args.engine == "unsloth":
        assert fast_model_loader is not None
        model, tokenizer = fast_model_loader.from_pretrained(
            model_name=args.base_model,
            revision=args.revision,
            auto_model=AutoModelForSequenceClassification,
            max_seq_length=args.max_length,
            dtype=None,
            num_labels=len(labels),
            full_finetuning=True,
            id2label=id2label,
            label2id=label2id,
            problem_type="multi_label_classification",
            load_in_4bit=False,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            revision=args.revision,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            revision=args.revision,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            problem_type="multi_label_classification",
        )

    raw_dataset = Dataset.from_dict(
        {
            "text": [record["text"] for record in records],
            "labels": binarizer.transform(combined).astype(np.float32),
        }
    )
    test_size = max(1, round(len(records) * args.validation_fraction))
    if test_size >= len(records):
        raise SystemExit("dataset needs at least two examples for a train/evaluation split")
    split = raw_dataset.train_test_split(test_size=test_size, seed=args.seed)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = split.map(tokenize, batched=True)
    risk_labels = sorted({label for record in records for label in record["risks"]})
    risk_indices = [label2id[label] for label in risk_labels]

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits, expected = eval_prediction
        if isinstance(logits, tuple):
            logits = logits[0]
        probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
        expected_array = np.asarray(expected)
        predicted = probabilities >= args.threshold
        macro_f1 = f1_score(
            expected_array,
            predicted,
            average="macro",
            zero_division=0,
        )
        security_positives = (
            int(expected_array[:, risk_indices].sum()) if risk_indices else 0
        )
        security_recall = (
            recall_score(
                expected_array[:, risk_indices],
                predicted[:, risk_indices],
                average="macro",
                zero_division=0,
            )
            if security_positives
            else 0.0
        )
        return {
            "macro_f1": float(macro_f1),
            "security_recall": float(security_recall),
            "security_positive_labels": float(security_positives),
            "expected_calibration_error": _expected_calibration_error(
                probabilities,
                expected_array,
            ),
        }

    bf16 = bool(torch.cuda.is_bf16_supported())
    run_manifest = {
        **plan_payload,
        "schema_version": "lfg.encoder-training-run.v1",
        "started_at": datetime.now(UTC).isoformat(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "labels": labels,
        "risk_labels": risk_labels,
        "train_examples": len(tokenized["train"]),
        "evaluation_examples": len(tokenized["test"]),
    }
    updates_per_epoch = math.ceil(
        len(tokenized["train"])
        / (args.batch_size * args.gradient_accumulation_steps)
    )
    total_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
    warmup_steps = round(total_updates * args.warmup_ratio)
    run_manifest["warmup_steps"] = warmup_steps
    _write_json(args.output / "run-manifest.json", run_manifest)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"),
            seed=args.seed,
            data_seed=args.seed,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="no" if args.development_run else "epoch",
            save_total_limit=2,
            load_best_model_at_end=not args.development_run,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=1,
            report_to="none",
            fp16=not bf16,
            bf16=bf16,
            tf32=True,
            use_cpu=False,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    evaluation = {
        key: float(value) if isinstance(value, (int, float)) else value
        for key, value in trainer.evaluate().items()
    }

    final_dir = args.output / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    _write_json(final_dir / "labels.json", {"labels": labels})
    _write_json(args.output / "evaluation.json", evaluation)

    policy = load_policy()
    quality_gate = (
        evaluation.get("eval_macro_f1", 0.0)
        >= policy["minimum_intent_capability_macro_f1"]
        and evaluation.get("eval_security_recall", 0.0)
        >= policy["minimum_security_recall"]
        and evaluation.get("eval_expected_calibration_error", 1.0)
        <= policy["maximum_expected_calibration_error"]
    )
    candidate = {
        "schema_version": "lfg.encoder-candidate.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "development_run": args.development_run,
        "sample_gate_passed": plan_payload["dataset_report"]["promotion_ready"],
        "quality_gate_passed": quality_gate,
        "promotion_eligible": False,
        "remaining_gates": plan_payload["remaining_gates"][1:],
        "evaluation": evaluation,
        "artifact": str(final_dir.resolve()),
    }
    _write_json(args.output / "candidate.json", candidate)
    print(
        f"trained candidate at {final_dir}; ONNX export and Rust CPU proof remain required"
    )


def export_onnx(args: argparse.Namespace) -> None:
    checkpoint = (args.checkpoint or (args.output / "final")).resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"trained checkpoint not found: {checkpoint}")
    onnx_output = (
        args.onnx_output or (args.output / "onnx" / "model.fp32.onnx")
    ).resolve()
    quantized_output = onnx_output.with_name("model.int8.onnx")
    existing = [path for path in (onnx_output, quantized_output) if path.exists()]
    if existing and not args.force_export:
        raise SystemExit(
            "export output already exists: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --force-export to replace these exact artifacts."
        )
    onnx_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install requirements.txt before ONNX export.") from error

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            return self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

    sample_texts = [
        "Research the current routing design and cite primary sources.",
        "Implement the component, run its tests, and do not expose secrets.",
    ]
    encoded = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_names = ["input_ids", "attention_mask"]
    sample_inputs = tuple(encoded[name] for name in input_names)
    wrapper = LogitsOnly(model).eval()
    with torch.inference_mode():
        reference = wrapper(*sample_inputs).detach().cpu().numpy()

    batch_dimension = torch.export.Dim("batch", min=1)
    sequence_dimension = torch.export.Dim(
        "sequence",
        min=1,
        max=args.max_length,
    )
    torch.onnx.export(
        wrapper,
        sample_inputs,
        str(onnx_output),
        input_names=input_names,
        output_names=["logits"],
        dynamic_shapes=(
            {0: batch_dimension, 1: sequence_dimension},
            {0: batch_dimension, 1: sequence_dimension},
        ),
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    onnx.checker.check_model(onnx.load(onnx_output))

    ort_inputs = {
        name: encoded[name].detach().cpu().numpy().astype(np.int64)
        for name in input_names
    }
    fp32_session = ort.InferenceSession(
        str(onnx_output),
        providers=["CPUExecutionProvider"],
    )
    fp32_logits = fp32_session.run(["logits"], ort_inputs)[0]
    fp32_max_difference = float(np.max(np.abs(reference - fp32_logits)))
    export_parity_passed = fp32_max_difference <= args.export_tolerance

    quantized: dict[str, Any] | None = None
    if not args.skip_quantization:
        # ORT's dynamic quantizer converts the two classifier Gemm nodes to
        # MatMul and reruns ONNX shape inference. Dynamo's intermediate
        # value_info currently conflicts with that conversion (768 hidden
        # features versus 13 labels), so let ORT infer fresh intermediates.
        quantization_source = onnx.load(onnx_output)
        del quantization_source.graph.value_info[:]
        quantize_dynamic(
            model_input=quantization_source,
            model_output=str(quantized_output),
            per_channel=True,
            weight_type=QuantType.QInt8,
        )
        int8_session = ort.InferenceSession(
            str(quantized_output),
            providers=["CPUExecutionProvider"],
        )
        int8_logits = int8_session.run(["logits"], ort_inputs)[0]
        quantized = {
            "path": str(quantized_output),
            "bytes": quantized_output.stat().st_size,
            "sha256": _file_sha256(quantized_output),
            "maximum_logit_difference_from_pytorch": float(
                np.max(np.abs(reference - int8_logits))
            ),
            "smoke_output_finite": bool(np.isfinite(int8_logits).all()),
            "quality_gate_passed": False,
            "quality_gate_note": (
                "Logit smoke parity is not the policy's held-out quantization quality gate."
            ),
        }

    manifest = {
        "schema_version": "lfg.encoder-onnx-export.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "opset": 18,
        "maximum_length": args.max_length,
        "sample_count": len(sample_texts),
        "fp32": {
            "path": str(onnx_output),
            "bytes": onnx_output.stat().st_size,
            "sha256": _file_sha256(onnx_output),
            "maximum_logit_difference_from_pytorch": fp32_max_difference,
            "tolerance": args.export_tolerance,
            "parity_passed": export_parity_passed,
        },
        "int8": quantized,
        "promotion_eligible": False,
        "remaining_gates": [
            "held-out quantization quality loss",
            "Rust target-CPU p95 latency",
        ],
    }
    manifest_path = args.output / "onnx-manifest.json"
    _write_json(manifest_path, manifest)

    candidate_path = args.output / "candidate.json"
    if candidate_path.is_file():
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate["onnx_export"] = manifest
        if export_parity_passed:
            candidate["remaining_gates"] = [
                gate
                for gate in candidate.get("remaining_gates", [])
                if gate != "ONNX export parity"
            ]
        candidate["promotion_eligible"] = False
        _write_json(candidate_path, candidate)

    if not export_parity_passed:
        raise SystemExit(
            f"ONNX parity failed: maximum logit difference {fp32_max_difference:.6g} "
            f"exceeds {args.export_tolerance:.6g}"
        )
    print(f"ONNX export proven: {manifest_path}")


def benchmark_onnx(args: argparse.Namespace) -> None:
    import platform
    import time

    try:
        import numpy as np
        import onnxruntime as ort
        import psutil
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install requirements.txt before the CPU preflight.") from error

    checkpoint = (args.checkpoint or (args.output / "final")).resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"trained checkpoint not found: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    text = (
        "Research the request, identify the required capabilities and risks, "
        "implement the work, verify the result, and preserve the evidence. "
    ) * 80
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_tensors="np",
    )
    ort_inputs = {
        name: np.asarray(encoded[name], dtype=np.int64)
        for name in ("input_ids", "attention_mask")
    }
    policy = load_policy()
    process = psutil.Process()

    def summarize(samples: list[float]) -> dict[str, float]:
        return {
            "p50_micros": float(np.percentile(samples, 50)),
            "p95_micros": float(np.percentile(samples, 95)),
            "minimum_micros": float(min(samples)),
            "maximum_micros": float(max(samples)),
            "mean_micros": float(np.mean(samples)),
        }

    def measure(path: Path) -> dict[str, Any]:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if args.cpu_threads:
            options.intra_op_num_threads = args.cpu_threads
            options.inter_op_num_threads = 1
        rss_before = process.memory_info().rss
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        for _ in range(args.benchmark_warmup):
            session.run(["logits"], ort_inputs)
        rss_after_warmup = process.memory_info().rss

        model_samples: list[float] = []
        end_to_end_samples: list[float] = []
        for _ in range(args.benchmark_iterations):
            started = time.perf_counter_ns()
            session.run(["logits"], ort_inputs)
            model_samples.append((time.perf_counter_ns() - started) / 1000.0)

            started = time.perf_counter_ns()
            current = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
                return_tensors="np",
            )
            session.run(
                ["logits"],
                {
                    name: np.asarray(current[name], dtype=np.int64)
                    for name in ("input_ids", "attention_mask")
                },
            )
            end_to_end_samples.append(
                (time.perf_counter_ns() - started) / 1000.0
            )

        model_latency = summarize(model_samples)
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "model_only": model_latency,
            "tokenization_and_model": summarize(end_to_end_samples),
            "rss_before_session_bytes": rss_before,
            "rss_after_warmup_bytes": rss_after_warmup,
            "rss_growth_bytes": max(0, rss_after_warmup - rss_before),
            "meets_python_preflight_latency": (
                model_latency["p95_micros"]
                <= policy["maximum_p95_micros_512_tokens"]
            ),
            "authoritative_rust_gate_passed": False,
        }

    onnx_dir = args.output / "onnx"
    artifacts = [onnx_dir / "model.fp32.onnx", onnx_dir / "model.int8.onnx"]
    existing = [path for path in artifacts if path.is_file()]
    if not existing:
        raise SystemExit(f"no ONNX artifacts found under {onnx_dir.resolve()}")
    results = [measure(path) for path in existing]
    report = {
        "schema_version": "lfg.encoder-cpu-preflight.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "authoritative": False,
        "why_not_authoritative": (
            "The promotion policy requires the Rust runtime on the pinned target CPU."
        ),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "onnxruntime": ort.__version__,
        "maximum_length": args.max_length,
        "iterations": args.benchmark_iterations,
        "warmup_iterations": args.benchmark_warmup,
        "intra_op_threads": args.cpu_threads or "onnxruntime-default",
        "policy_p95_micros": policy["maximum_p95_micros_512_tokens"],
        "artifacts": results,
        "promotion_eligible": False,
    }
    report_path = args.output / "cpu-preflight.json"
    _write_json(report_path, report)

    candidate_path = args.output / "candidate.json"
    if candidate_path.is_file():
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate["cpu_preflight"] = report
        candidate["promotion_eligible"] = False
        _write_json(candidate_path, candidate)
    print(f"CPU preflight recorded: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["validate", "plan", "train", "export", "benchmark"],
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", choices=["unsloth", "transformers"], default="unsloth")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--onnx-output", type=Path)
    parser.add_argument("--export-tolerance", type=float, default=1e-4)
    parser.add_argument("--skip-quantization", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iterations", type=int, default=30)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="0 uses ONNX Runtime's default; otherwise pin this thread count",
    )
    parser.add_argument(
        "--development-run",
        "--allow-below-gate",
        dest="development_run",
        action="store_true",
        help="allow a below-gate compatibility run; artifacts remain ineligible",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be between 0 and 1")
    if not 0.0 < args.threshold < 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.benchmark_warmup < 0 or args.benchmark_iterations < 1:
        raise SystemExit("benchmark warmup must be non-negative and iterations positive")
    if args.cpu_threads < 0:
        raise SystemExit("--cpu-threads cannot be negative")
    actions = {
        "validate": validate,
        "plan": plan,
        "train": train,
        "export": export_onnx,
        "benchmark": benchmark_onnx,
    }
    actions[args.command](args)


if __name__ == "__main__":
    main()
