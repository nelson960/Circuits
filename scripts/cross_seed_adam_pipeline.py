from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _format_step(step: int) -> str:
    return f"{step:06d}"


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _run_command(command: list[str], *, dry_run: bool) -> None:
    print(f"[cross-seed] {_command_text(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def _seed_dir(run_root: Path, seed: int) -> Path:
    return run_root / f"seed_{seed:04d}"


def _seed_config_path(run_root: Path, seed: int) -> Path:
    return _seed_dir(run_root, seed) / "run_config.json"


def _scan_trace_dir(run_root: Path, seed: int, end_step: int, scan_every: int) -> Path:
    return (
        _seed_dir(run_root, seed)
        / "analysis"
        / "optimizer_update_trace"
        / f"from_init_000000_{_format_step(end_step)}_scan_every_{_format_step(scan_every)}"
    )


def _adam_trace_dir(run_root: Path, seed: int, start_step: int, end_step: int) -> Path:
    return (
        _seed_dir(run_root, seed)
        / "analysis"
        / "optimizer_update_trace"
        / f"from_init_{_format_step(start_step)}_{_format_step(end_step)}_stepwise"
    )


def _scan_dir(run_root: Path, seed: int, layer: int, head: int, end_step: int) -> Path:
    return (
        _seed_dir(run_root, seed)
        / "analysis"
        / "bilinear_qk_match_separation"
        / f"l{layer}h{head}_support_value_scan_000000_{_format_step(end_step)}"
    )


def _adam_dir(
    run_root: Path,
    seed: int,
    layer: int,
    head: int,
    rank: int,
    start_step: int,
    end_step: int,
    candidate_label: str,
) -> Path:
    if not candidate_label:
        raise ValueError("candidate_label must be non-empty.")
    prefix = "" if candidate_label == "winner" else f"{candidate_label}_"
    return (
        _seed_dir(run_root, seed)
        / "analysis"
        / "bilinear_qk_rank_adam_state_attribution"
        / f"{prefix}l{layer}h{head}_rank{rank}_support_value_{_format_step(start_step)}_{_format_step(end_step)}_stepwise"
    )


def _checkpoint_args(checkpoint_dir: Path, *, end_step: int, every: int) -> list[str]:
    if every <= 0:
        raise ValueError(f"scan_every must be positive, got {every}")
    steps = list(range(0, end_step + 1, every))
    if steps[-1] != end_step:
        steps.append(end_step)
    args: list[str] = []
    for step in steps:
        path = checkpoint_dir / f"step_{step:06d}.pt"
        args.extend(["--checkpoint", str(path)])
    return args


def _prepare_configs(
    *,
    base_config_path: Path,
    run_root: Path,
    seeds: list[int],
    end_step: int,
    overwrite: bool,
) -> list[Path]:
    base_config = _read_json(base_config_path)
    original_run_name = str(base_config.get("run_name", "run"))
    config_paths: list[Path] = []
    for seed in seeds:
        seed_config_path = _seed_config_path(run_root, seed)
        if seed_config_path.exists() and not overwrite:
            raise FileExistsError(f"Seed config already exists: {seed_config_path}")
        seed_run_dir = _seed_dir(run_root, seed)
        seed_config = dict(base_config)
        seed_config["seed"] = int(seed)
        seed_config["run_name"] = f"{original_run_name}_seed_{seed:04d}"
        seed_config["output_dir"] = str(seed_run_dir)
        seed_config["num_steps"] = int(end_step)
        _write_json(seed_config_path, seed_config)
        config_paths.append(seed_config_path)
    manifest = {
        "base_config": str(base_config_path),
        "run_root": str(run_root),
        "seeds": seeds,
        "end_step": end_step,
        "config_paths": [str(path) for path in config_paths],
    }
    _write_json(run_root / "cross_seed_manifest.json", manifest)
    return config_paths


def _trace_command(
    *,
    python_executable: str,
    config_path: Path,
    output_dir: Path,
    device: str,
    end_step: int,
    train_split: str,
    checkpoint_every: int,
    checkpoint_start_step: int,
    progress_every: int,
    top_k_parameters: int,
    overwrite: bool,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "circuit.cli",
        "optimizer-update-trace",
        "--config",
        str(config_path),
        "--from-initialization",
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--end-step",
        str(end_step),
        "--train-split",
        train_split,
        "--checkpoint-every",
        str(checkpoint_every),
        "--checkpoint-start-step",
        str(checkpoint_start_step),
        "--progress-every",
        str(progress_every),
        "--top-k-parameters",
        str(top_k_parameters),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def _scan_command(
    *,
    python_executable: str,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    device: str,
    layer: int,
    head: int,
    end_step: int,
    scan_every: int,
    context_stage: str,
    score_query_role: str,
    support_role: str,
    distractor_role: str,
    layernorm_mode: str,
    rank: int,
    group_by: str,
    split_filters: list[str],
    window_start: int,
    window_end: int,
    overwrite: bool,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "circuit.cli",
        "bilinear-qk-match-separation",
        "--config",
        str(config_path),
        "--probe-set",
        str(probe_set_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
        *_checkpoint_args(checkpoint_dir, end_step=end_step, every=scan_every),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--head-layer",
        str(layer),
        "--head",
        str(head),
        "--context-stage",
        context_stage,
        "--score-query-role",
        score_query_role,
        "--support-role",
        support_role,
        "--distractor-role",
        distractor_role,
        "--layernorm-mode",
        layernorm_mode,
        "--rank",
        str(rank),
        "--group-by",
        group_by,
        "--window-start",
        str(window_start),
        "--window-end",
        str(window_end),
    ]
    for split_filter in split_filters:
        command.extend(["--split", split_filter])
    if overwrite:
        command.append("--overwrite")
    return command


def _adam_command(
    *,
    python_executable: str,
    config_path: Path,
    probe_set_path: Path,
    optimizer_trace_dir: Path,
    output_dir: Path,
    device: str,
    layer: int,
    head: int,
    rank: int,
    context_stage: str,
    layernorm_mode: str,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    record_side: str,
    route_pair_type: str,
    route_pair_source_type: str,
    max_route_pairs_per_type: int,
    min_route_pairs_per_type: int,
    loss_scope: str,
    overwrite: bool,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "circuit.cli",
        "bilinear-qk-rank-adam-state-attribution",
        "--config",
        str(config_path),
        "--probe-set",
        str(probe_set_path),
        "--optimizer-trace-dir",
        str(optimizer_trace_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--head-layer",
        str(layer),
        "--head",
        str(head),
        "--rank",
        str(rank),
        "--context-stage",
        context_stage,
        "--layernorm-mode",
        layernorm_mode,
        "--score-query-role",
        score_query_role,
        "--support-key-role",
        support_key_role,
        "--distractor-key-role",
        distractor_key_role,
        "--record-side",
        record_side,
        "--route-pair-type",
        route_pair_type,
        "--route-pair-source-type",
        route_pair_source_type,
        "--max-route-pairs-per-type",
        str(max_route_pairs_per_type),
        "--min-route-pairs-per-type",
        str(min_route_pairs_per_type),
        "--loss-scope",
        loss_scope,
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def _scan_report_path(run_root: Path, seed: int, layer: int, head: int, end_step: int) -> Path:
    return _scan_dir(run_root, seed, layer, head, end_step) / "bilinear_qk_match_separation_report.json"


def _select_winner(
    *,
    run_root: Path,
    seed: int,
    layers: int,
    heads: int,
    end_step: int,
    context_stage: str,
    rank: int,
) -> dict[str, Any]:
    projection = f"rank_{rank}"
    candidates: list[dict[str, Any]] = []
    for layer in range(layers):
        for head in range(heads):
            report_path = _scan_report_path(run_root, seed, layer, head, end_step)
            report = _read_json(report_path)
            summary_rows = report.get("summary_rows")
            if not isinstance(summary_rows, list):
                raise TypeError(f"Missing summary_rows list in {report_path}")
            matches = [
                row
                for row in summary_rows
                if row.get("context_stage") == context_stage and row.get("projection") == projection
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one {context_stage}/{projection} summary row in {report_path}, got {len(matches)}"
                )
            row = matches[0]
            metric_name = "window_delta_qk_match_separation"
            if metric_name not in row:
                raise KeyError(f"Missing {metric_name} in {report_path}")
            candidates.append(
                {
                    "seed": seed,
                    "head_layer": layer,
                    "head": head,
                    "head_label": f"L{layer}H{head}",
                    "context_stage": context_stage,
                    "projection": projection,
                    "score": float(row[metric_name]),
                    "window_delta_qk_match_separation": float(row[metric_name]),
                    "window_qk_match_separation_vs_qk_singular_value_top": row.get(
                        "window_qk_match_separation_vs_qk_singular_value_top"
                    ),
                    "window_qk_match_separation_vs_answer_margin": row.get(
                        "window_qk_match_separation_vs_answer_margin"
                    ),
                    "window_delta_support_beats_all_rate": row.get("window_delta_support_beats_all_rate"),
                    "report": str(report_path),
                }
            )
    candidates.sort(key=lambda item: (float(item["score"]), -int(item["head_layer"]), -int(item["head"])), reverse=True)
    if not candidates:
        raise RuntimeError(f"No scan candidates found for seed {seed}")
    winner = candidates[0]
    payload = {"seed": seed, "winner": winner, "candidates": candidates}
    _write_json(_seed_dir(run_root, seed) / "analysis" / "cross_seed_head_selection.json", payload)
    return payload


def _write_winners_csv(path: Path, winners: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "head_label",
        "head_layer",
        "head",
        "score",
        "window_delta_qk_match_separation",
        "window_qk_match_separation_vs_qk_singular_value_top",
        "window_qk_match_separation_vs_answer_margin",
        "window_delta_support_beats_all_rate",
        "report",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for winner_payload in winners:
            winner = winner_payload["winner"]
            writer.writerow({field: winner.get(field) for field in fields})


def _candidate_indices(*, candidate_count: int, requested_labels: list[str]) -> list[tuple[str, int]]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")
    resolved: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label in requested_labels:
        if label == "winner":
            index = 0
        elif label == "runner_up":
            index = 1
        elif label == "median":
            index = candidate_count // 2
        elif label == "bottom":
            index = candidate_count - 1
        else:
            raise ValueError(f"Unsupported adam candidate label: {label}")
        if index >= candidate_count:
            raise ValueError(f"Candidate label {label!r} requires index {index}, but only {candidate_count} candidates exist.")
        if index in seen:
            continue
        seen.add(index)
        resolved.append((label, index))
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-seed exact-trace QK route scans and Adam-state attribution.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--probe-set", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--top-k-parameters", type=int, default=0)
    parser.add_argument("--context-stage", type=str, default="layer_1_post_mlp")
    parser.add_argument("--score-query-role", type=str, default="prediction")
    parser.add_argument("--support-role", type=str, default="support_value")
    parser.add_argument("--distractor-role", type=str, default="value_distractors")
    parser.add_argument("--layernorm-mode", type=str, default="head_ln1")
    parser.add_argument("--group-by", type=str, default="query_key")
    parser.add_argument("--split", type=str, action="append", default=[])
    parser.add_argument("--window-start", type=int, required=True)
    parser.add_argument("--window-end", type=int, required=True)
    parser.add_argument("--record-side", type=str, default="clean")
    parser.add_argument("--route-pair-type", type=str, default="support_value")
    parser.add_argument("--route-pair-source-type", type=str, default="support_value")
    parser.add_argument("--max-route-pairs-per-type", type=int, default=64)
    parser.add_argument("--min-route-pairs-per-type", type=int, default=16)
    parser.add_argument("--loss-scope", type=str, default="full_lm")
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        choices=["configs", "trace-scan", "scan", "select", "trace-adam", "adam"],
        help="Pipeline stage to run. Repeat for multiple stages.",
    )
    parser.add_argument("--scan-checkpoint-every", type=int, default=250)
    parser.add_argument("--adam-start-step", type=int, default=None)
    parser.add_argument("--adam-end-step", type=int, default=None)
    parser.add_argument(
        "--adam-candidate",
        type=str,
        action="append",
        default=None,
        choices=["winner", "runner_up", "median", "bottom"],
        help="Candidate head to run Adam attribution for. Repeat for controls. Defaults to winner.",
    )
    parser.add_argument("--cleanup-adam-trace", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.end_step <= 0:
        raise ValueError("--end-step must be positive.")
    if args.window_start < 0 or args.window_end < args.window_start or args.window_end > args.end_step:
        raise ValueError("--window-start/--window-end must satisfy 0 <= start <= end <= end-step.")
    if args.layers <= 0 or args.heads <= 0:
        raise ValueError("--layers and --heads must be positive.")
    if args.scan_checkpoint_every <= 0:
        raise ValueError("--scan-checkpoint-every must be positive.")
    if len(set(args.seed)) != len(args.seed):
        raise ValueError(f"Duplicate seeds are not allowed: {args.seed}")
    adam_start_step = args.window_start if args.adam_start_step is None else int(args.adam_start_step)
    adam_end_step = args.window_end if args.adam_end_step is None else int(args.adam_end_step)
    if adam_start_step < 0 or adam_end_step <= adam_start_step or adam_end_step > args.end_step:
        raise ValueError("--adam-start-step/--adam-end-step must satisfy 0 <= start < end <= end-step.")

    run_root = args.run_root
    seeds = [int(seed) for seed in args.seed]
    stages = list(dict.fromkeys(args.stage))
    adam_candidate_labels = args.adam_candidate if args.adam_candidate is not None else ["winner"]

    if "configs" in stages:
        _prepare_configs(
            base_config_path=args.base_config,
            run_root=run_root,
            seeds=seeds,
            end_step=args.end_step,
            overwrite=args.overwrite,
        )

    if "trace-scan" in stages:
        for seed in seeds:
            config_path = _seed_config_path(run_root, seed)
            command = _trace_command(
                python_executable=args.python,
                config_path=config_path,
                output_dir=_scan_trace_dir(run_root, seed, args.end_step, args.scan_checkpoint_every),
                device=args.device,
                end_step=args.end_step,
                train_split=args.train_split,
                checkpoint_every=args.scan_checkpoint_every,
                checkpoint_start_step=0,
                progress_every=args.progress_every,
                top_k_parameters=args.top_k_parameters,
                overwrite=args.overwrite,
            )
            _run_command(command, dry_run=args.dry_run)

    if "scan" in stages:
        for seed in seeds:
            config_path = _seed_config_path(run_root, seed)
            trace_checkpoint_dir = _scan_trace_dir(run_root, seed, args.end_step, args.scan_checkpoint_every) / "checkpoints"
            for layer in range(args.layers):
                for head in range(args.heads):
                    command = _scan_command(
                        python_executable=args.python,
                        config_path=config_path,
                        probe_set_path=args.probe_set,
                        checkpoint_dir=trace_checkpoint_dir,
                        output_dir=_scan_dir(run_root, seed, layer, head, args.end_step),
                        device=args.device,
                        layer=layer,
                        head=head,
                        end_step=args.end_step,
                        scan_every=args.scan_checkpoint_every,
                        context_stage=args.context_stage,
                        score_query_role=args.score_query_role,
                        support_role=args.support_role,
                        distractor_role=args.distractor_role,
                        layernorm_mode=args.layernorm_mode,
                        rank=args.rank,
                        group_by=args.group_by,
                        split_filters=args.split,
                        window_start=args.window_start,
                        window_end=args.window_end,
                        overwrite=args.overwrite,
                    )
                    _run_command(command, dry_run=args.dry_run)

    winners: list[dict[str, Any]] = []
    if "select" in stages or "adam" in stages:
        for seed in seeds:
            winner_payload = _select_winner(
                run_root=run_root,
                seed=seed,
                layers=args.layers,
                heads=args.heads,
                end_step=args.end_step,
                context_stage=args.context_stage,
                rank=args.rank,
            )
            winners.append(winner_payload)
        _write_json(run_root / "cross_seed_winners.json", {"winners": winners})
        _write_winners_csv(run_root / "cross_seed_winners.csv", winners)
        for winner_payload in winners:
            winner = winner_payload["winner"]
            print(
                "[cross-seed] "
                f"seed={winner['seed']} winner={winner['head_label']} "
                f"score={float(winner['score']):.6g}",
                flush=True,
            )

    if "trace-adam" in stages and "adam" in stages:
        for winner_payload in winners:
            seed = int(winner_payload["seed"])
            trace_command = _trace_command(
                python_executable=args.python,
                config_path=_seed_config_path(run_root, seed),
                output_dir=_adam_trace_dir(run_root, seed, adam_start_step, adam_end_step),
                device=args.device,
                end_step=adam_end_step,
                train_split=args.train_split,
                checkpoint_every=1,
                checkpoint_start_step=adam_start_step,
                progress_every=args.progress_every,
                top_k_parameters=args.top_k_parameters,
                overwrite=args.overwrite,
            )
            _run_command(trace_command, dry_run=args.dry_run)
            candidate_indices = _candidate_indices(
                candidate_count=len(winner_payload["candidates"]),
                requested_labels=adam_candidate_labels,
            )
            for candidate_label, candidate_index in candidate_indices:
                candidate = winner_payload["candidates"][candidate_index]
                layer = int(candidate["head_layer"])
                head = int(candidate["head"])
                command = _adam_command(
                    python_executable=args.python,
                    config_path=_seed_config_path(run_root, seed),
                    probe_set_path=args.probe_set,
                    optimizer_trace_dir=_adam_trace_dir(run_root, seed, adam_start_step, adam_end_step),
                    output_dir=_adam_dir(
                        run_root,
                        seed,
                        layer,
                        head,
                        args.rank,
                        adam_start_step,
                        adam_end_step,
                        candidate_label,
                    ),
                    device=args.device,
                    layer=layer,
                    head=head,
                    rank=args.rank,
                    context_stage=args.context_stage,
                    layernorm_mode=args.layernorm_mode,
                    score_query_role=args.score_query_role,
                    support_key_role=args.support_role,
                    distractor_key_role=args.distractor_role,
                    record_side=args.record_side,
                    route_pair_type=args.route_pair_type,
                    route_pair_source_type=args.route_pair_source_type,
                    max_route_pairs_per_type=args.max_route_pairs_per_type,
                    min_route_pairs_per_type=args.min_route_pairs_per_type,
                    loss_scope=args.loss_scope,
                    overwrite=args.overwrite,
                )
                _run_command(command, dry_run=args.dry_run)
            if args.cleanup_adam_trace:
                trace_dir = _adam_trace_dir(run_root, seed, adam_start_step, adam_end_step)
                print(f"[cross-seed] cleanup {trace_dir}", flush=True)
                if not args.dry_run:
                    shutil.rmtree(trace_dir)
    elif "trace-adam" in stages:
        for seed in seeds:
            command = _trace_command(
                python_executable=args.python,
                config_path=_seed_config_path(run_root, seed),
                output_dir=_adam_trace_dir(run_root, seed, adam_start_step, adam_end_step),
                device=args.device,
                end_step=adam_end_step,
                train_split=args.train_split,
                checkpoint_every=1,
                checkpoint_start_step=adam_start_step,
                progress_every=args.progress_every,
                top_k_parameters=args.top_k_parameters,
                overwrite=args.overwrite,
            )
            _run_command(command, dry_run=args.dry_run)
    elif "adam" in stages:
        for winner_payload in winners:
            seed = int(winner_payload["seed"])
            candidate_indices = _candidate_indices(
                candidate_count=len(winner_payload["candidates"]),
                requested_labels=adam_candidate_labels,
            )
            for candidate_label, candidate_index in candidate_indices:
                candidate = winner_payload["candidates"][candidate_index]
                layer = int(candidate["head_layer"])
                head = int(candidate["head"])
                command = _adam_command(
                    python_executable=args.python,
                    config_path=_seed_config_path(run_root, seed),
                    probe_set_path=args.probe_set,
                    optimizer_trace_dir=_adam_trace_dir(run_root, seed, adam_start_step, adam_end_step),
                    output_dir=_adam_dir(
                        run_root,
                        seed,
                        layer,
                        head,
                        args.rank,
                        adam_start_step,
                        adam_end_step,
                        candidate_label,
                    ),
                    device=args.device,
                    layer=layer,
                    head=head,
                    rank=args.rank,
                    context_stage=args.context_stage,
                    layernorm_mode=args.layernorm_mode,
                    score_query_role=args.score_query_role,
                    support_key_role=args.support_role,
                    distractor_key_role=args.distractor_role,
                    record_side=args.record_side,
                    route_pair_type=args.route_pair_type,
                    route_pair_source_type=args.route_pair_source_type,
                    max_route_pairs_per_type=args.max_route_pairs_per_type,
                    min_route_pairs_per_type=args.min_route_pairs_per_type,
                    loss_scope=args.loss_scope,
                    overwrite=args.overwrite,
                )
                _run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
