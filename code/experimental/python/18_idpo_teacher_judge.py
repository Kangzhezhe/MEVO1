"""阶段 18：Teacher 对同一 Prompt 的 on-policy responses 做偏好判断。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from common.concurrency import BoundedJobError, run_bounded  # noqa: E402
from pipeline_common import load_config, read_jsonl, teacher_client, write_json, write_jsonl  # noqa: E402
from idpo_common import idpo_path  # noqa: E402


def _prompt(rollout: dict[str, Any]) -> str:
    candidates = []
    for item in rollout["responses"]:
        trace = dict(item["trace"])
        trace.pop("parent_id", None)
        trace.pop("parent_a_id", None)
        trace.pop("parent_b_id", None)
        candidates.append(
            {
                "response_id": item["response_id"],
                "output": item["output"],
                "edit_summary": trace,
            }
        )
    payload = {
        "current_input": rollout["current_input"],
        "retrieved_history": rollout["retrieved_history"],
        "parent_a": rollout["parent_a"],
        "parent_b": rollout.get("parent_b"),
        "operation_type": rollout["operation_type"],
        "responses": candidates,
    }
    return (
        "You are a preference judge for personalized scholarly output editing.\n"
        "Choose the response that best preserves the current task contribution while applying "
        "a user preference actually supported by the visible history. Prefer faithful, concise, "
        "natural output. Do not assume a hidden answer, do not mention gold, reference, ROUGE, "
        "or any evaluation metric. If the responses are indistinguishable or unsupported, return tie.\n\n"
        "Return JSON only with this schema:\n"
        '{"decision":"prefer_a|prefer_b|tie","chosen_id":"...","rejected_id":"...",'
        '"confidence":0.0,"evidence_ids":[],"reason":"..."}\n\n'
        "PAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _mock_judge(rollout: dict[str, Any]) -> dict[str, Any]:
    responses = rollout["responses"]
    if len(responses) < 2:
        return {"decision": "tie", "chosen_id": "", "rejected_id": "", "confidence": 0.0, "evidence_ids": [], "reason": "Not enough responses."}
    return {
        "decision": "prefer_a",
        "chosen_id": str(responses[0]["response_id"]),
        "rejected_id": str(responses[1]["response_id"]),
        "confidence": 0.8,
        "evidence_ids": [],
        "reason": "The first response is selected for the deterministic smoke test.",
    }


def _validate(value: Any, rollout: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Judge response 必须是 JSON object")
    ids = {str(item["response_id"]) for item in rollout["responses"]}
    decision = str(value.get("decision", "")).lower()
    confidence = float(value.get("confidence", 0.0))
    if decision not in {"prefer_a", "prefer_b", "tie"}:
        raise ValueError("decision 必须是 prefer_a/prefer_b/tie")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence 必须在 [0,1]")
    chosen = str(value.get("chosen_id", ""))
    rejected = str(value.get("rejected_id", ""))
    if decision != "tie":
        if chosen not in ids or rejected not in ids or chosen == rejected:
            raise ValueError("chosen/rejected 必须是两个不同的有效 response_id")
        if confidence < float(settings["minimum_confidence"]):
            raise ValueError("Judge confidence 低于门槛")
    evidence = [str(item) for item in value.get("evidence_ids", [])]
    valid_evidence = {str(item["id"]) for item in rollout["retrieved_history"]}
    if set(evidence) - valid_evidence:
        raise ValueError("Judge 使用了不可见的 evidence_id")
    reason = str(value.get("reason", "")).strip()
    if not reason or len(reason) > 500:
        raise ValueError("Judge reason 必须是简短非空文本")
    return {
        "decision": decision,
        "chosen_id": chosen,
        "rejected_id": rejected,
        "confidence": confidence,
        "evidence_ids": evidence,
        "reason": reason,
    }


def _judge_one(rollout: dict[str, Any], config: dict) -> dict[str, Any]:
    settings = config["idpo"]
    if not bool(rollout.get("minimum_responses_met", False)):
        return {**rollout, "judge_status": "insufficient_responses", "judge": None}
    prompt = _prompt(rollout)
    client = teacher_client(config)
    last_error: Exception | None = None
    for attempt in range(int(settings["judge_retries"]) + 1):
        current = prompt
        try:
            if str(config["teacher"].get("provider")) == "mock":
                raw = _mock_judge(rollout)
            else:
                if attempt:
                    current += (
                        "\n\nSCHEMA RETRY: Generate a fresh JSON response. Do not include any previous "
                        f"answer. Previous validation error: {last_error}"
                    )
                raw, _ = client.json(f"idpo_pairwise_judge_retry_{attempt}", current, {
                    "retrieved_profile": rollout["retrieved_history"],
                    "responses": rollout["responses"],
                })
            judge = _validate(raw, rollout, settings)
            status = "accepted" if judge["decision"] != "tie" else "tie"
            return {**rollout, "judge_status": status, "judge": judge}
        except Exception as error:
            last_error = error
            if str(config["teacher"].get("provider")) != "mock":
                try:
                    client.invalidate(f"idpo_pairwise_judge_retry_{attempt}", current)
                except Exception:
                    pass
    return {**rollout, "judge_status": "invalid", "judge": None, "judge_error": str(last_error)}


def run(config: dict, split: str = "validation") -> dict[str, Any]:
    settings = config["idpo"]
    round_index = int(settings["round"])
    source_path = idpo_path(config, round_index, f"{split}_rollouts.jsonl")
    destination = idpo_path(config, round_index, f"{split}_judged.jsonl")
    source = read_jsonl(source_path)
    existing = read_jsonl(destination) if bool(settings.get("resume_existing", True)) and destination.exists() else []
    done = {str(row["rollout_id"]): row for row in existing}
    jobs = [row for row in source if str(row["rollout_id"]) not in done]
    print(f"IDPO judge jobs={len(jobs)} resume={len(done)}/{len(source)} concurrency={settings['judge_concurrency']}", flush=True)

    def checkpoint() -> None:
        write_jsonl(destination, [done[key] for key in sorted(done)])

    def complete(row, result, completed):
        done[str(row["rollout_id"])] = result
        if completed % int(settings.get("checkpoint_every", 10)) == 0:
            checkpoint()
        print(f"IDPO judge progress {completed}/{len(jobs)} rollout={row['rollout_id']} status={result['judge_status']}", flush=True)

    try:
        run_bounded(
            jobs,
            lambda row: _judge_one(row, config),
            complete,
            max_workers=int(settings["judge_concurrency"]),
            thread_name_prefix="idpo-judge",
        )
    except BoundedJobError as error:
        checkpoint()
        raise RuntimeError(f"IDPO Judge failed rollout={error.job.get('rollout_id')}: {error.error}") from error.error
    checkpoint()
    rows = [done[key] for key in sorted(done)]
    accepted = sum(row.get("judge_status") == "accepted" for row in rows)
    report = {
        "round": round_index,
        "split": split,
        "rollouts": len(rows),
        "accepted_pairs": accepted,
        "tie": sum(row.get("judge_status") == "tie" for row in rows),
        "insufficient_responses": sum(row.get("judge_status") == "insufficient_responses" for row in rows),
        "invalid": sum(row.get("judge_status") == "invalid" for row in rows),
        "teacher_sees_hidden_target": False,
        "minimum_confidence": float(settings["minimum_confidence"]),
    }
    write_json(idpo_path(config, round_index, f"{split}_judge_report.json"), report)
    print(f"IDPO judged -> {destination}; report={report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="18 - IDPO Teacher pairwise judge")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    run(load_config(args.config), args.split)


if __name__ == "__main__":
    main()
