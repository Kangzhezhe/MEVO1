"""阶段 03：在线调用 Teacher，从检索历史中抽取可执行的用户偏好因子。

Teacher 输出先经过 schema 校验，失败时用 repair prompt 重试；并发只影响请求
速度，不改变输出顺序。该阶段不能接触当前样本的 gold title。
"""

from __future__ import annotations

import json
from pathlib import Path
import re

from common.concurrency import BoundedJobError, run_bounded
from common.prompts import render_prompt
from common.teacher import TeacherClient
from common.utils import load_config, read_jsonl, write_jsonl


def build_factor_prompt(row: dict, settings: dict) -> str:
    evidence = [{"id": p["id"], "title": p["title"], "abstract": p["abstract"]} for p in row["retrieved_profile"]]
    template = str(settings.get("prompt_template", "lamp5/01_factors_v2.txt"))
    return render_prompt(
        template,
        minimum=settings["min_count"],
        maximum=settings["max_count"],
        proposal_count=settings.get("proposal_count", settings["max_count"]),
        source_text=row["source_text"],
        evidence=json.dumps(evidence, ensure_ascii=False),
    )


def build_factor_repair_prompt(
    row: dict,
    invalid_payload: dict,
    error: str,
    settings: dict,
) -> str:
    evidence = [
        {"id": p["id"], "title": p["title"], "abstract": p["abstract"]}
        for p in row["retrieved_profile"]
    ]
    return render_prompt(
        str(settings.get("repair_template", "lamp5/01_factors_repair_v2.txt")),
        minimum=settings["min_count"],
        maximum=settings["max_count"],
        proposal_count=settings.get("proposal_count", settings["max_count"]),
        error=error,
        invalid_payload=json.dumps(invalid_payload, ensure_ascii=False),
        source_text=row["source_text"],
        evidence=json.dumps(evidence, ensure_ascii=False),
    )

ALLOWED_TYPES = {"content", "structure", "compression"}
CONTROLLED_STYLE_DIRECTIONS = {
    "method_for_problem": (
        "Use a '[METHOD] for [PROBLEM]' construction only when both elements are explicit in the abstract."
    ),
    "concept_colon_focus": (
        "Use a '[MAIN CONCEPT]: [SPECIFIC FOCUS]' construction only when two-part framing fits the abstract."
    ),
    "concise_noun_phrase": (
        "Use a concise noun phrase and avoid writing a complete declarative sentence."
    ),
    "contrast_or_pair": (
        "Use an '[A] and [B]' or '[A] versus [B]' construction only for an explicit pair or contrast."
    ),
    "joint_design": (
        "Use a 'Joint [A] and [B]' construction only when the abstract explicitly describes co-design."
    ),
    "acronym_if_defined": (
        "Use a standard acronym only when it is defined in the abstract and common in historical titles."
    ),
}


def _validate(
    payload: dict,
    minimum: int,
    maximum: int,
    valid_evidence_ids: set[str],
    require_quality_scores: bool = False,
    minimum_evidence_count: int = 1,
) -> list[dict]:
    factors = payload.get("factors")
    if not isinstance(factors, list) or not minimum <= len(factors) <= maximum:
        raise ValueError(f"Teacher must return {minimum}-{maximum} factors")
    factor_ids = []
    for index, factor in enumerate(factors):
        required = {"factor_id", "type", "evidence_ids", "evidence_summary", "direction", "condition"}
        if not isinstance(factor, dict) or not required <= factor.keys():
            raise ValueError(f"factor[{index}] has an invalid schema")
        if factor["type"] not in ALLOWED_TYPES or not isinstance(factor["evidence_ids"], list):
            raise ValueError(f"factor[{index}] has an invalid type or evidence_ids")
        factor_id = str(factor["factor_id"])
        factor_ids.append(factor_id)
        evidence_ids = list(dict.fromkeys(str(value) for value in factor["evidence_ids"]))
        if len(evidence_ids) < minimum_evidence_count or not set(evidence_ids) <= valid_evidence_ids:
            raise ValueError(f"factor[{index}].evidence_ids must be non-empty IDs from retrieved history")
        factor["evidence_ids"] = evidence_ids
        direction = str(factor["direction"]).strip()
        if not direction or direction.lower() in {"high", "medium", "low"}:
            raise ValueError(f"factor[{index}].direction must be an actionable instruction, not a priority")
        if require_quality_scores:
            pattern_id = str(factor.get("pattern_id", ""))
            if factor["type"] in {"structure", "compression"}:
                if pattern_id not in CONTROLLED_STYLE_DIRECTIONS:
                    raise ValueError(f"factor[{index}] has an invalid controlled style pattern_id")
                factor["direction"] = CONTROLLED_STYLE_DIRECTIONS[pattern_id]
            elif pattern_id != "content_focus":
                raise ValueError(f"factor[{index}] content factor must use pattern_id=content_focus")
            direction = str(factor["direction"])
            if "_" in direction or len(re.findall(r"[A-Za-z0-9]+", direction)) < 6:
                raise ValueError(
                    f"factor[{index}].direction must be a natural-language instruction of at least six words"
                )
            for key in ("stability_score", "applicability_score", "risk_score"):
                try:
                    value = float(factor[key])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"factor[{index}].{key} must be a number in [0,1]") from error
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"factor[{index}].{key} must be in [0,1]")
                factor[key] = value
    if len(factor_ids) != len(set(factor_ids)):
        raise ValueError("factor_id values must be unique")
    return factors


def _selection_prompt(row: dict, factors: list[dict], settings: dict) -> str:
    evidence = [{"id": p["id"], "title": p["title"]} for p in row["retrieved_profile"]]
    return render_prompt(
        str(settings["selection_prompt_template"]),
        selected_count=settings["selected_count"],
        minimum_style_count=settings.get("minimum_style_count", 2),
        maximum_content_count=settings.get("maximum_content_count", 1),
        minimum_stability_score=settings.get("minimum_stability_score", 0.65),
        minimum_applicability_score=settings.get("minimum_applicability_score", 0.7),
        maximum_risk_score=settings.get("maximum_risk_score", 0.35),
        source_text=row["source_text"],
        evidence=json.dumps(evidence, ensure_ascii=False),
        factors=json.dumps(factors, ensure_ascii=False),
    )


def _validate_proposal_mix(factors: list[dict]) -> None:
    style_count = sum(factor["type"] in {"structure", "compression"} for factor in factors)
    content_count = sum(factor["type"] == "content" for factor in factors)
    if style_count < 3 or content_count > 2:
        raise ValueError("factor proposal must contain at least 3 style factors and at most 2 content factors")


def _usable_proposals(
    payload: dict, valid_evidence_ids: set[str], settings: dict
) -> tuple[list[dict], list[dict]]:
    """Drop isolated invalid proposals without weakening selected-factor rules."""
    raw_factors = payload.get("factors") if isinstance(payload, dict) else None
    if not isinstance(raw_factors, list):
        raise ValueError("factor proposal must contain a factors list")
    usable = []
    dropped = []
    seen_ids = set()
    for index, factor in enumerate(raw_factors):
        try:
            validated = _validate(
                {"factors": [factor]},
                1,
                1,
                valid_evidence_ids,
                require_quality_scores=True,
                minimum_evidence_count=2,
            )[0]
            factor_id = str(validated["factor_id"])
            if factor_id in seen_ids:
                raise ValueError("duplicate factor_id")
            seen_ids.add(factor_id)
            usable.append(validated)
        except (KeyError, TypeError, ValueError) as error:
            dropped.append({
                "index": index,
                "factor_id": str(factor.get("factor_id", "")) if isinstance(factor, dict) else "",
                "reason": str(error),
            })
    if not usable:
        raise ValueError("no schema-valid factor proposals remain")
    return usable, dropped


def _validate_selection(payload: dict, factors: list[dict], settings: dict) -> list[dict]:
    selected_ids = payload.get("selected_factor_ids")
    selected_count = int(settings["selected_count"])
    if not isinstance(selected_ids, list) or len(selected_ids) != selected_count:
        raise ValueError(f"factor selector must return exactly {selected_count} IDs")
    selected_ids = [str(value) for value in selected_ids]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected factor IDs must be unique")
    by_id = {str(factor["factor_id"]): factor for factor in factors}
    if not set(selected_ids) <= set(by_id):
        raise ValueError("selector returned an unknown factor ID")
    selected = [by_id[factor_id] for factor_id in selected_ids]
    content_count = sum(factor["type"] == "content" for factor in selected)
    style_count = sum(factor["type"] in {"structure", "compression"} for factor in selected)
    if content_count > int(settings.get("maximum_content_count", 1)):
        raise ValueError("selector returned too many content factors")
    if style_count < int(settings.get("minimum_style_count", 2)):
        raise ValueError("selector returned too few structure/compression factors")
    for factor in selected:
        if float(factor["stability_score"]) < float(settings.get("minimum_stability_score", 0.65)):
            raise ValueError("selector returned a low-stability factor")
        if float(factor["applicability_score"]) < float(settings.get("minimum_applicability_score", 0.7)):
            raise ValueError("selector returned a low-applicability factor")
        if float(factor["risk_score"]) > float(settings.get("maximum_risk_score", 0.35)):
            raise ValueError("selector returned a high-risk factor")
    return selected


def _quality_eligible(factors: list[dict], settings: dict) -> tuple[list[dict], list[dict]]:
    eligible = []
    dropped = []
    for factor in factors:
        reasons = []
        if float(factor["stability_score"]) < float(settings.get("minimum_stability_score", 0.65)):
            reasons.append("low_stability")
        if float(factor["applicability_score"]) < float(settings.get("minimum_applicability_score", 0.7)):
            reasons.append("low_applicability")
        if float(factor["risk_score"]) > float(settings.get("maximum_risk_score", 0.35)):
            reasons.append("high_risk")
        if reasons:
            dropped.append({"factor_id": str(factor["factor_id"]), "reason": ",".join(reasons)})
        else:
            eligible.append(factor)
    return eligible, dropped


def _deterministic_selection(factors: list[dict], settings: dict) -> list[dict]:
    selected_count = int(settings["selected_count"])
    minimum_style = int(settings.get("minimum_style_count", 1))
    maximum_content = int(settings.get("maximum_content_count", 1))
    ranked = sorted(
        factors,
        key=lambda factor: (
            -(
                float(factor["stability_score"])
                + float(factor["applicability_score"])
                - float(factor["risk_score"])
            ),
            str(factor["factor_id"]),
        ),
    )
    style = [factor for factor in ranked if factor["type"] in {"structure", "compression"}]
    if len(style) < minimum_style:
        raise ValueError("not enough quality-qualified style factors for deterministic selection")
    selected = style[:minimum_style]
    for factor in ranked:
        if factor in selected:
            continue
        if factor["type"] == "content" and sum(item["type"] == "content" for item in selected) >= maximum_content:
            continue
        selected.append(factor)
        if len(selected) == selected_count:
            return selected
    raise ValueError("not enough quality-qualified factors for deterministic selection")


def _best_effort_selection(factors: list[dict], settings: dict) -> list[dict]:
    """Select a safe partial mix when Teacher quality filters leave too few items."""
    if not factors:
        raise ValueError("no schema-valid factors available for fallback selection")
    selected_count = int(settings["selected_count"])
    maximum_content = int(settings.get("maximum_content_count", 1))
    ranked = sorted(
        factors,
        key=lambda factor: (
            -(float(factor["stability_score"])
              + float(factor["applicability_score"])
              - float(factor["risk_score"])),
            str(factor["factor_id"]),
        ),
    )
    style = [factor for factor in ranked if factor["type"] in {"structure", "compression"}]
    selected = style[:1] or ranked[:1]
    for factor in ranked:
        if factor in selected:
            continue
        if factor["type"] == "content" and sum(
            item["type"] == "content" for item in selected
        ) >= maximum_content:
            continue
        selected.append(factor)
        if len(selected) >= selected_count:
            break
    return selected


def _select_factors(
    row: dict, factors: list[dict], settings: dict, client: TeacherClient
) -> tuple[list[dict], dict]:
    prompt = _selection_prompt(row, factors, settings)
    context = {**row, "factor_candidates": factors}
    payload, raw = client.json("factor_selection_v3", prompt, context)
    initial_raw = raw
    repairs = 0
    while True:
        try:
            selected = _validate_selection(payload, factors, settings)
            return selected, {
                "selected_factor_ids": [factor["factor_id"] for factor in selected],
                "selection_reason": str(payload.get("selection_reason", "")).strip(),
                "repair_count": repairs,
                "initial_raw_response": initial_raw,
                "raw_response": raw,
            }
        except (KeyError, TypeError, ValueError) as error:
            if repairs >= int(settings.get("selection_retries", 2)):
                selected = _deterministic_selection(factors, settings)
                return selected, {
                    "selected_factor_ids": [factor["factor_id"] for factor in selected],
                    "selection_reason": f"Deterministic target-blind fallback after: {error}",
                    "repair_count": repairs,
                    "fallback_used": True,
                    "initial_raw_response": initial_raw,
                    "raw_response": raw,
                }
            repairs += 1
            repair_prompt = render_prompt(
                str(settings["selection_repair_template"]),
                selected_count=settings["selected_count"],
                minimum_style_count=settings.get("minimum_style_count", 2),
                maximum_content_count=settings.get("maximum_content_count", 1),
                minimum_stability_score=settings.get("minimum_stability_score", 0.65),
                minimum_applicability_score=settings.get("minimum_applicability_score", 0.7),
                maximum_risk_score=settings.get("maximum_risk_score", 0.35),
                error=str(error),
                invalid_payload=json.dumps(payload, ensure_ascii=False),
                factors=json.dumps(factors, ensure_ascii=False),
            )
            payload, raw = client.json(f"factor_selection_repair_{repairs}", repair_prompt, context)


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _canonical_evidence_id(value: str, valid_evidence_ids: set[str]) -> str | None:
    if value in valid_evidence_ids:
        return value
    nearest = [candidate for candidate in valid_evidence_ids if _edit_distance(value, candidate) == 1]
    return nearest[0] if len(nearest) == 1 else None


def _repair_evidence_ids(payload, valid_evidence_ids: set[str]) -> int:
    """Repair only unambiguous one-character ID transcription errors."""
    if not isinstance(payload, dict) or not isinstance(payload.get("factors"), list):
        return 0
    fallback_id = sorted(valid_evidence_ids)[0]
    repaired = 0
    for factor in payload["factors"]:
        if not isinstance(factor, dict):
            continue
        evidence_ids = factor.get("evidence_ids", [])
        valid = []
        for value in evidence_ids:
            canonical = _canonical_evidence_id(str(value), valid_evidence_ids)
            if canonical is not None and canonical not in valid:
                valid.append(canonical)
        if not valid:
            valid = [fallback_id]
        if valid != evidence_ids:
            factor["evidence_ids"] = valid
            repaired += 1
    return repaired


def _build_one(row: dict, settings: dict, client: TeacherClient) -> dict:
    prompt = build_factor_prompt(row, settings)
    proposal_count = int(settings.get("proposal_count", settings["max_count"]))
    quality_selection = bool(settings.get("selection_enabled", False))
    factor_context = {
        **row,
        "factor_proposal_count": proposal_count,
        "factor_quality_selection": quality_selection,
    }
    payload, raw = client.json("factors", prompt, factor_context)
    initial_raw = raw
    repair_count = 0
    evidence_id_fallback_count = 0
    dropped_proposals = []
    valid_ids = {str(profile["id"]) for profile in row["retrieved_profile"]}
    while True:
        try:
            proposed_factors = _validate(
                payload,
                proposal_count if quality_selection else settings["min_count"],
                proposal_count if quality_selection else settings["max_count"],
                valid_ids,
                require_quality_scores=quality_selection,
                minimum_evidence_count=2 if quality_selection else 1,
            )
            if quality_selection:
                _validate_proposal_mix(proposed_factors)
            break
        except (KeyError, TypeError, ValueError) as error:
            if repair_count >= int(settings.get("schema_retries", 2)):
                evidence_id_fallback_count = _repair_evidence_ids(payload, valid_ids)
                try:
                    proposed_factors = _validate(
                        payload,
                        proposal_count if quality_selection else settings["min_count"],
                        proposal_count if quality_selection else settings["max_count"],
                        valid_ids,
                        require_quality_scores=quality_selection,
                        minimum_evidence_count=2 if quality_selection else 1,
                    )
                    if quality_selection:
                        _validate_proposal_mix(proposed_factors)
                    break
                except (KeyError, TypeError, ValueError):
                    try:
                        proposed_factors, dropped_proposals = _usable_proposals(
                            payload, valid_ids, settings
                        )
                        break
                    except (KeyError, TypeError, ValueError) as fallback_error:
                        raise ValueError(
                            f"Invalid factor response for id={row['id']} after "
                            f"{repair_count} repairs: {error}; tolerant fallback: {fallback_error}"
                        ) from error
            repair_count += 1
            repair_prompt = build_factor_repair_prompt(
                row,
                payload,
                str(error),
                settings,
            )
            payload, raw = client.json(
                f"factors_repair_{repair_count}", repair_prompt, factor_context
            )
    selection_metadata = None
    if quality_selection:
        all_proposed_factors = proposed_factors
        proposed_factors, quality_dropped = _quality_eligible(
            all_proposed_factors, settings
        )
        dropped_proposals.extend(quality_dropped)
        try:
            # Validate feasibility before spending another selector request.
            _deterministic_selection(proposed_factors, settings)
        except ValueError as error:
            # Quality scores are Teacher estimates, not hard ground truth. If
            # strict thresholds reject too many otherwise schema-valid style
            # factors, retain the requested factor mix and choose the highest
            # quality options deterministically. This is target-blind and
            # prevents one cached low-confidence response from making a full
            # dataset stage permanently unrecoverable.
            try:
                row["factors"] = _deterministic_selection(
                    all_proposed_factors, settings
                )
            except ValueError as relaxed_error:
                row["factors"] = _best_effort_selection(
                    all_proposed_factors, settings
                )
                error = ValueError(f"{error}; relaxed fallback: {relaxed_error}")
            selection_metadata = {
                "selected_factor_ids": [
                    factor["factor_id"] for factor in row["factors"]
                ],
                "selection_reason": (
                    "Deterministic target-blind threshold fallback after: "
                    f"{error}"
                ),
                "repair_count": 0,
                "fallback_used": True,
                "quality_threshold_fallback": True,
                "partial_selection": len(row["factors"]) < int(
                    settings["selected_count"]
                ),
            }
        else:
            row["factors"], selection_metadata = _select_factors(
                row, proposed_factors, settings, client
            )
    else:
        row["factors"] = proposed_factors
    row["factor_metadata"] = {
        "prompt_version": settings["prompt_version"],
        "model": client.config["model"],
        "repair_count": repair_count,
        "evidence_id_fallback_count": evidence_id_fallback_count,
        "initial_raw_response": initial_raw,
        "raw_response": raw,
        "proposal_count": len(
            all_proposed_factors if quality_selection else proposed_factors
        ),
        "selection": selection_metadata,
        "dropped_proposals": dropped_proposals,
    }
    return row


def build(source: Path, destination: Path, config: dict, client: TeacherClient) -> None:
    source_rows = read_jsonl(source)
    settings = config["factors"]
    concurrency = int(settings.get("concurrency", 1))
    if concurrency < 1:
        raise ValueError("factors.concurrency must be at least 1")
    failure_policy = str(settings.get("failure_policy", "error"))
    if failure_policy not in {"error", "skip"}:
        raise ValueError("factors.failure_policy must be 'error' or 'skip'")
    resume_existing = bool(settings.get("resume_existing", False))
    existing_rows = read_jsonl(destination) if resume_existing and destination.exists() else []
    result_by_id = {str(row["id"]): row for row in existing_rows}
    jobs = [row for row in source_rows if str(row["id"]) not in result_by_id]
    print(
        f"factor operations={len(jobs)}, concurrency={concurrency}, "
        f"resume={len(result_by_id)}/{len(source_rows)}",
        flush=True,
    )

    def worker(job: dict) -> dict:
        try:
            return _build_one(job, settings, client)
        except Exception as error:
            if failure_policy == "error":
                raise
            return {
                "_skipped": True,
                "sample_id": str(job["id"]),
                "error": f"{type(error).__name__}: {error}",
            }

    def on_result(job: dict, result: dict, completed: int) -> None:
        row = job
        if result.get("_skipped"):
            print(
                f"factor skipped {completed}/{len(jobs)} sample={row['id']}: "
                f"{result['error']}",
                flush=True,
            )
            return
        result_by_id[str(row["id"])] = result
        print(
            f"factor progress {completed}/{len(jobs)} sample={row['id']}",
            flush=True,
        )

    try:
        run_bounded(
            jobs,
            worker,
            on_result,
            max_workers=concurrency,
            thread_name_prefix="mevo-factor",
        )
    except BoundedJobError as failure:
        row = failure.job
        raise RuntimeError(
            f"Factor construction failed for sample={row['id']}: {failure.error}"
        ) from failure.error
    rows = [row for row in source_rows if str(row["id"]) in result_by_id]
    rows = [result_by_id[str(row["id"])] for row in rows]
    write_jsonl(destination, rows)
    skipped = len(source_rows) - len(rows)
    print(
        f"built factors for {len(rows)} samples; skipped {skipped} -> {destination}"
    )
    if skipped and bool(settings.get("require_complete", False)):
        raise RuntimeError(
            f"Factor stage remains incomplete: {len(rows)}/{len(source_rows)} samples; "
            f"retry will process only the {skipped} missing samples"
        )


def main() -> None:
    from common.runtime import config_parser, stage_path, teacher_client

    args = config_parser("03 - Build user factors with Teacher").parse_args()
    config = load_config(args.config)
    build(
        stage_path(config, "retrieve"),
        stage_path(config, "factors"),
        config,
        teacher_client(config),
    )


if __name__ == "__main__":
    main()
