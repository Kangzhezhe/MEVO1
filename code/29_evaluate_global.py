"""评估全局单模型预测，并输出 ROUGE / BLEU 汇总。"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline_common import load_config, read_jsonl, resolve_path, score, write_json  # noqa: E402
from common.metrics import corpus_bleu  # noqa: E402


def run(config: dict) -> dict:
    source = resolve_path(config["paths"]["prediction_dir"]) / "test_predictions.jsonl"
    rows = read_jsonl(source)
    # 与旧 MEVO 的 first-50 评估保持同一用户口径。默认不筛选，完整 Global
    # 实验仍可评估全部 test 用户；设置 user_limit 后按稳定的 user_id 排序取前 N 个。
    user_limit = int(config.get("evaluation", {}).get("user_limit", 0))
    if user_limit > 0:
        selected = sorted({str(row.get("user_id", "")) for row in rows})[:user_limit]
        rows = [row for row in rows if str(row.get("user_id", "")) in selected]
        print(f"Global evaluation user filter: users={len(selected)} queries={len(rows)}", flush=True)
    users = {str(row.get("user_id", "")).strip() for row in rows}
    users.discard("")
    expected_users = int(config.get("evaluation", {}).get("expected_users", 0))
    expected_queries = int(config.get("evaluation", {}).get("expected_queries", 0))
    if expected_users > 0 and len(users) != expected_users:
        raise ValueError(
            f"评估用户口径错误：期望 {expected_users} 用户，实际 {len(users)} 用户、{len(rows)} Query"
        )
    if expected_queries > 0 and len(rows) != expected_queries:
        raise ValueError(
            f"评估 Query 口径错误：期望 {expected_queries}，实际 {len(rows)}（用户数 {len(users)}）"
        )
    # 无效或空生成也是模型失败，必须以零分计入均值，不能从分母中删除。
    values = [score(str(row.get("prediction", "")), str(row["target"])) for row in rows]
    if not values:
        raise ValueError("没有待评估预测")
    predictions = [str(row.get("prediction", "")) for row in rows]
    references = [str(row["target"]) for row in rows]
    bleu = corpus_bleu(predictions, references)
    report = {"protocol": "single_output", "users": len(users), "queries": len(rows), "valid_predictions": sum(bool(str(row.get("prediction", "")).strip()) for row in rows), "rouge_1": statistics.mean(item["rouge_1"] for item in values), "rouge_l": statistics.mean(item["rouge_l"] for item in values), "sacrebleu": float(bleu["score"]), "prediction_error_count": sum(bool(row.get("error")) for row in rows)}
    destination = resolve_path(config["paths"]["reports_dir"]) / "global_test_report.json"
    write_json(destination, report)
    print(f"Global evaluation -> {destination}; report={report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="29 - evaluate global predictions")
    parser.add_argument("--config", default=str(HERE.parent / "config_global.yaml"))
    parser.add_argument("--prediction-subdir", default="")
    parser.add_argument("--report-subdir", default="")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.prediction_subdir:
        config["paths"]["prediction_dir"] = str(resolve_path(config["paths"]["prediction_dir"]) / args.prediction_subdir)
    if args.report_subdir:
        config["paths"]["reports_dir"] = str(resolve_path(config["paths"]["reports_dir"]) / args.report_subdir)
    run(config)
