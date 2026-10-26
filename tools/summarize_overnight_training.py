"""Select an isolated candidate by public mask AP and assemble a review entrypoint.

This script does not train, infer, modify incumbent weights, or promote a model.
The small same-video factory split is diagnostic evidence, never the selector.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
from urllib.parse import quote


METRICS = ("precision", "recall", "map50", "map50_95")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def metric(metrics, split, group, key):
    return number(metrics.get(split, {}).get(group, {}).get(key))


def relative_url(path, output):
    path = Path(path).resolve()
    try:
        return quote(os.path.relpath(path, output).replace("\\", "/"), safe="/.:_-~")
    except ValueError:
        return path.as_uri()


def experiment_label(experiment):
    name = experiment.get("name", experiment.get("stage", "candidate"))
    text = f"{name} {experiment.get('data', '')}".lower()
    if "copypaste" in text or "synthetic" in text:
        return "合成增强组", "synthetic"
    if "control" in text or "baseline" in text or "real" in text:
        return "真实数据对照组", "real_control"
    return name, "candidate"


def check_screening(baseline, candidate):
    checks = []
    for split, group, key, threshold, name in (
        ("public_validation", "mask", "map50_95", 0.0, "公共 mask mAP50-95 未超过原基线"),
        ("public_validation", "box", "map50_95", -0.01, "公共 box mAP50-95 下降超过 0.01"),
        ("factory_labeled", "mask", "recall", -0.05, "现场 mask recall 下降超过 0.05"),
        ("factory_labeled", "box", "recall", -0.05, "现场 box recall 下降超过 0.05"),
    ):
        b, c = metric(baseline, split, group, key), metric(candidate, split, group, key)
        delta = None if b is None or c is None else c - b
        triggered = None if delta is None else (delta <= threshold if threshold == 0 else delta < threshold - 1e-12)
        checks.append({"name": name, "split": split, "group": group, "metric": key,
                       "baseline": b, "candidate": c, "delta": delta,
                       "threshold": threshold, "triggered": triggered,
                       "meaning": "screening_heuristic_not_acceptance_criterion"})
    b = number(baseline.get("factory_operating_point", {}).get("recall"))
    c = number(candidate.get("factory_operating_point", {}).get("recall"))
    if b is not None and c is not None:
        checks.append({"name": "现场固定阈值 mask recall 下降超过 0.05",
                       "split": "factory_operating_point", "group": "mask", "metric": "recall",
                       "baseline": b, "candidate": c, "delta": c - b, "threshold": -0.05,
                       "triggered": c - b < -0.05 - 1e-12,
                       "meaning": "screening_heuristic_not_acceptance_criterion"})
    if "factory_operating_point" in baseline or "factory_operating_point" in candidate:
        baseline_point = baseline.get("factory_operating_point", {})
        candidate_point = candidate.get("factory_operating_point", {})
        for key, threshold, name in (
            ("precision", -0.05, "现场固定阈值 precision 下降超过 0.05"),
            ("fp", 0.0, "现场固定阈值误报 FP 增加"),
        ):
            b, c = number(baseline_point.get(key)), number(candidate_point.get(key))
            comparable = b is not None and c is not None
            delta = c - b if comparable else None
            triggered = (delta < threshold - 1e-12 if key == "precision" else delta > threshold) if comparable else None
            if comparable and triggered:
                name += f"（{b:.4f} → {c:.4f}）" if key == "precision" else f"（{b:g} → {c:g}）"
            if not comparable:
                name = f"现场固定阈值 {key} 未定义或缺失，无法比较，需人工查看诊断结果"
            checks.append({"name": name, "split": "factory_operating_point", "group": "mask",
                           "metric": key, "baseline": b, "candidate": c, "delta": delta,
                           "threshold": threshold, "triggered": triggered,
                           "review_required": not comparable,
                           "comparison_status": "evaluated" if comparable else "undefined_or_missing",
                           "meaning": "small_sample_same_video_regression_alert_not_independent_accuracy"})
    return checks


def gather(config, output):
    status_path = output / "STATUS.json"
    state = read_json(status_path) if status_path.is_file() else {}
    completed_path = output / "completed_experiments.json"
    completed = read_json(completed_path) if completed_path.is_file() else []
    if not isinstance(completed, list):
        raise ValueError("completed_experiments.json must contain a list")
    planned = {row["name"]: row for row in config["experiments"]}
    incumbent_digest = sha256(config["baseline"])
    results, failures = [], list(state.get("failures", []))
    for item in completed:
        name = item.get("stage", item.get("name"))
        if name not in planned:
            failures.append({"experiment": name, "error": "Completed experiment is absent from config; excluded from selection."})
            continue
        evaluation = output / "evaluation" / name
        metrics_path = evaluation / "metrics.json"
        try:
            evaluation_report = read_json(metrics_path)
            if evaluation_report.get("status") != "complete_acceptance_pending":
                raise ValueError(f"Evaluation not complete: {evaluation_report.get('status')}")
            model_metrics = evaluation_report["metrics"]
            baseline, candidate = model_metrics["baseline"], model_metrics["candidate"]
            if metric(candidate, "public_validation", "mask", "map50_95") is None:
                raise ValueError("Candidate public mask mAP50-95 is not finite")
            if metric(baseline, "public_validation", "mask", "map50_95") is None:
                raise ValueError("Baseline public mask mAP50-95 is not finite")
            # Only a fully populated public segmentation evaluation is eligible.
            for role, values in model_metrics.items():
                for group in ("box", "mask"):
                    for key in METRICS:
                        if metric(values, "public_validation", group, key) is None:
                            raise ValueError(f"Missing {role} public {group} {key}")
            weight = Path(item["best"]).resolve()
            evaluated_weight = evaluation_report.get("weights", {}).get("candidate", {})
            evaluated_baseline = evaluation_report.get("weights", {}).get("baseline", {})
            if evaluated_baseline.get("sha256") and evaluated_baseline["sha256"] != incumbent_digest:
                raise ValueError("Evaluated baseline checksum differs from configured incumbent")
            digest = sha256(weight)
            if evaluated_weight.get("sha256") and evaluated_weight["sha256"] != digest:
                raise ValueError("Candidate checksum differs from evaluated checkpoint")
            if item.get("sha256") and item["sha256"] != digest:
                raise ValueError("Candidate checksum differs from training result")
            label, kind = experiment_label(planned[name])
            checks = check_screening(baseline, candidate)
            results.append({"name": name, "label": label, "kind": kind,
                            "weight": str(weight), "sha256": digest,
                            "training_result": item, "evaluation": str(evaluation),
                            "evaluation_metadata": {key: evaluation_report.get(key) for key in
                                ("created_at", "settings", "runtime", "evaluation_scope", "artifacts")},
                            "baseline": baseline, "candidate": candidate,
                            "audit": evaluation_report.get("audit", {}), "screening": checks,
                            "screening_flags": [check["name"] for check in checks
                                                if check["triggered"] or check.get("review_required")]})
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append({"experiment": name, "error": f"Excluded from final selection: {error}"})
    results.sort(key=lambda row: (-metric(row["candidate"], "public_validation", "mask", "map50_95"), row["name"]))
    missing = sorted(set(planned) - {row["name"] for row in results})
    return state, results, failures, missing


def consistency_issues(results):
    if len(results) < 2:
        return []
    issues = []
    reference = results[0]
    ref_audit = reference["audit"].get("public_validation", {})
    ref_hashes = sorted(item["sha256"] for item in ref_audit.get("items", []))
    for row in results[1:]:
        values = row["audit"].get("public_validation", {})
        hashes = sorted(item["sha256"] for item in values.get("items", []))
        if ref_hashes and hashes and hashes != ref_hashes:
            issues.append(f"{row['name']} 与其他实验公共验证图哈希不同；横向对比受限。")
        for group in ("box", "mask"):
            for key in METRICS:
                a = metric(reference["baseline"], "public_validation", group, key)
                b = metric(row["baseline"], "public_validation", group, key)
                if a is not None and b is not None and abs(a - b) > 1e-6:
                    issues.append(f"{row['name']} 的基线 {group} {key} 与另一轮差异超过 1e-6。")
    return issues


def copy_selected(selected, output, incumbent):
    source = Path(selected["weight"]).resolve()
    target = output / "selected_candidate.pt"
    if target.resolve() == Path(incumbent).resolve():
        raise ValueError("Selected candidate path must never equal incumbent checkpoint")
    if target.exists():
        if sha256(target) != selected["sha256"]:
            raise FileExistsError(f"Refusing to replace a different existing selected candidate: {target}")
    else:
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    digest = sha256(target)
    if digest != selected["sha256"]:
        raise RuntimeError("Selected checkpoint copy checksum verification failed")
    (output / "selected_candidate.sha256").write_text(f"{digest}  selected_candidate.pt\n", encoding="ascii")
    return {"path": str(target), "sha256": digest, "bytes": target.stat().st_size,
            "source": str(source), "source_experiment": selected["name"],
            "selection_metric": "public_validation.mask.map50_95",
            "selection_value": metric(selected["candidate"], "public_validation", "mask", "map50_95"),
            "factory_metrics_used_for_selection": False,
            "promoted_to_incumbent": False, "acceptance": "pending_user_review"}


def optional_json(path):
    if not Path(path).is_file():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def build_summary(config_path):
    config = read_json(config_path)
    output = Path(config["output"]).resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    incumbent = Path(config["baseline"]).resolve()
    initial_hash = sha256(incumbent)
    state, results, failures, missing = gather(config, output)
    issues = consistency_issues(results)
    expected_hash = state.get("incumbent_sha256")
    if expected_hash and expected_hash != initial_hash:
        raise RuntimeError("Incumbent checksum differs from the workflow start checksum")
    source_dataset = Path(config["source_dataset"]).resolve() if config.get("source_dataset") else None
    dataset = {"path": str(source_dataset) if source_dataset else None,
               "ready": optional_json(source_dataset / "READY.json") if source_dataset else None,
               "validation": optional_json(source_dataset / "validation.json") if source_dataset else None,
               "visual_qa": optional_json(source_dataset / "visual_qa.json") if source_dataset else None}
    acceptance_sources = optional_json(output / "acceptance_sources.json")
    source_groups = []
    if isinstance(acceptance_sources, list):
        for group, label, note in (
            ("old_1fps", "旧bag的1 FPS抽样图", "含已见训练帧与同视频邻近帧，仅作定性回归。"),
            ("orbbec_new", "新采集Orbbec彩色图", "未用于本次训练，但没有审核真值；仅作新采集画面的定性观察。"),
            ("shoe_10fps", "鞋子专项回归图", "来自旧视频，重点检查鞋子误检；不作独立测试。")):
            items = [item for item in acceptance_sources if item.get("group") == group]
            if items:
                source_groups.append({"group": group, "label": label, "images": len(items), "note": note,
                    "exact_training_matches": sum(bool(item.get("exact_match_any_training_source")) for item in items)})
    selected = copy_selected(results[0], output, incumbent) if results else None
    selected_flags = results[0]["screening_flags"] if results else []
    conclusion = "没有完成可用评估的候选，本次不能选出验收权重。"
    if selected:
        conclusion = f"已按公共 mask mAP50-95 选出{results[0]['label']}作为验收候选，未替换原基线。"
        if selected_flags:
            conclusion += "存在启发式筛查提示，不能据此声称模型整体更好。"
        else:
            conclusion += "预设筛查未触发，仍需现场目视验收；这不等于已证明整体更好。"
    synthesis = {"comparison_available": False}
    controls = [row for row in results if row["kind"] == "real_control"]
    synthetics = [row for row in results if row["kind"] == "synthetic"]
    if controls and synthetics:
        delta = metric(synthetics[0]["candidate"], "public_validation", "mask", "map50_95") - metric(controls[0]["candidate"], "public_validation", "mask", "map50_95")
        synthesis = {"comparison_available": True, "synthetic_experiment": synthetics[0]["name"],
                     "real_control_experiment": controls[0]["name"], "public_mask_map50_95_delta": delta,
                     "interpretation": "本次单种子实验中，合成组公共 mask AP 高于真实对照。" if delta > 0 else
                                       "本次单种子实验中，合成组公共 mask AP 未超过真实对照。",
                     "limitation": "只有一个训练随机种子；该差异不能证明收益稳定或能迁移到真实工厂。"}
    summary = {"schema_version": 1, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "status": "candidate_ready_review_pending" if selected else "no_eligible_candidate",
               "conclusion": conclusion, "config": str(config_path.resolve()),
               "output": str(output), "selected_candidate": selected,
               "selected_screening_flags": selected_flags, "screening_consistency_warnings": issues,
               "selection_policy": "Rank eligible candidates only by public mask mAP50-95; factory diagnostics do not determine ranking.",
               "screening_policy": "Heuristics only: public mask AP <= baseline; public box AP drop > 0.01; factory box or mask recall drop > 0.05; fixed-threshold factory precision drop > 0.05 or any FP increase. Undefined precision/FP require review rather than passing. Factory alerts describe small-sample same-video regression, not independent accuracy.",
               "partial_completion": bool(failures or missing), "planned_experiments": [row["name"] for row in config["experiments"]],
               "completed_evaluated_experiments": len(results), "missing_or_ineligible_experiments": missing,
               "failures": failures, "runner_status_snapshot": state,
               "runner_status_note": "Summary is created before the final video stage. STATUS.json is authoritative for the final workflow state.",
               "incumbent": {"path": str(incumbent), "sha256": initial_hash, "preserved": True},
               "experiments": results, "synthetic_vs_real_control": synthesis, "dataset": dataset,
               "acceptance_source_groups": source_groups,
               "acceptance_sources_manifest": str(output / "acceptance_sources.json") if acceptance_sources else None,
               "planned_full_video": {"video": "full_video/comparison.mp4", "metadata": "full_video/video_report.json",
                   "expected_frames": 909, "expected_fps": 10, "timing": "original_10fps_sequence",
                   "claim": "Qualitative same-source-video regression only; not an independent test.",
                   "status_at_summary": "available" if (output / "full_video" / "video_report.json").is_file() else "subsequent_stage"},
               "limitations": ["公共验证集参与过多轮模型选择，仅作版本比较，不是未触碰的最终测试集。",
                   "现场真值诊断仅 8 张、46 个实例，来自同一视频；与历史训练帧存在约 1–2 秒近邻关系。",
                   "8张诊断图只做了内容哈希精确隔离，没有设置训练/诊断时间缓冲区；相邻训练帧仍存在。",
                   "未标注现场图、新 Orbbec 彩色图和完整视频均仅作定性观察，不报告现场准确率提升。",
                   "已见训练图不能当作独立测试；预测数量减少不能证明误检减少。",
                   "本次没有新增大量人工描边；合成标签来自已审核多边形的确定性变换，未把模型预测当训练真值。",
                   "所选权重是离线验收候选；现用 best.pt、部署配置和旧数据保持不变。",
                   "未执行 Jetson Orin Nano 板端时延、显存和长期稳定性验收。"]}
    if sha256(incumbent) != initial_hash:
        raise RuntimeError("Incumbent checkpoint changed during summary creation")
    write_json(output / "SUMMARY.json", summary)
    render_html(summary, output)
    render_readme(summary, output)
    return summary


def render_html(summary, output):
    esc = html.escape
    def link(path, label):
        return f'<a href="{esc(relative_url(path, output))}">{esc(label)}</a>'

    def table(split):
        rows = []
        experiments = summary["experiments"]
        if not experiments:
            return "<p>本次没有完整评估数据。</p>"
        models = [("原基线", experiments[0]["baseline"], False)] + [
            (row["label"], row["candidate"], bool(summary["selected_candidate"] and row["name"] == summary["selected_candidate"]["source_experiment"])) for row in experiments]
        for label, values, selected in models:
            for group in ("box", "mask"):
                cells = []
                for key in METRICS:
                    value = metric(values, split, group, key)
                    cells.append("—" if value is None else f"{value:.4f}")
                mark = " · 本次候选" if selected else ""
                rows.append(f'<tr class="{"chosen" if selected else ""}"><th>{esc(label + mark)}</th><td>{group}</td>' + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        return '<div class="scroll"><table><thead><tr><th>模型</th><th>任务</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"

    def operating_table():
        experiments = summary["experiments"]
        if not experiments or "factory_operating_point" not in experiments[0]["baseline"]:
            return ""
        models = [("原基线", experiments[0]["baseline"])] + [(row["label"], row["candidate"]) for row in experiments]
        rows = []
        for label, metrics in models:
            values = metrics.get("factory_operating_point", {})
            cells = [str(values.get(key, "—")) for key in ("tp", "fp", "fn")]
            cells += ["—" if number(values.get(key)) is None else f"{float(values[key]):.4f}" for key in ("precision", "recall")]
            cells += [str(values.get("negative_image_fp", "—"))]
            rows.append(f"<tr><th>{esc(label)}</th>" + "".join(f"<td>{esc(value)}</td>" for value in cells) + "</tr>")
        point = experiments[0]["baseline"]["factory_operating_point"]
        return f'<h3>固定现场阈值的实例匹配</h3><p>conf={point["conf"]}、NMS IoU={point["nms_iou"]}；原图像素mask IoU≥0.5，按置信度降序一对一匹配。负图FP来自{point["negative_images"]}张审核空图。此处P/R对应固定阈值，同样仅作诊断。</p><div class="scroll"><table><tr><th>模型</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>负图FP</th></tr>{"".join(rows)}</table></div>'

    selected = summary["selected_candidate"]
    checkpoint = ""
    if selected:
        checkpoint = f'<div class="downloads">{link(selected["path"], "验收候选 selected_candidate.pt")} · {link(output / "selected_candidate.sha256", "SHA-256")}</div><p class="small">来源实验：{esc(selected["source_experiment"])}<br>SHA-256：<code>{selected["sha256"]}</code></p>'
    failures = ""
    if summary["partial_completion"]:
        failures = '<section class="alert"><h2>部分实验未完成</h2><p>可用候选只在已完成评估的实验中排序。</p><ul>'
        failures += "".join(f'<li>{esc(str(row.get("experiment", "unknown")))}：{esc(str(row.get("error", row)))}</li>' for row in summary["failures"])
        failures += "".join(f'<li>{esc(name)}：缺失或未通过汇总校验。</li>' for name in summary["missing_or_ineligible_experiments"])
        failures += "</ul></section>"
    flag_rows = []
    for row in summary["experiments"]:
        flags = row["screening_flags"] or ["预设筛查未触发；仍需目视验收。"]
        flag_rows.append(f'<li><b>{esc(row["label"])}</b>：{esc("；".join(flags))}</li>')
    for warning in summary["screening_consistency_warnings"]:
        flag_rows.append(f"<li>{esc(warning)}</li>")
    screening = "".join(flag_rows) or "<li>没有可用候选。</li>"
    experiment_cards = []
    for row in summary["experiments"]:
        folder = Path(row["evaluation"])
        artifacts = row["evaluation_metadata"].get("artifacts") or {}
        sheets = artifacts.get("contact_sheets", [])
        preview = f'<a href="{esc(relative_url(folder / sheets[0], output))}"><img loading="lazy" src="{esc(relative_url(folder / sheets[0], output))}" alt="基线与候选的现场对照联系表"></a>' if sheets else ""
        sheet_links = " · ".join(link(folder / name, f"联系表 {i + 1}") for i, name in enumerate(sheets))
        paired = artifacts.get("paired_images", [])
        pairs = " · ".join(link(folder / name, str(i + 1)) for i, name in enumerate(paired))
        experiment_cards.append(f'''<article><h3>{esc(row['label'])} <span class="small">{esc(row['name'])}</span></h3>
<p>{link(folder / 'report.html', '打开完整逐图验收报告')} · {link(folder / 'metrics.json', '指标及参数')} · {link(folder / 'audit.json', '训练曝光审计')} · {link(output / 'training' / row['name'] / 'results.csv', '训练曲线数据')}</p>
<p>{sheet_links}</p>{preview}<details><summary>逐张对照图（{len(paired)}张）</summary><p>{pairs}</p></details>
<details><summary>筛查数值与训练来源</summary><pre>{esc(json.dumps({'screening': row['screening'], 'weight': row['weight'], 'sha256': row['sha256']}, ensure_ascii=False, indent=2))}</pre></details></article>''')
    dataset = summary["dataset"]
    dataset_section = ""
    if dataset["path"]:
        path = Path(dataset["path"])
        ready = (dataset.get("ready") or {}).get("summary", {})
        counts = f"真实训练图 {ready.get('unique_real_training_images', '—')} 张；合成图 {ready.get('synthetic_images', '—')} 张；两组训练采样量分别 {ready.get('baseline_train_samples', '—')} / {ready.get('synthetic_train_samples', '—')}。"
        sources = [("contact_sheet.jpg", "合成样本联系表"), ("factory_holdout_contact_sheet.jpg", "8张现场诊断图与真值"),
                   ("validation.json", "数据校验"), ("manifest.json", "完整来源与变换记录"),
                   ("visual_qa.json", "合成视觉检查"), ("READY.json", "数据准备状态"),
                   ("dataset_baseline.yaml", "真实对照数据配置"), ("dataset_synthetic.yaml", "合成组数据配置"),
                   ("factory_holdout.yaml", "现场诊断配置")]
        links = " · ".join(link(path / filename, label) for filename, label in sources if (path / filename).is_file())
        dataset_section = f'<h2>数据与复现来源</h2><p>{counts}</p><p>{links}</p><details><summary>数据校验详情</summary><pre>{esc(json.dumps(dataset.get("validation"), ensure_ascii=False, indent=2))}</pre></details>'
    synthesis = summary["synthetic_vs_real_control"]
    synthesis_note = "合成组与真实对照未同时完成，暂不能比较合成收益。"
    if synthesis["comparison_available"]:
        synthesis_note = f"{synthesis['interpretation']} 差值 {synthesis['public_mask_map50_95_delta']:+.4f}。{synthesis['limitation']}"
    video_stills = "".join(
        f'<a href="full_video/{name}.jpg"><img loading="lazy" src="full_video/{name}.jpg" '
        f'alt="{label}" onerror="this.parentElement.style.display=\'none\'"></a>'
        for name, label in [("first", "首帧"), ("middle", "中间帧"), ("last", "末帧")])
    source_description = ""
    if summary.get("acceptance_source_groups"):
        source_description = "<ul>" + "".join(
            f'<li><b>{esc(group["label"])}：{group["images"]}张。</b>{esc(group["note"])}</li>'
            for group in summary["acceptance_source_groups"]) + "</ul>"
        source_description += "<p>" + link(summary["acceptance_sources_manifest"], "逐图来源、SHA-256及训练曝光标记") + "</p>"
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>夜间垃圾分割训练 · 验收入口</title><style>
:root{{color-scheme:dark}}body{{max-width:1300px;margin:32px auto;padding:0 24px;background:#101827;color:#e5e7eb;font:16px/1.7 system-ui,sans-serif}}
h1,h2,h3{{color:#fff;line-height:1.35}}h1{{font-size:32px;margin-bottom:10px}}h2{{margin-top:32px}}a{{color:#7dd3fc}}.lead{{font-size:20px}}.small{{font-size:13px;color:#acbad0}}
section,article{{padding:20px;margin:24px 0;background:#1c293b;border:1px solid #334155;border-radius:12px}}.alert{{border-color:#c99d43;background:#302d24}}.downloads{{padding:16px;background:#153245;border-radius:8px}}
table{{width:100%;border-collapse:collapse;white-space:nowrap}}td,th{{text-align:right;padding:10px 12px;border-bottom:1px solid #34435a}}th:first-child{{text-align:left}}thead{{color:#a5c9ef}}.chosen{{background:#13372f}}.scroll{{overflow:auto}}
img,video{{width:100%;border-radius:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}code{{overflow-wrap:anywhere}}details{{margin:12px 0}}.strip{{display:flex;gap:12px}}.strip a{{width:33.333%}}
</style><h1>夜间垃圾分割训练 · 验收入口</h1><p class="small">{esc(summary['created_at'])} · 候选待验收 · 原模型已保留</p>
<p class="lead">{esc(summary['conclusion'])}</p>{checkpoint}
<p>{link(output / 'SUMMARY.json', '完整汇总JSON')} · {link(output / 'README.md', '使用说明')} · {link(output / 'STATUS.json', '最终运行状态')} · {link(Path(summary['config']), '训练配置')}</p>
{failures}<section class="alert"><h2>验收前的筛查提示</h2><ul>{screening}</ul><p class="small">启发式筛查包括：公共 mask AP 未超过原基线、公共 box AP 下降大于 0.01、现场 box/mask recall 下降大于 0.05、固定阈值 precision 下降大于 0.05，或误报 FP 数量增加。precision 未定义时标记为无法比较，不计为通过；FP仍单独检查。现场提示只是8张同视频近邻图的小样本回归警报，不代表独立精度。未触发也不代表部署验收通过。</p></section>
<h2>公共验证集：唯一选模依据</h2><p>仅按候选的公共 mask mAP50-95 排序，选出最高者。现场诊断没有参与排序。P/R为Ultralytics曲线选取的工作点，AP越高越好。</p>{table('public_validation')}
<p>{esc(synthesis_note)}</p>
<h2>8张现场审核图：有限诊断</h2><p>46个标注实例，来自同一视频；只用于发现回归，不能代表新工厂场景的泛化精度。</p>{table('factory_labeled')}
{operating_table()}
<h2>完整原时序视频对照</h2><p>同一训练来源视频的定性回归。计划为 909 帧、10 FPS；以视频元数据验收记录为准。该阶段在本汇总之后执行，最终完成状态见 STATUS.json。</p>
<p>{link(output / 'full_video' / 'comparison.mp4', '打开完整左右对照视频')} · {link(output / 'full_video' / 'video_report.json', '视频验收记录')}</p>
<video controls preload="metadata" src="full_video/comparison.mp4" onerror="this.style.display='none'"></video>
<div class="strip">{video_stills}</div>
<h2>两组逐图验收与元数据</h2><p>完整报告中左侧为原图，中间为原基线，右侧为候选。图片幻灯预览使用人为播放速度，不能代表原始视频时序。优先检查鞋子、规整纸箱、亮斑误检，以及小碎纸和薄膜漏检。</p>{source_description}{''.join(experiment_cards)}
{dataset_section}<h2>结果边界</h2><ul>{''.join('<li>' + esc(note) + '</li>' for note in summary['limitations'])}</ul>
<details><summary>汇总时的流程状态快照</summary><p>该快照可能早于完整视频验收，最终状态请打开 STATUS.json。</p><pre>{esc(json.dumps(summary['runner_status_snapshot'], ensure_ascii=False, indent=2))}</pre></details></html>'''
    (output / "acceptance.html").write_text(content, encoding="utf-8")


def render_readme(summary, output):
    selected = summary["selected_candidate"]
    lines = ["# 夜间垃圾分割训练验收包", "", summary["conclusion"], "",
             "打开 `acceptance.html` 查看公共验证指标、现场诊断和两组逐图对照。", "",
             "- 最终流程状态：`STATUS.json`；完整结构化汇总：`SUMMARY.json`。",
             "- 完整原时序视频：`full_video/comparison.mp4`，验收记录：`full_video/video_report.json`。",
             "- 完整视频阶段在汇总后执行；这些文件是否完成，以最终 `STATUS.json` 为准。", ""]
    if selected:
        lines.extend([f"验收候选：`selected_candidate.pt`，来自 `{selected['source_experiment']}`。",
                      f"SHA-256：`{selected['sha256']}`。", "",
                      "仅按重新评估的公共 mask mAP50-95 选模；现场 8 张诊断没有参与排序。",
                      "候选待用户验收，没有修改当前 best.pt 或部署配置。", ""])
    if summary["selected_screening_flags"]:
        lines.extend(["筛查提示：", "", *("- " + note for note in summary["selected_screening_flags"]), ""])
    if summary["partial_completion"]:
        lines.extend(["部分实验失败或不完整，详见 SUMMARY.json 的 failures 和 STATUS.json。", ""])
    lines.extend(["验收边界：", "", *("- " + note for note in summary["limitations"]), "",
                  "重建汇总命令（不训练、不推理；所选权重若不同则拒绝覆盖）：", "", "```powershell",
                  f"python tools/summarize_overnight_training.py --config '{summary['config']}'", "```", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.config.resolve())
    print(json.dumps({"status": summary["status"], "selected": summary["selected_candidate"],
                      "partial_completion": summary["partial_completion"],
                      "report": str(Path(summary["output"]) / "acceptance.html")}, ensure_ascii=False), flush=True)
    if summary["selected_candidate"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
