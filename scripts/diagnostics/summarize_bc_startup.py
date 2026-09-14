"""Summarize startup transitions and preserve missing-evidence limitations."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


MARKER = "BC_STARTUP_DIAG "


def summarize(root: Path) -> str:
    """Extract ordered events and the first recorded fault latch."""
    lines = ["# 閉循環啟動診斷", "", f"測試：`{root.name}`", ""]
    outcomes = Counter()
    for episode in sorted(root.glob("episode_*")):
        if not episode.is_dir():
            continue
        result_path = episode / "result.json"
        result = (
            json.loads(result_path.read_text()) if result_path.exists() else {}
        )
        outcome = result.get("terminal_reason", "missing result")
        outcomes[outcome] += 1
        lines.extend([
            f"## {episode.name}", "",
            f"Seed：{result.get('seed', '未知')}；結果：{outcome}；"
            f"步數：{result.get('steps', '未知')}", "",
            f"原始失敗原因：{result.get('failure_reason') or '無紀錄'}", "",
        ])
        log_path = episode / "flight.log"
        events = []
        malformed = 0
        if log_path.exists():
            log_lines = log_path.read_text(errors="replace").splitlines()
            for number, line in enumerate(log_lines, 1):
                if MARKER not in line:
                    continue
                try:
                    event = json.loads(line.split(MARKER, 1)[1])
                    if not isinstance(event, dict):
                        raise ValueError("diagnostic must be an object")
                    events.append((number, event))
                except (ValueError, TypeError):
                    malformed += 1
        first_fault = None
        for number, event in events:
            status = event.get("result", {})
            state = status.get("state", event.get("state", ""))
            reason = (
                status.get("stop_reason") or status.get("hold_reason")
                or event.get("message") or event.get("failure_reason", "")
            )
            fault = event.get("fault_latched") or status.get("fault_latched")
            if fault and first_fault is None:
                first_fault = (number, event)
            lines.append(
                f"- flight.log:{number}: {event.get('component')} "
                f"{event.get('event')} → {state}: {reason}"
            )
        if first_fault:
            number, event = first_fault
            lines.extend([
                "", f"首次故障鎖定（flight.log:{number}）：",
                "", "```json", json.dumps(event, indent=2), "```", "",
            ])
        else:
            lines.extend([
                "", "未記錄首次故障快照，原因尚未確認。", "",
            ])
        if not events:
            lines.append(
                "沒有結構化診斷紀錄（舊版或不完整測試）。\n"
            )
        if malformed:
            lines.append(f"無法解析的診斷紀錄：{malformed}\n")
    lines[4:4] = [
        "結果統計：" + ", ".join(
            f"{key}={value}" for key, value in sorted(outcomes.items())
        ), "",
    ]
    return "\n".join(lines)


def timing_summary(root: Path) -> dict:
    """Join identical topic/stamp pairs on the host monotonic clock only."""
    events = []
    malformed = 0
    files = sorted((root / "timing").glob("*.jsonl"))
    for path in files:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
                if event.get("schema") != "uav_timing/v1":
                    raise ValueError("unknown schema")
                if not isinstance(event.get("mono_ns"), int):
                    raise ValueError("missing monotonic clock")
                events.append({**event, "file": path.name, "line": line_number})
            except (ValueError, AttributeError):
                malformed += 1
    events.sort(key=lambda event: event["mono_ns"])
    publications = {}
    metrics = {}
    unmatched = Counter()
    timeline = []
    origin = events[0]["mono_ns"] if events else 0

    def add(label, value, event):
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            metrics.setdefault(label, []).append((value, event))

    for event in events:
        kind, node = event["event"], event["node"]
        topic = event.get("topic", "")
        key = (topic, event.get("key"))
        if kind == "publish" and key[1]:
            publications[key] = event
        elif kind == "receive":
            sent = publications.get(key)
            if sent:
                add(f"{sent['node']} → {node}｜{topic}",
                    (event["mono_ns"] - sent["mono_ns"]) / 1e6, event)
            else:
                unmatched[f"{node}｜{topic}"] += 1
            add(f"接收間隔｜{node}｜{topic}", event.get("receive_gap_ms"), event)
        if kind == "tick":
            add(f"timer 執行間隔｜{node}", event.get("timer_interval_ms"), event)
        elif kind == "consume":
            for name, entry in event.get("inputs", {}).items():
                add(f"判斷時快取年齡｜{node}｜{name}", entry.get("residence_ms"), event)
        elif kind == "inference_end":
            add(f"推論耗時｜{node}", event.get("inference_ms"), event)
        elif kind == "image_read":
            add(f"影像讀取與 JPEG 編碼｜{node}", event.get("read_encode_ms"), event)
        if kind in {"service_request", "service_response", "fault", "state"}:
            timeline.append({"time_s": (event["mono_ns"] - origin) / 1e9,
                             "node": node, "event": kind,
                             "detail": event.get("action", event.get("reason", "")),
                             "state": event.get("state"),
                             "thresholds_s": event.get("thresholds_s", {})})
    statistics = []
    for label, samples in sorted(metrics.items()):
        values = sorted(value for value, _ in samples)
        worst = max(samples, key=lambda pair: pair[0])[1]
        statistics.append({
            "segment": label, "count": len(values),
            "mean_ms": sum(values) / len(values),
            "p95_ms": values[math.ceil(len(values) * 0.95) - 1],
            "max_ms": values[-1],
            "maximum_at_s": (worst["mono_ns"] - origin) / 1e9,
            "evidence": f"timing/{worst['file']}:{worst['line']}",
        })
    manifest = root / "px4_ulog/manifest.json"
    return {
        "schema": "uav_timing_summary/v1", "clock": "host_monotonic_ns",
        "event_count": len(events), "files": len(files),
        "malformed_records": malformed, "statistics": statistics,
        "unmatched_receives": dict(unmatched), "timeline": timeline,
        "ulog": json.loads(manifest.read_text()) if manifest.exists() else None,
        "limitations": [
            "發布至 callback 包含傳輸與排程，無法單獨歸因 DDS。",
            "接收間隔與判斷時快取年齡，不等於封包傳輸耗時；快照可能包含未被選用的控制來源。",
            "PX4 wire timestamp 與 ULog boot time 未校時；不直接相減。",
            "影像 header 是發布時標記，尚未量到 GPU 真正曝光或渲染完成時間。",
            "未量測 PX4 接收至實際執行；相同 topic/stamp 僅關聯已記錄的觀測點。",
            "每 32 筆寫入緩衝 flush；強制終止可能缺少尾端事件。",
        ],
    }


def write_timing_report(root: Path) -> dict:
    """Write a Chinese reader report plus machine-readable measurements."""
    summary = timing_summary(root)
    result_path = root / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    first_fault = next((event for event in summary["timeline"]
                        if event["event"] == "fault"), None)
    lines = ["# 閉循環時間診斷", "", f"回合：`{root.name}`", "",
             f"已記錄 {summary['event_count']} 個事件、{summary['files']} 個 node 檔案；"
             f"無法解析 {summary['malformed_records']} 筆。", "",
             "## 快速重點", "",
             f"- 回合結果：{result.get('terminal_reason', '尚無結果紀錄')}。",
             f"- 原始失敗原因：{result.get('failure_reason') or '無紀錄'}。",
             (f"- 首次記錄故障：{first_fault['time_s']:.3f} s，"
              f"{first_fault['node']}：{first_fault['detail']}。"
              if first_fault else "- 沒有記錄到故障事件；不代表已排除所有故障。"),
             "- 以下為觀測結果，尚未判定延遲根因；未改動控制或門檻。", "",
             "## 如何閱讀", "",
             "單位為毫秒。P95 表示 95% 的樣本不超過此值。最大值提供原始 JSONL 行號，"
             "可回查當時的 source key、接收時間與 node 狀態。", "",
             "只有同一主機 monotonic clock 的時間可以直接相減；"
             "未找到相同 topic/stamp 的來源時，不猜測傳輸延遲。", "",
             "## 各段實測", "",
             "| 觀測區段 | 筆數 | 平均 ms | P95 ms | 最大 ms | 最大值發生於 s | 原始證據 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in summary["statistics"]:
        label = row["segment"].replace("|", "／")
        lines.append(f"| {label} | {row['count']} | {row['mean_ms']:.3f} | "
                     f"{row['p95_ms']:.3f} | {row['max_ms']:.3f} | "
                     f"{row['maximum_at_s']:.3f} | `{row['evidence']}` |")
    if not summary["statistics"]:
        lines.extend(["", "沒有時間事件；可能為舊版紀錄或 node 未成功啟動。"])
    lines.extend(["", "## 事件順序", "", "以首次記錄事件為 0 秒。", ""])
    for event in summary["timeline"]:
        lines.append(f"- {event['time_s']:.3f} s：{event['node']} "
                     f"{event['event']} {event['detail']}")
        if event["thresholds_s"]:
            lines.append(f"  當時有效門檻（秒）：`{event['thresholds_s']}`")
    lines.extend(["", "## ULog 保存", ""])
    ulog = summary["ulog"] or {}
    lines.append(f"保存檔案數：{len(ulog.get('copied_files', []))}。")
    for saved in ulog.get("copied_files", []):
        lines.extend(["", f"- [{saved['path']}](px4_ulog/{saved['path']})："
                      f"{saved['size']} bytes。"])
    lines.extend(["", "檔案存在不代表無 dropout；本摘要未自動解析 ULog 的資料完整性。"])
    if not ulog.get("copied_files"):
        lines.append(f"未取得 ULog：{ulog.get('failure_reason', '沒有 manifest')}。")
    lines.extend(["", "## 尚未證明", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.append("- 本報告提供定位依據，最大值或時間相鄰本身不代表根因；沒有自動調整門檻。")
    (root / "timing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (root / "timing_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Write a report alongside the existing episode artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    if not root.is_dir():
        parser.error("run_directory must be an existing directory")
    output = root / "startup_diagnosis.md"
    output.write_text(summarize(root), encoding="utf-8")
    episodes = sorted(root.glob("episode_*"))
    for episode in episodes or [root]:
        if episode.is_dir():
            write_timing_report(episode)
    print(output)


if __name__ == "__main__":
    main()
