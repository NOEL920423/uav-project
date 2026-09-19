"""Summarize startup transitions and preserve missing-evidence limitations."""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path


MARKER = "BC_STARTUP_DIAG "
CONTROL_SUMMARY_FILENAME = "control_summary.md"
RUN_CONTROL_SUMMARY_FILENAME = "closed_loop_control_summary.md"
IMAGE_SOURCE_TOPICS = {
    "top_rgb": "/uav/isaac/observer/image/compressed",
    "fpv_rgb": "/uav/isaac/fpv/image/compressed",
    "fpv_depth": "/uav/isaac/fpv/depth/compressed",
}


def _statistic(summary: dict, segment: str) -> dict | None:
    """Return one named statistic without relying on display-table order."""
    return next(
        (row for row in summary.get("statistics", []) if row["segment"] == segment),
        None,
    )


def _image_interval_statistic(summary: dict, image_source: str) -> dict | None:
    """Find the BC image-input cadence for the evaluated source."""
    topic = IMAGE_SOURCE_TOPICS.get(image_source)
    if topic is None:
        return None
    return next(
        (row for row in summary.get("statistics", [])
         if row["segment"].startswith("接收間隔｜bc_policy｜")
         and row["segment"].endswith(topic)),
        None,
    )


def _maximum_timer_statistic(summary: dict) -> dict | None:
    """Find the worst observed timer gap across the recorded nodes."""
    rows = [row for row in summary.get("statistics", [])
            if row["segment"].startswith("timer 執行間隔｜")]
    return max(rows, key=lambda row: row["max_ms"], default=None)


def _format_ms(statistic: dict | None, key: str = "p95_ms") -> str:
    return "未取得" if statistic is None else f"{statistic[key]:.1f} ms"


def _result_label(result: dict) -> str:
    labels = {
        "success": "成功到達目標", "collision": "碰撞", "out_of_bounds": "飛出邊界",
        "timeout": "逾時", "runtime_failure": "執行階段失敗",
    }
    return labels.get(result.get("terminal_reason"), result.get("terminal_reason", "尚無結果"))


def reader_summary(summary: dict, result: dict) -> dict:
    """Create the small, Chinese-first set of facts for each flight review."""
    image_source = result.get("image_source", "")
    image_interval = _image_interval_statistic(summary, image_source)
    image_rate_hz = (
        1000.0 / image_interval["mean_ms"]
        if image_interval and image_interval["mean_ms"] > 0 else None
    )
    image_to_px4 = _statistic(summary, "影像接收至動作首次 PX4 發布")
    image_age = _statistic(summary, "PX4 發布時影像接收年齡（含重送）")
    timer = _maximum_timer_statistic(summary)
    lineage = summary.get("action_lineage", {})
    new_actions = lineage.get("bc_new_actions", 0)
    repeats = lineage.get("bc_repeats", 0)
    action_lineage_available = "bc_new_actions" in lineage
    ulog_topics = {
        topic["topic"]
        for entry in summary.get("ulog_inspection", {}).get("files", [])
        for topic in entry.get("topics", [])
    }
    tracking_available = {
        "trajectory_setpoint", "vehicle_local_position"
    }.issubset(ulog_topics)
    hypothesis = "資料不足：先確認本回合是否成功寫入 timing 與 ULog。"
    if image_rate_hz is not None and not action_lineage_available:
        hypothesis = "此回合是舊格式，沒有動作來源串接；下一次 closed-loop 才能量到影像到 PX4 的完整時間。"
    if image_rate_hz is not None and image_to_px4 is not None:
        hypothesis = "先檢查影像更新率與舊動作重送是否使控制依據過舊。"
    if timer and timer["max_ms"] > 2.0 * timer["mean_ms"]:
        hypothesis = "先檢查此回合的排程停頓，再判定是否需要調整控制週期或 timeout。"
    return {
        "result": _result_label(result),
        "image_source": image_source or "未記錄",
        "image_rate_hz": image_rate_hz,
        "image_interval_p95_ms": None if image_interval is None else image_interval["p95_ms"],
        "image_to_px4_p95_ms": None if image_to_px4 is None else image_to_px4["p95_ms"],
        "image_age_p95_ms": None if image_age is None else image_age["p95_ms"],
        "new_actions": new_actions,
        "repeated_actions": repeats,
        "action_lineage_available": action_lineage_available,
        "maximum_timer_ms": None if timer is None else timer["max_ms"],
        "maximum_timer_node": None if timer is None else timer["segment"].removeprefix("timer 執行間隔｜"),
        "tracking_available": tracking_available,
        "hypothesis": hypothesis,
    }


def write_reader_summary(root: Path, summary: dict, result: dict) -> dict:
    """Write the one-page Chinese reading order for a single episode."""
    reader = reader_summary(summary, result)
    rate = (f"{reader['image_rate_hz']:.1f} Hz"
            if reader["image_rate_hz"] is not None else "未取得")
    timer = (f"{reader['maximum_timer_ms']:.1f} ms（{reader['maximum_timer_node']}）"
             if reader["maximum_timer_ms"] is not None else "未取得")
    px4_following = (
        "ULog 已保存目標與實際位置資料；本報告尚未自動判定是否追隨。"
        if reader["tracking_available"] else
        "尚無可同時比較的 ULog 目標與實際位置資料。"
    )
    lines = [
        "# Closed-loop 控制摘要", "", "這是本回合優先閱讀的內容；"
        "原始 JSONL、ULog 與 flight.log 只在需要追查時再查看。", "",
        "## 本回合結論", "",
        f"- 結果：{reader['result']}。",
        f"- 模型影像來源：{reader['image_source']}。",
        f"- 新影像率：{rate}" + (
            f"（影像接收間隔 P95 {_format_ms(_image_interval_statistic(summary, reader['image_source']))}）。"
            if reader["image_rate_hz"] is not None else "。"),
        f"- 影像接收 → 該動作首次送往 PX4：P95 {_format_ms(_statistic(summary, '影像接收至動作首次 PX4 發布'))}。",
        f"- 送往 PX4 時，該動作依據影像的年齡：P95 {_format_ms(_statistic(summary, 'PX4 發布時影像接收年齡（含重送）'))}。",
        (f"- 新動作／重送舊動作：{reader['new_actions']}／{reader['repeated_actions']} 次。"
         if reader["action_lineage_available"] else
         "- 新動作／重送舊動作：此回合為舊格式，未記錄。"),
        f"- 最大 timer 間隔：{timer}。",
        f"- PX4 是否追隨速度／位置指令：{px4_following}",
        f"- 原始失敗原因：{result.get('failure_reason') or '無紀錄'}。", "",
        "## 這些數字代表什麼", "",
        "- 新影像率：模型每秒大約有幾次機會根據新畫面重新決策。",
        "- 影像 → PX4：從程式收到影像，到該次動作第一次呼叫 PX4 發布 API 的等待；不包含 PX4 接收與執行。",
        "- 影像年齡：指令發出時，所依據畫面距離 BC 收到它已過多久；重送舊動作會使這個數字上升。",
        "- 最大 timer 間隔：某個定時工作最久隔了多久才再次執行；它是排程停頓線索，不單獨證明根因。", "",
        "## 下一個要驗證的假說", "", f"{reader['hypothesis']}", "",
        "不要只根據單一 P95 或最大值改程式。先和同 checkpoint、同 seed 的對照回合比較，"
        "再回到 timing_summary.md 的原始證據列與 ULog CSV。",
    ]
    (root / CONTROL_SUMMARY_FILENAME).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return reader


def write_run_reader_summary(root: Path) -> dict:
    """Write one Chinese overview after every closed-loop evaluation finishes."""
    records = []
    for episode in sorted(root.glob("episode_*")):
        result_path = episode / "result.json"
        timing_path = episode / "timing_summary.json"
        if not result_path.exists() or not timing_path.exists():
            continue
        result = json.loads(result_path.read_text())
        timing = json.loads(timing_path.read_text())
        records.append({"episode": episode.name, "seed": result.get("seed"),
                        **reader_summary(timing, result)})
    outcomes = Counter(record["result"] for record in records)
    lines = ["# Closed-loop 控制總覽", "", "先讀這份；只有異常回合才開各回合的 `control_summary.md`。", "",
             f"回合數：{len(records)}；結果：" + "、".join(
                 f"{name} {count}" for name, count in sorted(outcomes.items())
             ), "", "| 回合 | 結果 | 新影像率 | 影像→PX4 P95 | 影像年齡 P95 | 最大 timer | 下一個假說 |",
             "|---|---|---:|---:|---:|---|---|"]
    for record in records:
        rate = "未取得" if record["image_rate_hz"] is None else f"{record['image_rate_hz']:.1f} Hz"
        first = "未取得" if record["image_to_px4_p95_ms"] is None else f"{record['image_to_px4_p95_ms']:.1f} ms"
        age = "未取得" if record["image_age_p95_ms"] is None else f"{record['image_age_p95_ms']:.1f} ms"
        timer = "未取得" if record["maximum_timer_ms"] is None else f"{record['maximum_timer_ms']:.1f} ms"
        lines.append(f"| [{record['episode']}]({record['episode']}/{CONTROL_SUMMARY_FILENAME}) | "
                     f"{record['result']} | {rate} | {first} | {age} | {timer} | {record['hypothesis']} |")
    if not records:
        lines.append("| 無完整回合 | 尚無資料 | - | - | - | - | - |")
    lines.extend(["", "數字是測量結果，不是自動結論。跨回合比較時請固定 checkpoint、seed 與程式版本。"])
    (root / RUN_CONTROL_SUMMARY_FILENAME).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {"episodes": records, "outcomes": dict(outcomes)}


def inspect_ulog(root: Path, export: bool = False) -> dict:
    """Inspect existing PX4 evidence without inventing host/boot alignment."""
    paths = sorted((root / "px4_ulog").glob("*.ulg"))
    report = {"files": [], "clock_alignment": "unavailable",
              "limitations": [
                  "ULog topic timestamp is not a measured command receive/actuation time.",
                  "Logged setpoints may be downsampled; missing samples are not proof of packet loss.",
                  "Host/PX4 clocks are not calibrated; no cross-clock latency is computed.",
              ]}
    if not paths:
        return report
    try:
        from pyulog import ULog
    except ImportError as error:
        report["error"] = str(error)
        print(f"ULOG_DIAGNOSTIC_ERROR: {error}", file=sys.stderr)
        return report
    topics = {
        "trajectory_setpoint", "vehicle_local_position_setpoint",
        "vehicle_local_position", "vehicle_attitude", "vehicle_attitude_setpoint",
        "vehicle_odometry", "timesync_status",
    }
    for path in paths:
        entry = {"path": path.name, "topics": [], "missing_topics": []}
        report["files"].append(entry)
        try:
            log = ULog(str(path), message_name_filter_list=sorted(topics))
            entry["dropouts"] = [
                {"timestamp_us": int(item.timestamp), "duration_ms": int(item.duration)}
                for item in log.dropouts
            ]
            for dataset in log.data_list:
                if dataset.name not in topics:
                    continue
                data = dataset.data
                stamps = data.get("timestamp", [])
                gaps = [int(b) - int(a) for a, b in zip(stamps, stamps[1:])]
                item = {
                    "topic": dataset.name, "instance": dataset.multi_id,
                    "samples": len(stamps), "fields": sorted(data),
                    "first_timestamp_us": int(stamps[0]) if len(stamps) else None,
                    "last_timestamp_us": int(stamps[-1]) if len(stamps) else None,
                    "maximum_sample_gap_ms": max(gaps) / 1000 if gaps else None,
                }
                if export:
                    target = path.with_name(
                        f"{path.stem}_{dataset.name}_{dataset.multi_id}.csv"
                    )
                    fields = ["timestamp"] + sorted(set(data) - {"timestamp"})
                    with target.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(fields)
                        writer.writerows(zip(*(data[field] for field in fields)))
                    item["csv"] = str(target.relative_to(root))
                entry["topics"].append(item)
            entry["missing_topics"] = sorted(topics - {
                item["topic"] for item in entry["topics"]
            })
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            print(f"ULOG_DIAGNOSTIC_ERROR: {path}: {entry['error']}", file=sys.stderr)
    return report


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
    lineage = {}
    first_streamed = set()
    lineage_counts = Counter()
    clock_witnesses = Counter()
    chain = {
        "/uav/control/selected_command": "/uav/control/bc_command",
        "/uav/px4/setpoint_candidate": "/uav/control/selected_command",
        "/fmu/in/trajectory_setpoint": "/uav/px4/setpoint_candidate",
    }

    def add(label, value, event):
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            metrics.setdefault(label, []).append((value, event))

    for event in events:
        kind, node = event["event"], event["node"]
        topic = event.get("topic", "")
        key = (topic, event.get("key"))
        if kind == "publish" and key[1]:
            publications[key] = event
            action = event.get("action_origin")
            if topic == "/uav/control/bc_command" and action:
                action = {**action, "publisher": (node, event.get("pid", event["file"]))}
                lineage[key] = action
                lineage_counts["bc_repeats" if event.get("repeated_action") else "bc_new_actions"] += 1
            elif topic in chain:
                parent = event.get("input_refs", {}).get(chain[topic], {}).get("key")
                action = lineage.get((chain[topic], parent))
                context = event.get("consume_context", {})
                if topic == "/uav/control/selected_command" and context.get("active_source") != "BC_POLICY":
                    action = None
                if topic == "/uav/px4/setpoint_candidate" and context.get("state") != "SAFE_TO_FORWARD":
                    action = None
                if action:
                    lineage[key] = action
            if topic == "/fmu/in/trajectory_setpoint":
                if action:
                    inputs = action.get("observation_inputs", {})
                    image = inputs.get(action.get("image_topic"), {})
                    received = image.get("receive_ns")
                    if received is not None:
                        age = (event["mono_ns"] - received) / 1e6
                        add("PX4 發布時影像接收年齡（含重送）", age, event)
                        identity = (action["publisher"], action["action_id"])
                        if identity not in first_streamed:
                            first_streamed.add(identity)
                            add("影像接收至動作首次 PX4 發布", age, event)
                        add("PX4 發布時距推論完成", (event["mono_ns"] - action["inference_completed_ns"]) / 1e6, event)
                        sent = publications.get((action.get("image_topic"), image.get("key")))
                        if sent:
                            add("PX4 發布時影像來源發布年齡", (event["mono_ns"] - sent["mono_ns"]) / 1e6, event)
                    lineage_counts["stream_linked"] += 1
                else:
                    lineage_counts["stream_unlinked_or_non_bc"] += 1
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
        elif kind == "inference_start":
            inputs = event.get("observation_inputs", {})
            images = [entry for name, entry in inputs.items() if "/image/" in name]
            odometry = inputs.get("/uav/vehicle/odometry")
            if images and odometry:
                add("推論輸入影像與里程計接收時間差（絕對值，非擷取差）",
                    abs(images[0]["receive_ns"] - odometry["receive_ns"]) / 1e6, event)
        elif kind == "image_read":
            source = event.get("topic", "unknown_camera_legacy")
            add(f"影像讀取與 JPEG 編碼｜{node}｜{source}", event.get("read_encode_ms"), event)
            add(f"影像讀取｜{source}", event.get("read_ms"), event)
            add(f"JPEG 編碼｜{source}", event.get("encode_ms"), event)
        elif kind == "publish_api_end":
            add(f"publish API wall time | {node} | {event.get('topic')}", event.get("duration_ms"), event)
            add(f"publish API thread CPU | {node} | {event.get('topic')}", event.get("thread_cpu_ms"), event)
        elif kind == "callback_end":
            add(f"callback 執行｜{node}｜{event.get('callback')}", event.get("duration_ms"), event)
            add(f"callback CPU｜{node}｜{event.get('callback')}", event.get("thread_cpu_ms"), event)
        elif kind == "scheduler" and event.get("delta"):
            delta = event["delta"]
            add(f"執行緒等待 CPU｜{node}", delta["thread_runqueue_wait_ns"] / 1e6, event)
            add(f"執行緒 block I/O 等待｜{node}", delta["thread_block_io_ticks"] * 1000 / event["block_io_tick_hz"], event)
        elif kind == "timesync":
            clock_witnesses[str(event.get("source_protocol"))] += 1
            add("PX4 timesync 往返時間（非控制延遲）", event.get("round_trip_time_us", 0) / 1000, event)
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
        "action_lineage": dict(lineage_counts),
        "timesync_witnesses_by_protocol": dict(clock_witnesses),
        "ulog": json.loads(manifest.read_text()) if manifest.exists() else None,
        "limitations": [
            "發布至 callback 包含傳輸與排程，無法單獨歸因 DDS。",
            "接收間隔與判斷時快取年齡，不等於封包傳輸耗時；快照可能包含未被選用的控制來源。",
            "PX4 wire timestamp 與 ULog boot time 未校時；不直接相減。",
            "影像 header 是發布時標記，尚未量到 GPU 真正曝光或渲染完成時間。",
            "未量測 PX4 接收至實際執行；相同 topic/stamp 僅關聯已記錄的觀測點。",
            "每 32 筆寫入緩衝 flush；強制終止可能缺少尾端事件。",
            "callback 耗時包含診斷成本；wall 減 CPU 不可直接稱為 I/O 等待。",
            "scheduler 紀錄需啟用 UAV_TIMING_SCHED_SECONDS；零值不保證核心 accounting 已啟用。",
            "影像與里程計接收時間差不等於感測器擷取時間差；發布編號不代表新渲染畫面。",
        ],
    }


def write_timing_report(root: Path) -> dict:
    """Write a Chinese reader report plus machine-readable measurements."""
    summary = timing_summary(root)
    summary["ulog_inspection"] = inspect_ulog(root, export=True)
    result_path = root / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    summary["reader_summary"] = write_reader_summary(root, summary, result)
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
    lines.extend(["", "## 動作來源串接", "", str(summary["action_lineage"]),
                  "", "未串接包含非 BC 控制、舊版紀錄及缺少來源；不可當作丟包數。",
                  "", "## ULog 內容檢查", ""])
    inspection = summary["ulog_inspection"]
    if inspection.get("error"):
        lines.append(f"解析不可用：{inspection['error']}")
    for entry in inspection["files"]:
        lines.append(f"- {entry['path']}：dropout {len(entry.get('dropouts', []))} 筆；"
                     f"缺少 topics：{', '.join(entry['missing_topics']) or '無'}")
        if entry.get("error"):
            lines.append(f"  原始解析錯誤：{entry['error']}")
        for topic in entry["topics"]:
            lines.append(f"  - {topic['topic']}：{topic['samples']} 筆，"
                         f"[CSV]({topic['csv']})")
    lines.extend(["", "ULog setpoint 紀錄不等於已量到接收或致動時間；可能降採樣。"
                  "PX4 與主機尚未校時，不計算跨時鐘延遲。"])
    lines.append(f"主機 timesync 證據（依 protocol）：{summary['timesync_witnesses_by_protocol']}；"
                 "空值表示未收到。offset／RTT 保存在原始事件，不自動視為完成校時。")
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
