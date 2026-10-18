"""Run bounded JSONL input through the local, non-LLM change-signal engine."""
from __future__ import annotations

import argparse
import hashlib
import importlib
from itertools import chain
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from datetime import datetime, timezone

from render_results import VERSION as RENDERER_VERSION, render_markdown, summarize_result, summary_counts

VERSION = "1.5.0"
__version__ = VERSION
MAX_PAIRS = 10000
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_SELECTED_BYTES = 128 * 1024 * 1024
ENGINE_SIBLING_FILES = ('formula_scope.py', 'static_tex_context.py', 'asymptotic_order.py')


def _read_pairs(path: Path, maximum: int) -> tuple[list[dict], dict]:
    """Read one line at a time; hash the entire file, even beyond a pair limit."""
    digest = hashlib.sha256()
    selected_digest = hashlib.sha256()
    pairs = []
    identifiers = set()
    lines = nonempty = selected_bytes = total_bytes = 0
    with path.open("rb") as stream:
        while True:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            lines += 1
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError(f"第 {lines} 行超过单行 {MAX_LINE_BYTES} 字节上限；未执行算法。")
            digest.update(raw)
            total_bytes += len(raw)
            if not raw.strip() or (lines == 1 and raw.strip() == b"\xef\xbb\xbf"):
                continue
            nonempty += 1
            if len(pairs) >= maximum:
                continue
            selected_bytes += len(raw)
            if selected_bytes > MAX_SELECTED_BYTES:
                raise ValueError(f"选中输入超过 {MAX_SELECTED_BYTES} 字节上限；请调低 --max-pairs 或分批输入。")
            try:
                pair = json.loads(raw.decode("utf-8-sig" if lines == 1 else "utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"第 {lines} 行不是有效 UTF-8 JSON：{exc}") from exc
            if not isinstance(pair, dict):
                raise ValueError(f"第 {lines} 行须为 JSON 对象。")
            identifier = pair.get("pair_id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"第 {lines} 行缺少非空字符串 pair_id。")
            if identifier in identifiers:
                raise ValueError(f"第 {lines} 行出现重复 pair_id：{identifier!r}。")
            if not isinstance(pair.get("old"), dict) or not isinstance(pair.get("new"), dict):
                raise ValueError(f"第 {lines} 行须提供 old 和 new 对象。")
            identifiers.add(identifier)
            selected_digest.update(raw)
            pairs.append(pair)
    return pairs, {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "selected_jsonl_lines_sha256": selected_digest.hexdigest(),
        "bytes": total_bytes,
        "selected_line_bytes": selected_bytes,
        "total_lines": lines,
        "nonempty_lines": nonempty,
        "selected_pairs": len(pairs),
        "max_pairs": maximum,
        "limit_applied": nonempty > len(pairs),
        "excluded_due_to_limit": nonempty - len(pairs),
        "validation_scope": "selected lines only; full file hashed",
    }


def _rss_at_end() -> dict:
    """Current resident memory, not a claimed peak measurement."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return {"rss_at_end_bytes": int(counters.WorkingSetSize), "method": "Windows current WorkingSetSize", "peak_memory_measured": False}
        except (ImportError, AttributeError, OSError, ValueError):
            pass
    else:
        try:
            resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
            return {"rss_at_end_bytes": resident_pages * os.sysconf("SC_PAGE_SIZE"), "method": "procfs current resident pages", "peak_memory_measured": False}
        except (OSError, ValueError, IndexError, AttributeError):
            pass
    return {"rss_at_end_bytes": None, "method": "current RSS unavailable on this platform", "peak_memory_measured": False}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, chunks) -> dict:
    digest = hashlib.sha256()
    size = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix="." + path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for chunk in chunks:
                data = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
                stream.write(data)
                digest.update(data)
                size += len(data)
        os.replace(temporary, path)
        return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _bounded_integer(value: str) -> int:
    number = int(value)
    if not 1 <= number <= MAX_PAIRS:
        raise argparse.ArgumentTypeError(f"须在 1—{MAX_PAIRS} 之间。")
    return number


def _export_groups(results: list[dict]) -> list[dict]:
    """Reference, rather than sum, counts repeated across pair rows."""
    groups = {}
    for result in results:
        support = result.get("support") if isinstance(result.get("support"), dict) else {}
        group_key = support.get("group_key")
        if not isinstance(group_key, str) or not group_key:
            identity = str(result.get("pair_id", ""))
            group_key = "ungrouped-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        group = groups.setdefault(group_key, {
            "group_key": group_key, "support": support, "member_pair_ids": [],
            "support_consistency": "identical_across_member_rows",
            "count_semantics": {"scope": "current input candidate observations only", "add_across_pair_rows": False,
                                "verified_adoption": None, "verified_adoption_status": "not_verified", "corpus_prevalence_estimated": False},
        })
        group["member_pair_ids"].append(result.get("pair_id"))
        if support != group["support"]:
            group["support_consistency"] = "inconsistent_member_rows_preserved"
            variants = group.setdefault("support_variants", [group["support"]])
            if support not in variants:
                variants.append(support)
        result["export"] = {
            "support_group_file": "groups.jsonl", "support_group_key": group_key,
            "support_counts_repeat_across_pair_rows": True, "support_counts_add_across_pair_rows": False,
            "compact_evidence_file": "summary.jsonl",
        }
    for group in groups.values():
        group["member_pair_ids"] = sorted(set(group["member_pair_ids"]), key=str)
        group["pair_rows_referencing_group"] = len(group["member_pair_ids"])
    return [groups[key] for key in sorted(groups)]


def _jsonl(items):
    return (json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for item in items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对有界、已有候选运行自动强度变化线索排序；不调用 LLM 或人工判定。")
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 JSONL，每行一个候选对。")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-pairs", type=_bounded_integer, default=MAX_PAIRS, help=f"最多分析的前缀候选数，默认 {MAX_PAIRS}。")
    parser.add_argument("--max-display", type=_bounded_integer, default=30, help="Markdown 展示条数，最多 200；完整 JSONL 保留全部结果。")
    args = parser.parse_args(argv)
    if args.max_display > 200:
        parser.error("--max-display 最多为 200。")
    started = time.perf_counter()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    try:
        pairs, input_info = _read_pairs(input_path, args.max_pairs)
        engine = importlib.import_module("strength_signals")
        results = engine.analyze_pairs(pairs)
        if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
            raise ValueError("引擎须返回由 JSON 对象组成的 list。")
        if len(results) > args.max_pairs:
            raise ValueError("引擎输出超过已声明的候选上限。")
        groups = _export_groups(results)
        summaries = [summarize_result(result) for result in results]
        metadata = {
            **input_info,
            "max_display": args.max_display,
            "input_description": input_path.name,
            "scope": "输入是旧候选的复查集合，只分析提供的声明、材料和源码指针；没有重新检索全库。后文候选观察仅在本批去重，不据剩余数估计全库召回。",
        }
        markdown = render_markdown(results, metadata=metadata)
        # Validate serializability before replacing a previous successful run.
        for result in chain(results, groups, summaries):
            json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = [output_dir / "results.jsonl", output_dir / "用户结果.md", output_dir / "run_manifest.json", output_dir / "groups.jsonl", output_dir / "summary.jsonl"]
        if input_path in output_paths:
            raise ValueError("输入文件不能与本次输出文件同路径。")
        json_info = _atomic_write(output_paths[0], _jsonl(results))
        markdown_info = _atomic_write(output_paths[1], [markdown])
        groups_info = _atomic_write(output_paths[3], _jsonl(groups))
        summary_info = _atomic_write(output_paths[4], _jsonl(summaries))
        code_info = {}
        engine_dir = Path(engine.__file__).resolve().parent
        engine_files = [engine_dir / name for name in ENGINE_SIBLING_FILES]
        for path in (Path(__file__).resolve(), Path(render_markdown.__code__.co_filename).resolve(), Path(engine.__file__).resolve(), *engine_files):
            code_info[path.name] = {"path": str(path), "sha256": _hash_file(path)}
        manifest = {
            "schema_version": "strength-change-discovery-v1",
            "status": "completed",
            "runner_version": VERSION,
            "renderer_version": RENDERER_VERSION,
            "compact_schema_version": "compact-user-evidence-v2",
            "engine_version": str(getattr(engine, "VERSION", getattr(engine, "__version__", "unknown"))),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "input": input_info,
            "result_count": len(results),
            "group_count": len(groups),
            "summary_count": len(summaries),
            "core_category_counts": summary_counts(results),
            "markdown_display_count": min(len(results), args.max_display),
            "markdown_order": "provided investigation_queue.rank ascending where present, priority_score descending, pair_id ascending",
            "elapsed_seconds_at_results_written": round(time.perf_counter() - started, 6),
            "memory_after_results_and_markdown_before_manifest": _rss_at_end(),
            "bounds": {"max_pairs": args.max_pairs, "max_line_bytes": MAX_LINE_BYTES, "max_selected_line_bytes": MAX_SELECTED_BYTES},
            "runtime_boundary": {"llm_calls": 0, "external_service_calls": 0, "human_decision_dependencies": False, "manual_event_labels_as_features": False, "full_corpus_rerun": False},
            "interpretation": {"priority_score_is_probability": False, "mathematical_weakening_certified": False, "population_recall_estimated": False, "input_scope": "recheck_of_supplied_preexisting_candidates", "support_scope": "input only", "disclosure_scope": "provided evidence documents and explicitly recorded scan coverage only", "full_tex_in_compact": False},
            "outputs": {"results": json_info, "markdown": markdown_info, "groups": groups_info, "summary": summary_info},
            "file_roles": {"results": "full bounded engine records in input order", "summary": "compact per-pair evidence in input order; no full text/context", "groups": "one group record per group_key; do not add repeated per-pair counts", "markdown": "ranked reading preview with bounded source windows"},
            "code": code_info,
        }
        _atomic_write(output_paths[2], [json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"])
    except (OSError, ValueError, TypeError, ImportError, AttributeError) as exc:
        print(f"运行未完成：{exc}", file=sys.stderr)
        return 1
    print(f"完成：{len(results)} 对，Markdown 展示 {min(len(results), args.max_display)} 条。输出：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
