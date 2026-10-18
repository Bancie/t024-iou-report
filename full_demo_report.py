# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.16",
#     "matplotlib>=3.8",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import json
    import textwrap
    from pathlib import Path

    import marimo as mo

    def resolve_report_dir():
        try:
            here = Path(__file__).resolve().parent
        except NameError:
            notebook = mo.notebook_dir()
            here = Path(notebook) if notebook is not None else Path.cwd()
        if (here / "results").is_dir():
            return here
        notebook = mo.notebook_dir()
        if notebook is not None and (Path(notebook) / "results").is_dir():
            return Path(notebook)
        return here

    REPORT_DIR = resolve_report_dir()

    # molab chỉ mirror đúng một file .py từ GitHub, nên thư mục results/ được tải
    # từ repo public khi không nằm cạnh notebook. Chạy local thì dùng file có sẵn.
    _RAW_BASE = "https://raw.githubusercontent.com/Bancie/t024-iou-report/main/"
    _REMOTE_FILES = ['results/edge_loop/baseline.json', 'results/edge_loop/comparisons.json', 'results/edge_loop/overlays.json', 'results/edge_loop/per_image_opt.json', 'results/edge_loop/per_image_other.json', 'results/edge_loop/per_image_validation.json', 'results/edge_loop/rounds.json', 'results/edge_loop/run_summary.json', 'results/full_demo/candidates_validation.json', 'results/full_demo/feedback.json', 'results/full_demo/final_test.json', 'results/full_demo/overlays.json', 'results/full_demo/per_image_test.json', 'results/full_demo/per_image_validation.json', 'results/full_demo/rounds.json', 'results/full_demo/run_summary.json']

    def _ensure_remote_files(report_dir):
        import os
        import tempfile
        import urllib.request

        if (report_dir / 'results/full_demo/run_summary.json').is_file():
            return report_dir
        target = report_dir
        try:
            (target / "results").mkdir(parents=True, exist_ok=True)
            (target / ".write_test").write_text("ok")
            (target / ".write_test").unlink()
        except OSError:
            target = Path(tempfile.mkdtemp(prefix="t024_iou_report_"))
        for rel in _REMOTE_FILES:
            dest = target / rel
            if dest.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                with urllib.request.urlopen(_RAW_BASE + rel, timeout=30) as response:
                    dest.write_bytes(response.read())
            except Exception:
                pass
        os.environ["T024_IOU_REPORT_DIR"] = str(target)
        return target

    REPORT_DIR = _ensure_remote_files(REPORT_DIR)
    RESULTS = REPORT_DIR / "results"
    DEMO = RESULTS / "full_demo"
    # URL endpoint và đường dẫn máy local không đưa vào object render.
    HIDDEN_CONFIG_KEYS = ("llm_base_url", "sam3_text_url", "dataset_root")

    def load_json(path):
        if not path.is_file():
            return None, "thiếu file"
        try:
            return json.loads(path.read_text(encoding="utf-8")), None
        except json.JSONDecodeError as exc:
            return None, "JSON không hợp lệ: " + exc.msg
        except OSError as exc:
            return None, "không đọc được: " + str(exc)

    def note(notes, path, error):
        if error:
            notes.append("`" + path.name + "` (" + error + ").")

    load_notes = []
    run_summary, run_error = load_json(DEMO / "run_summary.json")
    note(load_notes, DEMO / "run_summary.json", run_error)
    rounds, rounds_error = load_json(DEMO / "rounds.json")
    note(load_notes, DEMO / "rounds.json", rounds_error)
    candidates, candidates_error = load_json(DEMO / "candidates_validation.json")
    note(load_notes, DEMO / "candidates_validation.json", candidates_error)
    final_test, final_error = load_json(DEMO / "final_test.json")
    note(load_notes, DEMO / "final_test.json", final_error)
    feedback, feedback_error = load_json(DEMO / "feedback.json")
    note(load_notes, DEMO / "feedback.json", feedback_error)
    overlays, overlays_error = load_json(DEMO / "overlays.json")
    note(load_notes, DEMO / "overlays.json", overlays_error)
    per_val, per_val_error = load_json(DEMO / "per_image_validation.json")
    note(load_notes, DEMO / "per_image_validation.json", per_val_error)
    per_test, per_test_error = load_json(DEMO / "per_image_test.json")
    note(load_notes, DEMO / "per_image_test.json", per_test_error)

    if not isinstance(run_summary, dict):
        run_summary = {}
    if not isinstance(rounds, list):
        rounds = []
    if not isinstance(candidates, list):
        candidates = []
    if not isinstance(final_test, dict):
        final_test = {}
    if not isinstance(feedback, dict):
        feedback = {}
    if not isinstance(overlays, list):
        overlays = []
    if not isinstance(per_val, dict):
        per_val = {}
    if not isinstance(per_test, dict):
        per_test = {}

    config = dict(run_summary.get("config") or {})
    for key in HIDDEN_CONFIG_KEYS:
        config.pop(key, None)
    run_summary = {key: value for key, value in run_summary.items() if key != "config"}
    run_summary["config"] = config

    def fmt(value, digits=4):
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    def scored_rows(rows):
        return [row for row in rows or [] if isinstance(row, dict) and row.get("iou") is not None]

    def mean_of(rows):
        values = [float(row["iou"]) for row in scored_rows(rows)]
        if not values:
            return None
        return sum(values) / len(values)

    def zero_count(rows):
        return sum(1 for row in scored_rows(rows) if float(row["iou"]) == 0.0)

    def close(got, expected):
        if got is None or expected is None:
            return got is None and expected is None
        try:
            return abs(float(got) - float(expected)) <= 1e-9
        except (TypeError, ValueError):
            return False

    def md_table(headers, table_rows):
        if not table_rows:
            return "_Không có dòng nào._"

        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")

        head = "| " + " | ".join(headers) + " |"
        rule = "| " + " | ".join("---" for _ in headers) + " |"
        body = ["| " + " | ".join(cell(item) for item in row) + " |" for row in table_rows]
        return "\n".join([head, rule, *body])

    def render(template, **kwargs):
        text = textwrap.dedent(template).strip()
        if kwargs:
            text = text.format(**kwargs)
        return mo.md(text)

    def row_by_label(rows, label):
        for row in rows or []:
            if isinstance(row, dict) and row.get("label") == label:
                return row
        return None

    def words(text):
        if not text:
            return 0
        return len(str(text).split())

    def short_id(image_id):
        text = str(image_id or "")
        if len(text) <= 24:
            return text
        return text[:24] + "…"

    def group_case(rows, case):
        return [row for row in rows or [] if row.get("case") == case]

    def delta_text(new, old, digits=4):
        if new is None or old is None:
            return "—"
        diff = float(new) - float(old)
        sign = "+" if diff > 0 else ""
        return sign + fmt(diff, digits)

    def count_change(new, old):
        if new is None or old is None:
            return "không đủ số"
        diff = int(new) - int(old)
        if diff < 0:
            return f"giảm {old} → {new} (Δ {diff})"
        if diff > 0:
            return f"tăng {old} → {new} (Δ +{diff})"
        return f"giữ nguyên {new}"

    def same_value(values):
        items = list(values)
        return bool(items) and all(item == items[0] for item in items)

    def align_rows(left, right):
        right_map = {row.get("image_id"): row for row in right or []}
        paired = []
        missing = 0
        for row in left or []:
            other = right_map.get(row.get("image_id"))
            if other is None or row.get("iou") is None or other.get("iou") is None:
                missing += 1
                continue
            paired.append((row, other))
        paired_ids = {row.get("image_id") for row, _other in paired}
        extra = sum(1 for row in right or [] if row.get("image_id") not in paired_ids)
        return paired, missing, extra

    def pair_stats(left, right):
        paired, missing, extra = align_rows(left, right)
        up = down = same = zero_to_pos = pos_to_zero = 0
        for row, other in paired:
            new = float(row["iou"])
            old = float(other["iou"])
            if new > old:
                up += 1
            elif new < old:
                down += 1
            else:
                same += 1
            if old == 0.0 and new > 0.0:
                zero_to_pos += 1
            if new == 0.0 and old > 0.0:
                pos_to_zero += 1
        return {
            "n": len(paired),
            "up": up,
            "down": down,
            "same": same,
            "zero_to_pos": zero_to_pos,
            "pos_to_zero": pos_to_zero,
            "missing": missing,
            "extra": extra,
        }

    def case_mean_cell(value, n):
        if not n:
            return f"{fmt(value)} (n = 0)"
        return f"{fmt(value)} (n = {n})"

    integrity = []

    def check_block(label, block, rows):
        if not isinstance(block, dict):
            integrity.append(f"{label}: thiếu block tổng.")
            return
        if not isinstance(rows, list):
            integrity.append(f"{label}: thiếu per_image.")
            return
        scored = scored_rows(rows)
        if block.get("n") != len(scored):
            integrity.append(
                f"{label}: `n` tổng {block.get('n')} khác per_image {len(scored)}."
            )
        failed = sum(1 for row in rows if row.get("iou") is None)
        if block.get("n_failed") != failed:
            integrity.append(
                f"{label}: `n_failed` tổng {block.get('n_failed')} khác per_image {failed}."
            )
        got = mean_of(rows)
        if not close(block.get("mean_iou"), got):
            integrity.append(
                f"{label}: mean tổng {fmt(block.get('mean_iou'))} khác per_image {fmt(got)}."
            )
        zeros = zero_count(rows)
        if zeros != block.get("n_iou_zero"):
            integrity.append(
                f"{label}: `n_iou_zero` tổng {block.get('n_iou_zero')} khác per_image {zeros}."
            )
        for case, value in (block.get("iou_by_case") or {}).items():
            group = group_case(rows, case)
            if not group:
                if value is not None:
                    integrity.append(
                        f"{label}: case `{case}` không có ảnh nhưng mean tổng là {fmt(value)}."
                    )
                continue
            if not close(value, mean_of(group)):
                integrity.append(
                    f"{label}: mean `{case}` tổng {fmt(value)} khác per_image {fmt(mean_of(group))}."
                )

    for block in candidates:
        check_block(
            "validation " + str(block.get("label")),
            block,
            per_val.get(block.get("label")),
        )
    for key in ("best", "baseline"):
        check_block("test " + key, final_test.get(key), per_test.get(key))

    split_counts = config.get("split_counts") or {}
    by_case_counts = split_counts.get("by_case") or {}
    opt_counts = by_case_counts.get("opt") or {}
    for rnd in rounds:
        version = rnd.get("prompt_version")
        num = 0.0
        den = 0
        skipped = False
        for case, n in opt_counts.items():
            if case == "other" or not n:
                continue
            value = (rnd.get("iou_by_case") or {}).get(case)
            if value is None:
                integrity.append(f"opt vòng {version}: thiếu mean `{case}`.")
                skipped = True
                continue
            num += float(value) * int(n)
            den += int(n)
        if not skipped and den and not close(num / den, rnd.get("iou_total")):
            integrity.append(
                f"opt vòng {version}: `iou_total` {fmt(rnd.get('iou_total'))} khác tổng có trọng số {fmt(num / den)}."
            )

    return (
        mo,
        DEMO,
        load_notes,
        integrity,
        run_summary,
        config,
        rounds,
        candidates,
        final_test,
        feedback,
        overlays,
        per_val,
        per_test,
        fmt,
        mean_of,
        zero_count,
        md_table,
        render,
        row_by_label,
        words,
        short_id,
        group_case,
        delta_text,
        count_change,
        same_value,
        pair_stats,
        case_mean_cell,
        split_counts,
        by_case_counts,
    )


@app.cell(hide_code=True)
def _(integrity, load_notes, mo, render):
    def _section_2():
        title = render(
            """
            # Báo cáo full live demo — vòng lặp prompt trên bộ review nuImages

            Team T024 · binary IoU của mask `car` · baseline viết tay và prompt của Prompt Engine

            Một lần chạy live: retrieve, generate, vòng lặp trên **opt**, `selectBest` trên **validation**, rồi **test** chấm một lần. Notebook chỉ đọc `results/full_demo/`. Không gọi Modal, không đọc ảnh gốc, không đọc dataset. Báo cáo baseline một prompt nằm ở `iou_eval_report.py`.

            **Điều hướng nhanh**

            - [Tổng quan](#overview)
            - [Baseline và prompt](#prompts)
            - [IoU theo split và case](#splits)
            - [Vòng lặp trên opt](#rounds)
            - [Chọn prompt trên validation](#validation)
            - [Test chấm một lần](#test)
            - [Overlay](#overlays)
            - [Kết luận](#conclusion)
            - [Hạn chế](#limitations)
            """
        )
        blocks = [title]
        notes = list(load_notes)
        if integrity:
            notes.append("Lệch giữa file tổng và per_image:")
            notes.extend(integrity)
        if notes:
            blocks.append(
                mo.callout(
                    mo.md("\n".join("- " + line for line in notes)),
                    kind="warn",
                    title="Một số file hoặc số liệu chưa khớp",
                )
            )
        return mo.vstack(blocks)

    _section_2()


@app.cell(hide_code=True)
def _(by_case_counts, config, fmt, md_table, render, run_summary, split_counts):
    def _section_3():
        timestamps = run_summary.get("timestamps") or {}
        retries = run_summary.get("retries") or {}
        steps = run_summary.get("steps") or []
        step_rows = [
            [index, step.get("name"), fmt(step.get("seconds"), 3), fmt(step.get("ok"))]
            for index, step in enumerate(steps, start=1)
        ]
        order = ("opt", "validation", "test", "held_out", "unscored")
        count_rows = []
        for name in order:
            cases = by_case_counts.get(name) or {}
            count_rows.append(
                [
                    name,
                    split_counts.get(name, "—"),
                    cases.get("far_small", "—"),
                    cases.get("blur_rain", "—"),
                    cases.get("van_truck", "—"),
                    cases.get("other", "—"),
                ]
            )
        held = split_counts.get("held_out")
        val_n = split_counts.get("validation")
        test_n = split_counts.get("test")
        if isinstance(held, int) and isinstance(val_n, int) and isinstance(test_n, int):
            if held == val_n + test_n:
                held_line = f"`held_out` = validation + test = **{held}**."
            else:
                held_line = (
                    f"`held_out` ghi **{held}**, validation + test = **{val_n + test_n}**."
                )
        else:
            held_line = "Không đủ count để đối chiếu `held_out` với validation + test."
        notes = run_summary.get("notes") or []
        note_lines = "\n".join(f"- {item}" for item in notes) or "- không có"
        llm_events = run_summary.get("llm_events") or []
        if llm_events:
            event = llm_events[0]
            llm_line = (
                f"LLM event đầu: ok {fmt(event.get('ok'))}, attempts {fmt(event.get('attempts'))}, "
                f"retries {fmt(event.get('retries'))}, {fmt(event.get('seconds'), 3)} giây."
            )
            if len(llm_events) != 1:
                llm_line += f" Tổng event: {len(llm_events)}."
        else:
            llm_line = "Không có `llm_events`."
        return render(
            """
            <a id="overview"></a>

            # Tổng quan

            Run `{mode}` trên dataset `{dataset}` (`doc_id` `{doc_id}`), `finished` = {finished}, `finish_reason` = `{finish}`. Model LLM `{llm}`, embedding `{embedding}`, SAM 3 `{sam}`. Prediction `{predictions}`. Seed **{seed}**.

            | | |
            | --- | --- |
            | k | {k} |
            | max_rounds | {max_rounds} |
            | early_stop_patience | {patience} |
            | bắt đầu | {started} |
            | export | {exported} |
            | lỗi SAM 3 | {sam_errors} |
            | lỗi Qwen | {qwen_errors} |
            | retry SAM 3 / Qwen | {sam_retries} / {qwen_retries} |

            {llm_line}

            ## Counts trong config

            {held_line} `unscored` là case `other`, không vào mean.

            {counts}

            ## Steps

            {steps}

            ## Notes trong run_summary

            {notes}
            """,
            mode=config.get("mode", "—"),
            dataset=config.get("dataset", "—"),
            doc_id=config.get("doc_id", "—"),
            finished=fmt(config.get("finished")),
            finish=config.get("finish_reason", "—"),
            llm=config.get("llm_model", "—"),
            embedding=config.get("embedding_model", "—"),
            sam=config.get("sam3_model", "—"),
            predictions=config.get("predictions", "—"),
            seed=fmt(config.get("seed")),
            k=fmt(config.get("k")),
            max_rounds=fmt(config.get("max_rounds")),
            patience=fmt(config.get("early_stop_patience")),
            started=timestamps.get("started", "—"),
            exported=timestamps.get("exported", "—"),
            sam_errors=len(run_summary.get("sam3_errors") or []),
            qwen_errors=len(run_summary.get("qwen_errors") or []),
            sam_retries=fmt(retries.get("sam3")),
            qwen_retries=fmt(retries.get("qwen")),
            llm_line=llm_line,
            held_line=held_line,
            counts=md_table(
                ["split", "n", "far_small", "blur_rain", "van_truck", "other"],
                count_rows,
            ),
            steps=md_table(["#", "step", "giây", "ok"], step_rows),
            notes=note_lines,
        )

    _section_3()


@app.cell(hide_code=True)
def _(candidates, feedback, final_test, md_table, render, rounds, row_by_label, same_value, words):
    def _section_4():
        baseline = row_by_label(candidates, "baseline") or final_test.get("baseline") or {}
        best_label = final_test.get("best_label")
        chosen = row_by_label(candidates, best_label) or final_test.get("best") or {}
        engine = chosen.get("prompt")
        if engine is None and rounds:
            engine = rounds[0].get("prompt")
        base_prompt = baseline.get("prompt")
        test_base = (final_test.get("best") or {}).get("prompt")
        prompt_rows = [
            [
                "baseline viết tay",
                f"`{base_prompt}`" if base_prompt else "—",
                words(base_prompt),
                len(base_prompt) if isinstance(base_prompt, str) else "—",
                "candidates_validation / final_test",
            ],
            [
                "Prompt Engine",
                f"`{engine}`" if engine else "—",
                words(engine),
                len(engine) if isinstance(engine, str) else "—",
                "vòng được chọn trên validation",
            ],
        ]
        if isinstance(engine, str) and isinstance(base_prompt, str):
            if words(engine) < words(base_prompt) and len(engine) < len(base_prompt):
                length_line = (
                    "Prompt của engine ngắn hơn baseline cả về số từ và số ký tự. "
                    "Chuỗi engine không thêm tính từ của baseline."
                )
            elif engine == base_prompt:
                length_line = "Hai prompt là cùng một chuỗi."
            else:
                length_line = "Số từ và số ký tự lấy từ chính chuỗi prompt trong JSON."
        else:
            length_line = "Thiếu một trong hai prompt."
        round_prompts = [rnd.get("prompt") for rnd in rounds]
        if rounds and same_value(round_prompts):
            round_line = (
                f"Cả **{len(rounds)}** vòng opt dùng cùng prompt `{round_prompts[0]}`."
            )
        elif rounds:
            listed = ", ".join(f"`{item}`" for item in round_prompts)
            round_line = f"Prompt từng vòng không giống nhau: {listed}."
        else:
            round_line = "Không có vòng nào trong `rounds.json`."
        if test_base and engine and test_base != engine:
            round_line += (
                f" Prompt khóa trên test (`{test_base}`) khác prompt validation (`{engine}`)."
            )
        yaml_parts = []
        seen_yaml = set()
        for label, block in (
            ("baseline", baseline),
            (str(best_label or "engine"), chosen),
        ):
            yaml = block.get("yaml")
            if not yaml or yaml in seen_yaml:
                continue
            seen_yaml.add(yaml)
            yaml_parts.append(f"**{label}**\n\n```yaml\n{str(yaml).strip()}\n```")
        for rnd in rounds:
            yaml = rnd.get("yaml")
            if not yaml or yaml in seen_yaml:
                continue
            seen_yaml.add(yaml)
            yaml_parts.append(
                f"**vòng {rnd.get('prompt_version')}**\n\n```yaml\n{str(yaml).strip()}\n```"
            )
        yaml_text = "\n\n".join(yaml_parts) or "_Không có YAML._"
        check_keys = []
        for rnd in rounds:
            check_keys.append(
                tuple(
                    (item.get("name"), item.get("ok"), item.get("reason"))
                    for item in (rnd.get("checks") or [])
                )
            )
        if check_keys and same_value(check_keys):
            check_rows = [
                [name, ok, reason] for name, ok, reason in check_keys[0]
            ]
            check_intro = "Mọi vòng opt có cùng kết quả guardrail."
            check_table = md_table(["check", "ok", "reason"], check_rows)
        else:
            check_rows = []
            for rnd in rounds:
                for item in rnd.get("checks") or []:
                    check_rows.append(
                        [
                            rnd.get("prompt_version"),
                            item.get("name"),
                            item.get("ok"),
                            item.get("reason"),
                        ]
                    )
            check_intro = "Guardrail theo từng vòng."
            check_table = md_table(["vòng", "check", "ok", "reason"], check_rows)
        query = rounds[0].get("query") if rounds else None
        fb_rounds = feedback.get("rounds") or []
        fb_round_1 = next(
            (item for item in fb_rounds if item.get("prompt_version") == 1),
            None,
        )
        if rounds and len(rounds) > 1 and fb_round_1:
            joined = "\n".join(
                item.get("error_text") or "" for item in (fb_round_1.get("items") or [])
            )
            second_query = rounds[1].get("query") or ""
            if second_query.strip() == joined.strip():
                query_2 = "Query vòng 2 trùng `error_text` của feedback vòng 1, nối bằng xuống dòng."
            else:
                query_2 = "Query vòng 2 không trùng nguyên văn feedback vòng 1."
        else:
            query_2 = "Không đủ vòng hoặc feedback để đối chiếu query vòng 2."
        return render(
            """
            <a id="prompts"></a>

            # Baseline và prompt

            Baseline là prompt viết tay, kind `baseline`. Prompt Engine là chuỗi trong vòng lặp, đưa vào SAM 3 sau guardrail. Không có prompt thứ ba trong các file này.

            {length_line}

            {round_line}

            {prompt_table}

            ## YAML

            {yaml_text}

            ## Guardrail

            {check_intro}

            {check_table}

            ## Query retrieve

            Vòng 1 query: `{query}`.

            {query_2}
            """,
            length_line=length_line,
            round_line=round_line,
            prompt_table=md_table(
                ["nguồn", "prompt", "số từ", "số ký tự", "file"],
                prompt_rows,
            ),
            yaml_text=yaml_text,
            check_intro=check_intro,
            check_table=check_table,
            query=query or "—",
            query_2=query_2,
        )

    _section_4()


@app.cell(hide_code=True)
def _(
    by_case_counts,
    candidates,
    case_mean_cell,
    final_test,
    fmt,
    group_case,
    md_table,
    per_test,
    per_val,
    render,
    rounds,
    row_by_label,
    run_summary,
    split_counts,
    zero_count,
):
    def _section_5():
        best_label = final_test.get("best_label")
        chosen = row_by_label(candidates, best_label) or {}
        baseline = row_by_label(candidates, "baseline") or {}
        test_best = final_test.get("best") or {}
        test_base = final_test.get("baseline") or {}
        opt_round = None
        for rnd in rounds:
            if rnd.get("prompt_version") == chosen.get("version"):
                opt_round = rnd
                break
        if opt_round is None and rounds:
            opt_round = rounds[0]
        opt_cases = by_case_counts.get("opt") or {}
        opt_n = split_counts.get("opt", "—")
        split_rows = [
            [
                "opt",
                (opt_round or {}).get("prompt", "—"),
                opt_n if opt_round else "—",
                fmt((opt_round or {}).get("iou_total")),
                "—",
                "—",
                "rounds.json, không có per_image opt",
            ],
            [
                "validation",
                chosen.get("prompt", "—"),
                fmt(chosen.get("n")),
                fmt(chosen.get("mean_iou")),
                fmt(chosen.get("n_iou_zero")),
                fmt(chosen.get("n_failed")),
                str(best_label or "—"),
            ],
            [
                "validation",
                baseline.get("prompt", "—"),
                fmt(baseline.get("n")),
                fmt(baseline.get("mean_iou")),
                fmt(baseline.get("n_iou_zero")),
                fmt(baseline.get("n_failed")),
                "baseline",
            ],
            [
                "test",
                test_best.get("prompt", "—"),
                fmt(test_best.get("n")),
                fmt(test_best.get("mean_iou")),
                fmt(test_best.get("n_iou_zero")),
                fmt(test_best.get("n_failed")),
                "best",
            ],
            [
                "test",
                test_base.get("prompt", "—"),
                fmt(test_base.get("n")),
                fmt(test_base.get("mean_iou")),
                fmt(test_base.get("n_iou_zero")),
                fmt(test_base.get("n_failed")),
                "baseline",
            ],
        ]
        case_names = ("far_small", "blur_rain", "van_truck")
        opt_case_rows = []
        stored = (opt_round or {}).get("iou_by_case") or {}
        for case in case_names:
            opt_case_rows.append(
                [case, fmt(opt_cases.get(case)), case_mean_cell(stored.get(case), opt_cases.get(case) or 0)]
            )
        notes = run_summary.get("notes") or []
        note_text = " ".join(str(item) for item in notes)
        val_rows = per_val.get(best_label) or []
        val_base_rows = per_val.get("baseline") or []
        test_rows = per_test.get("best") or []
        test_base_rows = per_test.get("baseline") or []

        def case_table(engine_rows, base_rows, engine_block, base_block):
            rows = []
            for case in case_names:
                eng = group_case(engine_rows, case)
                base = group_case(base_rows, case)
                eng_mean = (engine_block.get("iou_by_case") or {}).get(case)
                base_mean = (base_block.get("iou_by_case") or {}).get(case)
                rows.append(
                    [
                        case,
                        len(eng),
                        case_mean_cell(eng_mean, len(eng)),
                        zero_count(eng) if eng else "—",
                        len(base),
                        case_mean_cell(base_mean, len(base)),
                        zero_count(base) if base else "—",
                    ]
                )
            return rows

        val_case = case_table(val_rows, val_base_rows, chosen, baseline)
        test_case = case_table(test_rows, test_base_rows, test_best, test_base)
        blur_bits = []
        for name in ("opt", "validation", "test"):
            n = (by_case_counts.get(name) or {}).get("blur_rain")
            blur_bits.append(f"{name} **{fmt(n)}**")
        return render(
            """
            <a id="splits"></a>

            # IoU theo split và case

            Mean là binary IoU của mask `car` đã gộp, trung bình theo ảnh. Opt trong thư mục này chỉ có prompt của vòng lặp. Baseline `passenger car` được chấm trên validation và test. Không có mean baseline của opt trong `results/full_demo/`, nên bảng không điền ô đó từ báo cáo khác.

            Cột IoU = 0 của opt để trống vì không có `per_image` opt.

            {split_table}

            ## Opt theo case

            `n` lấy từ `split_counts`. Mean lấy từ `iou_by_case` của vòng được chọn. Ô `n = 0` là giá trị ghi trong file, không phải mean của một ảnh.

            {opt_cases}

            Notes của run: {note_text}

            ## Validation theo case

            Mean lấy từ `candidates_validation.json`. IoU = 0 đếm từ `per_image_validation.json`.

            {val_cases}

            ## Test theo case

            Mean lấy từ `final_test.json`. IoU = 0 đếm từ `per_image_test.json`.

            {test_cases}

            `blur_rain` scored: {blur_line}.
            """,
            split_table=md_table(
                ["split", "prompt", "n", "mean IoU", "IoU = 0", "failed", "nhãn"],
                split_rows,
            ),
            opt_cases=md_table(["case", "n", "mean ghi trong rounds"], opt_case_rows),
            note_text=note_text or "—",
            val_cases=md_table(
                [
                    "case",
                    "n engine",
                    "mean engine",
                    "IoU = 0 engine",
                    "n baseline",
                    "mean baseline",
                    "IoU = 0 baseline",
                ],
                val_case,
            ),
            test_cases=md_table(
                [
                    "case",
                    "n engine",
                    "mean engine",
                    "IoU = 0 engine",
                    "n baseline",
                    "mean baseline",
                    "IoU = 0 baseline",
                ],
                test_case,
            ),
            blur_line=", ".join(blur_bits) if blur_bits else "—",
        )

    _section_5()


@app.cell(hide_code=True)
def _(by_case_counts, candidates, final_test, group_case, per_test, rounds, row_by_label, zero_count):
    def _section_6():
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "figure.facecolor": "#fffdf8",
            "axes.facecolor": "#fffdf8",
            "savefig.facecolor": "#fffdf8",
            "text.color": "#102033",
            "axes.labelcolor": "#102033",
            "xtick.color": "#102033",
            "ytick.color": "#102033",
            "axes.edgecolor": "#16324f",
        })
        plt.close("all")
        best_label = final_test.get("best_label")
        chosen = row_by_label(candidates, best_label) or {}
        baseline = row_by_label(candidates, "baseline") or {}
        test_best = final_test.get("best") or {}
        test_base = final_test.get("baseline") or {}
        opt_round = None
        for rnd in rounds:
            if rnd.get("prompt_version") == chosen.get("version"):
                opt_round = rnd
                break
        if opt_round is None and rounds:
            opt_round = rounds[0]
        engine_values = [
            None if opt_round is None else opt_round.get("iou_total"),
            chosen.get("mean_iou"),
            test_best.get("mean_iou"),
        ]
        base_values = [None, baseline.get("mean_iou"), test_base.get("mean_iou")]
        labels = ["opt", "validation", "test"]
        figure, axes = plt.subplots(1, 2, figsize=(13.4, 6.4))
        width = 0.36
        engine_name = chosen.get("prompt") or "engine"
        base_name = baseline.get("prompt") or "baseline"
        eng_x = [index - width / 2 for index, value in enumerate(engine_values) if value is not None]
        eng_h = [float(value) for value in engine_values if value is not None]
        base_x = [index + width / 2 for index, value in enumerate(base_values) if value is not None]
        base_h = [float(value) for value in base_values if value is not None]
        if eng_h:
            axes[0].bar(eng_x, eng_h, width=width, color="#c2410c", label=str(engine_name))
            for x_pos, height in zip(eng_x, eng_h):
                axes[0].text(x_pos, height + 0.03, f"{height:.4f}", ha="center", va="bottom", fontsize=14, color="#102033")
        if base_h:
            axes[0].bar(base_x, base_h, width=width, color="#16324f", label=str(base_name))
            for x_pos, height in zip(base_x, base_h):
                axes[0].text(x_pos, height + 0.03, f"{height:.4f}", ha="center", va="bottom", fontsize=14, color="#102033")
        axes[0].set_xticks(range(len(labels)), labels)
        axes[0].set_ylim(0, 1.22)
        axes[0].set_ylabel("mean IoU")
        axes[0].set_title("Mean IoU theo split")
        if eng_h or base_h:
            axes[0].legend(loc="upper right", framealpha=0.95)

        cases = ["far_small", "blur_rain", "van_truck"]
        best_rows = per_test.get("best") or []
        base_rows = per_test.get("baseline") or []
        best_zeros = [zero_count(group_case(best_rows, case)) for case in cases]
        base_zeros = [zero_count(group_case(base_rows, case)) for case in cases]
        positions = list(range(len(cases)))
        if any(best_rows):
            best_pos = [index - width / 2 for index in positions]
            axes[1].bar(
                best_pos,
                best_zeros,
                width=width,
                color="#c2410c",
                label=str(test_best.get("prompt") or "best"),
            )
            for x_pos, height in zip(best_pos, best_zeros):
                axes[1].text(x_pos, height, str(height), ha="center", va="bottom", fontsize=14, color="#102033")
        if any(base_rows):
            base_pos = [index + width / 2 for index in positions]
            axes[1].bar(
                base_pos,
                base_zeros,
                width=width,
                color="#16324f",
                label=str(test_base.get("prompt") or "baseline"),
            )
            for x_pos, height in zip(base_pos, base_zeros):
                axes[1].text(x_pos, height, str(height), ha="center", va="bottom", fontsize=14, color="#102033")
        axes[1].set_xticks(positions, cases)
        axes[1].set_ylabel("số ảnh IoU = 0")
        blur_n = (by_case_counts.get("test") or {}).get("blur_rain")
        axes[1].set_title(f"Test, IoU = 0 theo case (blur_rain n = {blur_n})")
        peak_zero = max([*best_zeros, *base_zeros], default=1)
        axes[1].set_ylim(0, (peak_zero * 1.28) if peak_zero else 1)
        if best_rows or base_rows:
            axes[1].legend(loc="upper right")
        figure.tight_layout()
        return axes[0]

    _section_6()


@app.cell(hide_code=True)
def _(
    by_case_counts,
    case_mean_cell,
    config,
    feedback,
    fmt,
    md_table,
    render,
    rounds,
    same_value,
    short_id,
):
    def _section_7():
        opt_cases = by_case_counts.get("opt") or {}
        case_names = ("far_small", "blur_rain", "van_truck")
        round_rows = []
        for rnd in rounds:
            stored = rnd.get("iou_by_case") or {}
            chunk_ids = rnd.get("chunk_ids") or []
            feedback_flag = "có" if rnd.get("feedback") else "không"
            round_rows.append(
                [
                    rnd.get("prompt_version"),
                    f"`{rnd.get('prompt')}`",
                    fmt(rnd.get("iou_total")),
                    *[case_mean_cell(stored.get(case), opt_cases.get(case) or 0) for case in case_names],
                    f"{len(chunk_ids)}/{fmt(rnd.get('k'))}",
                    fmt(rnd.get("regenerate_count")),
                    feedback_flag,
                ]
            )
        ious = [rnd.get("iou_total") for rnd in rounds]
        if len(ious) >= 2 and all(value is not None for value in ious):
            best = float(ious[0])
            later_better = False
            for value in ious[1:]:
                if float(value) > best:
                    later_better = True
                    best = float(value)
            if later_better:
                stop_line = "Có vòng sau cao hơn `iou_total` tốt nhất trước đó."
            else:
                stop_line = "Không có vòng sau nào cao hơn `iou_total` tốt nhất trước đó."
        else:
            stop_line = "Không đủ `iou_total` để nói vòng sau có cải thiện hay không."
        prompts = [rnd.get("prompt") for rnd in rounds]
        if prompts and same_value(prompts):
            prompt_line = f"Prompt không đổi (`{prompts[0]}`)."
        else:
            prompt_line = "Prompt thay đổi giữa các vòng."
        max_rounds = config.get("max_rounds")
        patience = config.get("early_stop_patience")
        finish = config.get("finish_reason")
        unique_ids = sorted({str(cid) for rnd in rounds for cid in (rnd.get("chunk_ids") or [])})
        chunk_counts = [len(rnd.get("chunk_ids") or []) for rnd in rounds]
        k_values = [rnd.get("k") for rnd in rounds]
        sources = sorted({str(item) for rnd in rounds for item in (rnd.get("chunk_id_sources") or [])})
        if chunk_counts and same_value(chunk_counts) and same_value(k_values):
            chunk_line = (
                f"Mỗi vòng trả **{chunk_counts[0]}** chunk trong khi k = **{fmt(k_values[0])}**."
            )
        else:
            chunk_line = (
                f"Số chunk từng vòng: {chunk_counts or '—'}. k từng vòng: {k_values or '—'}."
            )
        chunk_rows = []
        for rnd in rounds:
            ids = rnd.get("chunk_ids") or []
            chunk_sources = rnd.get("chunk_id_sources") or []
            chunks = rnd.get("chunks") or []
            for index, cid in enumerate(ids):
                source = chunk_sources[index] if index < len(chunk_sources) else "—"
                meta = chunks[index] if index < len(chunks) and isinstance(chunks[index], dict) else {}
                chunk_rows.append(
                    [
                        rnd.get("prompt_version"),
                        cid,
                        source,
                        meta.get("source", "—"),
                        fmt(meta.get("score"), 6),
                    ]
                )
        low_rows = []
        for rnd in rounds:
            for item in rnd.get("low_iou_images") or []:
                low_rows.append(
                    [
                        rnd.get("prompt_version"),
                        item.get("case"),
                        fmt(item.get("iou")),
                        short_id(item.get("image_id")),
                    ]
                )
        fb_rounds = feedback.get("rounds") or []
        attached = sum(1 for rnd in rounds if rnd.get("feedback"))
        fb_rows = []
        for fb_round in fb_rounds:
            version = fb_round.get("prompt_version")
            rnd = next((item for item in rounds if item.get("prompt_version") == version), {})
            low_ids = list(rnd.get("low_iou_image_ids") or [])
            item_ids = [item.get("image_id") for item in (fb_round.get("items") or [])]
            match = "khớp" if item_ids == low_ids else "không khớp"
            for item in fb_round.get("items") or []:
                fb_rows.append(
                    [
                        version,
                        fb_round.get("reviewer"),
                        item.get("case"),
                        match,
                        short_id(item.get("image_id")),
                        item.get("error_text"),
                    ]
                )
        author = feedback.get("author") or "—"
        return render(
            """
            <a id="rounds"></a>

            # Vòng lặp trên opt

            File ghi **{n_rounds}** vòng. `max_rounds` = **{max_rounds}**. `early_stop_patience` = **{patience}**. `finish_reason` = `{finish}`. {stop_line} {prompt_line}

            `iou_by_case.blur_rain` vẫn hiện trong bảng. Opt có **{blur_n}** ảnh `blur_rain`, nên ô đó không phải IoU đo được. Case không có ảnh không nằm trong mẫu số của `iou_total`.

            {round_table}

            ## Retriever

            {chunk_line} Id khác nhau trên cả run: **{n_unique}** ({unique_ids}). `chunk_id_sources`: {sources}.

            Bảng dưới chỉ có id, source và score. Không chép nội dung chunk guideline vào báo cáo.

            {chunk_table}

            ## Ảnh low-IoU trên opt

            {low_table}

            ## Feedback

            `feedback.json` ghi author `{author}`, **{n_fb}** vòng. `rounds.json` gắn field `feedback` ở **{attached}** vòng.

            {fb_table}
            """,
            n_rounds=len(rounds),
            max_rounds=fmt(max_rounds),
            patience=fmt(patience),
            finish=finish or "—",
            stop_line=stop_line,
            prompt_line=prompt_line,
            blur_n=fmt(opt_cases.get("blur_rain")),
            round_table=md_table(
                [
                    "vòng",
                    "prompt",
                    "iou_total",
                    "far_small",
                    "blur_rain",
                    "van_truck",
                    "chunk/k",
                    "regenerate",
                    "feedback gắn",
                ],
                round_rows,
            ),
            chunk_line=chunk_line,
            n_unique=len(unique_ids),
            unique_ids=", ".join(unique_ids) if unique_ids else "—",
            sources=", ".join(sources) if sources else "—",
            chunk_table=md_table(
                ["vòng", "chunk id", "id source", "source", "score"],
                chunk_rows,
            ),
            low_table=md_table(["vòng", "case", "IoU", "image_id"], low_rows),
            author=author,
            n_fb=len(fb_rounds),
            attached=attached,
            fb_table=md_table(
                ["vòng", "reviewer", "case", "so với low_iou_image_ids", "image_id", "error_text"],
                fb_rows,
            ),
        )

    _section_7()


@app.cell(hide_code=True)
def _(candidates, count_change, delta_text, final_test, fmt, md_table, pair_stats, per_val, render, rounds, same_value):
    def _section_8():
        best_label = final_test.get("best_label")
        locked = final_test.get("locked")
        rows = []
        for block in candidates:
            stored = block.get("iou_by_case") or {}
            rows.append(
                [
                    block.get("label"),
                    block.get("kind"),
                    fmt(block.get("version")),
                    f"`{block.get('prompt')}`",
                    fmt(block.get("mean_iou")),
                    fmt(stored.get("far_small")),
                    fmt(stored.get("blur_rain")),
                    fmt(stored.get("van_truck")),
                    fmt(block.get("n")),
                    fmt(block.get("n_iou_zero")),
                    fmt(block.get("n_failed")),
                    fmt(block.get("reused")),
                ]
            )
        by_label = {block.get("label"): block for block in candidates}
        chosen = by_label.get(best_label) or {}
        baseline = by_label.get("baseline") or {}
        others = [block for block in candidates if block.get("label") not in {best_label, "baseline"}]
        reused_same = []
        for block in others:
            same_prompt = block.get("prompt") == chosen.get("prompt")
            same_mean = block.get("mean_iou") == chosen.get("mean_iou")
            reused_same.append(
                f"`{block.get('label')}` prompt {'trùng' if same_prompt else 'khác'}, "
                f"mean {'bằng' if same_mean else 'khác'}, `reused` = {fmt(block.get('reused'))}"
            )
        other_line = "; ".join(reused_same) if reused_same else "không có candidate khác"
        prompts = [rnd.get("prompt") for rnd in rounds]
        if chosen and prompts and same_value(prompts) and chosen.get("prompt") == prompts[0]:
            prompt_line = "Prompt được chọn trùng prompt của mọi vòng opt."
        elif chosen:
            prompt_line = "Prompt được chọn không trùng hết các vòng opt."
        else:
            prompt_line = "Không tìm thấy candidate của `best_label`."
        stats = pair_stats(per_val.get(best_label), per_val.get("baseline"))
        if stats["missing"] or stats["extra"]:
            pair_line = (
                f"Ghép ảnh validation lệch: thiếu {stats['missing']} ở nhãn được chọn, "
                f"thừa {stats['extra']} ở baseline."
            )
        else:
            pair_line = (
                f"Trên **{stats['n']}** ảnh ghép được: IoU nhãn được chọn cao hơn baseline ở **{stats['up']}** ảnh, "
                f"thấp hơn **{stats['down']}**, bằng **{stats['same']}**. "
                f"Ảnh baseline IoU = 0 thành dương: **{stats['zero_to_pos']}**. "
                f"Ảnh dương thành IoU = 0: **{stats['pos_to_zero']}**."
            )
        higher = (
            chosen.get("mean_iou") is not None
            and baseline.get("mean_iou") is not None
            and float(chosen.get("mean_iou")) > float(baseline.get("mean_iou"))
        )
        if higher:
            why = (
                f"`{best_label}` có mean validation cao hơn baseline "
                f"({fmt(chosen.get('mean_iou'))} so với {fmt(baseline.get('mean_iou'))}, "
                f"Δ {delta_text(chosen.get('mean_iou'), baseline.get('mean_iou'))})."
            )
        elif chosen and baseline:
            why = (
                f"`{best_label}` không cao hơn baseline "
                f"({fmt(chosen.get('mean_iou'))} so với {fmt(baseline.get('mean_iou'))})."
            )
        else:
            why = "Thiếu mean để giải thích nhãn được chọn."
        return render(
            """
            <a id="validation"></a>

            # Chọn prompt trên validation

            `final_test.best_label` = `{best_label}`. `locked` = {locked}. {why} Candidate khác: {other_line}. {prompt_line}

            IoU = 0: {zero_line}. `n` của nhãn được chọn là **{n}**.

            {table}

            {pair_line}

            `blur_rain` trên validation là null trong `iou_by_case` khi không có ảnh. Ô đó không dùng để chọn prompt.
            """,
            best_label=best_label or "—",
            locked=fmt(locked),
            why=why,
            other_line=other_line,
            prompt_line=prompt_line,
            zero_line=count_change(chosen.get("n_iou_zero"), baseline.get("n_iou_zero")),
            n=fmt(chosen.get("n")),
            table=md_table(
                [
                    "nhãn",
                    "kind",
                    "version",
                    "prompt",
                    "mean IoU",
                    "far_small",
                    "blur_rain",
                    "van_truck",
                    "n",
                    "IoU = 0",
                    "failed",
                    "reused",
                ],
                rows,
            ),
            pair_line=pair_line,
        )

    _section_8()


@app.cell(hide_code=True)
def _(
    by_case_counts,
    count_change,
    delta_text,
    final_test,
    fmt,
    group_case,
    md_table,
    pair_stats,
    per_test,
    render,
    zero_count,
):
    def _section_9():
        best = final_test.get("best") or {}
        baseline = final_test.get("baseline") or {}
        best_rows = per_test.get("best") or []
        base_rows = per_test.get("baseline") or []
        case_names = ("far_small", "blur_rain", "van_truck")
        case_rows = []
        for case in case_names:
            eng = group_case(best_rows, case)
            base = group_case(base_rows, case)
            eng_mean = (best.get("iou_by_case") or {}).get(case)
            base_mean = (baseline.get("iou_by_case") or {}).get(case)
            case_rows.append(
                [
                    case,
                    len(eng),
                    fmt(eng_mean),
                    zero_count(eng),
                    len(base),
                    fmt(base_mean),
                    zero_count(base),
                    delta_text(eng_mean, base_mean),
                    count_change(zero_count(eng), zero_count(base)),
                ]
            )
        stats = pair_stats(best_rows, base_rows)
        if stats["missing"] or stats["extra"]:
            pair_line = (
                f"Ghép ảnh test lệch: thiếu {stats['missing']} ở best, thừa {stats['extra']} ở baseline."
            )
        else:
            pair_line = (
                f"Trên **{stats['n']}** ảnh ghép được: IoU best cao hơn baseline ở **{stats['up']}** ảnh, "
                f"thấp hơn **{stats['down']}**, bằng **{stats['same']}**. "
                f"Ảnh baseline IoU = 0 thành dương: **{stats['zero_to_pos']}**. "
                f"Ảnh dương thành IoU = 0: **{stats['pos_to_zero']}**."
            )
        keys = ", ".join(str(key) for key in per_test.keys()) or "—"
        test_cases = by_case_counts.get("test") or {}
        return render(
            """
            <a id="test"></a>

            # Test chấm một lần

            `locked` = {locked}. `best_label` = `{best_label}`. `per_image_test.json` có các khóa: {keys}. Không có khóa cho từng vòng opt. Test chấm prompt đã chọn và baseline, không chấm lại mọi vòng.

            Mean `{best_prompt}` = **{best_mean}** (`n` {best_n}, failed {best_failed}). Baseline `{base_prompt}` = **{base_mean}** (`n` {base_n}, failed {base_failed}). Δ mean = **{delta}**.

            Ảnh IoU = 0: {zero_line}.

            {summary}

            ## Theo case

            `n` theo case trên test, từ config: far_small **{n_far}**, blur_rain **{n_blur}**, van_truck **{n_van}**.

            {cases}

            {pair_line}
            """,
            locked=fmt(final_test.get("locked")),
            best_label=final_test.get("best_label") or "—",
            keys=keys,
            best_prompt=best.get("prompt") or "—",
            best_mean=fmt(best.get("mean_iou")),
            best_n=fmt(best.get("n")),
            best_failed=fmt(best.get("n_failed")),
            base_prompt=baseline.get("prompt") or "—",
            base_mean=fmt(baseline.get("mean_iou")),
            base_n=fmt(baseline.get("n")),
            base_failed=fmt(baseline.get("n_failed")),
            delta=delta_text(best.get("mean_iou"), baseline.get("mean_iou")),
            zero_line=count_change(best.get("n_iou_zero"), baseline.get("n_iou_zero")),
            summary=md_table(
                ["nhãn", "prompt", "n", "mean IoU", "IoU = 0", "failed"],
                [
                    ["best", f"`{best.get('prompt')}`", fmt(best.get("n")), fmt(best.get("mean_iou")), fmt(best.get("n_iou_zero")), fmt(best.get("n_failed"))],
                    ["baseline", f"`{baseline.get('prompt')}`", fmt(baseline.get("n")), fmt(baseline.get("mean_iou")), fmt(baseline.get("n_iou_zero")), fmt(baseline.get("n_failed"))],
                ],
            ),
            n_far=fmt(test_cases.get("far_small")),
            n_blur=fmt(test_cases.get("blur_rain")),
            n_van=fmt(test_cases.get("van_truck")),
            cases=md_table(
                [
                    "case",
                    "n best",
                    "mean best",
                    "IoU = 0 best",
                    "n baseline",
                    "mean baseline",
                    "IoU = 0 baseline",
                    "Δ mean",
                    "IoU = 0",
                ],
                case_rows,
            ),
            pair_line=pair_line,
        )

    _section_9()


@app.cell(hide_code=True)
def _(DEMO, close, fmt, mo, overlays, per_test, per_val, render, short_id):
    def _section_10():
        def lookup(item):
            image_id = item.get("image_id")
            if item.get("split") == "validation":
                rows = per_val.get(item.get("label")) or []
            elif item.get("split") == "test":
                rows = per_test.get("best") or []
            else:
                rows = []
            for row in rows:
                if row.get("image_id") == image_id:
                    return row
            return None

        blocks = [
            render(
                """
                <a id="overlays"></a>

                # Overlay

                Các ảnh dưới đây là best, median và worst đã ghi trong `overlays.json` cho validation và test của nhãn được chọn. Feedback trong file mô tả viền xanh lá là gold và mảng xanh dương là prediction. Ảnh thu tới tối đa 640 px.
                """
            )
        ]
        if not overlays:
            blocks.append(mo.md("Không có overlay trong `overlays.json`."))
        for item in overlays:
            row = lookup(item)
            case = row.get("case") if row else "—"
            bits = [
                f"**{item.get('tag')}**",
                f"`{item.get('split')}`",
                f"nhãn `{item.get('label')}`",
                f"case `{case}`",
                f"IoU {fmt(item.get('iou'))}",
            ]
            if row is not None:
                bits.append(f"pred {row.get('pred_area')} px")
                bits.append(f"gold {row.get('gold_area')} px")
                if not close(row.get("iou"), item.get("iou")):
                    bits.append(f"per_image IoU {fmt(row.get('iou'))} lệch overlay")
            bits.append(f"`{short_id(item.get('image_id'))}`")
            blocks.append(mo.md(" · ".join(bits)))
            path = DEMO / item["file"] if item.get("file") else None
            if path is not None and path.is_file():
                blocks.append(mo.image(src=path, alt=str(item.get("tag")), width=480))
            else:
                blocks.append(mo.md("_Không có file overlay trong repo public này (ảnh nuImages không được đưa lên)._"))
        return mo.vstack(blocks)

    _section_10()


@app.cell(hide_code=True)
def _(
    candidates,
    config,
    count_change,
    delta_text,
    final_test,
    fmt,
    render,
    rounds,
    row_by_label,
    same_value,
):
    def _section_11():
        best_label = final_test.get("best_label")
        chosen = row_by_label(candidates, best_label) or {}
        baseline = row_by_label(candidates, "baseline") or {}
        test_best = final_test.get("best") or {}
        test_base = final_test.get("baseline") or {}
        prompts = [rnd.get("prompt") for rnd in rounds]
        ious = [fmt(rnd.get("iou_total")) for rnd in rounds]
        if prompts and same_value(prompts):
            prompt_line = f"Prompt các vòng opt cùng là `{prompts[0]}`."
        else:
            prompt_line = "Prompt các vòng opt không cùng một chuỗi."
        return render(
            """
            <a id="conclusion"></a>

            # Kết luận

            Baseline viết tay là `{base_prompt}`. Prompt Engine được khóa là `{engine}` (nhãn `{best_label}`).

            Trên validation (`n` {val_n}), mean `{best_label}` = **{val_engine}**, baseline = **{val_base}** (Δ {val_delta}).

            Test `locked` = {locked}. Mean `{test_prompt}` = **{test_engine}**, baseline = **{test_base}** (Δ {test_delta}). Ảnh IoU = 0: {zero_line}.

            Opt chạy **{n_rounds}/{max_rounds}** vòng, `finish_reason` = `{finish}`, patience **{patience}**. `iou_total` theo vòng: {ious}. {prompt_line} Mean cao hơn baseline, trên lần chạy này, đi cùng một prompt ngắn không đổi sau feedback.
            """,
            base_prompt=baseline.get("prompt") or "—",
            engine=chosen.get("prompt") or test_best.get("prompt") or "—",
            best_label=best_label or "—",
            val_n=fmt(chosen.get("n")),
            val_engine=fmt(chosen.get("mean_iou")),
            val_base=fmt(baseline.get("mean_iou")),
            val_delta=delta_text(chosen.get("mean_iou"), baseline.get("mean_iou")),
            locked=fmt(final_test.get("locked")),
            test_prompt=test_best.get("prompt") or "—",
            test_engine=fmt(test_best.get("mean_iou")),
            test_base=fmt(test_base.get("mean_iou")),
            test_delta=delta_text(test_best.get("mean_iou"), test_base.get("mean_iou")),
            zero_line=count_change(test_best.get("n_iou_zero"), test_base.get("n_iou_zero")),
            n_rounds=len(rounds),
            max_rounds=fmt(config.get("max_rounds")),
            finish=config.get("finish_reason") or "—",
            patience=fmt(config.get("early_stop_patience")),
            ious=", ".join(ious) if ious else "—",
            prompt_line=prompt_line,
        )

    _section_11()


@app.cell(hide_code=True)
def _(
    by_case_counts,
    candidates,
    config,
    feedback,
    final_test,
    fmt,
    overlays,
    per_test,
    render,
    rounds,
    row_by_label,
    same_value,
    split_counts,
    words,
):
    def _section_12():
        best_label = final_test.get("best_label")
        chosen = row_by_label(candidates, best_label) or {}
        baseline = row_by_label(candidates, "baseline") or {}
        prompts = [rnd.get("prompt") for rnd in rounds]
        ious = [rnd.get("iou_total") for rnd in rounds]
        improved = False
        if len(ious) >= 2 and all(value is not None for value in ious):
            best = float(ious[0])
            for value in ious[1:]:
                if float(value) > best:
                    improved = True
                    best = float(value)
        if improved:
            improve_line = "Có vòng sau cải thiện `iou_total`."
        elif len(ious) >= 2:
            improve_line = "Không có vòng sau nào cải thiện `iou_total`."
        else:
            improve_line = "Chưa đủ vòng để nói cải thiện."
        max_rounds = config.get("max_rounds")
        if isinstance(max_rounds, int) and len(rounds) < max_rounds:
            improve_line += " Số vòng ghi trong file nhỏ hơn `max_rounds`."
        elif isinstance(max_rounds, int) and len(rounds) == max_rounds:
            improve_line += " Đã chạy hết `max_rounds`."
        engine = chosen.get("prompt")
        base_prompt = baseline.get("prompt")
        if isinstance(engine, str) and isinstance(base_prompt, str):
            if engine != base_prompt and words(engine) < words(base_prompt) and same_value(prompts):
                wording = (
                    f"Prompt thắng là `{engine}` ({words(engine)} từ, {len(engine)} ký tự), "
                    f"ngắn hơn baseline `{base_prompt}` ({words(base_prompt)} từ, {len(base_prompt)} ký tự), "
                    "và không đổi sau feedback. Phần cải thiện so với baseline đến từ cách diễn đạt prompt, "
                    "chưa phải do vòng lặp học thêm quy tắc trong guideline."
                )
            else:
                wording = (
                    f"Prompt được chọn là `{engine}`, baseline là `{base_prompt}`. "
                    "Không kết luận thêm về việc prompt có học guideline hay không từ riêng độ dài chuỗi."
                )
        else:
            wording = "Thiếu prompt để so độ dài."
        author = feedback.get("author") or "—"
        fb_rounds = feedback.get("rounds") or []
        reviewers = sorted({str(item.get("reviewer")) for item in fb_rounds if item.get("reviewer")})
        n_items = [len(item.get("items") or []) for item in fb_rounds]
        attached = sum(1 for rnd in rounds if rnd.get("feedback"))
        low_n = len((rounds[0].get("low_iou_images") if rounds else []) or [])
        overlay_ids = {item.get("image_id") for item in overlays}
        low_ids = list((rounds[0].get("low_iou_image_ids") if rounds else []) or [])
        low_in_overlays = sum(1 for image_id in low_ids if image_id in overlay_ids)
        unique_ids = sorted({str(cid) for rnd in rounds for cid in (rnd.get("chunk_ids") or [])})
        chunk_counts = [len(rnd.get("chunk_ids") or []) for rnd in rounds]
        k_values = [rnd.get("k") for rnd in rounds]
        reused = [
            f"`{block.get('label')}` reused {fmt(block.get('reused'))}"
            for block in candidates
            if block.get("label") != best_label and block.get("kind") != "baseline"
        ]
        blur = {
            name: (by_case_counts.get(name) or {}).get("blur_rain")
            for name in ("opt", "validation", "test")
        }
        blur_total = sum(int(value) for value in blur.values() if isinstance(value, int))
        other_n = (by_case_counts.get("unscored") or {}).get("other")
        unscored_n = split_counts.get("unscored")
        if other_n == unscored_n:
            other_line = f"Case `other` unscored: **{fmt(other_n)}** ảnh, không chấm."
        else:
            other_line = (
                f"`other` ghi **{fmt(other_n)}**, `unscored` ghi **{fmt(unscored_n)}**."
            )
        opt_stored = None
        for rnd in rounds:
            if rnd.get("prompt_version") == chosen.get("version"):
                opt_stored = (rnd.get("iou_by_case") or {}).get("blur_rain")
                break
        if opt_stored is None and rounds:
            opt_stored = (rounds[0].get("iou_by_case") or {}).get("blur_rain")
        val_blur = (chosen.get("iou_by_case") or {}).get("blur_rain")
        test_keys = list(per_test.keys())
        if set(test_keys) == {"best", "baseline"}:
            test_key_line = "Test chỉ có khóa `best` và `baseline`."
        else:
            listed = ", ".join(f"`{key}`" for key in test_keys) or "—"
            test_key_line = f"Khóa trong `per_image_test.json`: {listed}."
        k_larger = bool(
            k_values
            and chunk_counts
            and len(k_values) == len(chunk_counts)
            and all(
                isinstance(k, int) and isinstance(count, int) and k > count
                for k, count in zip(k_values, chunk_counts)
            )
        )
        if k_larger:
            index_line = "Mỗi vòng trả ít chunk hơn k."
        else:
            index_line = "Đối chiếu k với số chunk trả về ở bảng vòng lặp."
        return render(
            """
            <a id="limitations"></a>

            # Hạn chế

            - Vòng lặp ghi **{n_rounds}/{max_rounds}** vòng rồi dừng. `finish_reason` = `{finish}`. `early_stop_patience` = **{patience}**. {improve_line}
            - {wording}
            - Feedback do `{author}`. Reviewer trong file: {reviewers}. Số mục feedback từng vòng: {n_items}. Vòng 1 có **{low_n}** ảnh low-IoU. **{low_in_overlays}** ảnh đó nằm trong `overlays.json`. Feedback là giả lập, viết từ mô tả overlay của các ảnh low-IoU đó, không phải reviewer người thật. `rounds.json` chỉ gắn feedback ở **{attached}** vòng; feedback còn lại không sinh thêm một prompt mới.
            - Validation có **{val_n}** ảnh. Chọn best trên mẫu này nhiễu. Nhãn khóa là `{best_label}` vì mean validation của nó cao hơn baseline trên đúng mẫu đó, không phải vì một tập lớn hơn.
            - `blur_rain`: opt **{blur_opt}**, validation **{blur_val}**, test **{blur_test}**, tổng ảnh scored **{blur_total}**. Không kết luận được cho mưa/mờ. Trên opt, `iou_by_case.blur_rain` = {opt_blur} trong khi số ảnh là {blur_opt}. Note của run nói `IouEvaluator` điền case rỗng, còn `evaluate_set` để mean null. Validation ghi {val_blur}.
            - Index RAG: id chunk khác nhau **{n_unique}** ({unique_ids}). k = {k_values}, số chunk trả về từng vòng = {chunk_counts}. {index_line}
            - Candidate sau prompt trùng nhau không được chấm lại trên validation: {reused}. {test_key_line}
            - {other_line}
            - `results/full_demo/` không có per_image opt và không có mean baseline `passenger car` trên opt. Báo cáo này không lấy hai số đó từ `results/summary.json`.
            """,
            n_rounds=len(rounds),
            max_rounds=fmt(config.get("max_rounds")),
            finish=config.get("finish_reason") or "—",
            patience=fmt(config.get("early_stop_patience")),
            improve_line=improve_line,
            wording=wording,
            author=author,
            reviewers=", ".join(reviewers) if reviewers else "—",
            n_items=", ".join(str(item) for item in n_items) if n_items else "—",
            low_n=low_n,
            low_in_overlays=low_in_overlays,
            attached=attached,
            val_n=fmt(chosen.get("n")),
            best_label=best_label or "—",
            blur_opt=fmt(blur.get("opt")),
            blur_val=fmt(blur.get("validation")),
            blur_test=fmt(blur.get("test")),
            blur_total=blur_total,
            opt_blur=fmt(opt_stored),
            val_blur=fmt(val_blur),
            n_unique=len(unique_ids),
            unique_ids=", ".join(unique_ids) if unique_ids else "—",
            k_values=", ".join(str(item) for item in k_values) if k_values else "—",
            chunk_counts=", ".join(str(item) for item in chunk_counts) if chunk_counts else "—",
            reused="; ".join(reused) if reused else "—",
            test_key_line=test_key_line,
            index_line=index_line,
            other_line=other_line,
        )

    _section_12()


@app.cell(hide_code=True)
def _():
    def _edge_section():
        import json
        from pathlib import Path

        import marimo as mo

        def _edge_report_dir():
            import os

            if os.environ.get("T024_IOU_REPORT_DIR"):
                return Path(os.environ["T024_IOU_REPORT_DIR"])
            try:
                here = Path(__file__).resolve().parent
            except NameError:
                notebook = mo.notebook_dir()
                here = Path(notebook) if notebook is not None else Path.cwd()
            if (here / "results").is_dir():
                return here
            return here

        edge_dir = _edge_report_dir() / "results" / "edge_loop"

        def _load(name):
            path = edge_dir / name
            if not path.is_file():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

        def _fmt(value, digits=4):
            if value is None:
                return "—"
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return str(value)

        def _ci(block):
            if not isinstance(block, dict) or block.get("mean") is None:
                return "—"
            return (
                f"{_fmt(block.get('mean'))} "
                f"[{_fmt(block.get('ci_low'))}, {_fmt(block.get('ci_high'))}] "
                f"(n={block.get('n')})"
            )

        def _verdict(val):
            lines = []
            for case in ("van_truck", "far_small"):
                delta = ((val or {}).get(case) or {}).get("delta") or {}
                mean = delta.get("mean")
                lo = delta.get("ci_low")
                hi = delta.get("ci_high")
                if mean is None:
                    lines.append(f"- `{case}`: không đủ ảnh để tính delta.")
                    continue
                if lo is not None and hi is not None and lo <= 0 <= hi:
                    direction = "khoảng tin cậy cắt qua 0, chưa tách được khỏi baseline"
                elif mean > 0:
                    direction = "trung bình cao hơn baseline"
                else:
                    direction = "trung bình thấp hơn baseline"
                lines.append(
                    f"- `{case}`: delta {_fmt(mean)} [{_fmt(lo)}, {_fmt(hi)}], {direction}."
                )
            return "Kết quả đo được trên validation, không chọn lọc:\n\n" + "\n".join(lines)

        def _opt_note(rounds, baseline):
            base = (baseline.get("opt") or {}).get("by_case") or {}
            lines = []
            for case in ("van_truck", "far_small"):
                base_mean = (base.get(case) or {}).get("mean_iou")
                best = None
                best_round = None
                for rnd in rounds or []:
                    iou = ((rnd.get("winners") or {}).get(case) or {}).get("iou")
                    if iou is None:
                        continue
                    if best is None or float(iou) > best:
                        best = float(iou)
                        best_round = rnd.get("round")
                if base_mean is None or best is None:
                    continue
                if best > float(base_mean):
                    lines.append(
                        f"Trên opt, `{case}` vòng {best_round} ({_fmt(best)}) cao hơn baseline `car` ({_fmt(base_mean)})."
                    )
                else:
                    lines.append(
                        f"Trên opt, IoU thắng cao nhất của `{case}` là vòng {best_round} ({_fmt(best)}), thấp hơn baseline `car` ({_fmt(base_mean)})."
                    )
            return "\n".join(lines)

        def _fallback_note(rounds):
            total = 0
            used = 0
            for rnd in rounds or []:
                for cand in rnd.get("candidates") or []:
                    total += 1
                    if cand.get("fallback"):
                        used += 1
            return (
                f"{used}/{total} candidate là fallback cố định vì Qwen không qua validator "
                "(quá 28 CLIP token hoặc có negation). Fallback không phải câu doc-rag vừa sinh."
            )

        def _other_note(comparisons):
            lines = []
            for label, block in ((comparisons or {}).get("other") or {}).items():
                delta = block.get("delta") or {}
                mean = delta.get("mean")
                lo = delta.get("ci_low")
                hi = delta.get("ci_high")
                if mean is None:
                    continue
                if lo is not None and hi is not None and hi < 0:
                    tone = "CI nằm dưới 0, regression so với baseline"
                elif lo is not None and hi is not None and lo > 0:
                    tone = "CI nằm trên 0"
                else:
                    tone = "CI cắt qua 0"
                lines.append(f"`other` / `{label}`: delta {_fmt(mean)} [{_fmt(lo)}, {_fmt(hi)}], {tone}.")
            return "\n".join(lines)

        def _md_table(headers, rows):
            if not rows:
                return "_Không có dòng nào._"

            def cell(value):
                return str(value).replace("|", "\\|").replace("\n", " ")

            head = "| " + " | ".join(headers) + " |"
            rule = "| " + " | ".join("---" for _ in headers) + " |"
            body = ["| " + " | ".join(cell(item) for item in row) + " |" for row in rows]
            return "\n".join([head, rule, *body])

        summary = _load("run_summary.json") or {}
        rounds = _load("rounds.json") or []
        baseline = _load("baseline.json") or {}
        comparisons = _load("comparisons.json") or {}
        overlays = _load("overlays.json") or []
        per_opt = _load("per_image_opt.json") or {}
        counts = (summary.get("counts") or {})
        blocks = []
        if not summary or not rounds:
            blocks.append(mo.md("## Vòng lặp edge case\n\nChưa có `results/edge_loop/`. Chạy `gdp-eval edge-loop` trước."))
        else:
            blur = (counts.get("blur_rain") or {})
            blocks.append(
                mo.md(
                    "\n".join(
                        [
                            "<a id=\"edge-loop\"></a>",
                            "",
                            "# Vòng lặp edge case",
                            "",
                            "Thí nghiệm này hỏi prompt sinh từ guideline, qua doc-rag và Qwen, có nâng IoU của SAM 3 trên `van_truck` và `far_small` hay không. Baseline là prompt `car`. Mỗi vòng sinh 4 candidate, chấm trên opt, và chọn candidate có IoU cao nhất **từng** edge case. Validation chỉ để báo cáo sau vòng 10. `other` chỉ để xem regression, không dùng để chọn prompt.",
                            "",
                            f"Retrieval: `{summary.get('retrieval_path')}`, doc id `{summary.get('doc_id')}`, embedding `{summary.get('embedding_model')}`. Guideline đưa vào index là `{summary.get('guideline')}`, cùng file mà live demo trước render thành PDF. LLM `{summary.get('llm_model')}`. SAM 3 `{summary.get('sam3_model')}`.",
                            "",
                            "SAM 3 dùng text encoder CLIP. `Sam3Processor` pad/truncate về **32** vị trí, gồm `<|startoftext|>` và `<|endoftext|>`. `max_position_embeddings` của tower này là 32 (không phải 77 của CLIP gốc). Prompt được phép tối đa **28** content token để còn chỗ cho hai special token và không bị cắt.",
                            "",
                            f"`blur_rain` bị bỏ vì cả bộ chỉ có **{blur.get('all', '—')}** ảnh (opt {blur.get('opt', '—')}, validation {blur.get('validation', '—')}, test {blur.get('test', '—')}). {summary.get('blur_rain_excluded') or ''}",
                            "",
                            "Số ảnh sau `gdp-data resplit` (không lấy từ kế hoạch cũ):",
                            "",
                            _md_table(
                                ["split", "n", "far_small", "van_truck", "blur_rain", "other"],
                                [
                                    [
                                        name,
                                        counts.get(f"n_{name}", "—") if name != "test" else counts.get("n_test"),
                                        (counts.get(name) or {}).get("far_small", "—"),
                                        (counts.get(name) or {}).get("van_truck", "—"),
                                        (counts.get(name) or {}).get("blur_rain", blur.get(name, "—") if name == "test" else "—"),
                                        (counts.get(name) or {}).get("other", "—"),
                                    ]
                                    for name in ("opt", "validation", "other", "test")
                                ],
                            ),
                        ]
                    )
                )
            )
            val = comparisons.get("validation") or {}
            base_opt = (baseline.get("opt") or {}).get("by_case") or {}
            base_val = (baseline.get("validation") or {}).get("by_case") or {}
            comp_rows = []
            for case in ("van_truck", "far_small"):
                block = val.get(case) or {}
                comp_rows.append(
                    [
                        case,
                        (base_opt.get(case) or {}).get("n"),
                        _fmt((base_opt.get(case) or {}).get("mean_iou")),
                        (base_val.get(case) or {}).get("n"),
                        _ci(block.get("baseline")),
                        block.get("prompt") or "—",
                        _ci(block.get("final")),
                        _ci(block.get("delta")),
                    ]
                )
            blocks.append(
                mo.md(
                    "\n".join(
                        [
                            "## Baseline `car` và prompt cuối trên validation",
                            "",
                            "Cột delta là hiệu IoU từng ảnh (prompt cuối trừ baseline), bootstrap 95% trên các ảnh đó. Seed và số lần rút nằm trong `comparisons.json`. Prompt cuối của mỗi case là prompt thắng ở vòng 10 trên opt, không phải prompt có IoU validation cao nhất.",
                            "",
                            _verdict(val),
                            "",
                            _opt_note(rounds, baseline),
                            "",
                            _fallback_note(rounds),
                            "",
                            _other_note(comparisons),
                            "",
                            _md_table(
                                [
                                    "edge case",
                                    "n opt",
                                    "baseline opt",
                                    "n validation",
                                    "baseline validation, CI 95%",
                                    "prompt cuối",
                                    "prompt cuối, CI 95%",
                                    "delta, CI 95%",
                                ],
                                comp_rows,
                            ),
                        ]
                    )
                )
            )
            curve_rows = []
            prompt_rows = []
            for rnd in rounds:
                winners = rnd.get("winners") or {}
                curve_rows.append(
                    [
                        rnd.get("round"),
                        _fmt((winners.get("van_truck") or {}).get("iou")),
                        _fmt((winners.get("far_small") or {}).get("iou")),
                    ]
                )
                for case in ("van_truck", "far_small"):
                    winner = winners.get(case) or {}
                    prompt_rows.append(
                        [
                            rnd.get("round"),
                            case,
                            winner.get("angle"),
                            winner.get("slot"),
                            winner.get("clip_tokens") if False else next(
                                (
                                    cand.get("clip_tokens")
                                    for cand in rnd.get("candidates") or []
                                    if cand.get("label") == winner.get("label")
                                ),
                                "—",
                            ),
                            f"`{winner.get('prompt')}`",
                        ]
                    )
            import matplotlib.pyplot as plt

            plt.rcParams.update({
                "font.family": "DejaVu Sans",
                "font.size": 14,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 13,
                "ytick.labelsize": 13,
                "legend.fontsize": 12,
                "figure.facecolor": "#fffdf8",
                "axes.facecolor": "#fffdf8",
                "savefig.facecolor": "#fffdf8",
                "text.color": "#102033",
                "axes.labelcolor": "#102033",
                "xtick.color": "#102033",
                "ytick.color": "#102033",
                "axes.edgecolor": "#16324f",
            })
            plt.close("all")
            figure, axis = plt.subplots(figsize=(11.2, 6.2))
            xs = [row[0] for row in curve_rows]
            axis.plot(xs, [None if row[1] == "—" else float(row[1]) for row in curve_rows], marker="o", color="#16324f", linewidth=2.2, label="van_truck winner")
            axis.plot(xs, [None if row[2] == "—" else float(row[2]) for row in curve_rows], marker="o", color="#c2410c", linewidth=2.2, label="far_small winner")
            van_base = (base_opt.get("van_truck") or {}).get("mean_iou")
            far_base = (base_opt.get("far_small") or {}).get("mean_iou")
            if van_base is not None:
                axis.axhline(float(van_base), color="#16324f", linestyle="--", linewidth=1.6, label="baseline van_truck opt")
            if far_base is not None:
                axis.axhline(float(far_base), color="#c2410c", linestyle="--", linewidth=1.6, label="baseline far_small opt")
            axis.set_xlabel("round")
            axis.set_ylabel("mean IoU on opt")
            axis.set_ylim(0, 1)
            axis.set_title("IoU opt của prompt thắng theo vòng")
            axis.legend()
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            blocks.append(mo.md("## Đường cong 10 vòng trên opt\n\nĐiểm trên đường là IoU của candidate được chọn cho case đó. Các candidate thua không bị xoá khỏi bảng phía dưới."))
            blocks.append(figure)
            blocks.append(mo.md(_md_table(["round", "van_truck", "far_small"], curve_rows)))
            cand_rows = []
            for rnd in rounds:
                for cand in rnd.get("candidates") or []:
                    by_case = cand.get("iou_by_case") or {}
                    cand_rows.append(
                        [
                            rnd.get("round"),
                            cand.get("slot"),
                            cand.get("angle"),
                            cand.get("clip_tokens"),
                            "fallback" if cand.get("fallback") else "",
                            _fmt(by_case.get("van_truck")),
                            _fmt(by_case.get("far_small")),
                            f"`{cand.get('prompt')}`",
                        ]
                    )
            blocks.append(mo.md("## Prompt từng vòng\n\n" + _md_table(
                ["round", "case", "angle", "slot", "CLIP tokens", "prompt thắng"],
                prompt_rows,
            )))
            blocks.append(mo.md("## Bốn candidate mỗi vòng\n\n" + _md_table(
                ["round", "slot", "angle", "CLIP tokens", "fallback", "van_truck", "far_small", "prompt"],
                cand_rows,
            )))
            chunk_rows = []
            for rnd in rounds:
                for chunk in rnd.get("chunks") or []:
                    text = str(chunk.get("text") or "").replace("\n", " ")
                    chunk_rows.append(
                        [
                            rnd.get("round"),
                            rnd.get("retrieval_path"),
                            chunk.get("id") or chunk.get("rank"),
                            chunk.get("source"),
                            _fmt(chunk.get("score"), 4),
                            text[:180] + ("…" if len(text) > 180 else ""),
                        ]
                    )
            blocks.append(
                mo.md(
                    "## Chunk doc-rag từng vòng\n\n"
                    + _md_table(
                        ["round", "path", "id", "source", "score", "text"],
                        chunk_rows,
                    )
                )
            )
            other = comparisons.get("other") or {}
            other_rows = []
            for label, block in other.items():
                other_rows.append(
                    [
                        label,
                        block.get("n"),
                        _ci(block.get("baseline")),
                        _ci(block.get("final")),
                        _ci(block.get("delta")),
                    ]
                )
            blocks.append(
                mo.md(
                    "## Regression trên `other`\n\n"
                    "Nhóm này không tham gia chọn prompt. Mỗi dòng là prompt thắng của một edge case, chấm trên ảnh `other`.\n\n"
                    + _md_table(
                        ["prompt", "n", "baseline CI", "prompt CI", "delta CI"],
                        other_rows,
                    )
                )
            )
            shots = []
            for item in overlays:
                path = edge_dir / str(item.get("file") or "")
                if path.is_file():
                    shots.append(
                        mo.md(
                            f"**{item.get('role')}** · `{item.get('case')}` · {item.get('split')} · IoU {_fmt(item.get('iou'))}"
                        )
                    )
                    shots.append(mo.image(src=path, alt=str(item.get("role")), width=480))
            if shots:
                blocks.append(mo.md("## Overlay\n\nẢnh gốc không nằm trong repo. Chỉ có overlay đã thu nhỏ (rộng tối đa 640 px)."))
                blocks.extend(shots)
            miss_rows = []
            last = rounds[-1] if rounds else {}
            for case in ("van_truck", "far_small"):
                label = ((last.get("winners") or {}).get(case) or {}).get("label")
                rows = [row for row in (per_opt.get(label) or []) if row.get("case") == case and row.get("iou") is not None]
                rows.sort(key=lambda row: float(row["iou"]))
                for row in rows[:3]:
                    miss_rows.append(
                        [
                            case,
                            _fmt(row.get("iou")),
                            row.get("error_kind") or "—",
                            str(row.get("image_id") or "")[:48],
                        ]
                    )
            if miss_rows:
                blocks.append(
                    mo.md(
                        "## Ảnh opt lỗi của prompt thắng vòng 10\n\n"
                        "Ba ảnh IoU thấp nhất mỗi edge case. `error_kind` lấy từ diện tích mask: "
                        "`miss` là mask rỗng, `fragment` là mask nhỏ hơn gold, `spill` là mask tràn ra ngoài gold, `shift` là mask lệch.\n\n"
                        + _md_table(["case", "IoU", "error_kind", "image_id"], miss_rows)
                    )
                )
            cost = summary.get("cost_estimate_before") or {}
            projected = summary.get("cost_projection_after_baseline") or {}
            tag_n = summary.get("tag_differences")
            blocks.append(
                mo.md(
                    "\n".join(
                        [
                            "## Hạn chế",
                            "",
                            f"- n nhỏ: opt `far_small` {(counts.get('opt') or {}).get('far_small')}, `van_truck` {(counts.get('opt') or {}).get('van_truck')}; validation cùng cỡ. Khoảng tin cậy rộng là điều kiện của mẫu này.",
                            "- Một lần chạy. Không lặp seed của SAM 3 hay của Qwen.",
                            f"- Tag edge case của Tiến và tag của tool khác nhau trên **{tag_n if tag_n is not None else '—'}** ảnh (`tag_differences` trong review report). IoU theo case dùng tag của tool (`decide_case`), không dùng group của Tiến.",
                            "- `blur_rain` không đủ ảnh để là một edge case.",
                            "- Chọn prompt chỉ trên opt. Validation không được dùng để đổi người thắng.",
                            (
                                "- Chi phí Modal là ước tính, không phải hoá đơn. "
                                f"Trước khi chạy (L4 ${cost.get('usd_per_gpu_second')}/GPU-giây, hai GPU, hệ số {cost.get('overhead')}): "
                                f"${_fmt(cost.get('estimated_usd'), 2)}. "
                                f"Sau baseline, phép chiếu còn lại là ${_fmt(projected.get('estimated_usd'), 2)}. "
                                f"Tường thời gian đo {_fmt(summary.get('elapsed_sec'), 1)} giây; nếu cả hai L4 đứng suốt khoảng đó thì khoảng "
                                f"${_fmt((summary.get('elapsed_sec') or 0) * 2 * 0.000222 * 1.15, 2)}. "
                                f"Cache miss SAM 3: {summary.get('sam3_cache_misses')}."
                            ),
                            "- Context CLIP 32 vị trí là giới hạn thật của SAM 3 trong code converter Hugging Face và `max_position_embeddings` của text tower. Prompt dài hơn bị cắt, nên 28 content token là trần của thí nghiệm.",
                        ]
                    )
                )
            )
        return mo.vstack(blocks)

    _edge_section()



if __name__ == "__main__":
    app.run()
