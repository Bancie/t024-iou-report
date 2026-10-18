# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.16",
#     "matplotlib>=3.8",
# ]
# ///

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    import json
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
    _REMOTE_FILES = ['results/check_gold.txt', 'results/examples.json', 'results/per_image.json', 'results/review_report.json', 'results/sam3_health.json', 'results/split_check.txt', 'results/splits_summary.json', 'results/summary.json']

    def _ensure_remote_files(report_dir):
        import os
        import tempfile
        import urllib.request

        if (report_dir / 'results/summary.json').is_file():
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

    def load_json(path):
        if not path.is_file():
            return None, "thiếu file"
        try:
            return json.loads(path.read_text(encoding="utf-8")), None
        except json.JSONDecodeError as exc:
            return None, "JSON không hợp lệ: " + exc.msg
        except OSError as exc:
            return None, "không đọc được: " + str(exc)

    def load_text(path):
        if not path.is_file():
            return None, "thiếu file"
        try:
            return path.read_text(encoding="utf-8"), None
        except OSError as exc:
            return None, "không đọc được: " + str(exc)

    def note(notes, path, error):
        if error:
            notes.append("`" + path.name + "` (" + error + ").")

    load_notes = []
    per_image, per_error = load_json(RESULTS / "per_image.json")
    note(load_notes, RESULTS / "per_image.json", per_error)
    summary, summary_error = load_json(RESULTS / "summary.json")
    note(load_notes, RESULTS / "summary.json", summary_error)
    review, review_error = load_json(RESULTS / "review_report.json")
    note(load_notes, RESULTS / "review_report.json", review_error)
    splits, splits_error = load_json(RESULTS / "splits_summary.json")
    note(load_notes, RESULTS / "splits_summary.json", splits_error)
    examples, examples_error = load_json(RESULTS / "examples.json")
    note(load_notes, RESULTS / "examples.json", examples_error)
    health, health_error = load_json(RESULTS / "sam3_health.json")
    note(load_notes, RESULTS / "sam3_health.json", health_error)
    check_gold_text, check_gold_error = load_text(RESULTS / "check_gold.txt")
    note(load_notes, RESULTS / "check_gold.txt", check_gold_error)
    split_check_text, split_check_error = load_text(RESULTS / "split_check.txt")
    note(load_notes, RESULTS / "split_check.txt", split_check_error)
    if not isinstance(per_image, list):
        per_image = []
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(review, dict):
        review = {}
    if not isinstance(splits, dict):
        splits = {}
    if not isinstance(examples, list):
        examples = []
    if not isinstance(health, dict):
        health = {}

    def fmt(value, digits=4):
        if value is None:
            return "—"
        return f"{float(value):.{digits}f}"

    def scored_rows(rows):
        return [row for row in rows if row.get("iou") is not None]

    def mean_of(rows):
        values = [float(row["iou"]) for row in scored_rows(rows)]
        if not values:
            return None
        return sum(values) / len(values)

    def md_table(headers, table_rows):
        if not table_rows:
            return "_Không có dòng nào._"
        head = "| " + " | ".join(headers) + " |"
        rule = "| " + " | ".join("---" for _ in headers) + " |"
        body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table_rows]
        return "\n".join([head, rule, *body])

    iou_rows = scored_rows(per_image)
    return (
        mo,
        RESULTS,
        load_notes,
        per_image,
        summary,
        review,
        splits,
        examples,
        health,
        check_gold_text,
        split_check_text,
        fmt,
        mean_of,
        md_table,
        iou_rows,
    )

@app.cell(hide_code=True)
def _(load_notes, mo):
    def _section_2():
        title = mo.md('\n# Báo cáo IoU baseline — bộ review nuImages\n\nTeam T024 · binary IoU của mask `car` · prompt baseline `passenger car`\n\nMột lần chạy SAM 3 trên các ảnh scored của split **opt**, **validation** và **test**. Notebook chỉ đọc `results/`. Không gọi Modal và không đọc ảnh gốc.\n\n**Điều hướng nhanh**\n\n- [Tổng quan](#overview)\n- [Dataset và review set](#dataset)\n- [Định nghĩa metric](#metric)\n- [Kết quả](#results)\n- [Ví dụ gold và prediction](#examples)\n- [Phân tích lỗi](#errors)\n- [Kết luận](#conclusion)\n- [Hạn chế và bước tiếp theo](#limitations)\n')
        blocks = [title]
        if load_notes:
            blocks.append(mo.callout(mo.md('\n'.join(('- ' + line for line in load_notes))), kind='warn', title='Một số file của báo cáo chưa được đọc'))
        return mo.vstack(blocks)
    _section_2()

@app.cell(hide_code=True)
def _(fmt, health, iou_rows, mean_of, mo, per_image, summary):
    def _section_3():
        service = health.get('service') or '—'
        cuda = health.get('cuda')
        if not health:
            cuda_text = 'thiếu file health'
        elif cuda:
            cuda_text = 'có CUDA'
        else:
            cuda_text = 'không thấy cuda'
        zero = sum((1 for row in iou_rows if float(row['iou']) == 0.0))
        failed = sum((1 for row in per_image if row.get('iou') is None))
        attempts = {}
        for row in per_image:
            attempts[row.get('attempts')] = attempts.get(row.get('attempts'), 0) + 1
        attempt_text = ', '.join((f'{count} ảnh × {n} attempt' for n, count in sorted(attempts.items(), key=lambda item: str(item[0])))) or '—'
        return mo.md(f'\n<a id="overview"></a>\n\n# Tổng quan\n\nBaseline gửi đúng một prompt, `passenger car`, tới SAM 3. Không chỉnh prompt. Model ghi trong summary là `{summary.get('model_id', '—')}`, service health là `{service}` ({cuda_text}).\n\n| | |\n| --- | --- |\n| Ảnh scored | {(len(iou_rows) if per_image else '—')} |\n| Ảnh failed (không có IoU) | {(failed if per_image else '—')} |\n| IoU = 0 | {(zero if per_image else '—')} |\n| Mean IoU (mọi ảnh scored) | {fmt(mean_of(iou_rows) if iou_rows else summary.get('mean_iou'))} |\n| Attempts | {attempt_text} |\n\nMean kéo xuống vì các ảnh IoU = 0. Median nằm ở phần kết quả. `gdp-eval score` chỉ chấm opt và không ghi từng ảnh; các số ở đây lấy từ `gdp-eval evaluate-set`.\n')
    _section_3()

@app.cell(hide_code=True)
def _(check_gold_text, md_table, mo, review, split_check_text, splits):
    def _section_4():
        images = review.get('images') or []
        by_case = review.get('by_case') or {}
        by_split = review.get('by_split') or {}
        reasons = {}
        for row in images:
            if row.get('case') == 'other':
                key = row.get('case_reason') or ''
                reasons[key] = reasons.get(key, 0) + 1
        reason_rows = [[name, count] for name, count in sorted(reasons.items())]
        case_rows = [[case, by_case.get(case, 0)] for case in ('far_small', 'blur_rain', 'van_truck', 'other')]
        split_rows = []
        for name in ('opt', 'validation', 'test', 'held_out', 'unscored'):
            counts = by_split.get(name) or {}
            split_rows.append([name, counts.get('far_small', 0), counts.get('blur_rain', 0), counts.get('van_truck', 0), counts.get('other', 0), sum((counts.get(case, 0) for case in ('far_small', 'blur_rain', 'van_truck', 'other')))])
        confusion = (review.get('confusion') or {}).get('counts') or {}
        columns = (review.get('confusion') or {}).get('columns') or []
        confusion_rows = [[group, *[counts.get(case, 0) for case in columns]] for group, counts in confusion.items()]
        warnings = review.get('warnings') or []
        warning_lines = '\n'.join((f'- {item.get('message')}' for item in warnings)) or '- không có'
        imbalance = splits.get('imbalance') or review.get('imbalance') or {}
        targets = imbalance.get('target_totals') or {}
        actual = imbalance.get('actual') or {}
        actual_rows = []
        for name in ('opt', 'validation', 'test'):
            got = actual.get(name) or {}
            actual_rows.append([name, targets.get(name, '—'), sum((got.get(case, 0) for case in ('far_small', 'blur_rain', 'van_truck'))) if got else '—', got.get('far_small', '—'), got.get('blur_rain', '—'), got.get('van_truck', '—')])
        missing = ''
        if not review or not splits:
            missing = '\n\n**Thiếu file.** Một phần dataset để trống cho đến khi đọc được `review_report.json` và `splits_summary.json`.\n'
        return mo.md(f'\n<a id="dataset"></a>\n\n# Dataset và review set\n{missing}\nBộ review có **{review.get('n_images', '—')}** ảnh trong `img/` và `gold/`. Case `other` là **{review.get('n_other', '—')}** ảnh, liệt kê ở key `unscored`, không vào opt / validation / test. Ảnh scored là **{review.get('n_scored', '—')}**.\n\n`other` không chỉ là ảnh không có `vehicle.car`. `decide_case` còn gán `other` khi có xe nhưng không rơi vào ba edge case (`reason = default`).\n\n{md_table(['case_reason của other', 'số ảnh'], reason_rows)}\n\n## Counts theo case\n\n{md_table(['case', 'số ảnh'], case_rows)}\n\n## Counts theo split\n\n`held_out` = validation + test. `unscored` không nằm trong `held_out`.\n\n{md_table(['split', 'far_small', 'blur_rain', 'van_truck', 'other', 'tổng'], split_rows)}\n\n## Leak check\n\nLog là logfile nuImages trong tên file (`nNNN-...±HHMM`). Một log không nằm ở hai split scored. Output của `gdp-data split --check`:\n\n```\n{(split_check_text or 'thiếu file').strip()}\n```\n\n`gdp-eval check-gold --require-complete` trên cả cây ảnh:\n\n```\n{(check_gold_text or 'thiếu file').strip()}\n```\n\n## Imbalance\n\nTarget là largest-remainder của opt 30 và validation 30 trên ảnh scored. `imbalance` ghi actual − target. `max_abs_delta` = **{imbalance.get('max_abs_delta', '—')}**. Đây là báo cáo, không phải lỗi.\n\n{md_table(['split', 'target tổng', 'actual tổng', 'far_small', 'blur_rain', 'van_truck'], actual_rows)}\n\n`blur_rain` chỉ có 1 ảnh scored. Target đặt ảnh đó vào test và 0 ảnh vào opt / validation. Không copy ảnh sang split khác.\n\n## Cảnh báo `decide_case`\n\n{warning_lines}\n\n## `decide_case` và group của Tiến\n\nGroup `far_close` / `rain_blur` / `van_truck` trong manifest của Tiến không phải case. Bảng dưới là số ảnh. Tag `rain_blur` và `van_truck` là heuristic, chưa được xác nhận. Số ảnh lệch tag: **{len(review.get('tag_differences') or [])}**.\n\n{md_table(['tien_group', *columns], confusion_rows)}\n')
    _section_4()

@app.cell(hide_code=True)
def _(mo, summary):
    def _section_5():
        return mo.md(f'\n<a id="metric"></a>\n\n# Định nghĩa metric\n\nMỗi ảnh scored có một mask gold: hợp (OR) mọi instance `vehicle.car`, PNG mode `L`, pixel 0 hoặc 255. Prediction là hợp các mask SAM 3 trả về cho prompt `{summary.get('prompt', 'passenger car')}`.\n\nBinary IoU của một ảnh là `|pred ∩ gold| / |pred ∪ gold|`. Hai mask rỗng thì IoU = 1. Mean là trung bình cộng các IoU đó, mỗi ảnh một phiếu. Đây không phải mIoU trên 19 class của CVAT, và không có IoU riêng cho truck, bus hay van.\n\nẢnh `other` / `unscored` không vào mean. Ảnh gọi SAM 3 thất bại (timeout sau retry) có `iou: null`, nằm trong `failures`, và cũng không vào mean. IoU = 0 chỉ khi request thành công và giao của hai mask rỗng.\n\nNgưỡng low-IoU trong summary là **{summary.get('low_iou_threshold', 0.5)}**. Segmenter chấm held-out không nhận `held_out_ids`, nên validation và test không bị `HeldOutLeak`.\n')
    _section_5()

@app.cell(hide_code=True)
def _(fmt, iou_rows, md_table, mean_of, mo, summary):
    def _section_6():
        split_rows = []
        for name in ('opt', 'validation', 'test'):
            group = [row for row in iou_rows if row['split'] == name]
            info = (summary.get('by_split') or {}).get(name) or {}
            split_rows.append([name, len(group), fmt(mean_of(group)), sum((1 for row in group if float(row['iou']) == 0.0)), info.get('n_failed', 0)])
        case_rows = []
        for case in ('far_small', 'blur_rain', 'van_truck'):
            group = [row for row in iou_rows if row['case'] == case]
            case_rows.append([case, len(group), fmt(mean_of(group)), sum((1 for row in group if float(row['iou']) == 0.0))])
        cross = []
        for name in ('opt', 'validation', 'test'):
            for case in ('far_small', 'blur_rain', 'van_truck'):
                group = [row for row in iou_rows if row['split'] == name and row['case'] == case]
                cross.append([name, case, len(group), fmt(mean_of(group))])
        if iou_rows:
            ordered = sorted(iou_rows, key=lambda row: (float(row['iou']), row['image_id']))
            median = ordered[len(ordered) // 2]
            median_text = f'**{fmt(median['iou'])}** — `{median['case']}` / `{median['split']}`'
        else:
            median_text = '—'
        bins = []
        for start in range(10):
            lo = start / 10
            hi = (start + 1) / 10
            if start == 9:
                count = sum((1 for row in iou_rows if lo <= float(row['iou']) <= hi))
                label = f'[{lo:.1f}, {hi:.1f}]'
            else:
                count = sum((1 for row in iou_rows if lo <= float(row['iou']) < hi))
                label = f'[{lo:.1f}, {hi:.1f})'
            bins.append([label, count])
        missing = ''
        if not iou_rows:
            missing = '\n\n**Thiếu file.** Chưa có `per_image.json`.\n'
        return mo.md(f'\n<a id="results"></a>\n\n# Kết quả\n{missing}\nMean mọi ảnh scored: **{fmt(mean_of(iou_rows))}**. Median (ảnh giữa sau khi sort theo IoU, hòa thì theo `image_id`): {median_text}.\n\n## Theo split\n\n{md_table(['split', 'n', 'mean IoU', 'IoU = 0', 'failed'], split_rows)}\n\n## Theo case\n\n{md_table(['case', 'n', 'mean IoU', 'IoU = 0'], case_rows)}\n\n## Split × case\n\nÔ `blur_rain` của opt và validation trống vì ảnh duy nhất nằm ở test.\n\n{md_table(['split', 'case', 'n', 'mean IoU'], cross)}\n\n## Histogram (đếm từ per_image)\n\n{md_table(['bin IoU', 'số ảnh'], bins)}\n')
    _section_6()

@app.cell(hide_code=True)
def _(iou_rows):
    def _section_7():
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 15,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'figure.facecolor': '#fffdf8',
            'axes.facecolor': '#fffdf8',
            'savefig.facecolor': '#fffdf8',
            'text.color': '#102033',
            'axes.labelcolor': '#102033',
            'xtick.color': '#102033',
            'ytick.color': '#102033',
            'axes.edgecolor': '#16324f',
        })
        plt.close('all')
        figure, axis = plt.subplots(figsize=(10.5, 5.6))
        if iou_rows:
            axis.hist([float(row['iou']) for row in iou_rows], bins=[index / 10 for index in range(11)], color='#16324f', edgecolor='white')
        axis.set_xlabel('IoU')
        axis.set_ylabel('số ảnh')
        axis.set_title('Histogram binary IoU')
        figure.tight_layout()
        return axis
    _section_7()

@app.cell(hide_code=True)
def _(iou_rows):
    def _section_8():
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 15,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'figure.facecolor': '#fffdf8',
            'axes.facecolor': '#fffdf8',
            'savefig.facecolor': '#fffdf8',
            'text.color': '#102033',
            'axes.labelcolor': '#102033',
            'xtick.color': '#102033',
            'ytick.color': '#102033',
            'axes.edgecolor': '#16324f',
        })
        plt.close('all')
        figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.8))
        panels = ((axes[0], 'split', ('opt', 'validation', 'test')), (axes[1], 'case', ('far_small', 'blur_rain', 'van_truck')))
        for axis, key, labels in panels:
            shown = []
            series = []
            for label in labels:
                values = [float(row['iou']) for row in iou_rows if row[key] == label]
                if values:
                    shown.append(label)
                    series.append(values)
            if series:
                drawn = axis.boxplot(series, tick_labels=shown, showfliers=True)
                for box in drawn['boxes']:
                    box.set(color='#16324f', linewidth=1.8)
                for whisker in drawn['whiskers']:
                    whisker.set(color='#16324f', linewidth=1.4)
                for cap in drawn['caps']:
                    cap.set(color='#16324f', linewidth=1.4)
                for median in drawn['medians']:
                    median.set(color='#c2410c', linewidth=2.2)
            axis.set_ylim(-0.05, 1.05)
            axis.set_ylabel('IoU')
            axis.set_title('Boxplot theo ' + key)
        figure.tight_layout()
        return axes[0]
    _section_8()

@app.cell(hide_code=True)
def _(RESULTS, examples, fmt, mo):
    def _section_9():
        blocks = [mo.md('\n<a id="examples"></a>\n\n# Ví dụ gold và prediction\n\nViền xanh lá là gold. Mảng xanh dương là prediction. Khung đỏ là ảnh IoU < 0.5. Ảnh đã thu xuống tối đa 640 px. Cùng một `image_id` chỉ xuất hiện một lần: best của `far_small` trùng best toàn bộ nên không vẽ lại.\n')]
        if not examples:
            blocks.append(mo.md('**Thiếu file.** Chưa có `examples.json`.'))
        for item in examples:
            path = RESULTS / item['file']
            caption = f'**{item['role']}** · `{item['case']}` · `{item['split']}` · IoU {fmt(item.get('iou'))} · pred {item.get('pred_area')} px · gold {item.get('gold_area')} px'
            blocks.append(mo.md(caption))
            if path.is_file():
                blocks.append(mo.image(src=path, alt=item['role'], width=480))
            else:
                blocks.append(mo.md('_Không có file overlay trong repo public này (ảnh nuImages không được đưa lên)._'))
        return mo.vstack(blocks)
    _section_9()

@app.cell(hide_code=True)
def _(fmt, iou_rows, md_table, mo):
    def _section_10():
        zeros = [row for row in iou_rows if float(row['iou']) == 0.0]
        empty = [row for row in zeros if row.get('pred_area') == 0]
        partial = [row for row in iou_rows if float(row['iou']) > 0 and row.get('gold_area')]
        over = [row for row in partial if row['pred_area'] > 1.5 * row['gold_area']]
        under = [row for row in partial if 0 < row['pred_area'] < 0.5 * row['gold_area']]
        zero_rows = []
        for case in ('far_small', 'blur_rain', 'van_truck'):
            group = [row for row in iou_rows if row['case'] == case]
            missed = [row for row in group if float(row['iou']) == 0.0]
            rate = len(missed) / len(group) if group else None
            zero_rows.append([case, len(group), len(missed), fmt(rate)])
        blur = [row for row in iou_rows if row['case'] == 'blur_rain']
        if blur:
            row = blur[0]
            blur_line = f'Ảnh `blur_rain` duy nhất nằm ở split `{row['split']}`, IoU **{fmt(row['iou'])}**, pred {row['pred_area']} px, gold {row['gold_area']} px, `attempts` = {row.get('attempts')}. Một ảnh không đủ để kết luận case mưa/mờ.'
        else:
            blur_line = 'Không có ảnh `blur_rain` trong per_image.'
        largest = sorted(empty, key=lambda row: (-row.get('gold_area', 0), row['image_id']))[:5]
        largest_rows = [[row['case'], row['split'], row['gold_area'], row['image_id'][:24] + '…'] for row in largest]
        missing = ''
        if not iou_rows:
            missing = '\n\n**Thiếu file.** Chưa có per_image để phân tích.\n'
        return mo.md(f'\n<a id="errors"></a>\n\n# Phân tích lỗi\n{missing}\n## IoU = 0\n\n**{len(zeros)}** ảnh có IoU = 0. Trong đó **{len(empty)}** ảnh có `pred_area` = 0: SAM 3 không trả mask `car` nào. Không có ảnh IoU = 0 mà prediction vẫn có pixel.\n\n{md_table(['case', 'n', 'IoU = 0', 'tỉ lệ'], zero_rows)}\n\nMiss có gold lớn nhất (vẫn `pred_area` = 0):\n\n{md_table(['case', 'split', 'gold px', 'image_id'], largest_rows)}\n\n`far_small` chiếm nhiều miss hơn `van_truck`. Xe nhỏ hoặc xa dễ bị prompt `passenger car` bỏ sót. Một số miss `van_truck` có gold rất lớn: model không bật mask dù xe chiếm nhiều pixel.\n\n## Over-segmentation và under-segmentation\n\nQuy ước trên các ảnh IoU > 0: over nếu `pred_area > 1.5 × gold_area`, under nếu `0 < pred_area < 0.5 × gold_area`. Đây là mô tả diện tích, không phải một metric thứ hai.\n\n- Over: **{len(over)}** ảnh\n- Under: **{len(under)}** ảnh\n\nOverlay `over-segmentation` và `under-segmentation` là ảnh IoU thấp nhất trong mỗi nhóm.\n\n## Van-truck và rainy-blur\n\n`van_truck` là case `decide_case` khi có category truck / bus / trailer / construction, hoặc attribute tên có `van`. IoU vẫn là IoU của mask `vehicle.car`, không phải IoU của truck. Mean của case này cao hơn `far_small`, nhưng vẫn có miss hoàn toàn.\n\n{blur_line}\n\n`decide_case` gán `blur_rain` vì giờ chụp ban đêm (`night_hour`), không phải vì một nhãn mưa đã được xác nhận. Group `rain_blur` của Tiến không trùng case này: bảng confusion ở trên cho thấy phần lớn group đó rơi vào `far_small`, `van_truck` hoặc `other`.\n')
    _section_10()

@app.cell(hide_code=True)
def _(fmt, iou_rows, mean_of, mo, summary):
    def _section_11():
        by_split = {}
        by_case = {}
        for name in ('opt', 'validation', 'test'):
            by_split[name] = mean_of([row for row in iou_rows if row['split'] == name])
        for case in ('far_small', 'blur_rain', 'van_truck'):
            by_case[case] = mean_of([row for row in iou_rows if row['case'] == case])
        return mo.md(f'\n<a id="conclusion"></a>\n\n# Kết luận\n\nPrompt baseline `passenger car` trên `{summary.get('model_id', 'facebook/sam3')}` cho mean IoU **{fmt(mean_of(iou_rows))}** trên {(len(iou_rows) if iou_rows else '—')} ảnh scored. Median cao hơn mean vì một nhóm ảnh bị miss hoàn toàn (IoU = 0, prediction rỗng).\n\nTheo split: opt **{fmt(by_split.get('opt'))}**, validation **{fmt(by_split.get('validation'))}**, test **{fmt(by_split.get('test'))}**. Validation cao hơn opt và test trên lần chạy này. Ba split không được dùng để chọn prompt: đây là một baseline, chưa có vòng `selectBest`.\n\nTheo case: `far_small` **{fmt(by_case.get('far_small'))}**, `van_truck` **{fmt(by_case.get('van_truck'))}**, `blur_rain` **{fmt(by_case.get('blur_rain'))}** (n = 1). Xe xa/nhỏ là nhóm yếu hơn. `van_truck` không có nghĩa model đang phân biệt van với truck; metric chỉ nhìn mask `car`.\n\nKhông có request thất bại trong `summary.failures` (số phần tử: {len(summary.get('failures') or [])}). Mọi ảnh scored trả lời ở attempt 1. Retry đã được bật nhưng không phải dùng.\n\nGroup của Tiến và `decide_case` lệch nhau trên nhiều ảnh. Case trong báo cáo là `decide_case`, không phải tên folder review gốc.\n')
    _section_11()

@app.cell(hide_code=True)
def _(mo):
    def _section_12():
        return mo.md('\n<a id="limitations"></a>\n\n# Hạn chế và bước tiếp theo\n\n- `blur_rain` chỉ có 1 ảnh scored dưới `decide_case`. IoU của ảnh đó không đại diện cho mưa/mờ. Cảnh báo "ít hơn 5 ảnh" được giữ. Không nhân bản ảnh và không mượn ảnh ngoài bộ này.\n- Case tag là heuristic. `blur_rain` trên bộ này đến từ giờ đêm. `van_truck` đến từ category truck/bus/trailer/construction. Group `rain_blur` và `van_truck` của Tiến chưa được xác nhận và không trùng `decide_case`.\n- Cỡ opt / validation (30 / 30) vẫn chờ mentor. `imbalance.max_abs_delta` của seed 24 trên bộ này là 0 so với target largest-remainder, nhưng target đó chưa được chốt.\n- Chỉ một prompt baseline, `passenger car`. Chưa có bản viết tay và chưa có bản tool sau vòng lặp.\n- nuImages có license riêng. Ảnh gốc và mask đầy đủ không commit. Repo public này không chứa ảnh overlay nào (kể cả bản thu nhỏ), nên phần ví dụ gold và prediction chỉ hiện tên và IoU. Bản đầy đủ có overlay nằm trong repo nội bộ của team.\n- Split theo logfile nên ảnh cùng scene không nằm ở hai split scored. Ảnh `other` của cùng logfile nằm ở `unscored`, không bị chấm.\n- Bước tiếp: chốt cỡ split với mentor, hỏi có được bù `blur_rain` từ nguồn khác hay không, rồi mới so baseline với prompt viết tay và prompt của tool trên cùng các ảnh này.\n')
    _section_12()

if __name__ == "__main__":
    app.run()
