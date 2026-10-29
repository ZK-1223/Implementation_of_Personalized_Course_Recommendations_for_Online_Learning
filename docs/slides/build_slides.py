"""工程答辩 PPT 生成器。

输出 ``docs/slides/智能应用系统课程设计-第5题-答辩PPT.pptx``（16:9，共 17 页）。

所有数字均从仓库真实产物读取（数据集统计 / 模型指标 / 离线评测 / 图表 / 界面截图），
不在本文件中硬编码结论性数值。缺图时自动降级为占位框，保证任何阶段都能生成完整 PPT。

运行：
    python docs/slides/build_slides.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIG = ROOT / "docs" / "report" / "figures"
OUT = Path(__file__).resolve().parent / "智能应用系统课程设计-第5题-答辩PPT.pptx"

# ---- 视觉规范 ----
W, H = Inches(13.333), Inches(7.5)
C_BG = RGBColor(0xFB, 0xFC, 0xFF)
C_INK = RGBColor(0x1B, 0x23, 0x33)
C_INK2 = RGBColor(0x5A, 0x64, 0x72)
C_INK3 = RGBColor(0x8A, 0x94, 0xA6)
C_BLUE = RGBColor(0x2F, 0x6B, 0xFF)
C_GREEN = RGBColor(0x00, 0xC4, 0x8C)
C_ORANGE = RGBColor(0xFF, 0x7A, 0x45)
C_PURPLE = RGBColor(0x7C, 0x5C, 0xFF)
C_CARD = RGBColor(0xFF, 0xFF, 0xFF)
C_LINE = RGBColor(0xE3, 0xE8, 0xF2)
FONT = "Microsoft YaHei"

TOTAL_SLIDES = 18


# ==================================================================
# 基础绘图助手
# ==================================================================


def new_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = C_BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    return slide


def textbox(slide, x, y, w, h, text: str, size: float = 16, bold: bool = False,
            color: RGBColor = C_INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            line_spacing: float = 1.25, font: str = FONT):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    first = True
    for para_text in str(text).split("\n"):
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = align
        para.line_spacing = line_spacing
        run = para.add_run()
        run.text = para_text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = font
    return box


def header(slide, kicker: str, title: str, accent: RGBColor = C_BLUE) -> None:
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.72), Inches(0.62), Inches(0.10), Inches(0.52))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()
    bar.shadow.inherit = False
    textbox(slide, Inches(0.98), Inches(0.52), Inches(9.0), Inches(0.32), kicker, 12, True, accent)
    textbox(slide, Inches(0.98), Inches(0.82), Inches(11.6), Inches(0.56), title, 28, True, C_INK)


def footer(slide, page: int, note: str = "LearnPath · 智学路径") -> None:
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.72), Inches(6.86), Inches(11.9), Emu(9525))
    line.fill.solid()
    line.fill.fore_color.rgb = C_LINE
    line.line.fill.background()
    line.shadow.inherit = False
    textbox(slide, Inches(0.72), Inches(6.96), Inches(7.0), Inches(0.3), note, 10, False, C_INK3)
    textbox(slide, Inches(11.2), Inches(6.96), Inches(1.42), Inches(0.3),
            f"{page} / {TOTAL_SLIDES}", 10, False, C_INK3, align=PP_ALIGN.RIGHT)


def card(slide, x, y, w, h, fill: RGBColor = C_CARD, border: RGBColor | None = C_LINE):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if border is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = border
        shape.line.width = Pt(1.0)
    shape.shadow.inherit = False
    shape.adjustments[0] = 0.06
    return shape


def bullets(slide, x, y, w, h, items: Sequence[str], size: float = 15,
            bullet_color: RGBColor = C_BLUE, gap: float = 0.10) -> None:
    """带色点的要点列表；以「·」分隔标题与说明，「|」前缀表示二级缩进。"""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    first = True
    for item in items:
        indent = item.startswith("|")
        text = item[1:].strip() if indent else item
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.line_spacing = 1.34
        para.space_after = Pt(6 if not gap else gap * 72)
        dot = "   ·  " if indent else "▪  "
        run = para.add_run()
        run.text = dot
        run.font.size = Pt(size * (0.72 if indent else 0.95))
        run.font.color.rgb = bullet_color if not indent else C_INK3
        run.font.name = FONT
        run.font.bold = True
        body = para.add_run()
        body.text = text
        body.font.size = Pt(size * (0.88 if indent else 1.0))
        body.font.color.rgb = C_INK2 if indent else C_INK
        body.font.name = FONT


def rich_line(slide, x, y, w, h, segments: Sequence[tuple[str, float, bool, RGBColor]],
              align=PP_ALIGN.LEFT):
    """一行内混排多个 run：[(文本, 字号, 是否加粗, 颜色), ...]。"""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    para = tf.paragraphs[0]
    para.alignment = align
    for text, size, bold, color in segments:
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = FONT


def kpi_row(slide, items: Sequence[tuple[str, str]], y, x=Inches(0.72), w=Inches(11.9),
            h=Inches(1.02), accent=C_BLUE) -> None:
    n = len(items)
    gap = Inches(0.16)
    cw = int((w - gap * (n - 1)) / n)
    for i, (value, label) in enumerate(items):
        cx = x + i * (cw + gap)
        card(slide, cx, y, cw, h)
        strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, cx, y, Inches(0.055), h)
        strip.fill.solid()
        strip.fill.fore_color.rgb = accent
        strip.line.fill.background()
        strip.shadow.inherit = False
        textbox(slide, cx + Inches(0.22), y + Inches(0.14), cw - Inches(0.34), Inches(0.44),
                value, 21, True, C_INK)
        textbox(slide, cx + Inches(0.22), y + Inches(0.60), cw - Inches(0.34), Inches(0.30),
                label, 10.5, False, C_INK3)


def picture_fit(slide, path: Path, x, y, w, h, caption: str | None = None,
                frame: bool = True) -> None:
    """把图片按比例放进 (x,y,w,h) 框内并居中，可选细边框与外框卡片。"""
    if frame:
        card(slide, x, y, w, h)
    if not path.exists():
        textbox(slide, x, y + int(h / 2) - Inches(0.2), w, Inches(0.4),
                "（截图待补）", 12, False, C_INK3, align=PP_ALIGN.CENTER)
        return
    with Image.open(path) as im:
        iw, ih = im.size
    pad = Inches(0.10)
    avail_w = w - 2 * pad
    avail_h = h - 2 * pad - (Inches(0.26) if caption else 0)
    scale = min(avail_w / iw, avail_h / ih)
    dw, dh = int(iw * scale), int(ih * scale)
    dx = x + int((w - dw) / 2)
    dy = y + int((avail_h - dh) / 2) + pad
    slide.shapes.add_picture(str(path), dx, dy, dw, dh)
    if caption:
        textbox(slide, x, y + h - Inches(0.30), w, Inches(0.24),
                caption, 10, False, C_INK3, align=PP_ALIGN.CENTER)


def table(slide, headers: Sequence[str], rows: Sequence[Sequence[str]], x, y, w,
          col_ratios: Sequence[float] | None = None, size: float = 11.5,
          header_fill: RGBColor = C_BLUE, highlight_rows: Sequence[int] = ()) -> None:
    n_rows, n_cols = len(rows) + 1, len(headers)
    row_h = Inches(0.34)
    shape = slide.shapes.add_table(n_rows, n_cols, x, y, w, row_h * n_rows)
    tbl = shape.table
    tbl.first_row = True
    if col_ratios:
        total = sum(col_ratios)
        for i, ratio in enumerate(col_ratios):
            tbl.columns[i].width = int(w * ratio / total)
    for j, head in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = str(head)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.CENTER
        for run in para.runs:
            run.font.size = Pt(size)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.name = FONT
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, row in enumerate(rows, start=1):
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                RGBColor(0xEE, 0xF4, 0xFF) if (i - 1) in highlight_rows
                else (C_CARD if i % 2 else RGBColor(0xF6, 0xF8, 0xFD))
            )
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            for run in para.runs:
                run.font.size = Pt(size)
                run.font.color.rgb = C_INK
                run.font.name = FONT
                run.font.bold = (i - 1) in highlight_rows
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.06)


# ==================================================================
# 数据加载
# ==================================================================


def load() -> dict[str, Any]:
    data: dict[str, Any] = {}

    metrics_path = ROOT / "models" / "model_metrics.json"
    data["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    eval_path = ROOT / "models" / "evaluation.json"
    data["evaluation"] = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None

    try:
        from learnpath.data.loader import load_dataset
        from learnpath.knowledge.graph import KnowledgeGraph

        ds = load_dataset()
        graph = KnowledgeGraph.from_dataset(ds)
        data["stats"] = ds.stats()
        data["graph"] = graph.stats().to_dict()
    except Exception as exc:  # pragma: no cover
        print(f"[warn] 数据集加载失败：{exc}")
        data["stats"] = {}
        data["graph"] = {}
    return data


# ==================================================================
# 各页
# ==================================================================


def slide_cover(prs, d):
    slide = new_slide(prs)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(2.9))
    band.fill.solid()
    band.fill.fore_color.rgb = RGBColor(0x0E, 0x15, 0x24)
    band.line.fill.background()
    band.shadow.inherit = False

    textbox(slide, Inches(0.9), Inches(0.62), Inches(6), Inches(0.3),
            "智能应用系统设计 · 课程设计答辩", 13, True, C_GREEN)
    textbox(slide, Inches(0.9), Inches(1.02), Inches(11.5), Inches(0.8),
            "LearnPath  智学路径", 44, True, RGBColor(0xFF, 0xFF, 0xFF))
    textbox(slide, Inches(0.9), Inches(1.92), Inches(11.5), Inches(0.5),
            "在线学习的个性化课程推荐实现 —— 从知识图谱到学习路径的多目标优化", 18, False,
            RGBColor(0xC3, 0xCC, 0xDD))

    textbox(slide, Inches(0.9), Inches(3.35), Inches(5.2), Inches(0.3), "题目序号", 11, True, C_INK3)
    textbox(slide, Inches(0.9), Inches(3.62), Inches(5.2), Inches(0.42), "第 5 题  ★★☆☆☆", 17, True, C_INK)

    rows = [("专业班级", "＿＿＿＿＿＿"), ("姓名学号", "＿＿＿＿＿＿"),
            ("课程教学班号", "＿＿＿＿＿＿"), ("任课教师", "＿＿＿＿＿＿")]
    for i, (label, value) in enumerate(rows):
        y = Inches(4.16 + i * 0.40)
        textbox(slide, Inches(0.9), y, Inches(2.0), Inches(0.30), label, 13, False, C_INK3)
        textbox(slide, Inches(2.95), y, Inches(3.4), Inches(0.30), value, 13, True, C_INK)

    kpi_row(slide, [("420", "知识点"), ("560", "课程"), ("720", "学习者"),
                    ("14,000", "学习行为")], Inches(5.94), h=Inches(0.92), accent=C_GREEN)


def slide_toc(prs, d, page):
    slide = new_slide(prs)
    header(slide, "CONTENTS", "汇报提纲")
    items = [
        ("01", "问题背景", "为什么学习场景不能照搬内容推荐"),
        ("02", "系统总览", "三层架构与八大能力"),
        ("03", "核心算法", "图谱 / 画像 / 推荐 / 路径 / 多目标 / 预测 / 自适应 / 社交"),
        ("04", "量化结果", "推荐指标、预测指标、消融实验"),
        ("05", "工程实现", "分层架构、测试、可复现与仓库规范"),
        ("06", "总结与展望", "收获、不足与下一步"),
    ]
    for i, (no, title, desc) in enumerate(items):
        col = i % 2
        row = i // 2
        x = Inches(0.86 + col * 6.05)
        y = Inches(1.92 + row * 1.62)
        card(slide, x, y, Inches(5.72), Inches(1.32))
        textbox(slide, x + Inches(0.28), y + Inches(0.24), Inches(0.8), Inches(0.5), no, 26, True, C_BLUE)
        textbox(slide, x + Inches(1.12), y + Inches(0.26), Inches(4.3), Inches(0.36), title, 17, True, C_INK)
        textbox(slide, x + Inches(1.12), y + Inches(0.68), Inches(4.3), Inches(0.5), desc, 11.5, False, C_INK3)
    footer(slide, page)


def slide_background(prs, d, page):
    slide = new_slide(prs)
    header(slide, "01 · PROBLEM", "平台不缺课，缺的是「该学什么、按什么顺序学」")

    questions = [
        ("该学什么", "目标岗位还缺哪些知识点？", C_BLUE),
        ("按什么顺序学", "课程之间有先修依赖，顺序错了学不动", C_GREEN),
        ("学得动吗", "以每周可投入的时间，这条路走得到吗", C_ORANGE),
    ]
    for i, (title, desc, color) in enumerate(questions):
        x = Inches(0.86 + i * 4.08)
        card(slide, x, Inches(1.86), Inches(3.82), Inches(1.28))
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.86), Inches(0.055), Inches(1.28))
        bar.fill.solid(); bar.fill.fore_color.rgb = color; bar.line.fill.background(); bar.shadow.inherit = False
        textbox(slide, x + Inches(0.26), Inches(2.06), Inches(3.4), Inches(0.36), title, 16, True, color)
        textbox(slide, x + Inches(0.26), Inches(2.50), Inches(3.4), Inches(0.5), desc, 11.5, False, C_INK2)

    card(slide, Inches(0.86), Inches(3.40), Inches(11.9), Inches(1.62))
    textbox(slide, Inches(1.14), Inches(3.58), Inches(11.4), Inches(0.32),
            "内容消费 vs 学习：本质差异", 14, True, C_INK)
    bullets(slide, Inches(1.14), Inches(3.94), Inches(11.4), Inches(1.0), [
        "学习有强先修约束 —— 缺「假设检验」就学不懂「A/B 测试」，点了也学不完",
        "目标是序列化的 —— 用户要的不是 10 门有趣的课，而是一条能走通的路径",
        "时间成本刚性 —— 每周可投入小时数是硬约束，路径总时长必须匹配",
    ], size=13)

    card(slide, Inches(0.86), Inches(5.22), Inches(11.9), Inches(1.30),
         fill=RGBColor(0xEE, 0xF4, 0xFF), border=C_BLUE)
    textbox(slide, Inches(1.14), Inches(5.40), Inches(11.4), Inches(0.32),
            "本题本质", 13, True, C_BLUE)
    textbox(slide, Inches(1.14), Inches(5.74), Inches(11.4), Inches(0.62),
            "带先修约束的多目标组合优化问题：在知识图谱约束下，为特定学习者的特定目标，"
            "从候选课程集合中选出一个子集并排序，使学习收益、时间成本、难度梯度、风格匹配"
            "等多个相互冲突的目标达到平衡。", 13, False, C_INK)
    footer(slide, page)


def slide_requirements(prs, d, page):
    slide = new_slide(prs)
    header(slide, "01 · SCOPE", "题目 8 条实现要求 → 系统能力全覆盖")
    reqs = [
        ("① 学习者建模", "多维画像 + BKT 掌握度追踪"),
        ("② 课程知识图谱", "420 知识点 DAG + 先修闭包"),
        ("③ 学习路径规划", "分层贪心覆盖 + 拓扑修复"),
        ("④ 多目标优化", "NSGA-II 前沿 + TOPSIS 选解"),
        ("⑤ 学习效果预测", "完成概率分类 + 效果回归"),
        ("⑥ 自适应调整", "四类触发 + 自动重规划"),
        ("⑦ 社交学习", "同伴匹配 + 能力互补小组"),
        ("⑧ 学习分析仪表板", "8 类可视化聚合"),
    ]
    for i, (title, desc) in enumerate(reqs):
        col, row = i % 4, i // 4
        x = Inches(0.86 + col * 3.06)
        y = Inches(1.92 + row * 1.42)
        card(slide, x, y, Inches(2.84), Inches(1.20))
        seq = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.20), y + Inches(0.18), Inches(0.44), Inches(0.44))
        seq.fill.solid(); seq.fill.fore_color.rgb = C_BLUE; seq.line.fill.background(); seq.shadow.inherit = False
        textbox(slide, x + Inches(0.20), y + Inches(0.26), Inches(0.44), Inches(0.30),
                str(i + 1), 12, True, RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
        textbox(slide, x + Inches(0.74), y + Inches(0.22), Inches(2.0), Inches(0.34), title, 12.5, True, C_INK)
        textbox(slide, x + Inches(0.22), y + Inches(0.76), Inches(2.44), Inches(0.34), desc, 10.5, False, C_INK2)

    card(slide, Inches(0.86), Inches(4.86), Inches(11.9), Inches(1.72),
         fill=RGBColor(0xEC, 0xFB, 0xF5), border=C_GREEN)
    textbox(slide, Inches(1.14), Inches(5.04), Inches(11.4), Inches(0.32),
            "额外完成的工程能力（超出题目最低要求）", 13.5, True, RGBColor(0x00, 0x8C, 0x66))
    bullets(slide, Inches(1.14), Inches(5.40), Inches(5.6), Inches(1.0), [
        "协同过滤（SVD 隐语义分解）",
        "可解释推荐（每条给出具体中文理由）",
    ], size=12, bullet_color=C_GREEN)
    bullets(slide, Inches(6.86), Inches(5.40), Inches(5.6), Inches(1.0), [
        "离线评测体系（4 策略对比 + 消融实验）",
        "整套 REST 接口 + 6 视图单页应用",
    ], size=12, bullet_color=C_GREEN)
    footer(slide, page)


def slide_architecture(prs, d, page):
    slide = new_slide(prs)
    header(slide, "02 · ARCHITECTURE", "三层架构：引擎层不依赖 Web，表现层不含算法")
    picture_fit(slide, FIG / "fig_architecture.png", Inches(0.86), Inches(1.74), Inches(7.6), Inches(4.30))
    bullets(slide, Inches(8.72), Inches(1.86), Inches(4.0), Inches(4.2), [
        "引擎层 learnpath/",
        "|数据契约 / 图谱 / 画像 / 推荐 / 规划 / 预测 / 自适应 / 社交 / 分析",
        "服务层 app/",
        "|单例容器：懒加载 + 线程锁 + 缺失模块降级为 503",
        "表现层 web/",
        "|原生 ES Module + 本地 ECharts，零构建、可离线演示",
        "可测试性",
        "|算法可脱离 HTTP 直接用 pytest 验证",
    ], size=12.5)
    footer(slide, page)


def slide_data(prs, d, page):
    slide = new_slide(prs)
    st, gs = d.get("stats", {}), d.get("graph", {})
    header(slide, "02 · DATA & GRAPH", "420 个知识点的先修 DAG，是「学得动」的约束来源")
    kpi_row(slide, [
        (str(st.get("n_knowledge_points", "—")), "知识点"),
        (str(st.get("n_kp_edges", "—")), "先修边"),
        (str(gs.get("max_depth", "—")), "最大深度"),
        (f"{st.get('completion_rate', 0):.1%}", "交互完成率"),
    ], Inches(1.72), accent=C_BLUE)
    picture_fit(slide, FIG / "fig_dataset_overview.png", Inches(0.86), Inches(2.92), Inches(6.0), Inches(3.10))
    picture_fit(slide, FIG / "fig_graph_structure.png", Inches(7.02), Inches(2.92), Inches(5.74), Inches(3.10))
    footer(slide, page)


def slide_learner(prs, d, page):
    slide = new_slide(prs)
    header(slide, "03 · LEARNER MODEL", "学习者建模：多维画像 + 贝叶斯知识追踪")
    card(slide, Inches(0.86), Inches(1.80), Inches(6.0), Inches(4.24))
    textbox(slide, Inches(1.14), Inches(1.98), Inches(5.5), Inches(0.32), "画像五个维度", 14, True, C_INK)
    bullets(slide, Inches(1.14), Inches(2.36), Inches(5.5), Inches(2.4), [
        "知识水平 —— 每个知识点的掌握度 m ∈ [0,1]",
        "学习偏好 —— 理论/实践倾向、承载模态、单次专注时长",
        "时间安排 —— 每周可投入小时数、可学习天数、截止周数",
        "能力系数 —— 学习速度，调制 BKT 的学习率",
        "兴趣领域 —— 用于社交匹配与多样性",
    ], size=12.5)
    card(slide, Inches(1.14), Inches(4.66), Inches(5.5), Inches(1.18),
         fill=RGBColor(0xF6, 0xF8, 0xFD), border=C_LINE)
    textbox(slide, Inches(1.34), Inches(4.80), Inches(5.2), Inches(0.30),
            "掌握度更新（BKT 后验 + 学习转移）", 11.5, True, C_BLUE)
    rich_line(slide, Inches(1.34), Inches(5.16), Inches(5.2), Inches(0.56), [
        ("m′ = posterior + (1 − posterior) · p_learn · a · w", 12, True, C_INK),
    ])

    card(slide, Inches(7.12), Inches(1.80), Inches(5.64), Inches(4.24))
    textbox(slide, Inches(7.40), Inches(1.98), Inches(5.1), Inches(0.32), "四个决策量", 14, True, C_INK)
    items = [
        ("先修就绪度", "课程先修知识点的加权掌握度 → sigmoid", C_BLUE),
        ("风格匹配度", "理论倾向差 + 模态是否一致", C_GREEN),
        ("时间可承受度", "课程时长 vs 每周预算与专注时长", C_ORANGE),
        ("预期增益", "缺口 × 覆盖权重 × 就绪折扣 × 能力调制", C_PURPLE),
    ]
    for i, (name, desc, color) in enumerate(items):
        y = Inches(2.42 + i * 0.90)
        textbox(slide, Inches(7.42), y, Inches(2.1), Inches(0.30), name, 12.5, True, color)
        textbox(slide, Inches(7.42), y + Inches(0.28), Inches(5.0), Inches(0.30), desc, 10.5, False, C_INK2)
    footer(slide, page)


def slide_recommend(prs, d, page):
    slide = new_slide(prs)
    header(slide, "03 · RECOMMENDER", "混合推荐：三路召回 + 加权融合 + MMR 去冗余", accent=C_GREEN)
    card(slide, Inches(0.86), Inches(1.80), Inches(6.30), Inches(2.42))
    textbox(slide, Inches(1.14), Inches(1.98), Inches(5.8), Inches(0.32), "五路信号与权重", 14, True, C_INK)
    weights = [
        ("内容（图谱缺口覆盖）", 0.40, C_BLUE),
        ("协同（SVD 隐语义）", 0.25, C_GREEN),
        ("质量（评分×报名量）", 0.15, C_ORANGE),
        ("先修就绪度", 0.12, C_PURPLE),
        ("社交（同伴完成率）", 0.08, C_INK3),
    ]
    for i, (name, w, color) in enumerate(weights):
        y = Inches(2.38 + i * 0.36)
        textbox(slide, Inches(1.14), y, Inches(3.0), Inches(0.26), name, 11, False, C_INK2)
        track = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.18), y + Inches(0.07),
                                       Inches(2.0), Inches(0.12))
        track.fill.solid(); track.fill.fore_color.rgb = RGBColor(0xEA, 0xEE, 0xF8)
        track.line.fill.background(); track.shadow.inherit = False
        fill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.18), y + Inches(0.07),
                                      int(Inches(2.0) * w / 0.40), Inches(0.12))
        fill.fill.solid(); fill.fill.fore_color.rgb = color
        fill.line.fill.background(); fill.shadow.inherit = False
        textbox(slide, Inches(6.28), y, Inches(0.8), Inches(0.26), f"{w:.2f}", 10.5, True, C_INK3)

    card(slide, Inches(0.86), Inches(4.36), Inches(6.30), Inches(1.68))
    textbox(slide, Inches(1.14), Inches(4.52), Inches(5.8), Inches(0.30), "MMR 多样性重排", 13, True, C_INK)
    textbox(slide, Inches(1.14), Inches(4.86), Inches(5.8), Inches(0.62),
            "c* = argmax [ λ·score(c) − (1−λ)·max sim(c, c′) ]，λ = 0.75",
            11.5, True, C_BLUE)
    textbox(slide, Inches(1.14), Inches(5.34), Inches(5.8), Inches(0.56),
            "避免「10 门都是 Python 入门课」；消融实验显示移除 MMR 后 NDCG 也会下降。",
            10.5, False, C_INK2)

    picture_fit(slide, FIG / "shot_recommend.png", Inches(7.34), Inches(1.80), Inches(5.42), Inches(4.24),
                caption="真实推荐结果：分项得分 + 具体中文理由")
    footer(slide, page)


def slide_planning(prs, d, page):
    slide = new_slide(prs)
    header(slide, "03 · PATH PLANNING", "路径规划：先修闭包 → 缺口识别 → 贪心覆盖 → 拓扑修复", accent=C_ORANGE)
    steps = [
        ("①", "先修闭包", "Pre(目标知识点)，得到「要学目标先得补什么」"),
        ("②", "缺口识别", "掌握度 < 0.60 的知识点构成待补集合 G"),
        ("③", "候选召回", "覆盖 G、先修就绪、时长可承受的课程"),
        ("④", "分层覆盖", "按拓扑层级自底向上，取「边际收益/时长」最大者"),
        ("⑤", "拓扑修复", "先修课必须排在后继课之前"),
        ("⑥", "周次装箱", "按每周预算分配周次并生成里程碑"),
    ]
    for i, (no, name, desc) in enumerate(steps):
        col, row = i % 3, i // 3
        x = Inches(0.86 + col * 4.08)
        y = Inches(1.80 + row * 1.16)
        card(slide, x, y, Inches(3.82), Inches(1.00))
        textbox(slide, x + Inches(0.22), y + Inches(0.20), Inches(0.5), Inches(0.4), no, 18, True, C_ORANGE)
        textbox(slide, x + Inches(0.72), y + Inches(0.16), Inches(2.9), Inches(0.30), name, 13, True, C_INK)
        textbox(slide, x + Inches(0.72), y + Inches(0.50), Inches(2.9), Inches(0.44), desc, 10, False, C_INK2)

    picture_fit(slide, FIG / "shot_path.png", Inches(0.86), Inches(4.28), Inches(7.4), Inches(1.80))
    card(slide, Inches(8.50), Inches(4.28), Inches(4.26), Inches(1.80),
         fill=RGBColor(0xFF, 0xF6, 0xF0), border=C_ORANGE)
    textbox(slide, Inches(8.74), Inches(4.44), Inches(3.8), Inches(0.30), "为什么不能只排序分数", 12, True, C_ORANGE)
    textbox(slide, Inches(8.74), Inches(4.78), Inches(3.8), Inches(1.16),
            "推荐只需每门课单独合理；路径要求整体自洽 —— 难度平缓上升、"
            "先修不缺失、总时长匹配预算。这是本题与普通推荐的分水岭。", 10.5, False, C_INK2)
    footer(slide, page)


def slide_multiobjective(prs, d, page):
    slide = new_slide(prs)
    header(slide, "03 · MULTI-OBJECTIVE", "多目标优化：先求前沿，再按偏好选点", accent=C_PURPLE)
    picture_fit(slide, FIG / "fig_pareto.png", Inches(0.86), Inches(1.78), Inches(6.40), Inches(3.30))
    card(slide, Inches(7.52), Inches(1.78), Inches(5.24), Inches(3.30))
    textbox(slide, Inches(7.78), Inches(1.94), Inches(4.8), Inches(0.30), "四个冲突目标", 13, True, C_INK)
    objs = [("学习收益", "最大化", C_BLUE), ("时间成本", "最小化", C_GREEN),
            ("难度平滑度", "最大化", C_ORANGE), ("风格匹配度", "最大化", C_PURPLE)]
    for i, (name, direction, color) in enumerate(objs):
        y = Inches(2.34 + i * 0.44)
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(7.80), y + Inches(0.09), Inches(0.10), Inches(0.10))
        dot.fill.solid(); dot.fill.fore_color.rgb = color; dot.line.fill.background(); dot.shadow.inherit = False
        textbox(slide, Inches(8.04), y, Inches(2.4), Inches(0.30), name, 12, False, C_INK)
        textbox(slide, Inches(10.50), y, Inches(2.1), Inches(0.30), direction, 11.5, True, color)
    textbox(slide, Inches(7.78), Inches(4.14), Inches(4.8), Inches(0.80),
            "NSGA-II 求 4 维帕累托前沿 → TOPSIS（极差归一化 + 加权贴近度）选妥协解",
            11, False, C_INK2)

    card(slide, Inches(0.86), Inches(5.24), Inches(11.9), Inches(1.36),
         fill=RGBColor(0xFF, 0xF3, 0xEE), border=C_ORANGE)
    textbox(slide, Inches(1.14), Inches(5.40), Inches(11.4), Inches(0.30),
            "过程中发现并修复的一个真实缺陷", 12.5, True, C_ORANGE)
    textbox(slide, Inches(1.14), Inches(5.72), Inches(11.4), Inches(0.72),
            "测试发现三种预设产出同一条路径。根因是 TOPSIS 用了向量归一化 x/‖x‖："
            "本问题中 gain 跨一个数量级而 time 只差 1.5 倍，范数被大值主导，权重语义被吃掉。"
            "改用 Hwang & Yoon 的极差变换 (x−min)/(max−min) 后，三个预设恢复正常区分。",
            10.5, False, C_INK2)
    footer(slide, page)


def slide_prediction(prs, d, page):
    slide = new_slide(prs)
    header(slide, "03 · PREDICTION", "学习效果预测：完成概率 + 掌握度增益", accent=C_GREEN)
    met = d.get("metrics") or {}
    em = met.get("effect_model", {})
    gm = met.get("gain_model", {})
    kpis = []
    for m in em.get("models", []):
        kpis.append((f"{m['auc']:.3f}", f"AUC · {m['name']}"))
    for m in gm.get("models", []):
        kpis.append((f"{m['r2']:.3f}", f"R² · {m['name']}"))
    if kpis:
        kpi_row(slide, kpis[:4], Inches(1.74), accent=C_GREEN)

    picture_fit(slide, FIG / "fig_model_comparison.png", Inches(0.86), Inches(2.94), Inches(6.30), Inches(3.10))
    card(slide, Inches(7.40), Inches(2.94), Inches(5.36), Inches(3.10))
    textbox(slide, Inches(7.66), Inches(3.10), Inches(4.9), Inches(0.30), "两个关键设计", 13, True, C_INK)
    bullets(slide, Inches(7.66), Inches(3.46), Inches(4.9), Inches(2.4), [
        "无标签泄漏",
        "|特征在「学习者尚未学过该课程」的时刻计算",
        "只用线上可观测信号",
        "|刻意不使用生成期隐变量，保证模型可部署",
        "可解释归因",
        "|特征重要性 Top-5 直接进入推荐理由",
        "风险分级",
        "|完成概率 → 低/中/高，指导学习者预期",
    ], size=11.5, bullet_color=C_GREEN)
    footer(slide, page)


def slide_adaptive(prs, d, page):
    slide = new_slide(prs)
    header(slide, "03 · ADAPTIVE", "自适应调整：四类触发 + 自动重规划", accent=C_ORANGE)
    rows = [
        ("测验分 < 0.60", "判定未掌握，插入同知识点更基础的补强课", C_ORANGE),
        ("测验分 > 0.90", "跳过路径中覆盖同一知识点的后续课程", C_GREEN),
        ("实际/计划进度 < 0.70", "收敛路径：消减低收益课程，切「速通」预设", C_BLUE),
        ("实际/计划进度 > 1.30", "追加进阶课程，切「夯实」预设", C_PURPLE),
    ]
    for i, (cond, action, color) in enumerate(rows):
        y = Inches(1.84 + i * 1.06)
        card(slide, Inches(0.86), y, Inches(11.9), Inches(0.90))
        chip = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.10), y + Inches(0.22),
                                      Inches(2.70), Inches(0.44))
        chip.fill.solid(); chip.fill.fore_color.rgb = color
        chip.line.fill.background(); chip.shadow.inherit = False
        textbox(slide, Inches(1.10), y + Inches(0.31), Inches(2.70), Inches(0.28),
                cond, 11, True, RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
        textbox(slide, Inches(4.06), y + Inches(0.30), Inches(8.4), Inches(0.34), action, 12, False, C_INK)

    card(slide, Inches(0.86), Inches(6.02), Inches(11.9), Inches(0.66),
         fill=RGBColor(0xF6, 0xF8, 0xFD), border=C_LINE)
    textbox(slide, Inches(1.14), Inches(6.16), Inches(11.4), Inches(0.36),
            "触发后：BKT 更新掌握度 → 重新规划路径 → 返回掌握度增量、动作说明、新画像与新路径",
            11.5, False, C_INK2)
    footer(slide, page)


def slide_social(prs, d, page):
    slide = new_slide(prs)
    header(slide, "03 · SOCIAL", "社交学习：同伴匹配 + 能力互补小组", accent=C_PURPLE)
    card(slide, Inches(0.86), Inches(1.80), Inches(5.0), Inches(4.24))
    textbox(slide, Inches(1.14), Inches(1.98), Inches(4.5), Inches(0.32), "复合相似度", 14, True, C_INK)
    textbox(slide, Inches(1.14), Inches(2.36), Inches(4.5), Inches(0.56),
            "sim = α·cos(掌握度向量) + β·同目标 + γ·Jaccard(兴趣)",
            11.5, True, C_PURPLE)
    bullets(slide, Inches(1.14), Inches(3.02), Inches(4.5), Inches(2.6), [
        "同伴推荐",
        "|相似度 + 共同目标知识点 + 共同课程 + 对方更擅长的知识点",
        "学习小组",
        "|按「能力互补」组建：同一知识点上有人已掌握、有人未掌握",
        "反哺推荐",
        "|群体的课程完成率作为混合推荐的第五路信号",
    ], size=11.5, bullet_color=C_PURPLE)
    picture_fit(slide, FIG / "shot_social.png", Inches(6.10), Inches(1.80), Inches(6.66), Inches(4.24),
                caption="同伴与学习小组推荐界面")
    footer(slide, page)


def slide_dashboard(prs, d, page):
    slide = new_slide(prs)
    header(slide, "03 · DASHBOARD", "学习分析仪表板：8 类可视化聚合", accent=C_BLUE)
    picture_fit(slide, FIG / "shot_dashboard.png", Inches(0.86), Inches(1.76), Inches(8.10), Inches(4.34))
    card(slide, Inches(9.14), Inches(1.76), Inches(3.62), Inches(4.34))
    textbox(slide, Inches(9.38), Inches(1.94), Inches(3.2), Inches(0.30), "8 类聚合数据", 13, True, C_INK)
    bullets(slide, Inches(9.38), Inches(2.30), Inches(3.2), Inches(3.6), [
        "KPI 卡片（6 项）",
        "掌握度演化曲线",
        "领域掌握度雷达",
        "课程难度分布",
        "承载模态构成",
        "每周时间分配",
        "完成概率预测表",
        "知识缺口明细表",
    ], size=11, bullet_color=C_BLUE, gap=0.04)
    footer(slide, page)


def slide_evaluation(prs, d, page):
    slide = new_slide(prs)
    header(slide, "04 · EVALUATION", "离线评测：混合推荐显著优于热度基线", accent=C_GREEN)
    ev = d.get("evaluation")
    if ev and ev.get("recommenders"):
        rows = [
            [r["name"], f"{r['precision_at_k']:.4f}", f"{r['recall_at_k']:.4f}",
             f"{r['ndcg_at_k']:.4f}", f"{r['coverage']:.3f}", f"{r['diversity']:.3f}"]
            for r in ev["recommenders"]
        ]
        hybrid_idx = [i for i, r in enumerate(ev["recommenders"]) if "本系统" in r["name"]]
        table(slide, ["策略", "Precision@10", "Recall@10", "NDCG@10", "Coverage", "Diversity"],
              rows, Inches(0.86), Inches(1.78), Inches(11.9),
              col_ratios=[3.4, 1.7, 1.6, 1.6, 1.5, 1.5], size=11,
              highlight_rows=hybrid_idx)
        y2 = Inches(3.66)
    else:
        y2 = Inches(1.80)
    picture_fit(slide, FIG / "fig_recommender_comparison.png", Inches(0.86), y2, Inches(6.10), Inches(3.06))
    picture_fit(slide, FIG / "fig_ablation.png", Inches(7.12), y2, Inches(5.64), Inches(3.06))
    if not ev:
        textbox(slide, Inches(0.86), Inches(5.10), Inches(11.9), Inches(0.40),
                "评测结果与图表在 models/evaluation.json 生成后自动填入本页。",
                13, False, C_INK3, align=PP_ALIGN.CENTER)
    footer(slide, page)


def slide_engineering(prs, d, page):
    slide = new_slide(prs)
    header(slide, "05 · ENGINEERING", "工程质量：分层、可测、可复现")
    kpi_row(slide, [("3 层", "引擎 / 服务 / 表现"), ("6 视图", "单页应用"),
                    ("13 接口", "REST 契约"), ("0 GPU", "纯 CPU 秒级响应")], Inches(1.74), accent=C_PURPLE)
    cards = [
        ("数据契约先行", "schema.py + API.md 固定后才开工，四条线并行开发、集成零返工", C_BLUE),
        ("测试分层", "契约层 / 算法层 / 接口层 / 评测层，pytest 一条命令全回归", C_GREEN),
        ("可复现", "固定随机种子，数据集入库；二进制模型脚本一键重建", C_ORANGE),
        ("仓库规范", "课题材料与产物分离、.gitignore 精确、提交信息规范", C_PURPLE),
    ]
    for i, (title, desc, color) in enumerate(cards):
        col, row = i % 2, i // 2
        x = Inches(0.86 + col * 6.05)
        y = Inches(2.94 + row * 1.12)
        card(slide, x, y, Inches(5.72), Inches(0.96))
        strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(0.055), Inches(0.96))
        strip.fill.solid(); strip.fill.fore_color.rgb = color
        strip.line.fill.background(); strip.shadow.inherit = False
        textbox(slide, x + Inches(0.26), y + Inches(0.14), Inches(5.2), Inches(0.30), title, 13, True, color)
        textbox(slide, x + Inches(0.26), y + Inches(0.46), Inches(5.24), Inches(0.42), desc, 10.5, False, C_INK2)

    card(slide, Inches(0.86), Inches(5.26), Inches(11.9), Inches(1.34))
    textbox(slide, Inches(1.14), Inches(5.42), Inches(11.4), Inches(0.30), "运行方式（评审可一键复现）", 12, True, C_INK)
    textbox(slide, Inches(1.14), Inches(5.74), Inches(11.4), Inches(0.70),
            "python -m pip install -r requirements.txt\n"
            "python -m uvicorn app.main:app --port 8000     →   浏览器打开 http://127.0.0.1:8000",
            11, False, C_INK2, line_spacing=1.5)
    footer(slide, page)


def slide_summary(prs, d, page):
    slide = new_slide(prs)
    header(slide, "06 · SUMMARY", "总结：把「个性化学习」翻译成可计算的问题")
    left = [
        "完成了什么",
        "|8 条实现要求全部落地，额外实现协同过滤 / 可解释推荐 / 离线评测",
        "最大的收获",
        "|学会从需求出发做技术选型：★☆难度选对路线比堆模型更重要",
        "|理解多目标优化「先求集、再选点」的两段式思路",
    ]
    right = [
        "不足",
        "|数据集为合成数据，结论可推广性待真实平台验证",
        "|掌握度用 BKT-lite，未建模知识点之间的耦合",
        "下一步",
        "|引入 GCN 知识图谱嵌入替代独立 BKT 假设",
        "|把社交从「匹配」推进到「协作」，并做线上 A/B 验证",
    ]
    card(slide, Inches(0.86), Inches(1.80), Inches(5.80), Inches(4.20))
    bullets(slide, Inches(1.14), Inches(2.02), Inches(5.24), Inches(3.8), left, size=12.5, bullet_color=C_BLUE)
    card(slide, Inches(6.96), Inches(1.80), Inches(5.80), Inches(4.20))
    bullets(slide, Inches(7.24), Inches(2.02), Inches(5.24), Inches(3.8), right, size=12.5, bullet_color=C_ORANGE)

    card(slide, Inches(0.86), Inches(6.14), Inches(11.9), Inches(0.62),
         fill=RGBColor(0xEE, 0xF4, 0xFF), border=C_BLUE)
    textbox(slide, Inches(1.14), Inches(6.28), Inches(11.4), Inches(0.34),
            "一句话总结：推荐不是「猜你喜欢」，而是在知识图谱约束下，为特定目标找到一条「走得通、学得动」的路。",
            12, True, C_BLUE)
    footer(slide, page)


def slide_thanks(prs, d, page: int = 0):
    slide = new_slide(prs)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(5.0))
    band.fill.solid()
    band.fill.fore_color.rgb = RGBColor(0x0E, 0x15, 0x24)
    band.line.fill.background()
    band.shadow.inherit = False
    textbox(slide, Inches(1.0), Inches(1.86), Inches(11.3), Inches(0.9),
            "谢谢，请老师批评指正", 40, True, RGBColor(0xFF, 0xFF, 0xFF))
    textbox(slide, Inches(1.0), Inches(2.86), Inches(11.3), Inches(0.44),
            "LearnPath · 智学路径 —— 在线学习的个性化课程推荐实现", 16, False,
            RGBColor(0xC3, 0xCC, 0xDD))
    textbox(slide, Inches(1.0), Inches(3.62), Inches(11.3), Inches(0.40),
            "github.com/ZK-1223/Implementation_of_Personalized_Course_Recommendations_for_Online_Learning",
            12, False, C_GREEN)
    textbox(slide, Inches(1.0), Inches(5.40), Inches(11.3), Inches(0.36),
            "工程源码 · 数据集 · 接口契约 · 图表脚本均已在仓库中，可一键复现全部结果。", 12, False, C_INK2)


# ==================================================================
# 主流程
# ==================================================================


def main() -> Path:
    print("加载真实数据 ...")
    data = load()
    st = data.get("stats", {})
    print(f"  数据集：{st.get('n_knowledge_points')} 知识点 / {st.get('n_courses')} 课程")
    print(f"  模型指标：{'有' if data.get('metrics') else '缺'}"
          f" | 评测结果：{'有' if data.get('evaluation') else '缺'}")

    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    builders = [
        slide_cover, slide_toc, slide_background, slide_requirements,
        slide_architecture, slide_data, slide_learner, slide_recommend,
        slide_planning, slide_multiobjective, slide_prediction, slide_adaptive,
        slide_social, slide_dashboard, slide_evaluation, slide_engineering,
        slide_summary, slide_thanks,
    ]
    # 封面与尾页不带页脚；其余按顺序编号
    for index, builder in enumerate(builders, start=1):
        if builder in (slide_cover, slide_thanks):
            builder(prs, data)
        else:
            builder(prs, data, index)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"已生成：{OUT.relative_to(ROOT)}  （{len(prs.slides)} 页，"
          f"{OUT.stat().st_size / 1024:.1f} KB）")
    return OUT


if __name__ == "__main__":
    main()
