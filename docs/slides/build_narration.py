"""讲解视频讲稿生成器（约 5 分钟版）。

生成 ``docs/slides/讲解视频讲稿.docx``：一份**可以照着念**的逐页讲稿，
与 ``智能应用系统课程设计-第5题-答辩PPT.pptx`` 的 18 页一一对应。

时长控制：
    讲稿正文约 1100 字，按每分钟 245 字的朗读速度 + 每页翻页停顿，合计约 5 分钟。
    为了压进 5 分钟：第 2 页（提纲）直接跳过；第 4、7、12、13、14、16 页各只讲一句；
    篇幅集中在第 3（问题定义）、9（路径规划）、10（多目标）、15（量化结果）四处。

设计目标（"一边翻 PPT 一边念"这个使用场景）：
    * 每页一个区块，页首标「第几页 / 页面标题 / 字数 / 建议秒数 / 累计时间」，抬头就能对上；
    * 讲稿用口语短句，方便朗读；
    * 每块末尾给明确的翻页提示；
    * 附术语读法与可能的提问，供答辩答疑用，不占视频时长。

所有数字与 PPT 上的数字一致（NDCG 0.1026 / AUC 0.8820 / R² 0.6431 / 244 个用例等）。

运行：
    python docs/slides/build_narration.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "讲解视频讲稿.docx"

FONT_CN = "宋体"
FONT_EN = "Times New Roman"
C_INK = RGBColor(0x1B, 0x23, 0x33)
C_GREY = RGBColor(0x6B, 0x74, 0x84)
C_BLUE = RGBColor(0x2F, 0x6B, 0xFF)

CHARS_PER_MIN = 265      # 朗读速度（字/分钟）：照着稿子念的正常讲课语速
FLIP_OVERHEAD = 2        # 每页翻页停顿（秒）
TARGET_SECONDS = 300     # 目标时长：5 分钟


# ==================================================================
# 讲稿内容： (页码, 页面标题, 讲稿段落)  —— 段落为空表示该页跳过
# ==================================================================


def _script() -> list[tuple[int, str, list[str]]]:
    return [
        (
            1,
            "封面 · LearnPath 智学路径",
            [
                "各位老师好，我是＿＿＿＿专业的＿＿＿＿。我的课程设计是第 5 题"
                "《在线学习的个性化课程推荐实现》，系统叫 LearnPath，"
                "含 420 个知识点、560 门课程、720 名学习者。",
            ],
        ),
        (2, "汇报提纲", []),  # 5 分钟版直接跳过
        (
            3,
            "问题背景",
            [
                "先看问题。平台不缺课，学习者卡住的是三件事：该学什么、按什么顺序学、学得动吗。",
                "所以这道题的本质是带先修约束的多目标组合优化：在知识图谱约束下选出课程序列并排序，"
                "让收益、时间、难度梯度、风格匹配这些冲突目标达到平衡。",
            ],
        ),
        (
            4,
            "题目 8 条实现要求",
            [
                "题目给的 8 条实现要求我逐条落地了，图上每张卡片对应一个独立模块。",
            ],
        ),
        (
            5,
            "系统总览：三层架构",
            [
                "系统分三层：引擎层纯 Python 不依赖 Web；中间 FastAPI 做缓存和降级；"
                "最上面是零构建、可离线的前端。好处是算法能脱离接口直接测试。",
            ],
        ),
        (
            6,
            "数据集与知识图谱",
            [
                "数据集是程序化合成的，固定种子可复现。420 个知识点、483 条先修边、最深 8 层——"
                "这就是「学得动」判断的约束来源。",
            ],
        ),
        (
            7,
            "学习者建模",
            [
                "学习者建模拆成五个维度，掌握度用贝叶斯知识追踪更新，"
                "再派生出先修就绪度等四个决策量。",
            ],
        ),
        (
            8,
            "混合推荐",
            [
                "推荐是三路召回再加权融合，最后用 MMR 做多样性重排。右边是真机截图，"
                "每条推荐都带具体数字的理由。",
            ],
        ),
        (
            9,
            "学习路径规划",
            [
                "路径规划是最关键的一步，分六步：先修闭包、缺口识别、候选召回、分层贪心覆盖、"
                "拓扑修复、周次装箱。",
                "关键设计是每周可投入时间做成硬约束，构造阶段就卡住总时长。截图这位学习者"
                "每周 3 小时、13 周共 39 小时，系统给出 35.1 小时、5 门课。",
                "推荐只要每门课单独合理；路径要求整体自洽——这是本题和普通推荐的分水岭。",
            ],
        ),
        (
            10,
            "多目标优化",
            [
                "多目标是两段式求解：NSGA-II 一次跑出整条帕累托前沿，再用 TOPSIS 按偏好选妥协解，"
                "所以切换三种预设不用重算。",
                "图里有些点看着被压制，这是正常的——非支配关系是在四个目标上判定的。",
            ],
        ),
        (
            11,
            "学习效果预测",
            [
                "效果预测用 23 维特征，都在「还没学过这门课」时计算，杜绝标签泄漏。"
                "完成概率 AUC 0.882，回归 R² 0.643。",
            ],
        ),
        (
            12,
            "自适应调整",
            [
                "自适应做了四类触发：低分补强、高分跳课、落后收敛、超前加速，触发后自动重规划。",
            ],
        ),
        (
            13,
            "社交学习",
            [
                "社交学习做同伴匹配，按能力互补组队。",
            ],
        ),
        (
            14,
            "学习分析仪表板",
            [
                "仪表板的这些曲线都是按周真实计算出来的。",
            ],
        ),
        (
            15,
            "离线评测结果",
            [
                "量化结果最关键。留出法评测 713 位学习者：混合推荐 NDCG@10 是 0.1026，"
                "热度基线 0.0328，是它的 3.13 倍。",
                "右边消融我说实话：只有知识图谱约束真正对精度有贡献，去掉它掉 38%；"
                "去掉 MMR 和协同过滤反而略升——MMR 是拿准确率换多样性。",
            ],
        ),
        (
            16,
            "工程质量",
            [
                "工程上三层架构、六个视图、13 个接口、不需要 GPU，一条 pytest 跑完 244 个用例。",
            ],
        ),
        (
            17,
            "总结",
            [
                "总结：8 条要求全部落地。最大收获是学会从需求出发做技术选型——两星难度，"
                "选对路线比堆模型更重要。不足是数据为合成、掌握度没建模耦合。",
                "我对这道题的理解是：推荐不是「猜你喜欢」，而是在图谱约束下"
                "为特定目标找到一条走得通、学得动的路。",
            ],
        ),
        (
            18,
            "致谢",
            [
                "汇报完毕，请老师批评指正，谢谢！",
            ],
        ),
    ]


PRONUNCIATION = [
    ("BKT", "B-K-T（贝叶斯知识追踪）", "读字母"),
    ("DAG", "D-A-G，也有人读「达格」", "有向无环图"),
    ("SVD", "S-V-D", "奇异值分解"),
    ("MMR", "M-M-R", "最大边际相关性重排"),
    ("NSGA-II", "N-S-G-A 二", "第二代非支配排序遗传算法"),
    ("TOPSIS", "托普西斯", "逼近理想解排序法"),
    ("Pareto", "帕累托", "帕累托前沿 / 帕累托最优"),
    ("NDCG", "N-D-C-G", "归一化折损累计增益"),
    ("AUC", "A-U-C，也有人读「奥克」", "ROC 曲线下面积"),
    ("FastAPI", "法斯特 A-P-I", "Python Web 框架"),
    ("pytest", "派-泰斯特", "Python 测试框架"),
    ("ECharts", "E-Charts", "开源图表库"),
]

QA = [
    (
        "为什么不用深度学习？",
        "两个原因。一是这道题两星难度，核心矛盾在图谱约束和多目标权衡，不在模型容量；"
        "二是环境没有 GPU，用深度模型会让评审无法复现。真要做，我只会在「掌握度估计」这一点上"
        "换成 GCN 知识图谱嵌入。",
    ),
    (
        "数据是合成的，结论可信吗？",
        "合成数据的价值在可控与可复现：生成机制公开，完成标签由 8 个特征的逻辑斯蒂模型生成，"
        "所以「任务有可学信号」成立。可推广性确实要真实平台验证，这一点写在报告不足里。",
    ),
    (
        "开发中遇到过什么问题？",
        "两个典型问题。一是 TOPSIS 用了向量归一化，导致三种预设产出同一条路径，"
        "改成极差变换后恢复正常；二是早期路径规划只按「收益除以时长」贪心，"
        "结果推出来的全是入门课、覆盖不到任何目标知识点，后来改成目标优先的加权最大覆盖，"
        "并把时间预算做成硬约束。",
    ),
    (
        "推荐理由是不是事后套的模板话术？",
        "不是。五条理由分别由五路信号的真实数值驱动：缺口理由带当前掌握度和距目标掌握线的差额，"
        "先修理由带「已就绪 3 比 4」这样的计数，同伴理由带相似学习者的完成率对比。"
        "测试里断言了理由必须含数字、禁止出现「为你推荐」这类空话。",
    ),
    (
        "系统响应速度怎么样？",
        "推荐单次 4.5 毫秒，路径规划单次 0.27 秒，加三档预设和帕累托前沿约 1 秒。"
        "唯一慢的是离线评测接口，首次几十秒，所以服务层做了进程级缓存，演示前预热一次即可。",
    ),
]


# ==================================================================
# 文档构建
# ==================================================================


def _set_font(run, size: float, bold: bool = False, color: RGBColor | None = None,
              cn: str = FONT_CN, en: str = FONT_EN) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = en
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:cs"), en)
    rfonts.set(qn("w:eastAsia"), cn)


def _shade(paragraph, hex_fill: str) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    ppr.append(shd)


def char_count(paragraphs: list[str]) -> int:
    return sum(len(p) for p in paragraphs)


def seconds_for(paragraphs: list[str]) -> int:
    if not paragraphs:
        return FLIP_OVERHEAD
    return int(round(char_count(paragraphs) / CHARS_PER_MIN * 60)) + FLIP_OVERHEAD


def _fmt(seconds: int) -> str:
    return f"{seconds // 60} 分 {seconds % 60:02d} 秒"


class _Doc:
    def __init__(self) -> None:
        self.doc = Document()
        section = self.doc.sections[0]
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Pt(72)
        section.bottom_margin = Pt(72)
        section.left_margin = Pt(90)
        section.right_margin = Pt(90)

        normal = self.doc.styles["Normal"]
        normal.font.name = FONT_EN
        normal.font.size = Pt(10.5)
        rpr = normal.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        rfonts.set(qn("w:ascii"), FONT_EN)
        rfonts.set(qn("w:hAnsi"), FONT_EN)
        rfonts.set(qn("w:eastAsia"), FONT_CN)

    def para(self, text: str = "", size: float = 11.0, bold: bool = False,
             color: RGBColor | None = None, align: str = "left",
             space_before: float = 0, space_after: float = 4,
             first_indent: bool = False, line_spacing: float = 1.5,
             shade: str | None = None):
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(space_before)
        pf.space_after = Pt(space_after)
        pf.line_spacing = line_spacing
        if first_indent:
            pf.first_line_indent = Pt(size * 2)
        p.alignment = {
            "left": WD_ALIGN_PARAGRAPH.LEFT,
            "center": WD_ALIGN_PARAGRAPH.CENTER,
            "right": WD_ALIGN_PARAGRAPH.RIGHT,
        }[align]
        if shade:
            _shade(p, shade)
        if text:
            run = p.add_run(text)
            _set_font(run, size, bold, color)
        return p

    def h(self, text: str, size: float = 15.0, space_before: float = 14) -> None:
        self.para(text, size=size, bold=True, color=C_INK,
                  space_before=space_before, space_after=6, line_spacing=1.3)

    def bullet(self, text: str, size: float = 10.5) -> None:
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Pt(size * 1.6)
        pf.first_line_indent = Pt(-size * 1.1)
        pf.line_spacing = 1.5
        pf.space_after = Pt(2)
        run = p.add_run("·  " + text)
        _set_font(run, size, color=C_INK)

    def table(self, headers: list[str], rows: list[list[str]],
              widths: list[float] | None = None, size: float = 9.0) -> None:
        table = self.doc.add_table(rows=1, cols=len(headers))
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
            borders.append(el)
        table._tbl.tblPr.append(borders)
        for i, head in enumerate(headers):
            cell = table.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(head)
            _set_font(run, size, bold=True)
        for row in rows:
            cells = table.add_row().cells
            for i, value in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(value)
                _set_font(run, size)
        if widths:
            table.autofit = False
            for i, w in enumerate(widths):
                if i < len(table.columns):
                    table.columns[i].width = Cm(w)
                for row in table.rows:
                    if i < len(row.cells):
                        row.cells[i].width = Cm(w)
        self.para("", size=6, space_after=2)

    def page_break(self) -> None:
        self.doc.add_page_break()

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        return path


def build() -> Path:
    d = _Doc()
    script = _script()
    all_chars = char_count([p for _, _, c in script for p in c])
    total_seconds = sum(seconds_for(core) for _, _, core in script)

    # ---------------- 标题区 ----------------
    d.para("智能应用系统课程设计 · 讲解视频讲稿", size=20, bold=True,
           align="center", space_before=44, space_after=8)
    d.para("第 5 题　在线学习的个性化课程推荐实现（LearnPath · 智学路径）",
           size=13, color=C_GREY, align="center", space_after=6)
    slow = int(total_seconds * CHARS_PER_MIN / 235)
    d.para(f"对应 PPT 共 {len(script)} 页　|　正文约 {all_chars} 字"
           f"　|　预计约 5 分钟（{_fmt(total_seconds)} ~ {_fmt(slow)}，随语速浮动）",
           size=11, color=C_BLUE, align="center", space_after=20)

    # ---------------- 使用说明 ----------------
    d.h("使用说明（录之前先看这 30 秒）", size=14, space_before=4)
    d.para("这份讲稿是照着念的：每块对应 PPT 的一页，页首写着「第几页、页面标题、字数、"
           "建议秒数、累计时间」，念到「▶ 翻页」就把 PPT 翻到下一页。",
           size=10.5, first_indent=True)
    d.bullet(f"全文 {all_chars} 字，按正常讲课语速约 5 分钟。念快了不到 5 分，念慢了 5 分半，都在范围内。")
    d.bullet("第 2 页（汇报提纲）直接跳过，不要念——5 分钟版没有时间讲提纲。")
    d.bullet("第 4、7、12、13、14、16 页各只有一句话，快速带过；"
             "篇幅集中在第 3、9、10、15 页。")
    d.bullet("语速比平时说话略慢一点，标点处停顿；不要逐字念，讲顺了只念每段第一句也行。")
    d.bullet("口误不用停，接着往下说；录完把明显卡壳的地方剪掉重录一小段即可。")
    d.bullet("录屏建议 1080p，PPT 用「幻灯片放映 → 从当前页开始」全屏录制。")
    d.bullet("若要现场跑系统：先访问一次「系统评测」页预热（首次约 30~60 秒），避免录到白屏。")

    d.h("时间分配一览", size=14)
    rows = []
    cumulative = 0
    for page, title, core in script:
        sec = seconds_for(core)
        cumulative += sec
        rows.append([
            str(page), title,
            "跳过" if not core else f"{char_count(core)} 字",
            f"{sec} 秒",
            _fmt(cumulative),
        ])
    rows.append(["—", "合计", f"{all_chars} 字", f"{total_seconds} 秒", _fmt(total_seconds)])
    d.table(["页码", "页面标题", "字数", "建议时长", "累计"], rows,
            widths=[1.2, 7.4, 2.2, 2.2, 3.0], size=8.5)

    # ---------------- 逐页讲稿 ----------------
    d.page_break()
    d.h("逐页讲稿", size=16, space_before=0)
    d.para("每块的抬头对应 PPT 当前页。下面的大字段落就是照着念的内容。",
           size=10.5, color=C_GREY, space_after=10)

    cumulative = 0
    for index, (page, title, core) in enumerate(script):
        cumulative += seconds_for(core)
        d.para(f"第 {page} 页　{title}", size=13, bold=True,
               color=RGBColor(0xFF, 0xFF, 0xFF), space_before=14, space_after=2, shade="2F6BFF")
        d.para(f"建议 {seconds_for(core)} 秒　·　累计 {_fmt(cumulative)}",
               size=9, color=C_GREY, space_after=6)

        if not core:
            d.para("（这页不念，直接翻到下一页。）", size=11, bold=True, color=C_GREY,
                   space_after=6)
        else:
            for line in core:
                d.para(line, size=12, first_indent=True, line_spacing=1.6, space_after=6)

        if index < len(script) - 1:
            d.para(f"▶ 讲完这页，翻到第 {script[index + 1][0]} 页。", size=10, bold=True,
                   color=C_BLUE, space_before=2, space_after=10)
        else:
            d.para("▶ 讲完这页，结束录制。", size=10, bold=True,
                   color=C_BLUE, space_before=2, space_after=10)

    # ---------------- 附一：术语读法 ----------------
    d.page_break()
    d.h("附一　术语怎么念", size=16, space_before=0)
    d.para("这些词第一次出现时念全称，之后可以只念缩写。", size=10.5, space_after=8)
    d.table(["缩写 / 术语", "建议读法", "中文含义"],
            [[a, b, c] for a, b, c in PRONUNCIATION],
            widths=[3.2, 5.0, 7.0], size=9)

    # ---------------- 附二：可能的提问 ----------------
    d.h("附二　老师可能会问的问题（不占视频时长）", size=16)
    d.para("视频里不用讲，但答辩或答疑时大概率会被问到，先看一眼心里有底。",
           size=10.5, color=C_GREY, space_after=8)
    for i, (question, answer) in enumerate(QA, start=1):
        d.para(f"Q{i}　{question}", size=11.5, bold=True, space_before=8, space_after=3)
        d.para(f"A：{answer}", size=10.5, first_indent=True, line_spacing=1.5, space_after=4)

    return d.save(OUT)


def main() -> Path:
    path = build()
    script = _script()
    total = sum(seconds_for(core) for _, _, core in script)
    chars = char_count([p for _, _, c in script for p in c])
    print(f"已生成：{path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.1f} KB)")
    print(f"共 {len(script)} 页（跳过 1 页）；正文 {chars} 字；预计 {_fmt(total)}")
    if total > TARGET_SECONDS + 20:
        print(f"⚠ 超过 5 分钟目标（{_fmt(TARGET_SECONDS)}），建议再压缩正文。")
    return path


if __name__ == "__main__":
    main()
