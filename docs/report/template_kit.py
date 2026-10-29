"""以课程设计报告模板为母版填充内容的构建器。

与「从零新建文档」的做法不同，本模块**打开老师下发的
``智能应用系统课程设计报告-2025模版.doc`` 转换出的 docx，直接在它自带的表格里写内容**，
因此封面、说明页、章节标题、页面设置、页眉页脚全部保持模板原样，读者看到的是
「把内容填进了模板」，而不是「照模板仿写了一份新文档」。

模板结构（实测）：
    * 正文段落：``计算机与信息学院``(26pt) / ``智能应用系统设计报告``(34pt) / ``说  明``(26pt)
      / 说明五条(14pt) / ``题目序号__   任务名称：``(15pt) / ``实践开展时间∶ 年 月 日``(15pt)
    * ``Table 0``：封面信息表 5 行 × 2 列（专业班级 / 姓名及学号 / 课程教学班号 / 任课教师 / 学年学期）
    * ``Table 1``：11 行 × 1 列，每行一个章节；行内首段是该章节标题（14pt 加粗），
      其后是一个空的占位段（12pt、行距 300）——正文就填在占位段的位置上。

字体规范（模板「说明」第 5 条）：中文宋体、英文 Times New Roman。
经 LibreOffice 转换后 ``Normal`` 样式的 ``rFonts`` 会被污染成
``宋体;汉仪书宋二KW`` / ``Times New Roman;Times New Roman`` 这类非法值，
本模块在填充前统一规范化，保证全文只有宋体与 Times New Roman 两种字体。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# ---- 字体常量：与模板一致 ----
FONT_CN = "宋体"
FONT_EN = "Times New Roman"

# ---- 从模板实测得到的字号（磅）----
SIZE_COVER_SCHOOL = 26.0
SIZE_COVER_TITLE = 34.0
SIZE_TASK_LINE = 15.0
SIZE_SECTION_TITLE = 14.0
SIZE_BODY = 12.0          # 模板章节占位段 sz=24 → 12pt（小四）
SIZE_CAPTION = 10.5
SIZE_TABLE = 9.0
SIZE_CODE = 9.0           # 小五
CODE_LINE_PT = 12.0       # 模板要求：附录源码行距 12 磅
LINE_SPACING_TWIPS = 300  # 模板 pPr: w:line=300 w:lineRule=auto（1.25 倍）

# ---- 章节行索引（Table 1 的行号 → 章节名）----
SECTION_ROWS: dict[str, int] = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "感想": 10,
}

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

#: LibreOffice Kit CLI（用于把 .doc 模板转成 .docx）
NODE_EXE = (
    r"C:\Users\zk002068\AppData\Local\Programs\DeepSeek Harness\resources\runtime"
    r"\primary-runtime\dependencies\node\bin\node.exe"
)
LO_KIT_CLI = (
    r"C:\Users\zk002068\AppData\Local\Programs\DeepSeek Harness\resources\app.asar.unpacked"
    r"\dsh\node_modules\@deepseek-ai\libreoffice-kit\lib\cli.js"
)


# ==================================================================
# 模板准备：.doc → .docx，并修正转换产生的格式噪声
# ==================================================================


def convert_template(src: Path, dst: Path) -> Path:
    """用 LibreOffice Kit 把旧版 ``.doc`` 模板转换为 ``.docx``。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    result = subprocess.run(
        [NODE_EXE, LO_KIT_CLI, "convert", "--input", str(src), "--output", str(dst)],
        capture_output=True,
        text=True,
        check=False,
    )
    if not dst.exists():
        raise RuntimeError(
            f"模板转换失败：{result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return dst


def _set_rfonts(element, cn: str = FONT_CN, en: str = FONT_EN) -> None:
    """把某个 ``w:rFonts`` 元素规范化为「中文宋体 + 西文 Times New Roman」。"""
    element.set(qn("w:ascii"), en)
    element.set(qn("w:hAnsi"), en)
    element.set(qn("w:cs"), en)
    element.set(qn("w:eastAsia"), cn)


def normalize_template(path: Path) -> None:
    """修正模板转换后的格式噪声，使全文只有宋体与 Times New Roman。

    需要修两类问题：
        1. ``w:rFonts`` 被写成 ``宋体;汉仪书宋二KW`` 这类「字体列表」——Word 不认，
           会回退到默认字体，导致中英文可能渲染成别的字形；
        2. ``w:jc`` 出现 ``start`` / ``end``（ODF 语义），python-docx 的枚举无法解析，
           读取 ``paragraph.alignment`` 会直接抛异常。
    """
    doc = Document(str(path))

    parts = [doc.element]
    for section in doc.sections:
        parts.extend([section.header._element, section.footer._element])
    parts.append(doc.styles.element)

    for root in parts:
        for rfonts in root.iter(qn("w:rFonts")):
            _set_rfonts(rfonts)
        for jc in root.iter(qn("w:jc")):
            value = jc.get(qn("w:val"))
            if value == "start":
                jc.set(qn("w:val"), "left")
            elif value == "end":
                jc.set(qn("w:val"), "right")

    doc.save(str(path))


def prepare_template(src: Path, work_dir: Path) -> Path:
    """转换 + 规范化，返回可直接填充的 docx 路径。"""
    work = work_dir / "template.docx"
    convert_template(src, work)
    normalize_template(work)
    return work


# ==================================================================
# 填充构建器
# ==================================================================


class TemplateReport:
    """在模板文档的表格单元格内写内容。

    提供的写作接口刻意与「从零新建文档」的构建器同名同义，
    因此报告正文的组装代码可以不改一行地复用：
        ``h`` / ``p`` / ``rich`` / ``bullets`` / ``numbered``
        / ``code`` / ``table`` / ``figure`` / ``caption`` / ``raw`` / ``page_break``
    """

    def __init__(self, template_path: Path) -> None:
        self.doc = Document(str(template_path))
        self._cell = None          # 当前写入的单元格
        self._figure_no = 0
        self._table_no = 0

        if len(self.doc.tables) < 2:
            raise RuntimeError("模板结构异常：未找到封面表与章节表")
        self.cover_table = self.doc.tables[0]
        self.section_table = self.doc.tables[1]

    # ---------------- 封面 / 抬头 ----------------

    def fill_cover(self, values: Sequence[str] = ("", "", "", "")) -> None:
        """填写封面信息表右侧四个单元格（专业班级 / 姓名学号 / 教学班号 / 任课教师）。

        参数留空则保持模板的空白状态，由本人手写或后续填写。
        """
        for index, value in enumerate(values[:4]):
            cell = self.cover_table.rows[index].cells[1]
            para = cell.paragraphs[0]
            if not value:
                continue
            run = para.add_run(value)
            self._style_run(run, size=15.0)
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def fill_task_line(self, sequence: str, task_name: str) -> None:
        """改写 ``题目序号__   任务名称：`` 这一行（保留原字号 15pt）。"""
        target = None
        for para in self.doc.paragraphs:
            if "题目序号" in para.text and "任务名称" in para.text:
                target = para
                break
        if target is None:
            raise RuntimeError("模板中未找到「题目序号 / 任务名称」行")
        runs = target.runs
        if len(runs) >= 3:
            runs[0].text = "题目序号 "
            runs[1].text = f"{sequence}    "
            runs[2].text = f"任务名称： {task_name}"
            for run in runs[3:]:
                run.text = ""
        else:  # 兜底：整体重写
            for run in runs:
                run.text = ""
            run = target.add_run(f"题目序号 {sequence}   任务名称： {task_name}")
            self._style_run(run, size=SIZE_TASK_LINE)

    def fill_practice_date(self, text: str) -> None:
        """填写 ``实践开展时间∶  年 月 日`` 这一行。"""
        for para in self.doc.paragraphs:
            if "实践开展时间" in para.text:
                for run in para.runs:
                    run.text = ""
                run = para.add_run(f"实践开展时间∶ {text}")
                self._style_run(run, size=SIZE_TASK_LINE)
                return
        raise RuntimeError("模板中未找到「实践开展时间」行")

    # ---------------- 章节切换 ----------------

    def open_section(self, key: str) -> None:
        """把写入位置切换到某个章节单元格，并清掉模板的空占位段。"""
        row_index = SECTION_ROWS[key]
        cell = self.section_table.rows[row_index].cells[0]
        self._cell = cell
        # 模板每行 = [标题段, 空占位段]；删掉占位段，正文从占位段的位置开始写
        if len(cell.paragraphs) >= 2:
            placeholder = cell.paragraphs[1]
            if not placeholder.text.strip():
                placeholder._p.getparent().remove(placeholder._p)
        self._figure_no = 0
        self._table_no = 0

    # ---------------- 底层工具 ----------------

    @staticmethod
    def _style_run(run, size: float, bold: bool = False,
                   color: RGBColor | None = None) -> None:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = FONT_EN
        if color is not None:
            run.font.color.rgb = color
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        _set_rfonts(rfonts)

    def _new_para(self, indent_first: bool = False, size: float = SIZE_BODY,
                  align: str = "left", space_before: float = 0.0,
                  space_after: float = 0.0, exact_line_pt: float | None = None):
        """在当前单元格追加段落，沿用模板的行距（1.25 倍）。"""
        if self._cell is None:
            raise RuntimeError("请先调用 open_section() 指定要填充的章节")
        para = self._cell.add_paragraph()
        para.style = self.doc.styles["Normal"]
        pf = para.paragraph_format
        if exact_line_pt is None:
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.25
        else:
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            pf.line_spacing = Pt(exact_line_pt)
        pf.space_before = Pt(space_before)
        pf.space_after = Pt(space_after)
        if indent_first:
            pf.first_line_indent = Pt(size * 2)
        para.alignment = {
            "left": WD_ALIGN_PARAGRAPH.LEFT,
            "center": WD_ALIGN_PARAGRAPH.CENTER,
            "right": WD_ALIGN_PARAGRAPH.RIGHT,
            "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
        }[align]
        return para

    def _add_runs(self, para, text: str, size: float, bold: bool = False) -> None:
        """写入文本，支持 ``**加粗**`` 行内标记。"""
        for index, segment in enumerate(text.split("**")):
            if not segment:
                continue
            run = para.add_run(segment)
            self._style_run(run, size=size, bold=(bold or index % 2 == 1))

    # ---------------- 写作接口（与 ReportBuilder 对齐）----------------

    def raw(self, text: str = "", size: float = SIZE_BODY, bold: bool = False,
            align: str = "left", space_before: float = 0.0, space_after: float = 0.0,
            indent_first: bool = False, line_spacing: float = 1.25):
        para = self._new_para(indent_first=indent_first, size=size, align=align,
                              space_before=space_before, space_after=space_after)
        if text:
            self._add_runs(para, text, size, bold)
        return para

    def h(self, text: str, level: int = 1):
        """写小节标题。

        模板已经提供了「一、…」「二、…」这类章节标题，因此 ``level == 1`` 的调用
        被忽略（避免与模板标题重复）；``level >= 2``（如「1.1 题目描述」）写为
        12pt 加粗段落，与模板正文风格一致。
        """
        if level <= 1:
            return None
        para = self._new_para(size=SIZE_BODY, space_before=6.0, space_after=2.0)
        self._add_runs(para, text, SIZE_BODY + 0.5, bold=True)
        return para

    def p(self, text: str, indent: bool = True, size: float = SIZE_BODY):
        para = self._new_para(indent_first=indent, size=size)
        self._add_runs(para, text, size)
        return para

    def rich(self, segments: Sequence[tuple[str, bool]], indent: bool = True,
             size: float = SIZE_BODY):
        para = self._new_para(indent_first=indent, size=size)
        for text, bold in segments:
            run = para.add_run(text)
            self._style_run(run, size=size, bold=bold)
        return para

    def bullets(self, items: Iterable[str], size: float = SIZE_BODY) -> None:
        for item in items:
            para = self._new_para(size=size)
            para.paragraph_format.left_indent = Pt(size * 1.6)
            para.paragraph_format.first_line_indent = Pt(-size * 1.1)
            self._add_runs(para, "•  " + item, size)

    def numbered(self, items: Iterable[str], size: float = SIZE_BODY) -> None:
        for index, item in enumerate(items, start=1):
            para = self._new_para(size=size)
            para.paragraph_format.left_indent = Pt(size * 1.9)
            para.paragraph_format.first_line_indent = Pt(-size * 1.4)
            self._add_runs(para, f"{index}.  " + item, size)

    def code(self, text: str) -> None:
        """源码块：小五号（9pt）、固定行距 12 磅 —— 与模板附录要求一致。"""
        for line in text.rstrip("\n").split("\n"):
            para = self._new_para(exact_line_pt=CODE_LINE_PT, size=SIZE_CODE)
            para.paragraph_format.left_indent = Pt(0)
            para.paragraph_format.first_line_indent = Pt(0)
            run = para.add_run(line if line.strip() else " ")
            self._style_run(run, size=SIZE_CODE)

    def caption(self, text: str) -> None:
        para = self._new_para(size=SIZE_CAPTION, align="center",
                              space_before=4.0, space_after=2.0)
        run = para.add_run(text)
        self._style_run(run, size=SIZE_CAPTION, bold=True)

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]],
              caption: str | None = None, widths: Sequence[float] | None = None,
              size: float = SIZE_TABLE, align_center_cols: Sequence[int] = (),
              header: bool = True) -> None:
        """在当前单元格内插入表格（沿用模板的 Table Grid 边框风格）。"""
        if self._cell is None:
            raise RuntimeError("请先调用 open_section()")
        self._table_no += 1
        if caption:
            self.caption(f"表 {self._table_no}  {caption}")

        table = self._cell.add_table(rows=1 if header else 0, cols=len(headers))
        # 模板经 LibreOffice 转换后不含 'Table Grid' 样式，直接写边框 XML，
        # 与模板自带表格的 0.5pt 单线边框保持一致
        self._grid_borders(table)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False

        if header:
            for i, text in enumerate(headers):
                cell = table.rows[0].cells[i]
                cell.text = ""
                para = cell.paragraphs[0]
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                para.paragraph_format.line_spacing = 1.0
                run = para.add_run(str(text))
                self._style_run(run, size=size, bold=True)
                self._shade_cell(cell, "EDF2FB")

        for row in rows:
            cells = table.add_row().cells
            for i, text in enumerate(row):
                cell = cells[i]
                cell.text = ""
                para = cell.paragraphs[0]
                para.paragraph_format.line_spacing = 1.0
                para.alignment = (
                    WD_ALIGN_PARAGRAPH.CENTER if i in align_center_cols
                    else WD_ALIGN_PARAGRAPH.LEFT
                )
                run = para.add_run(str(text))
                self._style_run(run, size=size)

        if widths:
            for i, w in enumerate(widths):
                if i < len(table.columns):
                    table.columns[i].width = Cm(w)
            for row in table.rows:
                for i, w in enumerate(widths):
                    if i < len(row.cells):
                        row.cells[i].width = Cm(w)

        # 表格后紧跟着一个新段（python-docx 为满足单元格 schema 自动插入），
        # 把它压成 6pt 固定行距，避免模板的行网格把它撑成一大段空白
        trailing = self._cell.paragraphs[-1]
        if not trailing.text.strip():
            pf = trailing.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            pf.line_spacing = Pt(6)
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            for run in trailing.runs:
                run.font.size = Pt(6)

    @staticmethod
    def _grid_borders(table) -> None:
        """给表格加上全网格单线边框（0.5pt 黑色），与模板自带表格风格一致。"""
        tbl_pr = table._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            element = OxmlElement(f"w:{edge}")
            element.set(qn("w:val"), "single")
            element.set(qn("w:sz"), "4")
            element.set(qn("w:space"), "0")
            element.set(qn("w:color"), "000000")
            borders.append(element)
        tbl_pr.append(borders)

    @staticmethod
    def _shade_cell(cell, hex_fill: str) -> None:
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_fill)
        tc_pr.append(shd)

    def figure(self, image_path: str | Path, caption: str, width_cm: float = 15.0) -> None:
        """插入图片 + 图题（父模板正文宽度约 15.3 cm）。"""
        self._figure_no += 1
        para = self._new_para(align="center", space_before=4.0, space_after=0.0)
        para.add_run().add_picture(str(image_path), width=Cm(width_cm))
        cap = self._new_para(size=SIZE_CAPTION, align="center", space_after=6.0)
        run = cap.add_run(f"图 {self._figure_no}  {caption}")
        self._style_run(run, size=SIZE_CAPTION)

    def page_break(self) -> None:
        """模板的章节是表格行，本就需要连续排布，无需分页。"""

    # ---------------- 保存 ----------------

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        return path


__all__ = [
    "TemplateReport",
    "prepare_template",
    "convert_template",
    "normalize_template",
    "SECTION_ROWS",
]
