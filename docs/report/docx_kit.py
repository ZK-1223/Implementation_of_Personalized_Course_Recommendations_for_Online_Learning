"""Word 报告渲染工具箱。

按《智能应用系统课程设计报告-2025 模版》的版式要求（A4、上下 2.54cm、左右 3.17cm、
中文宋体 / 英文 Times New Roman、正文五号 10.5pt、附录源码小五号行距 12 磅）提供
一套简洁的构建接口，供 :mod:`docs.report.build_report` 组装报告正文。

设计要点：
    * 所有 run 都显式设置 ``w:eastAsia`` 字体，避免中文回退到默认西文字体；
    * 标题层级用真实 Heading 样式，便于插入自动目录（TOC 域）；
    * 图片、表格自动居中并带题注，题注编号由调用方给出（报告里统一「图 x-y」「表 x-y」）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# ---- 版式常量（与模版一致）----
PAGE_WIDTH_PT = 595.3      # A4 宽
PAGE_HEIGHT_PT = 841.9     # A4 高
MARGIN_TB_PT = 72.0        # 上下 2.54 cm
MARGIN_LR_PT = 90.0        # 左右 3.17 cm

FONT_CN = "宋体"
FONT_EN = "Times New Roman"
BODY_SIZE = 10.5           # 五号
CODE_SIZE = 9.0            # 小五
CODE_LINE_PT = 12.0        # 附录源码行距 12 磅

HEADING_SIZES = {1: 14.0, 2: 12.0, 3: 10.5}   # 四号 / 小四 / 五号


def _set_run_font(run, size: float, bold: bool = False, cn: str = FONT_CN,
                  en: str = FONT_EN, color: RGBColor | None = None) -> None:
    """同时设置西文与东亚字体，确保中文不回退。"""
    run.font.name = en
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)
    rfonts.set(qn("w:cs"), en)


def _shade_cell(cell, hex_fill: str) -> None:
    """给表格单元格加底纹（如表头浅灰）。"""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def _add_field(paragraph, instr: str) -> None:
    """向段落插入 Word 域（用于目录、页码）。"""
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "（目录将在文档打开时自动生成；若未显示，请右键 → 更新域，或按 F9）"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr_el)
    run._r.append(fld_sep)
    run._r.append(placeholder)
    run._r.append(fld_end)
    _set_run_font(run, BODY_SIZE)


def _add_runs(paragraph, text: str, size: float, bold: bool = False) -> None:
    """向段落添加文本，支持 ``**加粗**`` 行内标记。

    报告正文里大量使用 ``**关键词**`` 强调，若不做解析会把星号原样打印出来。
    这里用最简的标记解析：按 ``**`` 切分后交替设置加粗。
    """
    segments = text.split("**")
    for index, seg in enumerate(segments):
        if not seg:
            continue
        run = paragraph.add_run(seg)
        _set_run_font(run, size, bold=(bold or index % 2 == 1))


class ReportBuilder:
    """课程设计报告构建器。"""

    def __init__(self) -> None:
        self.doc = Document()
        self._setup_page()
        self._setup_default_style()
        self._setup_footer_page_number()
        self._enable_update_fields()
        self._figure_no = 0
        self._table_no = 0

    # ========================================================
    # 版式初始化
    # ========================================================
    def _setup_page(self) -> None:
        for section in self.doc.sections:
            section.page_width = Pt(PAGE_WIDTH_PT)
            section.page_height = Pt(PAGE_HEIGHT_PT)
            section.top_margin = Pt(MARGIN_TB_PT)
            section.bottom_margin = Pt(MARGIN_TB_PT)
            section.left_margin = Pt(MARGIN_LR_PT)
            section.right_margin = Pt(MARGIN_LR_PT)

    def _setup_default_style(self) -> None:
        style = self.doc.styles["Normal"]
        style.font.name = FONT_EN
        style.font.size = Pt(BODY_SIZE)
        rpr = style.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        rfonts.set(qn("w:ascii"), FONT_EN)
        rfonts.set(qn("w:hAnsi"), FONT_EN)
        rfonts.set(qn("w:eastAsia"), FONT_CN)
        pf = style.paragraph_format
        pf.line_spacing = 1.5
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)

        # 标题样式统一字体，颜色改为黑色（Word 默认蓝）
        for level, size in HEADING_SIZES.items():
            name = f"Heading {level}"
            try:
                hstyle = self.doc.styles[name]
            except KeyError:  # pragma: no cover
                continue
            hstyle.font.name = FONT_EN
            hstyle.font.size = Pt(size)
            hstyle.font.bold = True
            hstyle.font.color.rgb = RGBColor(0, 0, 0)
            hrpr = hstyle.element.get_or_add_rPr()
            hrfonts = hrpr.find(qn("w:rFonts"))
            if hrfonts is None:
                hrfonts = OxmlElement("w:rFonts")
                hrpr.append(hrfonts)
            hrfonts.set(qn("w:ascii"), FONT_EN)
            hrfonts.set(qn("w:hAnsi"), FONT_EN)
            hrfonts.set(qn("w:eastAsia"), FONT_CN)
            hstyle.paragraph_format.space_before = Pt(10 if level == 1 else 6)
            hstyle.paragraph_format.space_after = Pt(6 if level == 1 else 4)
            hstyle.paragraph_format.line_spacing = 1.3

    def _setup_footer_page_number(self) -> None:
        for section in self.doc.sections:
            footer = section.footer
            para = footer.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run()
            _set_run_font(run, BODY_SIZE)
            fld_begin = OxmlElement("w:fldChar")
            fld_begin.set(qn("w:fldCharType"), "begin")
            instr_el = OxmlElement("w:instrText")
            instr_el.set(qn("xml:space"), "preserve")
            instr_el.text = "PAGE"
            fld_end = OxmlElement("w:fldChar")
            fld_end.set(qn("w:fldCharType"), "end")
            run._r.append(fld_begin)
            run._r.append(instr_el)
            run._r.append(fld_end)

    def _enable_update_fields(self) -> None:
        """让 Word 打开文档时自动更新目录与页码域。"""
        settings = self.doc.settings.element
        el = settings.find(qn("w:updateFields"))
        if el is None:
            el = OxmlElement("w:updateFields")
            settings.append(el)
        el.set(qn("w:val"), "true")

    # ========================================================
    # 基础块
    # ========================================================
    def raw(self, text: str = "", size: float = BODY_SIZE, bold: bool = False,
            align: str = "left", space_before: float = 0, space_after: float = 0,
            indent_first: bool = False, line_spacing: float = 1.5):
        para = self.doc.add_paragraph()
        para.alignment = _ALIGN[align]
        pf = para.paragraph_format
        pf.space_before = Pt(space_before)
        pf.space_after = Pt(space_after)
        pf.line_spacing = line_spacing
        if indent_first:
            pf.first_line_indent = Pt(size * 2)
        if text:
            run = para.add_run(text)
            _set_run_font(run, size, bold)
        return para

    def h(self, text: str, level: int = 1):
        """标题。level 1 = 一级章节（四号加粗）。"""
        para = self.doc.add_heading(level=level)
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = para.add_run(text)
        _set_run_font(run, HEADING_SIZES.get(level, BODY_SIZE), bold=True)
        return para

    def p(self, text: str, indent: bool = True, size: float = BODY_SIZE):
        """正文段落，默认首行缩进 2 字符；支持 ``**加粗**`` 行内标记。"""
        para = self.doc.add_paragraph()
        normalize = para.paragraph_format
        normalize.line_spacing = 1.5
        normalize.space_after = Pt(0)
        if indent:
            normalize.first_line_indent = Pt(size * 2)
        _add_runs(para, text, size)
        return para

    def rich(self, segments: Sequence[tuple[str, bool]], indent: bool = True,
             size: float = BODY_SIZE):
        """一段内混排粗体：segments = [("文字", 是否加粗), ...]。"""
        para = self.doc.add_paragraph()
        pf = para.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(0)
        if indent:
            pf.first_line_indent = Pt(size * 2)
        for text, bold in segments:
            run = para.add_run(text)
            _set_run_font(run, size, bold)
        return para

    def bullets(self, items: Iterable[str], size: float = BODY_SIZE) -> None:
        for item in items:
            para = self.doc.add_paragraph()
            pf = para.paragraph_format
            pf.line_spacing = 1.5
            pf.space_after = Pt(0)
            pf.left_indent = Pt(size * 1.6)
            pf.first_line_indent = Pt(-size * 1.1)
            _add_runs(para, "•  " + item, size)

    def numbered(self, items: Iterable[str], size: float = BODY_SIZE) -> None:
        """有序列表（手写序号，避免 Word 列表样式跨章连续编号）。"""
        for index, item in enumerate(items, start=1):
            para = self.doc.add_paragraph()
            pf = para.paragraph_format
            pf.line_spacing = 1.5
            pf.space_after = Pt(0)
            pf.left_indent = Pt(size * 1.9)
            pf.first_line_indent = Pt(-size * 1.4)
            _add_runs(para, f"{index}.  " + item, size)

    def code(self, text: str) -> None:
        """源码块：小五号、行距 12 磅（模版对附录的要求）。"""
        for line in text.rstrip("\n").split("\n"):
            para = self.doc.add_paragraph()
            pf = para.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            pf.line_spacing = Pt(CODE_LINE_PT)
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            pf.left_indent = Pt(0)
            run = para.add_run(line if line else " ")
            _set_run_font(run, CODE_SIZE, en="Consolas", cn=FONT_CN)

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]],
              caption: str | None = None, widths: Sequence[float] | None = None,
              size: float = 9.0, align_center_cols: Sequence[int] = (),
              header: bool = True) -> None:
        """数据表，带题注（题注在表上方）。

        Args:
            header: 是否渲染表头行。封面信息表等无表头的表格传 ``False``。
        """
        self._table_no += 1
        if caption:
            self.caption(f"表 {self._table_no}  {caption}")
        if header:
            table = self.doc.add_table(rows=1, cols=len(headers))
        else:
            table = self.doc.add_table(rows=0, cols=len(headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        if header:
            hdr = table.rows[0].cells
            for i, text in enumerate(headers):
                cell = hdr[i]
                cell.text = ""
                para = cell.paragraphs[0]
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                para.paragraph_format.line_spacing = 1.0
                run = para.add_run(str(text))
                _set_run_font(run, size, bold=True)
                _shade_cell(cell, "EDF2FB")
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
                _set_run_font(run, size)
        if widths:
            table.autofit = False
            for i, w in enumerate(widths):
                if i < len(table.columns):
                    table.columns[i].width = Cm(w)
            for row in table.rows:
                for i, w in enumerate(widths):
                    if i < len(row.cells):
                        row.cells[i].width = Cm(w)
        self.raw("", size=6)

    def figure(self, image_path: str | Path, caption: str, width_cm: float = 14.5) -> None:
        """插入图片（居中）+ 图题（图下方）。"""
        self._figure_no += 1
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(6)
        para.paragraph_format.space_after = Pt(2)
        para.add_run().add_picture(str(image_path), width=Cm(width_cm))
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(8)
        run = cap.add_run(f"图 {self._figure_no}  {caption}")
        _set_run_font(run, 9.0)

    def caption(self, text: str) -> None:
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(8)
        para.paragraph_format.space_after = Pt(3)
        run = para.add_run(text)
        _set_run_font(run, 9.0, bold=True)

    def page_break(self) -> None:
        self.doc.add_page_break()

    def toc(self, levels: str = "1-2") -> None:
        """插入自动目录域（Word 打开时自动更新）。"""
        para = self.doc.add_paragraph()
        _add_field(para, f'TOC \\o "{levels}" \\h \\z \\u')

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        return path


_ALIGN = {
    "left": WD_ALIGN_PARAGRAPH.LEFT,
    "center": WD_ALIGN_PARAGRAPH.CENTER,
    "right": WD_ALIGN_PARAGRAPH.RIGHT,
    "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
}

__all__ = ["ReportBuilder"]
