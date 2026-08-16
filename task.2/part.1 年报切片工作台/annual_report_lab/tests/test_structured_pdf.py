# -*- coding: utf-8 -*-
"""结构化 PDF 切片自测：验证首尾裁切后仍保留图形/表格与正文。"""
from __future__ import annotations
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pdf_slice import save_pdf_section


def main() -> int:
    try:
        import fitz
    except Exception:
        print("SKIP: 未安装 PyMuPDF")
        return 0
    with tempfile.TemporaryDirectory(prefix="arlab_pdf_") as d:
        src = os.path.join(d, "report.pdf")
        out = os.path.join(d, "000750.SZ_国海证券_2025_管理层讨论与分析.pdf")
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((72, 80), "Previous Section")
        p.insert_text((72, 150), "Management Discussion and Analysis")
        p.draw_rect(fitz.Rect(72, 220, 420, 360))
        p.insert_text((90, 250), "Table content 123")
        p2 = doc.new_page()
        p2.insert_text((72, 80), "Future outlook and operating analysis")
        p2.draw_circle((200, 240), 50)
        p2.insert_text((72, 500), "Corporate Governance")
        doc.save(src); doc.close()
        info = save_pdf_section(src, out, {
            "start_heading": "Management Discussion and Analysis",
            "end_heading": "Corporate Governance", "start_page": 0, "end_page": 1,
        })
        assert info["ok"] and os.path.isfile(out), info
        cut = fitz.open(out)
        text = "\n".join(p.get_text() for p in cut)
        drawings = sum(len(p.get_drawings()) for p in cut)
        cut.close()
        assert "Table content 123" in text
        assert "Corporate Governance" not in text
        assert drawings >= 1
        print("PASS:", info["mode"], os.path.basename(out))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
