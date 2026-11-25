"""Render proposal_*.md to PDF (Chrome headless). Normalizes lists for Python-Markdown."""
import re, pathlib, subprocess, markdown
from pypdf import PdfReader
CSS = """@page{size:letter;margin:0.75in}
body{font-family:'Times New Roman',serif;font-size:10.5pt;line-height:1.28;color:#111}
h1{font-size:16pt;margin:0 0 4pt} h2{font-size:12pt;margin:9pt 0 3pt;border-bottom:1px solid #999;break-after:avoid}
p{margin:5pt 0} ul,ol{margin:2pt 0 3pt 18pt;padding:0} li{margin:1pt 0} li>ul,li>ol{margin-top:1pt}
table{border-collapse:collapse;width:100%;font-size:8.8pt;line-height:1.15;margin:4pt 0} th,td{border:1px solid #bbb;padding:2pt 4pt;vertical-align:top} th{background:#eee} table.nums td,table.nums th+th{white-space:nowrap}
img{max-width:100%;display:block;margin:4pt auto} p:has(> img){break-after:avoid;margin-bottom:0} .refs{font-size:8.8pt;line-height:1.2;column-count:2;column-gap:14pt} .refs p{margin:0} .refs li{margin:0;break-inside:avoid} .refs .ref{display:block;break-inside:avoid}"""
LIST = re.compile(r"^(\s*)([-*]|\d+\.)\s")
def normalize(text):
    out, prev = [], ""
    for line in text.splitlines():
        m = LIST.match(line)
        if m:
            ind = len(m.group(1))
            line = " " * (4 * round(ind / 3) if ind else 0) + line.lstrip()   # 3-space -> 4-space nesting
            if prev.strip() and not LIST.match(prev) and (ind == 0 or not prev.startswith(" ")):
                out.append("")                                                # blank line before a list
        elif line.startswith("   ") and not line.startswith("    ") and out and out[-1] == "":
            line = "    " + line.lstrip()                                     # continuation paragraph inside item
        out.append(line); prev = line
    return "\n".join(out)
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for md in sorted(pathlib.Path(".").glob("*.md")):
    html = markdown.markdown(normalize(md.read_text()), extensions=["tables", "sane_lists"])
    html = re.sub(r"(<h2>References.*?</h2>)(.*)", r"\1<div class='refs'>\2</div>", html, flags=re.S)
    html = re.sub(r"(<div class='refs'>)(.*)(</div>)", lambda m: m.group(1) + "".join(
        f"<span class='ref'>{r.strip()}</span>" for r in re.sub(r"</?p>", "", m.group(2)).split("<br />")) + m.group(3), html, flags=re.S)
    html = re.sub(r"<table>(\s*<thead>\s*<tr>\s*<th[^>]*>Component</th>)",
                  r"<table style='table-layout:fixed'><colgroup><col style='width:44%'><col style='width:22%'><col style='width:34%'></colgroup>\1", html)
    html = re.sub(r"<td>(\d{1,2}/\d{1,2}\s*[–-]\s*\d{1,2}/\d{1,2})</td>", r"<td style='white-space:nowrap'>\1</td>", html)
    html = re.sub(r"<table>(\s*<thead>\s*<tr>\s*<th[^>]*>Study</th>)",
                  r"<table style='table-layout:fixed'><colgroup><col style='width:30%'><col style='width:70%'></colgroup>\1", html)
    html = re.sub(r"<table>(\s*<thead>\s*<tr>\s*<th[^>]*>Method</th>)", r"<table class='nums'>\1", html)   # results table: numbers on one line
    h = md.with_suffix(".html")
    h.write_text(f"<html><head><meta charset='utf-8'><title>{md.stem}</title><style>{CSS}</style></head><body>{html}</body></html>")
    pdf = md.with_suffix(".pdf").resolve()
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", f"file://{h.resolve()}"], capture_output=True, timeout=120)
    print(pdf.name, len(PdfReader(pdf).pages), "pages")
