"""
Convert pytest text output to a styled HTML report.

Called automatically by: make test
Reads:   reports/figures/pytest_section1.txt
Writes:  reports/figures/pytest_section1.html

Open the HTML in a browser for a clean screenshot,
or link it directly in PHASE3.md as evidence.
"""

from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "reports" / "figures"
TXT_PATH = FIGURES_DIR / "pytest_section1.txt"
HTML_PATH = FIGURES_DIR / "pytest_section1.html"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def colorize(text: str) -> str:
    safe = html.escape(text)
    replacements = [
        ("PASSED", '<span class="passed">PASSED</span>'),
        ("FAILED", '<span class="failed">FAILED</span>'),
        ("ERROR", '<span class="error">ERROR</span>'),
        ("SKIPPED", '<span class="skipped">SKIPPED</span>'),
        ("passed", '<span class="passed">passed</span>'),
        ("failed", '<span class="failed">failed</span>'),
        ("warning", '<span class="warning">warning</span>'),
    ]
    for word, span in replacements:
        safe = safe.replace(word, span)
    return safe


def build_html(output: str) -> str:
    failed = "failed" in output.lower() and "0 failed" not in output.lower()
    status = "FAILED" if failed else "PASSED"
    color = "#d93025" if failed else "#2d9e4f"
    lines = colorize(output).split("\n")
    rows = "\n".join(f'<div class="line">{ln}</div>' for ln in lines)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>pytest — Section 1 — {TIMESTAMP}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Menlo','Consolas','Monaco',monospace;
      background: #1e1e1e; color: #d4d4d4;
      padding: 28px 32px; font-size: 13px; line-height: 1.65;
    }}
    header {{
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 16px; padding-bottom: 12px;
      border-bottom: 1px solid #333;
    }}
    h1   {{ font-size: 14px; color: #9cdcfe; font-weight: 600; }}
    .badge {{
      font-size: 12px; font-weight: 700; padding: 3px 12px;
      border-radius: 4px; color: #fff; background: {color};
    }}
    .meta  {{ font-size: 11px; color: #666; margin-bottom: 16px; }}
    .output {{ background: #141414; border-radius: 6px;
               padding: 16px 20px; overflow-x: auto; }}
    .line   {{ white-space: pre; min-height: 1.65em; }}
    .passed  {{ color: #2d9e4f; font-weight: 600; }}
    .failed  {{ color: #d93025; font-weight: 600; }}
    .error   {{ color: #e37400; font-weight: 600; }}
    .skipped {{ color: #888;    font-weight: 600; }}
    .warning {{ color: #e37400; }}
  </style>
</head>
<body>
  <header>
    <h1>pytest tests/ -v &nbsp;·&nbsp; Section 1 CI Evidence</h1>
    <span class="badge">{status}</span>
  </header>
  <p class="meta">
    Generated: {TIMESTAMP} &nbsp;·&nbsp;
    mlops_frauddetection &nbsp;·&nbsp;
    Python {sys.version.split()[0]}
  </p>
  <div class="output">{rows}</div>
</body>
</html>"""


def main() -> None:
    if not TXT_PATH.exists():
        print(f"✗ Not found: {TXT_PATH}")
        print("  Run: make test   (which runs pytest and tee's to the txt file)")
        sys.exit(1)

    output = TXT_PATH.read_text(encoding="utf-8")
    HTML_PATH.write_text(build_html(output), encoding="utf-8")
    print(f"✓ HTML report → {HTML_PATH}")

    import webbrowser

    webbrowser.open(HTML_PATH.as_uri())


if __name__ == "__main__":
    main()
