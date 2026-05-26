#!/usr/bin/env python3

from __future__ import annotations

import html
import re
import sys
from pathlib import Path


def render_inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
        escaped,
    )
    return escaped


def markdown_to_html(markdown: str) -> tuple[str, str]:
    lines = markdown.splitlines()
    blocks: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    in_code = False
    code_lines: list[str] = []
    title = "Project"

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(part.strip() for part in paragraph).strip()
            blocks.append(f"<p>{render_inline(text)}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            items = "".join(f"<li>{render_inline(item)}</li>" for item in list_items)
            blocks.append(f"<ul>{items}</ul>")
            list_items = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            blocks.append(
                "<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>"
            )
            code_lines = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            flush_list()
            if in_code:
                flush_code()
            in_code = not in_code
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            flush_list()
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if level == 1:
                title = text
            level = min(level, 3)
            blocks.append(f"<h{level}>{render_inline(text)}</h{level}>")
            continue

        if stripped.startswith(("- ", "* ")):
            flush_paragraph()
            list_items.append(stripped[2:].strip())
            continue

        paragraph.append(stripped)

    flush_paragraph()
    flush_list()
    if in_code:
        flush_code()

    if blocks and blocks[0].startswith("<h1>"):
        blocks = blocks[1:]

    return title, "\n".join(blocks)


def build_page(readme_path: Path, output_path: Path) -> None:
    title, body = markdown_to_html(readme_path.read_text())
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0c1020;
      --panel: #121933;
      --panel-2: #0f152b;
      --fg: #eef3ff;
      --muted: #aab7d4;
      --accent: #9ad1ff;
      --border: rgba(154, 209, 255, 0.18);
      --code: rgba(154, 209, 255, 0.12);
      --link: #8ed0ff;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.3);
    }}
    @media (prefers-color-scheme: light) {{
      :root {{
        --bg: #f5f8ff;
        --panel: #ffffff;
        --panel-2: #f1f5ff;
        --fg: #14203a;
        --muted: #51617f;
        --accent: #1e5eff;
        --border: rgba(30, 94, 255, 0.14);
        --code: rgba(30, 94, 255, 0.08);
        --link: #1e5eff;
        --shadow: 0 24px 60px rgba(30, 94, 255, 0.08);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, var(--panel-2), var(--bg) 55%);
      color: var(--fg);
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 2rem 1.25rem 4rem;
    }}
    .hero, .content {{
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .hero {{
      padding: 1.5rem;
      margin-bottom: 1.25rem;
    }}
    .hero h1 {{
      margin: 0 0 0.5rem;
      font-size: clamp(2rem, 5vw, 3.25rem);
      line-height: 1.02;
    }}
    .hero p {{
      margin: 0.4rem 0;
      color: var(--muted);
      max-width: 60rem;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-top: 1.25rem;
    }}
    .actions a {{
      padding: 0.8rem 1.1rem;
      border-radius: 999px;
      border: 1px solid var(--border);
      text-decoration: none;
      font-weight: 700;
      color: var(--fg);
      background: transparent;
    }}
    .actions a.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }}
    .content {{
      padding: 1.5rem;
    }}
    h1, h2, h3 {{
      letter-spacing: -0.03em;
    }}
    h2 {{
      margin-top: 1.75rem;
      margin-bottom: 0.5rem;
      font-size: 1.35rem;
    }}
    h3 {{
      margin-top: 1.2rem;
      margin-bottom: 0.35rem;
      font-size: 1.05rem;
    }}
    p, li {{
      line-height: 1.7;
      color: var(--fg);
    }}
    ul {{
      padding-left: 1.25rem;
    }}
    pre {{
      margin: 1rem 0;
      padding: 1rem;
      overflow-x: auto;
      border-radius: 16px;
      background: var(--panel-2);
      border: 1px solid var(--border);
    }}
    code {{
      font-family: "SFMono-Regular", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: var(--code);
      border-radius: 6px;
      padding: 0.12rem 0.3rem;
    }}
    pre code {{
      padding: 0;
      background: transparent;
    }}
    a {{
      color: var(--link);
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p>Repository landing page built from <code>README.md</code>, with the latest compiled report published alongside it.</p>
      <div class="actions">
        <a class="primary" href="report.pdf">Open report PDF</a>
        <a href="report.pdf" download>Download PDF</a>
        <a href="https://github.com/kovacoj/attention">View repository</a>
      </div>
    </section>
    <section class="content">
{body}
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(page)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: build_pages.py README.md output/index.html", file=sys.stderr)
        return 1

    readme_path = Path(argv[1])
    output_path = Path(argv[2])
    build_page(readme_path, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
