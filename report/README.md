# Report Template Notes

This directory uses the TeXtured thesis template as a base, but it has been adapted for a short course report.

## Main entry file

Build:

```sh
latexmk paper
```

from inside the `report/` directory.

The main file is:

- `paper.tex`

## Structure

- `paper.tex`: report entry point using the TeXtured preamble
- `frontmatter/report-title.tex`: compact first page for the report
- `chapters/Report.tex`: current report body
- `preamble/data.tex`: document metadata and abstract

## GitHub Pages

The repository root contains the GitHub Actions workflow that builds this report and publishes a Pages site with the compiled PDF.

## Notes

- The original `thesis.tex` entry point is still present as an upstream reference, but it is no longer the primary build target for this project.
- The current setup intentionally avoids thesis-only front matter such as the formal title page, declaration, and dedication.
