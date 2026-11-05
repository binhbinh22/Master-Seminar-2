# Seminar 2 academic presentation

25-slide English Beamer deck based on the latest report, using the requested HUST theme.

- `main.pdf`: presentation PDF, 16:9.
- `main.tex`: editable source, including source notes for substantive slides.
- `figures/`: existing report evidence figures; no models retrained.
- `*.sty`, `assets/`: upstream HUST theme files, retained without modifications.

Build inside this directory with XeLaTeX twice:

```sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Set Overleaf compiler to XeLaTeX. Font: Arial (theme option); install it locally if absent. Slide IDs and names come from the report cover. The source retains proper Vietnamese diacritics.

## Provenance

Theme: https://github.com/k1m1nonawa4869/HUST-Beamer-Theme
Pinned upstream commit: `15102a28bdd3ec1fde97538bb7567e277862ba1c`.
Upstream theme copyright/ownership belongs to its authors. No LICENSE file was present in the fetched repository; this attribution does not assert a redistribution license. Brand artwork is retained from the user-requested template.

Content: `../report_rebuilt/main.tex`, its 19-entry bibliography and stored result figures. The deck retains unequal tuning budgets, selection dependence and limited replication as scientific limitations. Official benchmark scores have a distinct evaluation protocol.

The source footers and Beamer notes identify supporting report sections/artifacts. The final slide gives selected references. `validation.json` records PDF checks; rendering intermediates stay under ignored `build/`.
