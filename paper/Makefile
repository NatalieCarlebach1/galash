PAPER = main
LATEXMK = latexmk
PDFLATEX = pdflatex
BIBTEX = bibtex

.PHONY: all pdf clean veryclean watch

all: pdf

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode $(PAPER).tex

# Run pdflatex directly (useful for debugging)
manual:
	$(PDFLATEX) -interaction=nonstopmode $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(PDFLATEX) -interaction=nonstopmode $(PAPER).tex
	$(PDFLATEX) -interaction=nonstopmode $(PAPER).tex

# Auto-rebuild on save
watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(PAPER).tex

clean:
	$(LATEXMK) -c
	rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz
	rm -f sections/*.aux

veryclean: clean
	rm -f $(PAPER).pdf
