# Makefile para compilar el libro TextReIDNet
# Uso: make         (compila el PDF)
#      make clean   (limpia archivos auxiliares)
#      make view    (abre el PDF con el visor por defecto)

MAIN    = main
LATEX   = pdflatex -shell-escape -interaction=nonstopmode
BIBTEX  = bibtex
PDFLATEX_FLAGS = -shell-escape -interaction=nonstopmode

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex chapters/*.tex figures/*.tikz refs.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log
	rm -f $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot
	rm -f $(MAIN).lol $(MAIN).fls $(MAIN).fdb_latexmk
	rm -f $(MAIN).synctex.gz
	rm -rf build/

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null
