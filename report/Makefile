# Smart Makefile that detects biblatex vs natbib
MAIN = main
TEX = pdflatex

all: clean compile

compile:
	$(TEX) -synctex=1 -interaction=nonstopmode $(MAIN)
	@if grep -q "addbibresource\|printbibliography" $(MAIN).tex; then \
		echo "Detected biblatex → Running biber"; \
		biber $(MAIN); \
	else \
		echo "Detected natbib → Running bibtex"; \
		bibtex $(MAIN); \
	fi
	$(TEX) -synctex=1 -interaction=nonstopmode $(MAIN)
	$(TEX) -synctex=1 -interaction=nonstopmode $(MAIN)

clean:
	rm -f *.aux *.log *.toc *.out *.bbl *.blg *.synctex.gz *.bcf *.run.xml

view:
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null

.PHONY: all clean compile view
