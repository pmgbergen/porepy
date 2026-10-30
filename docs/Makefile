# You can set these variables from the command line, and also
# from the environment for the first two.
# $(O) is meant as a shortcut for $(SPHINXOPTS).
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = ./_build
POREPYDIR	  = ../src/porepy/
APIDOCDIR     = ./_build/rst/

# Use the UTF-8 locale available in minimal Ubuntu containers.
export LC_ALL := C.UTF-8

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help html pdf full clean apidoc-if-missing

# Regenerate API stubs, then build HTML and PDF documentation.
full:
	@mkdir -p "$(APIDOCDIR)"
	@rm -f "$(APIDOCDIR)"*.rst
	@$(SPHINXAPIDOC) -f -e -M -d 1 --implicit-namespaces -o "$(APIDOCDIR)" "$(POREPYDIR)"
	@rm -f "$(APIDOCDIR)modules.rst"
	@$(MAKE) pdf
	@$(MAKE) html

# Generate API stubs on a fresh checkout, then reuse them for ordinary builds.
apidoc-if-missing:
	@test -f "$(APIDOCDIR)porepy.rst" || $(MAKE) full

html: apidoc-if-missing
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build LaTeX sources and archive them. Compile the PDF when LaTeX is installed.
pdf: apidoc-if-missing
	@$(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@if command -v latexmk >/dev/null 2>&1 && command -v pdflatex >/dev/null 2>&1; then \
		$(MAKE) -C "$(BUILDDIR)/latex" all-pdf; \
	else \
		echo "latexmk or pdflatex not found; skipping PDF compilation."; \
	fi
	@tar -cf "$(BUILDDIR)/latex/porepy-latex-source.tar" \
		-C "$(BUILDDIR)" \
		--exclude=latex/porepy.pdf \
		--exclude=latex/porepy-latex-source.tar \
		--exclude=latex/*.aux --exclude=latex/*.log \
		--exclude=latex/*.fls --exclude=latex/*.fdb_latexmk \
		--exclude=latex/*.idx --exclude=latex/*.ind --exclude=latex/*.ilg \
		--exclude=latex/*.out --exclude=latex/*.toc --exclude=latex/*.synctex.gz \
		latex
	@find "$(BUILDDIR)/latex" -mindepth 1 -maxdepth 1 \
		! -name porepy.pdf ! -name porepy-latex-source.tar \
		-exec rm -rf -- {} +

# Remove generated API pages, Sphinx caches, and builder output for a fresh build.
clean:
	@rm -rf "$(BUILDDIR)"
