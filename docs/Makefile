# PorePy documentation Makefile
# =============================
#
# Run these commands from docs/ (or set the working directory to docs/).
#
# COMMON WORKFLOWS
# ----------------
#
#   make
#       Show Sphinx's list of available build targets.
#
#   make html
#       Build the HTML site in _build/html without building a PDF. The PDF
#       download link is omitted, so this works on machines without LaTeX.
#       Existing API stubs are reused; missing stubs are generated first.
#       Sphinx may reuse unchanged pages and doctrees from an earlier build.
#
#   make pdf
#       Generate API stubs if needed, build LaTeX sources, compile porepy.pdf,
#       and create porepy-latex-source.tar in _build/latex. Requires latexmk
#       and pdflatex. The target fails if either tool is absent, a build command
#       fails, or the PDF is not produced. After archiving, temporary LaTeX
#       output is removed, leaving the PDF and source archive in _build/latex.
#
#   make html INCLUDE_PDF=1
#       Build HTML with the PDF download link. A PDF must already exist; run
#       `make pdf` first. This adds the Sphinx tag `pdf`. The link in index.rst
#       is inside `.. only:: html and pdf`, so it appears only for an HTML
#       build that explicitly enables that tag. Plain `make html` does not.
#
#   make full
#       Regenerate all API stubs, build the PDF, then build HTML with the PDF
#       link. Use this for the complete site and deployment. It requires LaTeX
#       and stops if any command fails or the expected PDF is missing.
#
#   make clean
#       Delete _build/ entirely: API stubs, HTML, PDF/LaTeX files, and Sphinx
#       caches. The next build will start from scratch.
#
# API stubs are generated from ../src/porepy/ into _build/rst/. Ordinary HTML
# and PDF builds reuse them when _build/rst/porepy.rst exists. Run `make full`
# or `make clean` after adding, removing, or renaming source modules so the
# generated API pages reflect the new module tree.
#
# Sphinx warnings are printed but are non-fatal by default. This is intentional
# while API docstrings are incomplete. Passing `-W` makes warnings fatal, e.g.
# `make html SPHINXOPTS=-W`; the deployment workflow clears optional Sphinx
# arguments so warnings alone do not fail its build. Build errors remain fatal.
#
# CONFIGURATION AND OUTPUT
# ------------------------
# SPHINXBUILD and SPHINXAPIDOC select the Sphinx commands. SPHINXOPTS (or its
# alias O) passes extra options. Q controls command echoing: commands are shown
# by default; `make full Q=@` hides them. LC_ALL is set to C.UTF-8 for minimal
# Ubuntu containers. Generated outputs live under _build/: rst/ for API stubs,
# html/ for the site, latex/ for the PDF/archive, plus Sphinx-managed caches.
#
# SPHINXOPTS and SPHINXBUILD may also be set in the environment. Command-line
# assignments override these variables.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = .
POREPYDIR     = ../src/porepy/
BUILDDIR      = ./_build
APIDOCDIR     = $(BUILDDIR)/rst/
LATEXDIR      = $(BUILDDIR)/latex
PDFNAME       = porepy.pdf
TARNAME       = porepy-latex-source.tar

# Command echo prefix. Empty = every command is printed (default).
# Use `make full Q=@` for quiet output.
Q ?=

# Use the UTF-8 locale available in minimal Ubuntu containers.
export LC_ALL := C.UTF-8

# Put it first so that "make" without argument is like "make help".
help:
	$(Q)$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help apidoc apidoc-if-missing html pdf full clean

# Regenerate the API stubs only (no Sphinx build).
apidoc:
	$(Q)mkdir -p "$(APIDOCDIR)"
	$(Q)rm -f "$(APIDOCDIR)"*.rst
	$(Q)$(SPHINXAPIDOC) -f -e -M -d 1 --implicit-namespaces -o "$(APIDOCDIR)" "$(POREPYDIR)"
	$(Q)rm -f "$(APIDOCDIR)modules.rst"

# Generate API stubs on a fresh checkout, then reuse them for ordinary builds.
# Note: stubs are NOT refreshed if they exist. Run `make full` or `make clean`
# after adding, removing, or renaming modules.
apidoc-if-missing:
	$(Q)test -f "$(APIDOCDIR)porepy.rst" || $(MAKE) apidoc

# Regenerate stubs, build the PDF, then build HTML with a PDF download link.
full: apidoc
	$(Q)$(MAKE) pdf
	$(Q)$(MAKE) html INCLUDE_PDF=1

# HTML only. The PDF download link is included only when INCLUDE_PDF=1.
html: apidoc-if-missing
	$(Q)if [ "$(INCLUDE_PDF)" = "1" ]; then \
		test -f "$(LATEXDIR)/$(PDFNAME)" || { echo "ERROR: $(LATEXDIR)/$(PDFNAME) not found. Run 'make pdf' first."; exit 1; }; \
		$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -t pdf $(SPHINXOPTS) $(O); \
	else \
		$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O); \
	fi
	$(Q)test -f "$(BUILDDIR)/html/index.html" || { echo "ERROR: HTML index was not produced"; exit 1; }

# Build LaTeX sources, compile the PDF, and archive the LaTeX sources.
# Fails if the LaTeX toolchain is missing or the PDF was not produced.
pdf: apidoc-if-missing
	$(Q)command -v latexmk >/dev/null 2>&1 || { echo "ERROR: latexmk not found"; exit 1; }
	$(Q)command -v pdflatex >/dev/null 2>&1 || { echo "ERROR: pdflatex not found"; exit 1; }
	$(Q)$(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	$(Q)$(MAKE) -C "$(LATEXDIR)" all-pdf
	$(Q)test -f "$(LATEXDIR)/$(PDFNAME)" || { echo "ERROR: $(LATEXDIR)/$(PDFNAME) was not produced"; exit 1; }
	$(Q)tar -cf "$(LATEXDIR)/$(TARNAME)" \
		-C "$(BUILDDIR)" \
		--exclude=latex/$(PDFNAME) \
		--exclude=latex/$(TARNAME) \
		--exclude=latex/*.aux --exclude=latex/*.log \
		--exclude=latex/*.fls --exclude=latex/*.fdb_latexmk \
		--exclude=latex/*.idx --exclude=latex/*.ind --exclude=latex/*.ilg \
		--exclude=latex/*.out --exclude=latex/*.toc --exclude=latex/*.synctex.gz \
		latex
	$(Q)find "$(LATEXDIR)" -mindepth 1 -maxdepth 1 \
		! -name "$(PDFNAME)" ! -name "$(TARNAME)" \
		-exec rm -rf -- {} +

# Remove generated API pages, Sphinx caches, and builder output for a fresh build.
clean:
	$(Q)rm -rf "$(BUILDDIR)"
