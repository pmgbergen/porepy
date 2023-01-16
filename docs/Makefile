# You can set these variables from the command line, and also
# from the environment for the first two.
# $(O) is meant as a shortcut for $(SPHINXOPTS).
SPHINXOPTS    ?= "-E"
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = .
POREPYDIR	  = ../src/porepy/
POREPYDOCSRC  = ./docsrc/porepy/

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html latex complete

# complete target first, to build rst structure
complete:
	@rm -f "$(POREPYDOCSRC)*.*"
	@$(SPHINXAPIDOC) -f -e -M -d 1 --implicit-namespaces -o "$(POREPYDOCSRC)" "$(POREPYDIR)"
	@rm -f "$(POREPYDOCSRC)modules.rst"

# html target
html:
	@rm -rf ./doctrees/
	@rm -rf ./html/
	@rm -rf ./latex/
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# latex target
latex:
	@rm -rf ./doctrees/
	@rm -rf ./html/
	@rm -rf ./latex/
	@$(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
