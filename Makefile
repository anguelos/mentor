VENV_BIN   = /home/anguelos_ro/venvs/p312/bin
PYTHON     = $(VENV_BIN)/python
PYTEST     = $(VENV_BIN)/python -m pytest
SETUP      = $(VENV_BIN)/python setup.py

export PATH := $(VENV_BIN):$(PATH)

.PHONY: test unittest build clean docs docs_single docs_pdf

test:
	$(PYTEST) tests/ -q

unittest:
	$(PYTEST) tests/unit_testing/ -q --cov=mentor --cov-report=term-missing

build:
	$(SETUP) sdist

clean:
	find . -path ./tmp -prune -o \( \
	    -name '__pycache__'      \
	    -o -name '*.pyc'         \
	    -o -name '*.pyo'         \
	    -o -name '*.pyd'         \
	    -o -name '*.egg-info'    \
	    -o -name '.pytest_cache' \
	\) -print -exec rm -rf {} +
	rm -rf dist/ build/ $(DOCBUILD)/

SPHINXBUILD = $(VENV_BIN)/sphinx-build
SPHINXOPTS  =
DOCSRC      = docs
DOCBUILD    = docs/_build

docs:
	$(SPHINXBUILD) -b html     $(SPHINXOPTS) $(DOCSRC) $(DOCBUILD)/html
	@echo "HTML docs: $(DOCBUILD)/html/index.html"

docs_single:
	$(SPHINXBUILD) -b singlehtml $(SPHINXOPTS) $(DOCSRC) $(DOCBUILD)/singlehtml
	@echo "Single-page HTML: $(DOCBUILD)/singlehtml/index.html"

docs_pdf:
	$(SPHINXBUILD) -b latex    $(SPHINXOPTS) $(DOCSRC) $(DOCBUILD)/latex
	$(MAKE) -C $(DOCBUILD)/latex all-pdf
	@echo "PDF: $(DOCBUILD)/latex/mentor.pdf"
