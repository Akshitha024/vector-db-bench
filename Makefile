.PHONY: install lint fmt typecheck test test-fast bench bench-fast bench-large report pdf clean

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -v -m "not slow"

test-fast:
	uv run pytest -v -m "not slow and not integration" -x

test-all:
	uv run pytest -v

# Default bench: 100k corpus x 10k queries x {10,50,100} k.
bench:
	uv run vdb bench --out-dir runs/latest \
	  --n-docs 100000 --n-queries 10000 --dim 128 \
	  --backends faiss_flat,faiss_hnsw,faiss_ivf_pq --ks 10,50,100

bench-fast:
	uv run vdb bench --out-dir runs/fast \
	  --n-docs 10000 --n-queries 1000 --dim 64 \
	  --backends faiss_flat,faiss_hnsw,faiss_ivf_pq --ks 10

bench-large:
	uv run vdb bench --out-dir runs/large \
	  --n-docs 500000 --n-queries 25000 --dim 256 \
	  --backends faiss_flat,faiss_hnsw,faiss_ivf_pq --ks 10,50,100

report:
	uv run vdb report --out-dir runs/latest

pdf:
	cd docs/_report && pandoc research_report.md \
	    -o ../research_report.pdf \
	    --pdf-engine=xelatex --toc --toc-depth=2 --number-sections \
	    -V geometry:margin=1in -V fontsize=11pt \
	    -V mainfont="Helvetica" -V monofont="Menlo" \
	    -V linkcolor=blue -V urlcolor=blue -V linestretch=1.15 \
	    || echo "pandoc + xelatex required"

clean:
	rm -rf runs/* .pytest_cache .mypy_cache .ruff_cache
