PY = uv run
NOTEBOOKS = 01_exposure_bias 02_relmf 03_offline_evaluation

.PHONY: all env data test grid kernel notebooks clean

all: env data test grid notebooks

env:
	uv sync

data:
	$(PY) python src/download.py

test:
	$(PY) python tests/test_core.py

# About 5 hours in one process on a laptop CPU. Resumable: finished fits are skipped on restart.
grid: data
	$(PY) python src/run_grid.py

# Jupyter kernelspec for the project venv, installed inside .venv so jupytext --execute finds it.
kernel:
	@ls .venv/share/jupyter/kernels/clicks-are-pu >/dev/null 2>&1 || $(PY) python -m ipykernel install --sys-prefix --name clicks-are-pu --display-name "clicks-are-pu"

# Each notebook is a jupytext percent-format .py (diff-able), executed into the committed .ipynb.
# Notebooks read data/raw and data/processed and never train, so they need make grid first.
notebooks: kernel
	for nb in $(NOTEBOOKS); do $(PY) jupytext --to ipynb --execute notebooks/$$nb.py || exit 1; done

clean:
	rm -rf data/processed results/*.md results/*.png notebooks/*.ipynb
