# glearning

Graph-based semi-supervised learning experiments (Laplace learning on k-NN graphs).

## Setup

```bash
conda env create -f environment.yml
conda activate glearning
pip install -e .
```

## Layout

```
src/glearning/     the package: graph construction, Laplacians, Laplace learning, checks
experiments/       scripts that run experiments and make figures
tests/             pytest tests for the package
plots/             figures written by the experiments (not committed)
```

## Running

```bash
python experiments/two_moons.py
python experiments/agreement_bowl.py
pytest
```
