# spbu-bachelor-thesis
Code and thesis document for my Bachelor’s thesis at St. Petersburg State University (SPbU)

## How to Use

Follow these steps to get started:
1. Read the thesis document (in Russian) located at `./reports/thesis.pdf`.
2. Review the prerequisites and install all required dependencies as described in the [Installation](#installation) section.
3. Navigate to the `./notebooks/` directory and run the notebooks sequentially.
4. After executing all notebooks, you will find:
   - Datasets and embeddings that were used in experiments in `./data/`;
   - Saved weights of the trained models in `./models/`;
   - A summary of the proposed models and their metrics (same as in the thesis) in `./reports/models_summary.csv`;

## Installation

**Pre-requisites:**

```bash
Python 3.9 or higher
pip
venv
```

**This project can be installed using pip as follows:**

```bash
git clone https://github.com/trxxxxkov/spbu-bachelor-thesis.git
cd spbu-bachelor-thesis
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

## Project Structure
```bash
spbu-bachelor-thesis/
├── .github/
│   └── workflows/
│       └── latex-compilation.yaml  #  CI workflow that auto-compiles LaTeX sources into PDFs
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Clean raw datasets and write embeddings for later use
│   ├── 02_proposed_models.ipynb     # Define and train the KAN-based model variants
│   ├── 03_continual_learning.ipynb  # Benchmark models under continual-learning setup
│   └── 04_report_preparation.ipynb  # Summarize results in plots and tables
├── reports/  
│   ├── tex/
│   │   ├── figures/
│   │   │   └── ...  # All graphics (plots, diagrams, logos) referenced in LaTeX sources
│   │   ├── beamerbasetitle.sty  # Redefining of the Beamer title page
│   │   ├── beamerthemespbu.sty  # SPbU-branded Beamer theme
│   │   ├── presentation.tex     # Beamer source for the defence slides
│   │   ├── references.bib       # BibTeX database for the thesis
│   │   ├── thesis.sty           # Custom SPbU-compliant LaTeX style
│   │   └── thesis.tex           # Main bachelor-thesis LaTeX document
│   ├── models_summary.csv  # CSV log of all experiment metrics across models
│   ├── presentation.pdf    # Pre-built PDF of the defence slides
│   └── thesis.pdf          # Pre-built PDF of the full thesis
├── spbu_bachelor_thesis/
│   ├── nn/
│   │   ├── ensemble_kan.py
│   │   ├── kan.py              # Kolmogorov-Arnold Network layer and model
│   │   ├── mlp.py              # Baseline multilayer perceptron implementation
│   │   ├── regularized_kan.py  # KAN variant with explicit weight regularisation
│   │   └── rehearsal_kan.py    # KAN augmented with SOM layer for CL studies
│   ├── datasets.py          # Dataset loaders and preprocessing pipelines
│   ├── global_constants.py  # Paths, seeds and hyper-params
│   ├── metrics.py           # Custom CL and metrics and MPC accuracy implementation
│   ├── train_test.py        # Implementation of train/test loops for CL experiments
│   └── visualization.py     # Utilities for plotting training progress graphics
├── .gitignore
├── LICENSE          
├── pyproject.toml  # Build-system and dependency specification
└── README.md        
```

The thesis template was adapted from [itonik/spbu_diploma](https://github.com/itonik/spbu_diploma),
and the presentation template was adapted from [spbu-se/report_presentation_template](https://github.com/spbu-se/report_presentation_template).