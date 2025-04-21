# spbu-bachelor-thesis
Code and thesis document for my Bachelor's diploma on Kolmogorov-Arnold Networks with the implementation and reports included

## How to Use

Follow these steps to get started:
1. Review the prerequisites and install all required dependencies as described in the [Installation](#installation) section.
2. Navigate to the `src/notebooks/` directory and run the notebooks sequentially.
3. After executing all notebooks, you will find:
   - Datasets and embeddings that were used in experiments in `src/data/`;
   - A summary of the proposed models and their metrics (same as in the thesis) in `src/data/report.csv`;
   - Saved weights of the trained models in `src/models/`;

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
spbu-bachelor-thesis/  # Main project directory
├── .github/
│   └── workflows/
│       └── latex-compilation.yaml  # GitHub Actions workflow for automatic LaTeX compilation
├── src/           # Directory containing the project's source code
│   ├── ...
│   └── README.md  # Codebase structure description
├── .gitignore
├── LICENSE
├── README.md
├── presentation.pdf  # Compiled presentation for the thesis (in Russian)
├── presentation.tex  # Source code for the presentation
├── pyproject.toml    # Configuration file for the build system of the project's source code
├── thesis.pdf        # Compiled bachelor thesis document (in Russian)
├── thesis.sty        # Custom LaTeX style file for the thesis
└── thesis.tex        # Source code for the bachelor thesis document
```