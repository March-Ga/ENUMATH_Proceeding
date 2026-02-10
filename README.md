# Comparison of Fully Coupled Immersed Boundary and Domain Schemes for Fluid–Structure Interaction

This repository contains the **code, numerical experiments, and supporting material** for the preceeding paper:

> **Comparison of Fully Coupled Immersed Boundary and Domain Schemes for  
> Fluid–Structure Interaction with Lagrange Multipliers**

The goal of this repository is to ensure **transparency, reproducibility, and reuse** of the methods and results presented in the paper.

---

## Repository structure

```text
.
├── paper/          # Manuscript PDF
├── code/           # Source code for IB and ID schemes
│   ├── utils.py
│   ├── fsi_solvers.py
│   └── main.py
├── results/        # Figures and tables shown in the manuscript
├── LICENSE
├── CITATION.cff
└── README.md
```
---

## Code section

The code section contains 3 files:
- `utils.py` – auxiliary functions for fluid, solid, and coupling operators  
- `fsi_solvers.py` – fully coupled IB and ID solvers  
- `main.py` – executable script to run the numerical experiments
  
---
## Instructions for Reproducing the Results

To reproduce the simulations and results presented in this repository, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```
### 2. Create a Python virtual environment
```bash
python -m venv venv
```
#### Activate the environment:

- Linux / macOS:
```
source venv/bin/activate

- Windows (PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```
