# HS-decision

## Software companion to the publication

**HS-decision** is a companion repository for research on **hydrosilylation (HS)**. It bundles (i) tabular experimental and structural data, (ii) machine-learning models and descriptor pipelines based on **reaction SMILES** and **Mordred** / **RDKit**, and (iii) a small **Streamlit** application for inference. The scientific rationale, experimental design, model validation, and chemical interpretation belong in the **peer-reviewed article**; this README records what is in the repository so readers can **find materials**, **cite a pinned release or DOI**, and **reproduce the software side** of the work.

When you cite the paper, please also reference this repository (ideally a **tagged release** or **archived snapshot**, e.g. Zenodo) and the **software version** used to obtain reported numbers.

---

## Data in `All_data/`

As of this revision, [`All_data/`](All_data/) contains only the items below.

| Path | Role |
|------|------|
| [`All_data/Names+SMILES_forML.csv`](All_data/Names+SMILES_forML.csv) | **Canonical reaction table for machine learning.** Each row is one hydrosilylation entry: ChemDraw-style labels (`input`, `output`), response columns **`a`** and **`b`**, and the full **reaction SMILES** in **`SMILES`** (`reactants>>products`, dot-separated species). **Only the reactions listed in this file define the training inventory** for the ML models described in the paper (scope of structures and labels the algorithms were fit to). |
| [`All_data/substrats_ML/`](All_data/substrats_ML/) | **ChemDraw documents** (`page1.cdx` … `page14.cdx`) containing the drawn reaction schemes that correspond to the dataset; they are the graphical source material aligned with the SMILES / label table above. |

---

## Repository map (`streamlit/` and helpers)

Below: what each entry is for and when you typically open it.

| Path | What it is |
|------|------------|
| [`streamlit/code.py`](streamlit/code.py) | **Streamlit UI** — upload CSV/Excel with a `SMILES` column, pick initiator regime **A** (DTBP, 130 °C) or **B** (DCP, 120 °C), compute descriptors, run inference, download predictions as CSV. |
| [`streamlit/requirements.txt`](streamlit/requirements.txt) | Minimal **pip** dependencies for the Streamlit app (Streamlit, pandas, scikit-learn, joblib, mordred, matplotlib, seaborn, RDKit). |
| [`streamlit/test_SMILE.csv`](streamlit/test_SMILE.csv) | **Smoke-test input** — a few reaction SMILES in one column; useful to verify the UI and parsing. |
| [`streamlit/allDataWrite.ipynb`](streamlit/allDataWrite.ipynb) | **Research notebook** — builds Mordred features from reaction SMILES in the same spirit as the app (good starting point if you extend the pipeline). |
| [`streamlit/last1.ipynb`](streamlit/last1.ipynb) | **Exploratory notebook** — loads CSVs from `../data/` (expects a sibling `data` folder in a fuller checkout); many cells are stubs or commented—use as a scratchpad. |
| [`streamlit/ProdactionML_SHAP.ipynb`](streamlit/ProdactionML_SHAP.ipynb) | **Evaluation / interpretability** — model metrics, plots, SHAP-oriented analysis (title reflects intent; adjust paths to your data). |
| [`requirements.txt`](requirements.txt) | **Root-level** dependency list (overlap with `streamlit/requirements.txt`; handy for one-shot installs or CI). |
| [`runtime.txt`](runtime.txt) | **Python version pin** for hosted deploys (currently **3.11**). Match this locally if you need identical behaviour. |
| [`packages.txt`](packages.txt) | Optional **extra pinning** — fill or generate from `pip freeze` when you cut a reproducible release. |

---

## Quick run (local)

1. Install deps: `pip install -r streamlit/requirements.txt`
2. Add trained artefacts next to `code.py` (e.g. `model_a2.pkl`, `model_b2.pkl`, `features_a2.pkl`, `features_b2.pkl`) **or** edit the load paths in `code.py` (defaults may target Streamlit Cloud: `/mount/src/hs-solution/streamlit/…`).
3. Start the app:

   ```bash
   cd streamlit
   streamlit run code.py
   ```

---

## Upstream repository

Canonical remote: **[github.com/cochievgeor-maker/HS-solution](https://github.com/cochievgeor-maker/HS-solution)**

After acceptance, consider updating this README with the **article DOI**, **journal reference**, and the **exact Git tag** archived with the paper.
