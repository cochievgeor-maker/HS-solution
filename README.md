# HS-solution

Machine-learning helpers for **hydrosilylation (HS)** workflows: predict from **reaction SMILES** (`reactants>>products`, dot-separated species) using pretrained models and Mordred/RDKit descriptors.

This README is a **navigation map** for the repository. For methodology, metrics, and chemistry, refer to your paper and notebooks.

---

## Repository map

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
