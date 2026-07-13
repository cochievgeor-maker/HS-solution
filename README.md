# HS-solution

## Software companion to the publication "Broad-scope Peroxide-initiated Hydrosilylation under Equimolar and Solvent-free Conditions"

---

**HS-solution** accompanies the hydrosilylation (**HS**) study. The repository groups three layers: (1) **tabular data** under [`All_data/`](All_data/), (2) **research code** under [`processing_and_ML_code/`](processing_and_ML_code/) (descriptor generation, genetic feature selection, model benchmarks, SHAP-style diagnostics), and (3) a lightweight **Streamlit** demo under [`streamlit/`](streamlit/) for loading new reaction SMILES and running the trained models. The article remains the authoritative description of chemistry, experiments, and statistics; this file is a **navigation map** for code and data.

When you cite the paper, also reference a **pinned commit or release** of this repository (and a Zenodo DOI if you archive one).

---

## Data in `All_data/`

| Path | Description |
|------|-------------|
| [`All_data/Names+SMILES_forML.csv`](All_data/Names+SMILES_forML.csv) | **A data table used for machine learning.** A data table used for machine learning. Please note that not all the reactions described in the article are included in the learning algorithm - only the series from this list. |
| [`All_data/allData.csv`](All_data/allData.csv) | **Full valid descriptor matrix derived from SMILES.** Built from the same reaction inventory: leading columns include the condensed reaction key **`inp+inp=out`**, targets **`a`**, **`b`**, followed by **all numerically valid Mordred descriptors** computed for each reactant/product block (`reagent1_*`, `reagent2_*`, `product1_*`, …). This is the wide feature table before aggressive feature selection (thousands of columns, one row per reaction). |
| [`All_data/substrats_ML/`](All_data/substrats_ML/) | **ChemDraw source pages** (`page1.cdx`–`page14.cdx`) containing the drawn schemes that the SMILES table was derived from. |

---

## `processing_and_ML_code/` — modelling & processing

These Jupyter notebooks contain the **end-to-end workflow** used for descriptor calculation, genetic feature selection, model comparison, and Streamlit-related experiments.

| Notebook | Purpose |
|----------|---------|
| [`allDataWrite.ipynb`](processing_and_ML_code/allDataWrite.ipynb) | Compute Mordred descriptors from reaction SMILES, assemble wide tables, PCA / parity-style diagnostics. |
| [`gereticALGO.ipynb`](processing_and_ML_code/gereticALGO.ipynb) | Genetic algorithm feature selection (`GAFeatureSelectionCV`), Extra Trees benchmarks, small-sample model comparison. |
| [`last1.ipynb`](processing_and_ML_code/last1.ipynb) | Data joins, exploratory analysis, and glue code between spreadsheet exports and modelling matrices. |
| [`ProdactionML_SHAP.ipynb`](processing_and_ML_code/ProdactionML_SHAP.ipynb) | “Production” ML evaluation, residual checks, SHAP-oriented analysis. |

---

## Repository map (`streamlit/` and helpers)

The **`streamlit/`** tree ships the runnable web UI and a **small dependency list**. Older copies of some notebooks also live here for historical Streamlit Cloud layouts

| Path | What it is |
|------|------------|
| [`streamlit/code.py`](streamlit/code.py) | **Streamlit UI** — upload CSV/Excel with a `SMILES` column, pick regime **A** (DTBP, 130 °C) or **B** (DCP, 120 °C), compute descriptors, run inference, download CSV. |
| [`streamlit/requirements.txt`](streamlit/requirements.txt) | Minimal **pip** dependencies for the app. |
| [`streamlit/test_SMILE.csv`](streamlit/test_SMILE.csv) | Tiny example file with a `SMILES` column. |
| [`requirements.txt`](requirements.txt) | Root-level dependency list. |
| [`runtime.txt`](runtime.txt) | Target **Python 3.11** for deployment. |
| [`packages.txt`](packages.txt) | Optional extra pinning. |

---

## Quick run (online)
The interactive Streamlit application for hydrosilylation prediction is available here:  
[Open HS-solution Streamlit App](https://hs-solution-d8jbbxadsjj8ex7qwy8s25.streamlit.app/)

The app allows users to upload CSV or Excel files containing a `SMILES` column, choose the reaction regime, and obtain model predictions. 

---

## Quick run (local)

1. `pip install -r streamlit/requirements.txt`
2. Place `model_a2.pkl`, `model_b2.pkl`, `features_a2.pkl`, `features_b2.pkl` next to `streamlit/code.py`, **or** edit the loader paths (defaults may point to Streamlit Cloud: `/mount/src/hs-solution/streamlit/…`).
3. `cd streamlit && streamlit run code.py`
4. 
---

## Upstream repository

**[github.com/cochievgeor-maker/HS-solution](https://github.com/cochievgeor-maker/HS-solution)**

# Authors

Anton P. Drozdov^[a]^, Irina K. Goncharova^[a,b]^, Maria S. Sokolova^[a]^, Georgy D. Kochiev^[d]^, Bogdan O. Protsenko^[d]^, Maxim A. Novikov^[e]^, Irina P. Beletskaya^[b,c]^, Ashot V. Arzumanyan^[a,b]^*

[a] A.N. Nesmeyanov Institute of Organoelement Compounds, Russian Academy of Sciences, 28 Vavilov St., Moscow 119991, Russian Federation  
[b] A.V. Topchiev Institute of Petrochemical Synthesis, Russian Academy of Sciences, 29 Leninsky Prospect, Moscow 119991, Russian Federation  
[c] M.V. Lomonosov Moscow State University, GSP-1, Leninskie Gory, Moscow 119991, Russian Federation  
[d] The Smart Materials Research Institute, Southern Federal University, Rostov-on-Don, 344090, Russian Federation  
[e] N.D. Zelinsky Institute of Organic Chemistry, Russian Academy of Sciences, 47 Leninsky Pr., Moscow 119991, Russian Federation  

*Corresponding author: Ashot V. Arzumanyan

<img width="294" height="274" alt="image" src="https://github.com/user-attachments/assets/618d4ad2-0533-43f0-b69a-6ed9e1a4da50" />  


