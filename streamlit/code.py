import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from rdkit.Chem import PandasTools
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import random
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing
from mordred import Calculator, descriptors

import os
import sys

st.write(
    "Project, models, and data on GitHub: "
    "https://github.com/cochievgeor-maker/HS-solution.git"
)

# Model and feature-list paths default to Streamlit Cloud; use local paths for offline runs.
@st.cache_resource
def load_models(ficha):
    try:
        if ficha == "A":
            model_a = joblib.load("/mount/src/hs-solution/streamlit/model_a2.pkl")
            return model_a
        if ficha == "B":
            model_b = joblib.load("/mount/src/hs-solution/streamlit/model_b2.pkl")
            return model_b
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None
    return None


@st.cache_resource
def load_scalers(ficha):
    try:
        if ficha == "A":
            feature_a = joblib.load("/mount/src/hs-solution/streamlit/features_a2.pkl")
            return feature_a
        if ficha == "B":
            feature_b = joblib.load("/mount/src/hs-solution/streamlit/features_b2.pkl")
            return feature_b
    except Exception:
        return None


def calculate_descriptors_for_molecule(smiles, prefix=""):
    """Compute Mordred descriptors for one SMILES string; return dict or None."""
    if pd.isna(smiles):
        return None

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        mol_3d = Chem.AddHs(mol)
        Chem.EmbedMolecule(mol_3d, randomSeed=0xF006D)

        try:
            Chem.MMFFOptimizeMolecule(mol_3d)
        except Exception:
            pass

        calc = Calculator(descriptors)
        desc_dict = calc(mol_3d)

        numeric_descriptors = {}
        for key, value in desc_dict.items():
            try:
                float_value = float(value)
                numeric_descriptors[f"{prefix}{key}"] = float_value
            except (ValueError, TypeError):
                continue

        return numeric_descriptors

    except Exception as e:
        print(f"Descriptor error for SMILES {smiles}: {e}")
        return None


def process_reactions(data):
    """Parse reaction SMILES rows and collect per-species descriptors."""
    all_descriptors = []

    for idx, reaction_smiles in enumerate(data["SMILES"]):
        if pd.isna(reaction_smiles):
            all_descriptors.append({})
            continue

        print(f"Reaction {idx + 1}/{len(data)}: {reaction_smiles}")

        reaction_descriptors = {}

        if ">>" in str(reaction_smiles):
            reagents_part, products_part = reaction_smiles.split(">>")

            reagents = [r.strip() for r in reagents_part.split(".") if r.strip()]
            for i, reagent in enumerate(reagents):
                prefix = f"reagent{i + 1}_"
                desc_dict = calculate_descriptors_for_molecule(reagent, prefix)
                if desc_dict:
                    reaction_descriptors.update(desc_dict)

            products = [p.strip() for p in products_part.split(".") if p.strip()]
            for i, product in enumerate(products):
                prefix = f"product{i + 1}_"
                desc_dict = calculate_descriptors_for_molecule(product, prefix)
                if desc_dict:
                    reaction_descriptors.update(desc_dict)

        all_descriptors.append(reaction_descriptors)
    print("Descriptor pass finished")
    return all_descriptors


def preprocess_data(raw_df, ficha, n):
    try:
        all_desc_data = process_reactions(raw_df)
        desc_df = pd.DataFrame(all_desc_data)

        features = load_scalers(ficha)

        available_features = [f for f in features[:n] if f in desc_df.columns]

        if len(available_features) == 0:
            st.error("None of the requested features exist in the computed descriptor table.")
            return None

        selected_data = desc_df[available_features]

        st.success(
            f"Selected {len(available_features)} features out of {n} requested."
        )

        return selected_data
    except Exception as e:
        st.error(f"Processing error: {e}")
        return None


def main():
    st.title("ML models — SMILES upload")
    st.write(
        "Upload a table for prediction. You can try `test_SMILE.csv` from this folder on GitHub."
    )

    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None

    st.sidebar.header("Model settings")

    target_choice = st.sidebar.selectbox(
        "Initiator / temperature regime: A=(DTBP, 130 °C), B=(DCP, 120 °C)",
        ["A", "B"],
        help="Labels follow the ChemDraw workbook.",
    )

    model = load_models(target_choice)
    if model is None:
        st.error("Models failed to load. Check model files and paths.")
        return

    st.header("1. Upload SMILES file")
    uploaded_file = st.file_uploader(
        "Choose data file",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel with a column named 'SMILES'. Each cell should hold the full reaction as "
        "reactants>>products (dot-separated species).",
    )

    raw_data = None
    processed_data = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_data = pd.read_csv(uploaded_file)
            else:
                raw_data = pd.read_excel(uploaded_file)

            st.success(f"Loaded {uploaded_file.name}")
            st.header("2. Process data")
            if st.button("Process data", type="primary"):
                with st.spinner("Computing descriptors..."):
                    processed_data = preprocess_data(raw_data, target_choice, 30)
                    st.session_state.processed_data = processed_data
                    st.session_state.raw_data = raw_data
                    if processed_data is not None:
                        with st.expander("Processed preview"):
                            st.write(
                                f"Shape: {processed_data.shape[0]} rows × "
                                f"{processed_data.shape[1]} columns"
                            )
                            st.dataframe(processed_data.head())

        except Exception as e:
            st.error(f"File load error: {e}")

    st.header("3. Prediction")

    processed_data = st.session_state.processed_data
    raw_data = st.session_state.raw_data

    print(f"processed_data = {processed_data}")
    if processed_data is not None:
        if st.button("Run prediction", type="primary"):
            try:
                with st.spinner("Running prediction..."):
                    predictions = model.predict(processed_data)

                    st.success("Prediction finished.")

                    results_df = processed_data.copy()
                    results_df["Predicted"] = predictions
                    results_df["SMILES"] = raw_data["SMILES"]

                    st.subheader("Results")
                    st.dataframe(results_df[["SMILES", "Predicted"]])

                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download results (CSV)",
                        data=csv,
                        file_name=f"smiles_predictions_{target_choice}.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Check that the input columns match the model expectations.")

    else:
        st.info("Upload a file and run processing to enable prediction.")


if __name__ == "__main__":
    main()

