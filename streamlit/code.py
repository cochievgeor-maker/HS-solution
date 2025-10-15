import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.preprocessing import StandardScaler

from rdkit.Chem import PandasTools
import numpy as np
import pandas as pd
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
# from genetic_selection import GeneticSelectionCV
from mordred import Calculator, descriptors

# Загрузка моделей с кэшированием
@st.cache_resource
def load_models(ficha):
    try:
        if ficha == "a":
            model_a = joblib.load('../streamlit/model_a2.pkl')
            return model_a
        if ficha == "b":
            model_b = joblib.load('../streamlit/model_b2.pkl')
            return model_b
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {e}")
        return None, None
    
    # Загрузка фичей 
@st.cache_resource
def load_scalers(ficha):
    try:
        if ficha == "a":
            feature_a = joblib.load('../streamlit/features_a2.pkl')
            return feature_a
        if ficha == "b":
            feature_b = joblib.load('../streamlit/features_b2.pkl')
            return feature_b
    except:
        return None, None
    
def calculate_descriptors_for_molecule(smiles, prefix=""):
    """
    Рассчитывает дескрипторы Mordred для одного SMILES
    Возвращает dict с дескрипторами или None если ошибка
    """
    if pd.isna(smiles):
        return None
        
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
            
        # Создаем 3D структуру
        mol_3d = Chem.AddHs(mol)
        Chem.EmbedMolecule(mol_3d, randomSeed=0xf006d)
        
        try:
            Chem.MMFFOptimizeMolecule(mol_3d)
        except:
            pass
            
        # Рассчитываем дескрипторы
        calc = Calculator(descriptors)
        desc_dict = calc(mol_3d)
        
        # СОХРАНЯЕМ NaN значения 
        numeric_descriptors = {}
        for key, value in desc_dict.items():
            try:
                float_value = float(value)
                numeric_descriptors[f"{prefix}{key}"] = float_value
                # Сохраняем ВСЕ числа, включая nan и inf
            except (ValueError, TypeError):
                # Отсеиваем только настоящие ошибки
                continue
                
        return numeric_descriptors
        
    except Exception as e:
        print(f"Ошибка для SMILES {smiles}: {e}")
        return None

def process_reactions(data):
    """
    Обрабатывает все реакции и возвращает DataFrame с дескрипторами
    """
    all_descriptors = []
    
    for idx, reaction_smiles in enumerate(data['SMILES']):
        if pd.isna(reaction_smiles):
            all_descriptors.append({})
            continue
        
        print(f"Обработка реакции {idx+1}/{len(data)}: {reaction_smiles}")
        
        reaction_descriptors = {}
        
        if '>>' in str(reaction_smiles):
            # Разделяем на реагенты и продукты
            reagents_part, products_part = reaction_smiles.split('>>')
            
            # Обрабатываем реагенты
            reagents = [r.strip() for r in reagents_part.split('.') if r.strip()]
            for i, reagent in enumerate(reagents):
                prefix = f"reagent{i+1}_"
                desc_dict = calculate_descriptors_for_molecule(reagent, prefix)
                if desc_dict:
                    reaction_descriptors.update(desc_dict)
            
            # Обрабатываем продукты
            products = [p.strip() for p in products_part.split('.') if p.strip()]
            for i, product in enumerate(products):
                prefix = f"product{i+1}_"
                desc_dict = calculate_descriptors_for_molecule(product, prefix)
                if desc_dict:
                    reaction_descriptors.update(desc_dict)
        
        all_descriptors.append(reaction_descriptors)
    print ("Обработка прошла успешно")
    return all_descriptors

# Ваши кастомные функции для обработки данных
def preprocess_data(raw_df, ficha, n):
    try:
        all_desc_data = process_reactions(raw_df) # тупо все дескрипторы 
        # Преобразуем список словарей в DataFrame
        desc_df = pd.DataFrame(all_desc_data)
        
        # Получаем фичи под запрос
        features = load_scalers(ficha)
        
        # Проверяем, что запрашиваемые фичи существуют в данных
        available_features = [f for f in features[:n] if f in desc_df.columns]
        
        if len(available_features) == 0:
            st.error("❌ Не найдено ни одного из запрашиваемых признаков в данных")
            return None
            
        # Отбираем только нужные колонки
        selected_data = desc_df[available_features]
        
        st.success(f"✅ Отобрано {len(available_features)} признаков из {n} запрошенных")
        
        return selected_data
    except Exception as e:
        st.error(f"Ошибка при обработке данных: {e}")
        return None

def main():

    st.title("📊 ML Models with SMILES Upload")
    st.write("Загрузите файл с данными для предсказания")

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    
    # Боковая панель для настроек
    st.sidebar.header("Настройки модели")
    
    # Выбор задания 
    target_choice = st.sidebar.selectbox(
        "Выберите объект предсказания:",
        ["a", "b"],
        help="Названия подобраны в соответсивии с данными из ChemDraw"
    )

    # Загрузка нужной модели
    model = load_models(target_choice)    
    if model is None:
        st.error("Модели не загружены. Проверьте файлы моделей.")
        return
    
    # Загрузка файла
    st.header("1. Загрузите файл со SMILES")
    uploaded_file = st.file_uploader(
        "Выберите файл с данными", 
        type=['csv', 'xlsx', 'xls'],
        help="Поддерживаемые форматы: CSV, Excel. В файле должна быть колонка с названием 'SMILES', под которой будут данные. " \
        "Подразумевается, что в SMILES будет записанна вся реакция: 2 продукта и 1 выход"
    )
    
    raw_data = None
    processed_data = None
    
    if uploaded_file is not None:
        try:
            # Определяем тип файла и загружаем
            if uploaded_file.name.endswith('.csv'):
                raw_data = pd.read_csv(uploaded_file)
            else:
                raw_data = pd.read_excel(uploaded_file)
            
            st.success(f"Файл {uploaded_file.name} успешно загружен!")
            # Обработка данных
            st.header("2. Обработка данных")
            if st.button("Обработать данные", type="primary"):
                with st.spinner("Обрабатываю данные..."):
                    processed_data = preprocess_data(raw_data, target_choice, 30)
                    st.session_state.processed_data = processed_data  #  СОХРАНЯЕМ В SESSION STATE
                    st.session_state.raw_data = raw_data  #  СОХРАНЯЕМ В SESSION STATE
                    if processed_data is not None:
                        with st.expander("🔧 Просмотр обработанных данных"):
                            st.write(f"Размер после обработки: {processed_data.shape[0]} строк, {processed_data.shape[1]} колонок")
                            st.dataframe(processed_data.head())
            
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {e}")
    
    # Предсказание
    st.header("3. Предсказание")

    processed_data = st.session_state.processed_data  #  ДОСТАЕМ ИЗ SESSION STATE
    raw_data = st.session_state.raw_data  #  ДОСТАЕМ ИЗ SESSION STATE

    print (f"наши данные --- {processed_data}")
    if processed_data is not None:
        
        if st.button("🚀 Запустить предсказание", type="primary"):
            try:
                with st.spinner("Выполняю предсказание..."):

                    # Используем выбранную модель 
                    predictions = model.predict(processed_data)

                    # Результаты для классификации
                    st.success("Предсказание завершено!")
                    
                    # Добавляем предсказания к данным
                    results_df = processed_data.copy()
                    results_df['Predicted'] = predictions
                    results_df['SMILES'] = raw_data['SMILES']
                    
                    # Показываем результаты
                    st.subheader("📈 Результаты предсказаний")
                    st.dataframe(results_df[['SMILES', 'Predicted']])

                    
                    # Кнопка скачивания
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Скачать результаты (CSV)",
                        data=csv,
                        file_name=f"smiles_predictions_{target_choice}.csv",
                        mime="text/csv"
                    )
                    
                    
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")
                st.info("Проверьте, что структура данных соответствует ожиданиям модели")
    
    else:
        st.info("👆 Загрузите файл и обработайте данные для получения предсказаний")
    
if __name__ == "__main__":
    main()
