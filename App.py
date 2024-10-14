import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Charger les données
@st.cache
def load_data():
    data = pd.read_csv('kidney_disease.csv')
    data.columns = ['id', 'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 
                    'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 
                    'pe', 'ane', 'classification']
    return data

# Prétraiter les données
def preprocess_data(data):
    data['classification'] = data['classification'].astype(str).str.strip()
    data['classification'] = data['classification'].map({'notckd': 0, 'ckd': 1})
    for col in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col].fillna(data[col].median(), inplace=True)
    for col in ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
        data[col].fillna(data[col].mode()[0], inplace=True)
        data[col] = LabelEncoder().fit_transform(data[col])
    X = data.drop(columns=['id', 'classification'])
    y = data['classification']
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner le modèle sélectionné
def train_model(model_name, X_train, y_train, X_test, y_test):
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Random Forest":
        n_estimators = st.sidebar.slider("Nombre d'arbres", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_name == "SVM":
        C = st.sidebar.slider("Paramètre C (Regularization)", 0.01, 10.0, 1.0)
        model = SVC(C=C, probability=True, random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    return y_pred, y_proba, model

# Afficher les résultats
def display_results(y_test, y_pred, y_proba):
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.write(f"### Précision: {acc:.4f}, AUC: {auc:.4f}")
    st.text("Rapport de classification :\n" + report)
    st.text("Matrice de confusion :")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Prévisions")
    ax.set_ylabel("Vérité")
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("Taux de Faux Positifs")
    ax.set_ylabel("Taux de Vrais Positifs")
    ax.set_title("Courbe ROC")
    ax.legend()
    st.pyplot(fig)

# Interface Streamlit
st.title("Prédiction de la Maladie Rénale Chronique")
st.sidebar.title("Paramètres")

data = load_data()
st.write("Aperçu des données :")
st.write(data.head())

model_name = st.sidebar.selectbox("Choisissez le modèle", ("Logistic Regression", "Random Forest", "SVM", "XGBoost"))
X_train, X_test, y_train, y_test = preprocess_data(data)
y_pred, y_proba, model = train_model(model_name, X_train, y_train, X_test, y_test)
display_results(y_test, y_pred, y_proba)