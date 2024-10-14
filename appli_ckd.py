# Import des bibliothèques
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression

# Chargement des données (assurez-vous que le fichier csv est dans le même répertoire que ce script)
kidney = pd.read_csv('C:/Users/TOURE/Documents/MASTER-2 IA DA/SEMESTRE II/PROJET PROFESSIONNEL/kidney_disease.csv') 

# Renommons les colonnes
kidney.columns = ['id','age','blood_pressure','specific_gravity', 'albumin','sugar','red_blood_cells','pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'pedal_edema',
              'anemia', 'class']

# Suppression de la colonne id
kidney.drop(["id"], axis=1, inplace=True)  

# Convertissons les types des colonnes en numériques 
kidney.packed_cell_volume = pd.to_numeric(kidney.packed_cell_volume, errors='coerce')
kidney.red_blood_cell_count = pd.to_numeric(kidney.red_blood_cell_count, errors='coerce')
kidney.white_blood_cell_count = pd.to_numeric(kidney.white_blood_cell_count, errors='coerce')

kidney['diabetes_mellitus'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'}, inplace=True)
kidney['coronary_artery_disease'] = kidney['coronary_artery_disease'].replace(to_replace='\tno', value='no')
kidney['class'] = kidney['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'})

# Encodage des données
kidney = kidney.apply(LabelEncoder().fit_transform)
kidney.head()




# Récupérer les valeurs min, max et par défaut pour chaque caractéristique
plage_min_haemoglobin = float(kidney['haemoglobin'].min())
plage_max_haemoglobin = float(kidney['haemoglobin'].max())
valeur_par_defaut_haemoglobin = float(kidney['haemoglobin'].mean())

plage_min_age= float(kidney['age'].min())
plage_max_age = float(kidney['age'].max())
valeur_par_defaut_age = float(kidney['age'].mean())

plage_min_albumin = float(kidney['albumin'].min())
plage_max_albumin = float(kidney['albumin'].max())
valeur_par_defaut_albumin = float(kidney['albumin'].mean())

plage_min_serum_creatinine = float(kidney['serum_creatinine'].min())
plage_max_serum_creatinine = float(kidney['serum_creatinine'].max())
valeur_par_defaut_serum_creatinine = float(kidney['serum_creatinine'].mean())

plage_min_sodium = float(kidney['sodium'].min())
plage_max_sodium = float(kidney['sodium'].max())
valeur_par_defaut_sodium = float(kidney['sodium'].mean())

plage_min_hypertension = float(kidney['hypertension'].min())
plage_max_hypertension = float(kidney['hypertension'].max())
valeur_par_defaut_hypertension = float(kidney['hypertension'].mean())

plage_min_potassium = float(kidney['potassium'].min())
plage_max_potassium = float(kidney['potassium'].max())
valeur_par_defaut_potassium = float(kidney['potassium'].mean())

plage_min_blood_glucose_random = float(kidney['blood_glucose_random'].min())
plage_max_blood_glucose_random = float(kidney['blood_glucose_random'].max())
valeur_par_defaut_blood_glucose_random = float(kidney['blood_glucose_random'].mean())

plage_min_packed_cell_volume = float(kidney['packed_cell_volume'].min())
plage_max_packed_cell_volume = float(kidney['packed_cell_volume'].max())
valeur_par_defaut_packed_cell_volume = float(kidney['packed_cell_volume'].mean())

plage_min_red_blood_cell_count = float(kidney['red_blood_cell_count'].min())
plage_max_red_blood_cell_count = float(kidney['red_blood_cell_count'].max())
valeur_par_defaut_red_blood_cell_count = float(kidney['red_blood_cell_count'].mean())

plage_min_blood_urea = float(kidney['blood_urea'].min())
plage_max_blood_urea = float(kidney['blood_urea'].max())
valeur_par_defaut_blood_urea = float(kidney['blood_urea'].mean())

# Variables sélectionnées
selected_features = ['age', 'albumin', 'blood_glucose_random',
                    'serum_creatinine', 'blood_urea', 'sodium', 'haemoglobin', 'packed_cell_volume',
                    'red_blood_cell_count', 'hypertension']

# Sélection des variables avec SelectKBest et ANOVA
X = kidney[selected_features]
y = kidney["class"]
selector = SelectKBest(score_func=f_classif, k=len(selected_features))  # Sélectionnez le nombre de caractéristiques souhaitées ici
X_new = selector.fit_transform(X, y)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)


# Normalisation des caractéristiques
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Créer un modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train_scaled, y_train)

# Prédire les étiquettes sur les données de test
y_pred = model.predict(X_test_scaled)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
f_score = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

#print("Précision du modèle:", accuracy)
#print("Rapport de classification:\n", classification_rep)
# Entraînement du modèle
#clf = RandomForestClassifier()
#clf.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
#y_pred = clf.predict(X_test)

# Évaluation du modèle
#accuracy = accuracy_score(y_test, y_pred)
#conf_matrix = confusion_matrix(y_test, y_pred)
#class_report = classification_report(y_test, y_pred)

# Début de l'application Streamlit
st.write('''
# App de prédiction de la maladie rénale chronique (CKD)
Cette application prédit si un patient a une maladie rénale chronique (CKD) ou non en fonction de certaines caractéristiques.

Pour utiliser l'application, veuillez ajuster les paramètres sur le panneau latéral gauche et cliquer sur le bouton "Prédire".
''')

st.sidebar.header('Paramètres')

# Collecte des données d'entrée à partir de l'utilisateur
age = st.sidebar.slider('age', plage_min_age, plage_max_age, valeur_par_defaut_age)
albumin = st.sidebar.slider('albumin', plage_min_albumin, plage_max_albumin, valeur_par_defaut_albumin)
blood_glucose_random = st.sidebar.slider('blood_glucose_random', plage_min_blood_glucose_random, plage_max_blood_glucose_random, valeur_par_defaut_blood_glucose_random)
serum_creatinine = st.sidebar.slider('serum_creatinine', plage_min_serum_creatinine, plage_max_serum_creatinine, valeur_par_defaut_serum_creatinine)
sodium = st.sidebar.slider('sodium', plage_min_sodium, plage_max_sodium, valeur_par_defaut_sodium)
haemoglobin = st.sidebar.slider('haemoglobin', plage_min_haemoglobin, plage_max_haemoglobin, valeur_par_defaut_haemoglobin)
packed_cell_volume = st.sidebar.slider('packed_cell_volume', plage_min_packed_cell_volume, plage_max_packed_cell_volume, valeur_par_defaut_packed_cell_volume)
red_blood_cell_count = st.sidebar.slider('red_blood_cell_count', plage_min_red_blood_cell_count, plage_max_red_blood_cell_count, valeur_par_defaut_red_blood_cell_count)
hypertension = st.sidebar.slider('hypertension', plage_min_hypertension, plage_max_hypertension, valeur_par_defaut_hypertension)
blood_urea = st.sidebar.slider('blood_urea', plage_min_blood_urea, plage_max_blood_urea, valeur_par_defaut_blood_urea)

# Création d'un DataFrame pour la prédiction
user_kidney = pd.DataFrame({
    'age': [age],
    'albumin': [albumin],
    'blood_glucose_random': [blood_glucose_random],
    'serum_creatinine': [serum_creatinine],
    'blood_urea': [blood_urea],
    'sodium': [sodium],
    'haemoglobin': [haemoglobin],
    'packed_cell_volume': [packed_cell_volume],
    'red_blood_cell_count': [red_blood_cell_count],
    'hypertension': [hypertension],
    
})


# Affichage des données d'entrée
st.subheader('Données d\'entrée utilisateur')
st.write(user_kidney)

# Prédiction
prediction = model.predict(user_kidney)
prediction_text = "Maladie rénale chronique (CKD)" if prediction[0] == 1 else "Pas de maladie rénale chronique (non-CKD)"
st.subheader('Résultat de la prédiction')
st.write(prediction_text)

st.subheader('Évaluation du modèle')
st.write(f'Précision : {accuracy}')
st.write('Matrice de confusion :')
st.write(conf_matrix)
st.write('Rapport de classification :')
st.write(classification_rep)


