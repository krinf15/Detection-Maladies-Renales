Bienvenue dans le projet de **détection des maladies rénales** ! Ce projet utilise des algorithmes de machine learning pour analyser et prédire les risques de maladies rénales basés sur des données médicales. 🎯

## 📁 Structure du Projet

Le projet est organisé en plusieurs dossiers pour une meilleure lisibilité et maintenabilité :

- **`notebooks/`** 📓 : Contient les notebooks Jupyter utilisés pour l'analyse exploratoire des données et le développement.
- **`data/`** 📊 : Contient les fichiers de données, y compris le fichier de données `kidney_disease.csv` utilisé pour l'entraînement des modèles.
- **`src/`** 💻 : Contient les scripts Python, tels que `App.py`, pour l'implémentation des modèles de prédiction.
- **`README.md`** 📄 : Ce fichier, décrivant le projet et fournissant des instructions détaillées pour la configuration et l'utilisation.

## Description de chaque colonne

Age: age in years

Blood Pressure: : BP in mm/Hg (MB: presumably diastolic blood pressure)

Specific Gravity: one of (1.005,1.010,1.015,1.020,1.025); (MB: see https://en.wikipedia.org/wiki/Urine_specific_gravity)

Albumin: one of (0,1,2,3,4,5) (MB: in urine)

Sugar: one of (0,1,2,3,4,5) (MB: in urine)

Red Blood Cells: one of (“normal”, “abnormal”) (MB: in urine)

Pus Cell: one of (“normal”, “abnormal”) (MB: in urine)

Pus Cell clumps: one of (“present”, “notpresent”) (MB: in urine)

Bacteria: one of (“present”, “notpresent”) (MB: in urine)

Blood Glucose Random: in mgs/dl

Blood Urea: in mgs/dl

Serum Creatinine: in mgs/dl

Sodium: in mEq/L

Potassium: in mEq/L

Hemoglobin: in gms

Packed Cell Volume: (MB: volume percentage; see https://en.wikipedia.org/wiki/Hematocrit)

White Blood Cell Count: in cells/cumm

Red Blood Cell Count: in millions/cmm

Hypertension: one of (“yes”, “no”)

Diabetes Mellitus: one of (“yes”, “no”)

Coronary Artery Disease: one of (“yes”, “no”)

Appetite: one of (“good”, “poor”)

Pedal Edema: one of (“yes”, “no”)

Anemia: one of (“yes”, “no”)

Class : one of (“ckd”, “notckd”) in ckd_full.csv) or (1,0) in ckd_clean.csv, where 1 corresponds to “ckd”).
  
## 🛠️ Technologies Utilisées

Les technologies et bibliothèques suivantes sont utilisées dans ce projet :

    Python 🐍 : Langage principal pour le développement.
    Pandas et NumPy 📈 : Manipulation et analyse de données.
    Scikit-learn 🔍 : Création et évaluation des modèles de machine learning.
    Matplotlib et Seaborn 📊 : Visualisation de données.

## 📊 Jeu de Données

Les données utilisées proviennent du fichier kidney_disease.csv dans le dossier data/. Ce fichier contient des informations médicales sur divers patients, utilisées pour entraîner les modèles de prédiction. Le fichier doit être analysé et préparé avant l’entraînement des modèles.
## 👥 Contributions

Les contributions sont les bienvenues ! Si vous souhaitez contribuer, veuillez suivre ces étapes :

    Fork le projet.
    Créez une branche pour votre fonctionnalité ou correction de bug (git checkout -b feature/NouvelleFeature).
    Committez vos modifications (git commit -m 'Ajout d'une nouvelle fonctionnalité').
    Poussez vers la branche (git push origin feature/NouvelleFeature).
    Ouvrez une Pull Request pour soumettre vos modifications.

##🧑‍💻 Auteur

    Votre Nom - krinf15. Vous pouvez me contacter via GitHub.

## 📄 Licence

Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus de détails.
