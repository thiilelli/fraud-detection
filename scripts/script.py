import os
import ast
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, roc_curve, auc



### Préparation des données 

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "..", "Downloads", "fraud_detection_data.csv")
try:
    df = pd.read_csv(data_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file {data_path} does not exist.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
    exit(1)
except pd.errors.ParserError:
    print("Error: The file could not be parsed.")
    exit(1)

# La conversion garantit la manipulation uniforme des numéros de carte comme des chaînes
try:
    df['card_number'] = df['card_number'].astype(str)
except KeyError:
    print("Error: 'card_number' column is missing.")
    exit(1)
except ValueError:
    print("Error: 'card_number' column could not be converted to string.")
    exit(1)

print(df.head())  # Affiche un aperçu des données pour confirmation visuelle rapide
print(df.info())  # Résume les informations sur le DataFrame, y compris la présence de valeurs manquantes
print(df.describe())  # Fournit des statistiques descriptives des données numériques
print(df.columns)  # Affiche les noms des colonnes pour aider à vérifier la structure du DataFrame

# Analyse de la distribution des indicateurs de fraude
print(Counter(df['fraud_flag']))

# Transformation des variables catégorielles pour la modélisation
categorical_columns = ['merchant_state', 'merchant_city', 'card_type', 'cardholder_name']
for col in categorical_columns:
    try:
        df[f'{col}_code'] = df[col].astype('category').cat.codes
    except KeyError:
        print(f"Error: '{col}' column is missing.")
        exit(1)
    except ValueError:
        print(f"Error: '{col}' column could not be converted to categorical codes.")
        exit(1)

# Conversion des listes d'articles achetés en nombre d'articles, évalué en toute sécurité
try:
    df['number_of_items'] = [len(ast.literal_eval(x)) for x in df['items']]
except KeyError:
    print("Error: 'items' column is missing.")
    exit(1)
except (ValueError, SyntaxError):
    print("Error: 'items' column could not be processed.")
    exit(1)

# Application de filtres pour retirer les valeurs aberrantes basées sur les scores Z
try:
    threshold = 3
    z_scores = np.abs(stats.zscore(df['transaction_amount']))
    df_no_outliers = df[(z_scores < threshold)]
except KeyError:
    print("Error: 'transaction_amount' column is missing.")
    exit(1)
except TypeError:
    print("Error: 'transaction_amount' column contains non-numeric data.")
    exit(1)

print("Categorical variables have been encoded and outliers removed.")

# Préparation des données pour la modélisation
features = ['merchant_state_code', 'merchant_city_code', 'card_type_code', 'cardholder_name_code', 'transaction_amount', 'number_of_items']
target = 'fraud_flag'
X = df_no_outliers[features]
y = df_no_outliers[target]

# Sauvegarde des jeux de données préparés pour une utilisation ultérieure
try:
    features_output_path = os.path.join(base_path, "features.csv")
    targets_output_path = os.path.join(base_path, "targets.csv")
    X.to_csv(features_output_path, index=False)
    y.to_csv(targets_output_path, index=False)
    print("Features and targets have been saved to CSV files.")
except Exception as e:
    print(f"Error while saving files: {e}")
    exit(1)


# Construction et entraînement du modèle

# Configuration des chemins pour assurer la portabilité du code sur différents environnements d'exécution
base_path = os.path.dirname(os.path.abspath(__file__))

# Définition des hyperparamètres pour le modèle RandomForest, optimisant le nombre d'arbres et la profondeur
n_estimators_range = np.arange(20, 100, 20)
max_depth_range = np.arange(5, 30, 5)
param_grid = {
    'n_estimators': n_estimators_range,
    'max_depth': max_depth_range,
}

# Initialisation du RandomForest avec un seed fixe pour garantir la reproductibilité des résultats
rf_classifier = RandomForestClassifier(random_state=64)

# Chargement des jeux de données depuis les fichiers CSV préparés
features_path = os.path.join(base_path, "features.csv")
targets_path = os.path.join(base_path, "targets.csv")

try:
    X = pd.read_csv(features_path)
    y = pd.read_csv(targets_path)
    print("Feature and target data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: One of the files is empty.")
    exit(1)
except pd.errors.ParserError:
    print("Error: Could not parse one of the files.")
    exit(1)

# Division des données en ensembles d'entraînement et de test pour valider l'efficacité du modèle
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=128)
    print("Data has been split into training and testing sets.")
except ValueError as e:
    print(f"Error in data splitting: {e}")
    exit(1)

print("Training data (first 5 rows):")
print(X_train.head())
print(y_train.head())

print("Test data (first 5 rows):")
print(X_test.head())
print(y_test.head())

# Configuration de GridSearchCV pour une optimisation précise des hyperparamètres
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='precision')

# Exécution de la recherche de grille pour identifier les meilleurs hyperparamètres
try:
    grid_search.fit(X_train, y_train.values.ravel())
    print("Grid search completed successfully.")
except ValueError as e:
    print(f"Error in grid search: {e}")
    exit(1)

print("Best Hyperparameters:", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_

# Prédiction avec le modèle optimisé et gestion des erreurs potentielles
try:
    y_pred = best_rf_model.predict(X_test)
    print("Predictions made on the test set.")
except ValueError as e:
    print(f"Error in making predictions: {e}")
    exit(1)

# Sauvegarde des prédictions et des données de test pour référence future
try:
    validation_data = X_test.copy()
    validation_data['actual'] = y_test
    validation_data['predicted'] = y_pred
    validation_data_path = os.path.join(base_path, "validation_data.csv")
    validation_data.to_csv(validation_data_path, index=False)
    print(f"Validation data saved to {validation_data_path}.")
except Exception as e:
    print(f"Error while saving validation data: {e}")
    exit(1)

# Enregistrement du modèle formé pour une utilisation ultérieure
try:
    model_filename = os.path.join(base_path, 'random_forest_model.pkl')
    with open(model_filename, 'wb') as model_file:
        pickle.dump(best_rf_model, model_file)
    print(f"Random Forest model saved to {model_filename}.")
except Exception as e:
    print(f"Error while saving the model: {e}")
    exit(1)


############## Evaluation du modèle ######

# Configuration des chemins pour la portabilité du script
base_path = os.path.dirname(os.path.abspath(__file__))
validation_data_path = os.path.join(base_path, "validation_data.csv")

# Chargement sécurisé des données de validation
try:
    validation_data = pd.read_csv(validation_data_path)
    print("Validation data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file {validation_data_path} does not exist.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: The validation file is empty.")
    exit(1)
except pd.errors.ParserError:
    print("Error: The validation file could not be parsed.")
    exit(1)

# Extraction des valeurs réelles pour évaluation
try:
    actual = validation_data['actual']
except KeyError:
    print("Error: 'actual' column is missing in the validation data.")
    exit(1)

# Chemin pour le modèle sauvegardé
model_filename = os.path.join(base_path, 'random_forest_model.pkl')

# Chargement du modèle pour l'évaluation
try:
    with open(model_filename, 'rb') as model_file:
        loaded_rf_model = pickle.load(model_file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: The model file {model_filename} does not exist.")
    exit(1)
except pickle.UnpicklingError:
    print("Error: The model file could not be unpickled.")
    exit(1)

# Prédiction des probabilités sur les données de validation
try:
    X_test = validation_data.drop(columns=['actual', 'predicted'])
    y_prob = loaded_rf_model.predict_proba(X_test)[:, 1]
    print("Predictions made successfully.")
except KeyError:
    print("Error: Necessary columns are missing in the validation data.")
    exit(1)
except ValueError as e:
    print(f"Error in making predictions: {e}")
    exit(1)

# Seuil pour la classification ajustée
threshold = 0.19  
predicted_adjusted = (y_prob >= threshold).astype(int)

# Calcul et affichage des métriques de performance
try:
    precision = precision_score(actual, predicted_adjusted)
    accuracy = accuracy_score(actual, predicted_adjusted)
    recall = recall_score(actual, predicted_adjusted)
    f1 = f1_score(actual, predicted_adjusted)
    print(f"Performance metrics: Precision={precision}, Accuracy={accuracy}, Recall={recall}, F1 Score={f1}")
except ValueError as e:
    print(f"Error in calculating performance metrics: {e}")
    exit(1)

# Matrice de confusion pour évaluer la performance du modèle
try:
    conf_matrix = confusion_matrix(actual, predicted_adjusted)
    print("Confusion Matrix:")
    print(conf_matrix)
except ValueError as e:
    print(f"Error in calculating confusion matrix: {e}")
    exit(1)

# Calcul et affichage de la courbe ROC
try:
    fpr, tpr, thresholds = roc_curve(actual, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
except ValueError as e:
    print(f"Error in calculating ROC curve: {e}")
    exit(1)

# Extraction et affichage de l'importance des caractéristiques
try:
    importances = loaded_rf_model.feature_importances_
    features = loaded_rf_model.feature_names_in_
    feature_importance_df = pd.DataFrame({"features": features, "importances": importances}).sort_values("importances", ascending=False)
    print(feature_importance_df)
    feature_importance_path = os.path.join(base_path, "feature_importance.csv")
    feature_importance_df.to_csv(feature_importance_path, index=False)
    print(f"Feature importances saved to {feature_importance_path}.")
except AttributeError:
    print("Error: The model does not have feature importances.")
    exit(1)
except Exception as e:
    print(f"Error while saving feature importances: {e}")
    exit(1)
