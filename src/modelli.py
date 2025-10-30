# src/modelli.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def ottieni_modelli(stato_casuale=42):
    modelli = {
        'Regressione Logistica': LogisticRegression(random_state=stato_casuale, max_iter=1000),
        'Albero Decisionale': DecisionTreeClassifier(random_state=stato_casuale),
        'SVM': SVC(random_state=stato_casuale, probability=True)
    }
    return modelli

def ottieni_griglie_parametri():
    griglie_parametri = {
        'Regressione Logistica': {
            'C': [0.01, 0.1, 1, 10, 100]
        },
        'Albero Decisionale': {
            'max_depth': [None, 5, 10, 15],
            'min_samples_leaf': [1, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1]
        }
    }
    return griglie_parametri