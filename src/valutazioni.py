import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def valuta_modelli_cv(dizionario_modelli, X, y, num_split=10, stato_casuale=42):
    #valutazione modelli con k-fold 10 fold
    risultati = {}
    metriche_punteggio = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, pos_label=1, zero_division=0),
        'recall': make_scorer(recall_score, pos_label=1, zero_division=0),
        'f1': make_scorer(f1_score, pos_label=1, zero_division=0),
        'roc_auc': 'roc_auc'
    }
    strategia_cv = StratifiedKFold(n_splits=num_split, shuffle=True, random_state=stato_casuale)

    for nome, modello in dizionario_modelli.items():
        print(f"Valutazione {nome}...")
        try:
            risultati_cv = cross_validate(modello, X, y, cv=strategia_cv, scoring=metriche_punteggio, n_jobs=-1)
            risultati[nome] = risultati_cv
            print(f"valutazione {nome} ok")
        except Exception as e:
            print(f"errore durante valutazione {nome}: {e}")
            risultati[nome] = None

    return risultati

def ottimizza_e_valuta_modelli_cv(dizionario_modelli, griglie_parametri, X, y,num_split_esterni=10, num_split_interni=5, stato_casuale=42):
    #tuning iperparametri
    risultati_ottimizzati = {}
    metriche_punteggio = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, pos_label=1, zero_division=0),
        'recall': make_scorer(recall_score, pos_label=1, zero_division=0),
        'f1': make_scorer(f1_score, pos_label=1, zero_division=0),
        'roc_auc': 'roc_auc'
    }
    cv_esterna = StratifiedKFold(n_splits=num_split_esterni, shuffle=True, random_state=stato_casuale)
    cv_interna = StratifiedKFold(n_splits=num_split_interni, shuffle=True, random_state=stato_casuale + 1)

    for nome, modello in dizionario_modelli.items():
        print(f"tuning-valutazione {nome}...")
        if nome in griglie_parametri:
            #setup gridsrc tuning con CV interna
            classificatore_gs = GridSearchCV(estimator=modello, param_grid=griglie_parametri[nome],cv=cv_interna, scoring='roc_auc', n_jobs=-1)
            try:
                punteggi_cv = cross_validate(classificatore_gs, X, y, cv=cv_esterna, scoring=metriche_punteggio, n_jobs=-1)
                risultati_ottimizzati[nome + " (Tuned)"] = punteggi_cv
                print(f"tuning-valutazione {nome} ok.")
            except Exception as e:
                 print(f"Errore tuning/valutazione {nome}: {e}")
                 risultati_ottimizzati[nome + " (Tuned)"] = None
        else:
            # valutazione semplice se non c'è griglia
            print(f"no griglia parametri per {nome} -> valutazione semplice")
            try:
                punteggi_cv = cross_validate(modello, X, y, cv=cv_esterna, scoring=metriche_punteggio, n_jobs=-1)
                risultati_ottimizzati[nome] = punteggi_cv
                print(f"valutazione semplice {nome} completata.")
            except Exception as e:
                 print(f"Errore valutazione {nome}: {e}")
                 risultati_ottimizzati[nome] = None

    return risultati_ottimizzati

def stampa_risultati_cv(risultati):
    metriche = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
    riepilogo = {}

    for nome, ris in risultati.items():
        if ris is None:
            print(f"\nModello: {nome} valutazione fallita")
            riepilogo[nome] = {metrica: "fallita" for metrica in metriche}
            continue

        riepilogo[nome] = {}
        print(f"\nmodello: {nome}")
        for metrica in metriche:
            if metrica in ris:
                punteggio_medio = np.mean(ris[metrica])
                dev_std_punteggio = np.std(ris[metrica])
                riepilogo[nome][metrica] = f"{punteggio_medio:.4f} +/- {dev_std_punteggio:.4f}"
                print(f"  {metrica.replace('test_', '')}: {punteggio_medio:.4f} (+/- {dev_std_punteggio:.4f})")
            else:
                 riepilogo[nome][metrica] = "N/D"
                 print(f"  {metrica.replace('test_', '')}: N/D")

    df_risultati = pd.DataFrame(riepilogo).T
    print("\nriepilogo:")
    print(df_risultati)
    return df_risultati