import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import Counter    

def valuta_modelli_cv(dizionario_modelli, X, y, num_split=10, stato_casuale=42):
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
    print(f"tuning-valutazione split esterni={num_split_esterni}, interni={num_split_interni})")
    risultati_finali = {}
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
        best_params_per_fold = []

        if nome in griglie_parametri:
            classificatore_gs = GridSearchCV(estimator=modello, param_grid=griglie_parametri[nome], cv=cv_interna, scoring='roc_auc', n_jobs=-1,refit=True)
            try:
                punteggi_cv = cross_validate(classificatore_gs, X, y, cv=cv_esterna, scoring=metriche_punteggio, n_jobs=-1, return_estimator=True)

                for estimator in punteggi_cv['estimator']:
                    best_params_per_fold.append(estimator.best_params_)

                risultati_finali[nome + " (Tuned)"] = {
                    'scores': punteggi_cv,
                    'best_params_per_fold': best_params_per_fold
                }
                print(f"tuning-valutazione {nome} ok.")

            except Exception as e:
                 print(f"errore tuning-valutazione {nome}: {e}")
                 risultati_finali[nome + " (Tuned)"] = None
        else:
            print(f"nessuna griglia per {nome} -> valutazione semplice")
            try:
                punteggi_cv = cross_validate(modello, X, y, cv=cv_esterna, scoring=metriche_punteggio, n_jobs=-1)
                risultati_finali[nome] = {
                    'scores': punteggi_cv,
                    'best_params_per_fold': ["N/A"] * num_split_esterni
                }
                print(f"valutazione semplice {nome} ok.")
            except Exception as e:
                 print(f"errore valutazione {nome}: {e}")
                 risultati_finali[nome] = None

    return risultati_finali

def stampa_risultati_cv(risultati):

    print("\nrisultati CV e parametri")
    
    metriche = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
    riepilogo = {}
    params_riepilogo = {}

    for nome, ris_dict in risultati.items():
        if ris_dict is None:
            print(f"\nvalutazione fallita modello: {nome} ")
            riepilogo[nome] = {metrica: "fallita" for metrica in metriche}
            params_riepilogo[nome] = "fallita"
            continue

        punteggi = ris_dict['scores'] 
        params_list = ris_dict['best_params_per_fold']

        riepilogo[nome] = {}
        print(f"\nmodello: {nome}")
        
        for metrica in metriche:

            if metrica in punteggi:
                punteggio_medio = np.mean(punteggi[metrica])
                dev_std_punteggio = np.std(punteggi[metrica])
                riepilogo[nome][metrica] = f"{punteggio_medio:.4f} +/- {dev_std_punteggio:.4f}"
                print(f"  {metrica.replace('test_', '')}: {punteggio_medio:.4f} (+/- {dev_std_punteggio:.4f})")
            else:
                riepilogo[nome][metrica] = "N/D"
                print(f"  {metrica.replace('test_', '')}: N/D")
        
        if params_list and params_list[0] != "N/A":
            params_str_list = [str(p) for p in params_list]
            params_counts = Counter(params_str_list)
            most_common_params_str, freq = params_counts.most_common(1)[0]
            
            summary_str = f" {most_common_params_str}"
            params_riepilogo[nome] = summary_str
            print(f"param ottimiali: {most_common_params_str}")
        else:
            params_riepilogo[nome] = "N/A"
            print("  param ottimali: N/A")

    df_risultati = pd.DataFrame(riepilogo).T
    df_risultati['param ottimali'] = pd.Series(params_riepilogo)
    print("\nriepilogo:")
    pd.set_option('display.max_colwidth', None)
    print(df_risultati)
    pd.reset_option('display.max_colwidth')

    return df_risultati