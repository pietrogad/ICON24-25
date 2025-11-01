"""
Script per dimostrare la capacità della Rete Bayesiana di fare predizioni
anche in presenza di dati mancanti.
"""

import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BIC
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

def discretizza_per_bn(df, colonne_numeriche):
    """Discretizza le feature numeriche per la rete bayesiana"""
    df_discrete = df.copy()
    print("Discretizzazione feature numeriche...")
    
    for col in colonne_numeriche:
        if col in df_discrete.columns and pd.api.types.is_numeric_dtype(df_discrete[col]):
            try:
                if col == 'oldpeak':
                    df_discrete[col] = pd.cut(df_discrete[col], 
                                            bins=[-1, 0, 1, 2, 10], 
                                            labels=['0', '1', '2', '3+'])
                elif col == 'age':
                    df_discrete[col] = pd.cut(df_discrete[col], 
                                            bins=[0, 40, 50, 60, 100], 
                                            labels=['<40', '40-50', '50-60', '60+'])
                else:
                    df_discrete[col] = pd.qcut(df_discrete[col], 4, labels=False, duplicates='drop')
            except Exception as e:
                print(f"Errore binning {col}: {e}")
                df_discrete[col] = pd.cut(df_discrete[col], 3, labels=['low', 'medium', 'high'])

    for col in df_discrete.columns:
        df_discrete[col] = df_discrete[col].astype(str)
    
    return df_discrete

def demo_predizione_con_dati_mancanti(df_dati, target_col='target'):
     
    colonne_numeriche = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    df_discrete = discretizza_per_bn(df_dati.copy(), colonne_numeriche)
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df_discrete, test_size=0.2, random_state=42, stratify=df_discrete[target_col])
    
    print(f"\nDati di training: {len(df_train)} esempi")
    print(f"Dati di test: {len(df_test)} esempi")

    hc = HillClimbSearch(df_train)
    best_model = hc.estimate(scoring_method=BIC(df_train), max_iter=500, show_progress=False)
    model = DiscreteBayesianNetwork(best_model.edges())

    if target_col not in model.nodes():
        print(f"   Aggiungo connessioni al target '{target_col}'")
        for feature in df_train.columns:
            if feature != target_col and feature in model.nodes():
                model.add_edge(feature, target_col)
    
    print(f"   Struttura appresa: {len(model.edges())} archi tra {len(model.nodes())} nodi")

    model.fit(df_train, estimator=MaximumLikelihoodEstimator)
    print("   Parametri appresi con successo")
    inference = VariableElimination(model)

    esempio_idx = None
    miglior_score = -1
    
    for idx in range(min(100, len(df_test))):
        esempio_test = df_test.iloc[idx]
        target_test = esempio_test[target_col]

        evidenze_test = {}
        for col in esempio_test.index:
            if col != target_col and col in model.nodes():
                evidenze_test[col] = str(esempio_test[col])
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                query_test = inference.query(variables=[target_col], evidence=evidenze_test)
            
            stati_test = list(query_test.state_names[target_col])
            prob_1 = 0.5
            
            if '1' in stati_test:
                idx_1 = stati_test.index('1')
                prob_1 = float(query_test.values[idx_1])
            elif len(stati_test) == 2:
                prob_1 = float(query_test.values[1])
            
            pred_test = '1' if prob_1 > 0.5 else '0'

            score = 0
            if pred_test == target_test:
                score += 10
                score += abs(prob_1 - 0.5) * 2
                
                if miglior_score < score:
                    miglior_score = score
                    esempio_idx = idx
        except:
            continue
    
    if esempio_idx is None:
        esempio_idx = 0
        miglior_score = 0
    else:
        print(f"esempio: paziente #{esempio_idx} (score: {miglior_score:.1f})")
    
    esempio = df_test.iloc[esempio_idx]
    target_reale = esempio[target_col]

    print(f"\nTarget reale: {target_reale}")
    print("\nFeature dell'esempio:")
    for col in esempio.index:
        if col != target_col:
            print(f"  {col}: {esempio[col]}")

    print("Predizione con tutti i dati")
    
    evidenze_complete = {}
    for col in esempio.index:
        if col != target_col and col in model.nodes():
            evidenze_complete[col] = str(esempio[col])
    
    print(f"\nEvidenze fornite: {len(evidenze_complete)} feature")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        risultato_completo = inference.query(variables=[target_col], evidence=evidenze_complete)
    
    print(f"\nDistribuzione di probabilità per '{target_col}':")
    stati = list(risultato_completo.state_names[target_col])
    for i, stato in enumerate(stati):
        prob = risultato_completo.values[i]
        marker = " predizione" if prob == max(risultato_completo.values) else ""
        print(f"  {stato}: {prob:.4f} ({prob*100:.2f}%){marker}")
    
    predizione_completa = stati[np.argmax(risultato_completo.values)]
    print(f"\nPredizione finale: {predizione_completa}")
    print(f"Target reale: {target_reale}")
    print(f"Corretto: {'SI' if predizione_completa == target_reale else 'NO'}")

    print("Predizione con un dato mancante")

    feature_da_rimuovere = 'cp' if 'cp' in evidenze_complete else list(evidenze_complete.keys())[0]
    
    evidenze_parziali_1 = evidenze_complete.copy()

    print(f"Evidenze fornite: {len(evidenze_parziali_1)} feature")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        risultato_parziale_1 = inference.query(variables=[target_col], evidence=evidenze_parziali_1)
    
    print(f"\nDistribuzione di probabilità per '{target_col}':")
    for i, stato in enumerate(stati):
        prob = risultato_parziale_1.values[i]
        marker = "predizione" if prob == max(risultato_parziale_1.values) else ""
        print(f"  {stato}: {prob:.4f} ({prob*100:.2f}%){marker}")
    
    predizione_parziale_1 = stati[np.argmax(risultato_parziale_1.values)]
    print(f"\nPredizione finale: {predizione_parziale_1}")
    print(f"Target reale: {target_reale}")
    print(f"Corretto: {'SI' if predizione_parziale_1 == target_reale else 'NO'}")

    print("predizione con più dati mancanti")
    features_da_rimuovere = list(evidenze_complete.keys())[:3]
    evidenze_parziali_multi = evidenze_complete.copy()
    
    print(f"\nFeature rimosse:")
    for feat in features_da_rimuovere:
        if feat in evidenze_parziali_multi:
            val = evidenze_parziali_multi.pop(feat)
            print(f"  - {feat} = {val}")
    
    print(f"\nevidenze fornite: {len(evidenze_parziali_multi)} feature (mancano {len(features_da_rimuovere)})")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        risultato_parziale_multi = inference.query(variables=[target_col], evidence=evidenze_parziali_multi)
    
    print(f"\nDistribuzione di probabilità per '{target_col}':")
    for i, stato in enumerate(stati):
        prob = risultato_parziale_multi.values[i]
        marker = " ← PREDIZIONE" if prob == max(risultato_parziale_multi.values) else ""
        print(f"  {stato}: {prob:.4f} ({prob*100:.2f}%){marker}")
    
    predizione_parziale_multi = stati[np.argmax(risultato_parziale_multi.values)]
    print(f"\nPredizione finale: {predizione_parziale_multi}")
    print(f"Target reale: {target_reale}")
    print(f"Corretto: {'SI' if predizione_parziale_multi == target_reale else 'NO'}")

    print("riepilogo")
    
    risultati = [
        ("Tutti i dati", len(evidenze_complete), predizione_completa, predizione_completa == target_reale),
        (f"Manca 1 dato ({feature_da_rimuovere})", len(evidenze_parziali_1), predizione_parziale_1, predizione_parziale_1 == target_reale),
        (f"Mancano {len(features_da_rimuovere)} dati", len(evidenze_parziali_multi), predizione_parziale_multi, predizione_parziale_multi == target_reale)
    ]
    
    print(f"\n{'Scenario':<30} {'Feature':<10} {'Predizione':<12} {'Corretto'}")
    for scenario, n_feat, pred, corretto in risultati:
        status = "SI" if corretto else "NO"
        print(f"{scenario:<30} {n_feat:<10} {pred:<12} {status}")
        
    return model, inference, esempio, risultati

def esegui_demo(percorso_csv='dataset/heart-disease/heart_disease_uci.csv'):

    from loader_dati import carica_e_pulisci_base, riempimento_dati

    df_dati_con_nan = carica_e_pulisci_base(percorso_csv)
    df_dati_riempiti = riempimento_dati(df_dati_con_nan.copy())
    
    if df_dati_riempiti is None:
        return None
    
    model, inference, esempio, risultati = demo_predizione_con_dati_mancanti(df_dati_riempiti)
    
    return model, inference, esempio, risultati

if __name__ == "__main__":
    esegui_demo()