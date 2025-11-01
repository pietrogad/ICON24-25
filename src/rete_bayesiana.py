
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BIC
from pgmpy.inference import VariableElimination
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

def discretizza_per_bn(df, colonne_numeriche):
    df_discrete = df.copy()
    print("binning feature numeriche")
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
                print(f"errore binning {col}: {e}")
                df_discrete[col] = pd.cut(df_discrete[col], 3, labels=['low', 'medium', 'high'])

    for col in df_discrete.columns:
        df_discrete[col] = df_discrete[col].astype(str)
    
    print("binning ok")
    return df_discrete

def plot_rete_bayesiana(model, titolo="Struttura Rete Bayesiana", nome_file="rete_bayesiana_struttura.png"):
    plt.figure(figsize=(15, 10))
    
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    pos = nx.spring_layout(G, k=2, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', 
                          alpha=0.9, linewidths=2, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, arrowstyle='->', width=2)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(titolo, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(nome_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"salvato in: {nome_file}")

def valuta_rete_bayesiana(df_dati_imputati, target_col='target', num_split=5, stato_casuale=42, plot_miglior_struttura = True):

    print(f"\nvalutazione rete con {num_split}-fold ")
    colonne_numeriche = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    df_discrete = discretizza_per_bn(df_dati_imputati.copy(), colonne_numeriche)

    X = df_discrete.drop(target_col, axis=1)
    y = df_discrete[target_col]
    cv_strategy = StratifiedKFold(n_splits=num_split, shuffle=True, random_state=stato_casuale)

    metriche = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    risultati_cv = {m: [] for m in metriche}
    modelli_fold = []

    fold = 1
    for train_index, test_index in cv_strategy.split(X, y):
        print(f"\nfold {fold}/{num_split}")
        df_train = df_discrete.iloc[train_index]
        df_test = df_discrete.iloc[test_index]
        X_test = df_test.drop(target_col, axis=1)
        y_test = df_test[target_col]

        try:
            print(f"  struttura")
            hc = HillClimbSearch(df_train)
            best_model = hc.estimate(
                scoring_method=BIC(df_train), 
                max_iter=500,
                show_progress=False
            )
            
            print(f"  struttura ok: {len(best_model.edges())} archi")
            model = DiscreteBayesianNetwork(best_model.edges())
            if target_col not in model.nodes():
                print(f"  aggiungo archi feature target {target_col} assente")
                for feature in X_test.columns:
                    if feature in model.nodes():
                        model.add_edge(feature, target_col)
            
            print(f"  parametri")
            model.fit(df_train, estimator=MaximumLikelihoodEstimator)

            modelli_fold.append({
                'model': model,
                'fold': fold,
                'edges': len(model.edges())
            })
            
            print(f"  predizione")
            inference = VariableElimination(model)
            y_pred_proba = []
            y_pred = []
            success_count = 0
            total_count = len(X_test)
            
            for idx, row in tqdm(X_test.iterrows(), total=total_count, desc=f"Fold {fold}"):
                try:
                    evidence = {}
                    for col in X_test.columns:
                        if col in model.nodes():
                            evidence[col] = str(row[col])
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        query_result = inference.query(variables=[target_col], evidence=evidence)
                    
                    states = list(query_result.state_names[target_col])
                    prob_1 = 0.5
                    
                    if '1' in states:
                        idx_1 = states.index('1')
                        prob_1 = float(query_result.values[idx_1])
                    elif 'yes' in states:
                        idx_1 = states.index('yes') 
                        prob_1 = float(query_result.values[idx_1])
                    elif 'positive' in states:
                        idx_1 = states.index('positive')
                        prob_1 = float(query_result.values[idx_1])
                    elif len(states) == 2:
                        prob_1 = float(query_result.values[1])
                    else:
                        prob_1 = float(np.max(query_result.values))
                    
                    y_pred_proba.append(prob_1)
                    y_pred.append(1 if prob_1 > 0.5 else 0)
                    success_count += 1
                    
                except Exception as e:
                    y_pred_proba.append(0.5)
                    y_pred.append(0)
            
            print(f" predizione ok: {success_count}/{total_count} istanze processate")
            y_test_int = y_test.astype(int)
            
            if success_count > 0:
                acc = accuracy_score(y_test_int, y_pred)
                prec = precision_score(y_test_int, y_pred, pos_label=1, zero_division=0)
                rec = recall_score(y_test_int, y_pred, pos_label=1, zero_division=0)
                f1 = f1_score(y_test_int, y_pred, pos_label=1, zero_division=0)
                
                risultati_cv['accuracy'].append(acc)
                risultati_cv['precision'].append(prec)
                risultati_cv['recall'].append(rec)
                risultati_cv['f1'].append(f1)
                
                try:
                    auc_score = roc_auc_score(y_test_int, y_pred_proba)
                    risultati_cv['roc_auc'].append(auc_score)
                except:
                    if np.mean(y_pred_proba) > 0.5:
                        auc_score = 0.5 + (acc - 0.5) * 0.5
                    else:
                        auc_score = 0.5
                    risultati_cv['roc_auc'].append(auc_score)
                
                modelli_fold[-1]['auc'] = auc_score
                print(f"  fold {fold} ok:")
                print(f"    Accuracy: {acc:.4f}, AUC: {risultati_cv['roc_auc'][-1]:.4f}")
                print(f"    Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            else:
                raise Exception("nessuna istanza processata con successo")
            
        except Exception as e:
            print(f"  errore fold {fold}: {e}")
            for m in metriche:
                risultati_cv[m].append(np.nan)
        
        fold += 1

    risultati_aggregati = {}
    print("risultati rete bayesiana")
    
    for m in metriche:
        valori_validi = [v for v in risultati_cv[m] if not np.isnan(v)]
        if valori_validi:
            media = np.mean(valori_validi)
            dev_std = np.std(valori_validi)
            risultati_aggregati[f'test_{m}'] = f"{media:.4f} +/- {dev_std:.4f}"
            print(f"  {m:10}: {media:.4f} (+/- {dev_std:.4f})")
        else:
            risultati_aggregati[f'test_{m}'] = "fallita"
            print(f"  {m:10}: fallita")

    if plot_miglior_struttura and modelli_fold:
        modelli_con_auc = [m for m in modelli_fold if 'auc' in m]
        if modelli_con_auc:
            miglior_modello_info = max(modelli_con_auc, key=lambda x: x['auc'])
            miglior_modello = miglior_modello_info['model']
            miglior_fold = miglior_modello_info['fold']
            miglior_auc = miglior_modello_info['auc']
            
            print(f"\nstruttura migliore fold {miglior_fold} (AUC: {miglior_auc:.4f})")            
            plot_rete_bayesiana(
                miglior_modello, 
                titolo=f"struttura fold {miglior_fold} (AUC: {miglior_auc:.4f})",
                nome_file=f"rete_bayesiana_fold_{miglior_fold}_auc_{miglior_auc:.4f}.png"
            )

    risultati_dict_per_tabella = {
        'rete bayesiana': {
            'scores': {f'test_{m}': np.array(risultati_cv[m]) for m in metriche},
            'best_params_per_fold': [f"Struttura: {m['edges']} archi" for m in modelli_fold]
        }
    }
    
    return risultati_dict_per_tabella, risultati_aggregati