import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from preprocessing import FEATURE_NUMERICHE, FEATURE_CATEGORICHE_OHE, FEATURE_BINARIE

def trova_k_ottimale(X_processato, max_k=10, stato_casuale=42):
    print(f"\nricerca del k Ottimale")
    inerzie = []
    range_k = range(1, max_k + 1)
    for k in range_k:
        kmeans = KMeans(n_clusters=k, random_state=stato_casuale, n_init=10)
        kmeans.fit(X_processato)
        inerzie.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range_k, inerzie, marker='o')
    plt.title('metodo elbow per K Ottimale')
    plt.xlabel('numero di cluster (K)')
    plt.ylabel('inerzia')
    plt.xticks(range_k)
    plt.grid(True)
    plt.show()

def applica_kmeans_e_aggiungi_feature(X_originale, X_processato, k, stato_casuale=42):
    print(f"\k-means con k={k}")

    kmeans = KMeans(
        n_clusters=k, 
        random_state=stato_casuale, 
        n_init=30,
        max_iter=500,
        algorithm='elkan'
    )
    
    etichette_cluster = kmeans.fit_predict(X_processato)
    X_con_cluster = X_originale.copy()
    X_con_cluster['cluster'] = etichette_cluster.astype(str)
    distanze = kmeans.transform(X_processato)
    distanza_proprio_cluster = np.array([distanze[i, etichette_cluster[i]] for i in range(len(etichette_cluster))])
    X_con_cluster['cluster_distance'] = distanza_proprio_cluster
    distanza_minima = distanze.min(axis=1)
    X_con_cluster['nearest_cluster_distance'] = distanza_minima

    with np.errstate(divide='ignore', invalid='ignore'):
        uncertainty_ratio = distanza_proprio_cluster / (distanza_minima + 1e-10)
        uncertainty_ratio[np.isnan(uncertainty_ratio)] = 1.0
    X_con_cluster['cluster_uncertainty'] = uncertainty_ratio

    cluster_sizes = pd.Series(etichette_cluster).value_counts().to_dict()
    X_con_cluster['cluster_size'] = [cluster_sizes[c] for c in etichette_cluster]
    
    X_con_cluster['cluster_density'] = X_con_cluster['cluster_size'] / len(etichette_cluster)

    for i in range(k):
        weights = 1.0 / (distanze[:, i] + 1e-10)
        weights_normalized = weights / weights.sum()
        X_con_cluster[f'cluster_prob_{i}'] = weights_normalized
    
    threshold = np.percentile(distanza_proprio_cluster, 90)
    X_con_cluster['is_outlier'] = (distanza_proprio_cluster > threshold).astype(int)

    nuove_feature_numeriche = [
        'cluster_distance', 
        'nearest_cluster_distance', 
        'cluster_uncertainty',
        'cluster_size',
        'cluster_density'
    ] + [f'cluster_prob_{i}' for i in range(k)]
    
    feature_numeriche_estese = FEATURE_NUMERICHE + nuove_feature_numeriche
    feature_categoriche_ohe_estese = FEATURE_CATEGORICHE_OHE + ['cluster']
    feature_binarie_estese = FEATURE_BINARIE + ['is_outlier']
    
    preprocessore_esteso = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_numeriche_estese),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feature_categoriche_ohe_estese)
        ],
        remainder='passthrough'
    )
    
    try:
        X_processato_con_cluster = preprocessore_esteso.fit_transform(X_con_cluster)
        
        nuovi_nomi_ohe = preprocessore_esteso.named_transformers_['cat'].get_feature_names_out(feature_categoriche_ohe_estese)
        colonne_rimanenti = [col for col in X_con_cluster.columns 
                            if col not in feature_numeriche_estese 
                            and col not in feature_categoriche_ohe_estese]
        
        nomi_feature_estese = list(feature_numeriche_estese) + list(nuovi_nomi_ohe) + colonne_rimanenti

        print(f"\nstat clustering")
        for i in range(k):
            mask = etichette_cluster == i
            size = mask.sum()
            avg_distance = distanza_proprio_cluster[mask].mean()
            print(f"  clust {i}: {size} punti, distanza media: {avg_distance:.3f}")
        
        print(f"\noutlier {X_con_cluster['is_outlier'].sum()} ({X_con_cluster['is_outlier'].mean()*100:.1f}%)")
        
        return X_processato_con_cluster, preprocessore_esteso, nomi_feature_estese, etichette_cluster
        
    except Exception as e:
        print(f"errore nel preprocessing: {e}")
        return None, None, None, None

def clustering_robust(X_processato, k, stato_casuale=42):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_processato)
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=50,
        max_iter=1000,
        random_state=stato_casuale,
        algorithm='elkan'
    )
    etichette = kmeans.fit_predict(X_scaled)
    inerzia = kmeans.inertia_
    print(f"Inerzia finale: {inerzia:.2f}")
    
    return etichette, kmeans, scaler