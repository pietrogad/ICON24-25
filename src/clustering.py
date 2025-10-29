import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
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

    print(f"\napplicazione k-means, k={k}")
    kmeans = KMeans(n_clusters=k, random_state=stato_casuale, n_init=10)
    etichette_cluster = kmeans.fit_predict(X_processato)

    X_con_cluster = X_originale.copy()
    X_con_cluster['cluster'] = etichette_cluster
    X_con_cluster['cluster'] = X_con_cluster['cluster'].astype(str)

    feature_categoriche_ohe_estese = FEATURE_CATEGORICHE_OHE + ['cluster']

    preprocessore_esteso = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), FEATURE_NUMERICHE),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None), feature_categoriche_ohe_estese)
        ],
        remainder='passthrough'
    )

    try:
        X_processato_con_cluster = preprocessore_esteso.fit_transform(X_con_cluster)
        nuovi_nomi_ohe = preprocessore_esteso.named_transformers_['cat'].get_feature_names_out(feature_categoriche_ohe_estese)
        colonne_rimanenti = [col for col in X_con_cluster.columns if col not in FEATURE_NUMERICHE and col not in feature_categoriche_ohe_estese]
        nomi_feature_estese = FEATURE_NUMERICHE + list(nuovi_nomi_ohe) + colonne_rimanenti

        print(f"aggiunta feature 'cluster' (K={k})")

        if len(nomi_feature_estese) != X_processato_con_cluster.shape[1]:
             print(f"discrepanza nomi feature ({len(nomi_feature_estese)}) e colonne ({X_processato_con_cluster.shape[1]}) in clustering")

        return X_processato_con_cluster, preprocessore_esteso, nomi_feature_estese, etichette_cluster
    except Exception as e:
        print(f"errore durante ri-preprocessing con cluster: {e}")
        return None, None, None
