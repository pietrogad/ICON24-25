import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

FEATURE_NUMERICHE = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
FEATURE_CATEGORICHE_OHE = ['cp', 'restecg', 'slope', 'ca', 'thal']
FEATURE_BINARIE = ['sex', 'fbs', 'exang']

def crea_preprocessore(feature_numeriche=FEATURE_NUMERICHE,feature_categoriche_ohe=FEATURE_CATEGORICHE_OHE):
    trasformatore_numerico = StandardScaler()
    trasformatore_categorico = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None)

    preprocessore = ColumnTransformer(
        transformers=[
            ('num', trasformatore_numerico, feature_numeriche),
            ('cat', trasformatore_categorico, feature_categoriche_ohe)
        ],
        remainder='passthrough'
    )
    return preprocessore

def preprocessa_dati(df, colonna_target='target'):
    if df is None or colonna_target not in df.columns:
        print("DataFrame non valido o colonna target mancante, salto preprocessing.")
        return None, None, None, None

    X = df.drop(colonna_target, axis=1)
    y = df[colonna_target]

    required_cols = FEATURE_NUMERICHE + FEATURE_CATEGORICHE_OHE
    missing_cols = [col for col in required_cols if col not in X.columns]
    if missing_cols:
        print(f"le colonne richieste mancano nel DataFrame X: {missing_cols}")
        return None, None, None, None
    
    preprocessore = crea_preprocessore()

    try:
        X_processato = preprocessore.fit_transform(X)

        # nomi feature dopo il preprocessing
        nomi_feature_ohe = preprocessore.named_transformers_['cat'].get_feature_names_out(FEATURE_CATEGORICHE_OHE)
        colonne_rimanenti = [col for col in X.columns if col not in FEATURE_NUMERICHE and col not in FEATURE_CATEGORICHE_OHE]
        nomi_feature_processate = FEATURE_NUMERICHE + list(nomi_feature_ohe) + colonne_rimanenti

        if len(nomi_feature_processate) != X_processato.shape[1]:
             print(f"discrepanza nomi feature ({len(nomi_feature_processate)}) e colonne ({X_processato.shape[1]})!")

        return X_processato, y, preprocessore, nomi_feature_processate
    except Exception as e:
        print(f"errore durante il preprocessing: {e}")
        return None, None, None, None
