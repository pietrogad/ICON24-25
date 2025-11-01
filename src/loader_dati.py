import pandas as pd
import numpy as np
import os

def carica_e_pulisci_base(percorso_file):
    print("load e pulizia")
    if not os.path.exists(percorso_file):
        print(f"file non trovato {percorso_file}")
        return None
    try:
        df = pd.read_csv(percorso_file, na_values='?')
        df.columns = df.columns.str.strip()
        print(f"dataset caricato")

        colonne_da_rimuovere = ['id', 'dataset']
        colonne_esistenti_da_rimuovere = [col for col in colonne_da_rimuovere if col in df.columns]
        if colonne_esistenti_da_rimuovere:
            df = df.drop(columns=colonne_esistenti_da_rimuovere)

        colonne_numeriche_da_convertire = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
        for col in colonne_numeriche_da_convertire:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'fbs' in df.columns:
            df['fbs'] = df['fbs'].astype(str).str.strip().str.upper()
            df['fbs'] = df['fbs'].apply(lambda x: 1 if x == 'TRUE' else 0 if x == 'FALSE' else np.nan if pd.isna(x) or x == 'NAN' else x)
            df['fbs'] = pd.to_numeric(df['fbs'], errors='coerce')
        if 'exang' in df.columns:
            df['exang'] = df['exang'].astype(str).str.strip().str.upper()
            df['exang'] = df['exang'].apply(lambda x: 1 if x == 'TRUE' else 0 if x == 'FALSE' else x)
            df['exang'] = pd.to_numeric(df['exang'], errors='coerce')
        if 'sex' in df.columns:
            df['sex'] = df['sex'].astype(str).str.strip()
            df['sex'] = df['sex'].apply(lambda x: 1 if x.upper() == 'MALE' else 0 if x.upper() == 'FEMALE' else np.nan)

        colonne_categoriche = ['cp', 'restecg', 'slope', 'thal']
        for col in colonne_categoriche:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', np.nan)

        if 'num' in df.columns:
            df = df.rename(columns={'num': 'target'})
            df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
            df['target'] = df['target'].astype('int64')
        else:
            print("colonna non trovata")
            return None
        
        for col in ['sex', 'fbs', 'exang', 'cp', 'restecg', 'slope', 'ca', 'thal']:
            if col in df.columns:
                if pd.api.types.is_float_dtype(df[col]):
                    try:
                        df[col] = df[col].astype('Int64')
                    except TypeError:
                         pass
                if col in ['cp', 'restecg', 'slope', 'thal']:
                    df[col] = df[col].astype(object)

        print("pulizia ok")
        df.info()
        print(f"\nvalori mancanti: {df.isnull().sum().sum()}")
        return df

    except Exception as e:
        print(f"errore load-pulizia: {e}")
        return None

def riempimento_dati(df_con_nan):
    if df_con_nan is None:
        return None
    df_imputato = df_con_nan.copy()
    
    colonne_con_nan = df_imputato.isnull().sum()[df_imputato.isnull().sum() > 0].index.tolist()
    colonne_numeriche_note = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    
    for col in colonne_con_nan:
        if col in colonne_numeriche_note and col in df_imputato.columns:
             mediana = df_imputato[col].median()
             df_imputato[col] = df_imputato[col].fillna(mediana)
        elif col in df_imputato.columns:
             if not df_imputato[col].mode().empty:
                 moda = df_imputato[col].mode()[0]
                 df_imputato[col] = df_imputato[col].fillna(moda)
             else:
                 df_imputato[col] = df_imputato[col].fillna(df_imputato[col].dtype.type())
    
    print(f"nan restanti: {df_imputato.isnull().sum().sum()}")
    return df_imputato