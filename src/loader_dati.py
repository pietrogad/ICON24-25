import pandas as pd
import numpy as np
import os

def pulisci_dati_iniziali(df):
    df.columns = df.columns.str.strip()

    #rimuovo colonne inutili
    colonne_da_rimuovere = ['id', 'dataset']
    colonne_esistenti_da_rimuovere = [col for col in colonne_da_rimuovere if col in df.columns]
    if colonne_esistenti_da_rimuovere:
        df = df.drop(columns=colonne_esistenti_da_rimuovere)

    #cnverto colonne numeriche
    colonne_numeriche_da_convertire = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    for col in colonne_numeriche_da_convertire:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Colonna '{col}' non trovata.")

    #valori booleani
    if 'fbs' in df.columns:
        df['fbs'] = df['fbs'].astype(str).str.strip().str.upper()
        df['fbs'] = df['fbs'].apply(lambda x: 1 if x == 'TRUE' else 0 if x == 'FALSE' else np.nan if pd.isna(x) else x)
        df['fbs'] = pd.to_numeric(df['fbs'], errors='coerce')
    if 'exang' in df.columns:
        df['exang'] = df['exang'].astype(str).str.strip().str.upper()
        df['exang'] = df['exang'].apply(lambda x: 1 if x == 'TRUE' else 0 if x == 'FALSE' else x)
        df['exang'] = pd.to_numeric(df['exang'], errors='coerce')
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.strip()
        df['sex'] = df['sex'].apply(lambda x: 1 if x.upper() == 'MALE' else 0 if x.upper() == 'FEMALE' else np.nan)
        print("convertita 'sex' in numerico 0/1.")
    if 'thal' in df.columns:
        df['thal'] = df['thal'].astype(str).str.strip()

    # valori NaN
    print(f"\nvalori mancanti prima \n{df.isnull().sum()[df.isnull().sum() > 0]}")
    colonne_con_nan = df.isnull().sum()[df.isnull().sum() > 0].index.tolist()
    colonne_numeriche_note = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in colonne_con_nan:
        if col in colonne_numeriche_note and col in df.columns:
             mediana = df[col].median()
             df[col] = df[col].fillna(mediana)
        elif col in df.columns :
             if not df[col].mode().empty:
                 moda = df[col].mode()[0]
                 df[col] = df[col].fillna(moda)
             else:
                 print(f"impossibile calcolare la moda per '{col}'.")
                 df[col] = df[col].fillna('Sconosciuto')

    #feature num -> target
    if 'num' in df.columns:
        df = df.rename(columns={'num': 'target'})
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    else:
        print("colonna 'num' non trovata.")
        return None
    
    print("\nverifica dopo pulizia")
    df.info() # Controlla che i Dtype siano corretti ora
    nan_restanti = df.isnull().sum().sum()
    print(f"\nvalori mancanti dopo la pulizia: {nan_restanti}")
    if nan_restanti > 0:
        print(df.isnull().sum()[df.isnull().sum() > 0])

    return df

def carica_e_pulisci_dati(percorso_file):
    if not os.path.exists(percorso_file):
        print(f"file non trovato")
        return None
    try:
        df = pd.read_csv(percorso_file, na_values='?')
        df_pulito = pulisci_dati_iniziali(df)
        return df_pulito
    except Exception as e:
        print(f"errore durante caricamento o pulizia: {e}")
        return None