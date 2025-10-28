import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def esegui_eda_base(df):
    """Esegue EDA di base e stampa informazioni."""
    if df is None:
        print("DataFrame è None, salto EDA.")
        return

    print("\n--- Ispezione Dati Iniziale ---")
    print("Prime 5 righe:")
    print(df.head())
    print("\nInfo Dataset:")
    df.info()
    print("\nStatistiche Descrittive:")
    print(df.describe())
    if 'target' in df.columns:
        print("\nConteggio Valori per 'target':")
        print(df['target'].value_counts())
    else:
        print("\nColonna 'target' non trovata.")

def traccia_distribuzioni(df):
    """Traccia istogrammi per feature numeriche e countplots per categoriche."""
    if df is None:
        print("DataFrame è None, salto plotting.")
        return

    print("\n--- Visualizzazioni di Base ---")
    colonne_numeriche = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    colonne_categoriche = [col for col in df.columns if col not in colonne_numeriche]

    colonne_numeriche_valide = [col for col in colonne_numeriche if col in df.columns]
    if colonne_numeriche_valide:
        print("Tracciamento istogrammi per feature numeriche...")
        df[colonne_numeriche_valide].hist(figsize=(12, 8), bins=15)
        plt.suptitle('Istogrammi Feature Numeriche')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print("Nessuna colonna numerica valida trovata per gli istogrammi.")

    print("\nTracciamento countplots per feature categoriche...")
    colonne_cat_da_tracciare = [col for col in colonne_categoriche if col != 'target' and col in df.columns and df[col].nunique() < 20]
    num_grafici = len(colonne_cat_da_tracciare)
    if num_grafici > 0:
        num_colonne_plot = 3
        num_righe_plot = (num_grafici + num_colonne_plot - 1) // num_colonne_plot
        figura, assi = plt.subplots(nrows=num_righe_plot, ncols=num_colonne_plot, figsize=(15, 4 * num_righe_plot))
        assi = assi.flatten()

        for i, col in enumerate(colonne_cat_da_tracciare):
             sns.countplot(x=col, data=df, ax=assi[i], palette='viridis')
             assi[i].set_title(f'Distribuzione di {col}')
             assi[i].tick_params(axis='x', rotation=45)

        for j in range(num_grafici, len(assi)):
            figura.delaxes(assi[j])

        plt.suptitle('Distribuzioni Feature Categoriche')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print("Nessuna feature categorica adatta trovata per i countplots.")