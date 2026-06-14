import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# KROK 1: Wczytanie surowych danych numerycznych
# ==========================================
print("1. Wczytywanie surowych danych i wyliczanie Delt...")
df = pd.read_csv(r'C:\Users\mikdy\Documents\uczelnia\4 semestr\eksploracja danych\endangered_species\project1\data\normalizacja\climate_deforestation_supervised_modeling_base.csv')

# Mapowanie Klastrów
cluster_mapping = {
    'AC': 0, 'AM': 0, 'AP': 0, 'MA': 0, 'PA': 0, 'RO': 0, 'TO': 0,
    'PR': 1, 'RS': 1, 'SC': 1, 'SP': 1,
    'DF': 2, 'ES': 2, 'GO': 2, 'MG': 2, 'MS': 2,
    'AL': 3, 'BA': 3, 'CE': 3, 'MT': 3, 'PB': 3, 'PE': 3, 'RN': 3, 'RR': 3, 'SE': 3
}
df['Working_Cluster'] = df['State'].map(cluster_mapping)

# ==========================================
# KROK 2: Wyliczenie Anomalii (Delt)
# ==========================================
# 1. Średnia historyczna dla stanu i miesiąca
baseline = df.groupby(['State', 'Month'])[['Max_Air_Temperature', 'Total_Precipitation', 'Min_Relative_Humidity']].mean().reset_index()
baseline.columns = ['State', 'Month', 'Norm_Temp', 'Norm_Precip', 'Norm_Hum']

# 2. Łączymy z główną tabelą i liczymy Deltę (Różnicę)
df = pd.merge(df, baseline, on=['State', 'Month'])
df['Delta_Temp'] = df['Max_Air_Temperature'] - df['Norm_Temp']
df['Delta_Precip'] = df['Total_Precipitation'] - df['Norm_Precip']
df['Delta_Hum'] = df['Min_Relative_Humidity'] - df['Norm_Hum']

# ==========================================
# KROK 3: Dyskretyzacja ZMIAN (przygotowanie pod Koszyk)
# ==========================================
print("2. Zamiana liczb na kategorie anomalii...")

# Używamy np. qcut, aby podzielić odchylenia na 3 grupy: Spadek, Norma, Wzrost
# Jeśli delta opadów jest mocno ujemna -> "Opad_Spadek"
df['Temp_Anomaly'] = pd.qcut(df['Delta_Temp'], q=3, labels=['Temp_Spadek', 'Temp_Norma', 'Temp_Wzrost'])
df['Precip_Anomaly'] = pd.qcut(df['Delta_Precip'], q=3, labels=['Opad_Spadek', 'Opad_Norma', 'Opad_Wzrost'])
df['Hum_Anomaly'] = pd.qcut(df['Delta_Hum'], q=3, labels=['Wilgotnosc_Spadek', 'Wilgotnosc_Norma', 'Wilgotnosc_Wzrost'])

# Kategoryzujemy wylesianie
df['Deforestation_Level'] = pd.qcut(df['Deforestation_ha'], q=3, labels=['Deforestation_Low', 'Deforestation_Medium', 'Deforestation_Critical'])

# Budujemy czystą tabelę pod Apriori (tylko klastry i nowe kategorie)
mba_data = df[['Working_Cluster', 'Deforestation_Level', 'Temp_Anomaly', 'Precip_Anomaly', 'Hum_Anomaly']].copy()

# ==========================================
# KROK 4: Algorytm Apriori dla Klastrów
# ==========================================
def run_mba_on_deltas(data, cluster_id, cluster_name):
    print(f"\n==========================================")
    print(f" ANALIZA ANOMALII KLIMATYCZNYCH: {cluster_name.upper()} (Klaster {cluster_id})")
    print(f"==========================================")
    
    # Wyciągamy dane dla klastra i usuwamy numer klastra
    cluster_df = data[data['Working_Cluster'] == cluster_id].drop(columns=['Working_Cluster'])
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(cluster_df)
    
    # Szukamy wzorców (wystarczy 3% by złapać rzadkie anomalie)
    frequent_itemsets = apriori(df_encoded, min_support=0.03, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("Zbyt mało powtarzalnych anomalii.")
        return
        
    # Szukamy powiązań
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
    
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # FILTR: Interesuje nas TYLKO wpływ krytycznego wylesiania na odchylenia od normy
    impact_rules = rules[rules['antecedents'].str.contains('Deforestation_Critical', na=False)]
    
    # Wyrzucamy z wyników sytuacje, gdzie konsekwencją jest "Norma" (szukamy pęknięć systemu)
    impact_rules = impact_rules[~impact_rules['consequents'].str.contains('Norma|Deforestation', na=False, regex=True)]
    
    impact_rules = impact_rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])
    
    if impact_rules.empty:
        print("Brak dowodów na to, że wylesianie wywołuje anomalie w tym regionie.")
    else:
        for index, row in impact_rules.head(5).iterrows():
            print(f"JEŚLI: [{row['antecedents']}]")
            print(f"--TO-> [{row['consequents']}]")
            print(f"       [Pewność: {row['confidence']*100:.1f}% | Moc (Lift): {row['lift']:.2f}]\n")

# Uruchamiamy
run_mba_on_deltas(mba_data, cluster_id=0, cluster_name="Amazonia i Łuk Wylesiania")
run_mba_on_deltas(mba_data, cluster_id=1, cluster_name="Podzwrotnikowe Południe")