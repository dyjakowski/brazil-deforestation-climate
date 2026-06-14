import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# ==========================================
# krok 1: wczytywanie danych
# ==========================================
print("Loading data...")
df = pd.read_csv(r'C:\Users\mikdy\Documents\uczelnia\4 semestr\eksploracja danych\endangered_species\project1\data\normalizacja\climate_deforestation_state_level_clustering_seasonal.csv')

# Wybór ustandaryzowanych cech klimatycznych oraz nowych zmiennych sezonowych.
# Algorytm grupuje stany tylko na podstawie klimatu.
clustering_features = [
    'Total_Precipitation_mean_z', 
    'Atmospheric_Pressure_Station_mean_z',
    'Global_Radiation_mean_z', 
    'Relative_Humidity_mean_z', 
    'Hourly_Wind_Speed_mean_z', 
    'Summer_Temperature', 
    'Winter_Temperature', 
    'Avg_Seasonal_Range'
]

X = df[clustering_features]
states = df['State'].values

# ==========================================
# krok 2: Metoda Łokcia - szukanie optymalnej liczby klastrów
# ==========================================
print("Calculating optimal number of clusters (Elbow Method)...")
wcss = []
k_range = range(2, 10)

# Obliczanie inercji dla różnej liczby klastrów (od 2 do 9)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Rysowanie wykresu wyboru liczby klastrów
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method: Choosing the number of clusters (k)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Sum of squared errors)')
plt.grid(True)
plt.show()

# ==========================================
# krok 3 i 4: metoda K-Means oraz Silhouette Score
# ==========================================
# Ustawiamy docelową liczbę klastrów (po analizie wykresu łokcia, k=4)
target_k = 4 
kmeans_final = KMeans(n_clusters=target_k, random_state=42, n_init=10)
df['Cluster_kMeans'] = kmeans_final.fit_predict(X)

# Ocena jakości podziału (im bliżej 1, tym lepiej odseparowane klastry)
sil_score = silhouette_score(X, df['Cluster_kMeans'])
print("\n--- K-MEANS CLUSTERING RESULTS ---")
print(f"Silhouette Score for k={target_k}: {sil_score:.3f}")

# ==========================================
# krok 5: Grupowanie hierarchiczne (HAC) - Dendrogram
# ==========================================
print("\nGenerating HAC Dendrogram...")

# Metoda Warda minimalizuje wariancję wewnątrz nowo tworzonych klastrów
Z = linkage(X, method='ward')

plt.figure(figsize=(12, 6))
plt.title('Hierarchical Dendrogram of Brazilian States (Climate & Seasonality)')
plt.xlabel('State')
plt.ylabel('Distance (Feature space)')
dendrogram(
    Z,
    labels=states,
    leaf_rotation=90.,
    leaf_font_size=10.,
    color_threshold=max(Z[:,2]) * 0.7 
)
plt.axhline(y=max(Z[:,2]) * 0.7, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# ==========================================
# krok 6: Profilowanie klastrów i interpretacja
# ==========================================
print("\n--- K-MEANS CLUSTER PROFILING ---")

# Agregacja danych w celu zbadania, czym charakteryzuje się dany klaster (i jakie ma wylesianie)
cluster_analysis = df.groupby('Cluster_kMeans').agg(
    State_Count=('State', 'count'),
    Mean_Deforestation_ha=('Deforestation_mean_ha', 'mean'),
    Mean_Summer_Temp=('Summer_Temperature', 'mean'),
    Mean_Winter_Temp=('Winter_Temperature', 'mean'),
    Mean_Seasonal_Range=('Avg_Seasonal_Range', 'mean'),
    Mean_Precipitation_Z=('Total_Precipitation_mean_z', 'mean')
).round(2)

print(cluster_analysis)

# Wyświetlenie przypisania konkretnych stanów do poszczególnych klastrów
for i in range(target_k):
    states_in_cluster = df[df['Cluster_kMeans'] == i]['State'].tolist()
    print(f"\nCluster {i}:")
    print(f"States: {', '.join(states_in_cluster)}")


# ==========================================
# krok 7: Określenie optymalnej liczby składowych PCA (Scree Plot)
# ==========================================
print("\nCalculating PCA explained variance for all components...")

# Tworzymy PCA bez limitu wymiarów, by zbadać wszystkie
pca_full = PCA(random_state=42)
pca_full.fit(X)

# Pobieranie ilości wiedzy (wariancji) wyjaśnianej przez każdą składową
explained_variance = pca_full.explained_variance_ratio_
# Obliczanie skumulowanej wiedzy (np. składowa 1 + składowa 2 + ...)
cumulative_variance = np.cumsum(explained_variance)

# Wyświetlanie statystyk w konsoli
print("--- PCA EXPLAINED VARIANCE ---")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i+1}: {var*100:.2f}% (Cumulative: {cum_var*100:.2f}%)")

# Wykres osypiska
plt.figure(figsize=(10, 6))

# Wykres słupkowy dla pojedynczych składowych
plt.bar(
    range(1, len(explained_variance) + 1), 
    explained_variance, 
    alpha=0.6, 
    align='center',
    label='Individual explained variance'
)

# Wykres liniowy dla wariancji skumulowanej
plt.step(
    range(1, len(cumulative_variance) + 1), 
    cumulative_variance, 
    where='mid',
    color='red', 
    marker='o',
    label='Cumulative explained variance'
)

# Rysowanie linii pomocniczej na poziomie 90%
plt.axhline(y=0.90, color='gray', linestyle='--', label='90% Information Threshold')

plt.title('Scree Plot: Explained Variance by Principal Components')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_variance) + 1))
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# ==========================================
# krok 8: Wizualizacja klastrów w 2D za pomocą PCA
# ==========================================
print("\nGenerating 2D K-Means Visualization using PCA...")

# Redukcja 8 wymiarów pogodowych do 2 składowych, aby narysować wykres 2D
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(X)

df['PCA_1'] = pca_result[:, 0]
df['PCA_2'] = pca_result[:, 1]

# Obliczenie, ile oryginalnej wiedzy o klimacie zachowaliśmy w 2D
explained_variance = pca.explained_variance_ratio_
print(f"PCA explained variance: PC1 = {explained_variance[0]*100:.2f}%, PC2 = {explained_variance[1]*100:.2f}%")

# Rysowanie wykresu punktowego
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PCA_1', 
    y='PCA_2', 
    hue='Cluster_kMeans', 
    palette='Set1', 
    data=df, 
    s=100, 
    alpha=0.8
)

# Podpisywanie poszczególnych kropek skrótami stanów
for i in range(len(df)):
    plt.text(
        df['PCA_1'][i] + 0.1, 
        df['PCA_2'][i] + 0.1, 
        df['State'][i], 
        fontsize=9
    )

plt.title('K-Means Clusters in 2D Space (PCA Projection)')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}%)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}%)')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()