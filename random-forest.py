import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# =====================================================
# 1. Wczytanie danych
# =====================================================
print("1. Wczytywanie i agregacja danych...")
df = pd.read_csv(r"C:\Users\mikdy\Documents\uczelnia\4 semestr\eksploracja danych\endangered_species\project1\data\normalizacja\climate_deforestation_supervised_modeling_base.csv")

# =====================================================
# 2. Agregacja miesięcy -> lata
# =====================================================
annual = (
    df.groupby(["State", "Year"])
      .agg(
          Annual_Deforestation=("Deforestation_ha", "sum"),
          Mean_Temperature=("Air_Temperature", "mean"),
          Mean_Precipitation=("Total_Precipitation", "mean"),
          Mean_Humidity=("Relative_Humidity", "mean"),
          Mean_Radiation=("Global_Radiation", "mean")
      )
      .reset_index()
)

# =====================================================
# 3. Roczna anomalia temperatury (Norma dla stanu)
# =====================================================
state_mean_temp = annual.groupby("State")["Mean_Temperature"].mean()

annual["Temperature_Anomaly"] = (
    annual["Mean_Temperature"] - annual["State"].map(state_mean_temp)
)

# =====================================================
# 4. Sortowanie i Grupowanie
# =====================================================
annual = annual.sort_values(["State", "Year"]).reset_index(drop=True)
group = annual.groupby("State")

# =====================================================
# 5. Opóźnienia wylesienia (Lagi)
# =====================================================
annual["Defor_t"] = annual["Annual_Deforestation"]
annual["Defor_t-1"] = group["Annual_Deforestation"].shift(1)
annual["Defor_t-2"] = group["Annual_Deforestation"].shift(2)
annual["Defor_t-3"] = group["Annual_Deforestation"].shift(3)

# =====================================================
# 6. Skumulowane wylesienie i 4-letnia suma krocząca
# =====================================================
annual["Cumulative_Deforestation"] = group["Annual_Deforestation"].cumsum()
annual["Rolling4_Deforestation"] = (
    group["Annual_Deforestation"].rolling(window=4, min_periods=1).sum().reset_index(level=0, drop=True)
)

# =====================================================
# 7. Ważone wylesienie
# =====================================================
annual["Weighted_Deforestation"] = (
    0.40 * annual["Defor_t"] + 
    0.30 * annual["Defor_t-1"] + 
    0.20 * annual["Defor_t-2"] + 
    0.10 * annual["Defor_t-3"]
)

# =====================================================
# 8. NOWY TARGET: Przewidywanie ZMIANY (Delty) na kolejny rok
# =====================================================
# O ile zmieni się anomalia w przyszłym roku względem anomalii w tym roku?
anomalia_przyszla = group["Temperature_Anomaly"].shift(-1)
annual["Target_Delta"] = anomalia_przyszla - annual["Temperature_Anomaly"]

# =====================================================
# 9. Usunięcie braków
# =====================================================
annual = annual.dropna().reset_index(drop=True)

# =====================================================
# 10. Podział czasowy (Train <= 2018, Test > 2018)
# =====================================================
print("2. Podział na zbiór treningowy i testowy...")
train = annual[annual["Year"] <= 2018].copy()
test = annual[annual["Year"] > 2018].copy()

# =====================================================
# 11. NAPRAWIONE FEATURES (Wersja dająca R² ~ 0.27)
# =====================================================
features = [
    # "Year" <- USUNIĘTE
    "Annual_Deforestation",
    "Defor_t-1",
    "Defor_t-2",
    "Defor_t-3",
    "Cumulative_Deforestation",
    "Rolling4_Deforestation",
    "Weighted_Deforestation",
    "Mean_Precipitation",
    "Mean_Humidity",
    "Mean_Radiation",
    "Temperature_Anomaly" # <-- OBECNA ANOMALIA WŁĄCZONA
]

X_train = train[features]
y_train = train["Target_Delta"]

X_test = test[features]
y_test = test["Target_Delta"]

# =====================================================
# 12. Modelowanie (Random Forest Regressor)
# =====================================================
print("3. Trenowanie modelu Random Forest...")
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# =====================================================
# 13. Ewaluacja Modelu
# =====================================================
y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- WYNIKI PREDYKCJI ZMIANY TEMPERATURY (DELTA) ---")
print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.4f} °C (Średni błąd przewidywania skoku temperatury)")

# =====================================================
# 14. Wizualizacja Feature Importance
# =====================================================
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Co napędza zmianę temperatury w kolejnym roku? (Feature Importance)")
plt.bar(range(X_train.shape[1]), importances[indices], align="center", color="coral")
plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
plt.ylabel("Waga (Znaczenie) zmiennej")
plt.tight_layout()
plt.show()