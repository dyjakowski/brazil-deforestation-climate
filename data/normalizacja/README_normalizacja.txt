Normalizacja i przygotowanie danych: climate-deforestation-merged(1).csv

Wykonane pliki:
1. climate_deforestation_row_level_zscore_features.csv
   - Dane miesięczne: 1 wiersz = obserwacja State-Year-Month.
   - Zmienione: dodano Month_sin, Month_cos oraz klasy Deforestation_Class_Median i Deforestation_Class_Q75.
   - Z-score wykonano wyłącznie na cechach klimatycznych: Total_Precipitation, Atmospheric_Pressure_Station, Global_Radiation, Air_Temperature, Max_Air_Temperature, Min_Air_Temperature, Relative_Humidity, Max_Relative_Humidity, Min_Relative_Humidity, Hourly_Wind_Speed, Max_Wind_Gust.
   - Nie standaryzowano Year, Month, State, Deforestation_ha ani CO2_Emissions.
   - Użycie: EDA, korelacje cech wejściowych, PCA, metody odległościowe na poziomie miesięcznych obserwacji.

2. climate_deforestation_supervised_modeling_base.csv
   - Bez globalnej standaryzacji cech, aby uniknąć data leakage w klasyfikacji.
   - Użycie: klasyfikacja z Pipeline, gdzie scaler.fit() robi się tylko na treningu.
   - CO2_Emissions usunięto z cech wejściowych, bo jest silnie zależne od Deforestation_ha i grozi przeciekiem informacyjnym.

3. climate_deforestation_state_level_clustering_zscore.csv
   - Dane zagregowane do poziomu stanu: 1 wiersz = 1 State.
   - Z-score wykonano po agregacji, na średnim wylesianiu i średnich cechach klimatycznych.
   - Użycie: k-Means/HAC do grupowania stanów.

4. climate_deforestation_association_rules_discretized.csv
   - Dane zdyskretyzowane do reguł asocjacyjnych.
   - Cechy klimatyczne: Low/Medium/High według tercyli.
   - Deforestation_Level: Low/Medium/High/Critical według kwartylów.

5. climate_deforestation_scaling_metadata.csv
   - Parametry skalowania: mean, std_pop, min, max, kwartyle/progi.

Progi Deforestation_ha:
- q25 = 565.780000
- median/q50 = 2334.630000
- q75 = 8851.500000

Uwaga metodyczna:
Do klasyfikacji nie używaj globalnie przeskalowanego pliku jako jedynego źródła trening/test. Poprawnie: train_test_split -> fit scaler tylko na X_train -> transform X_train i X_test. Najbezpieczniej zrobić to przez sklearn Pipeline.
