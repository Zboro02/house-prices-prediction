# Sprawozdanie z projektu: predykcja cen nieruchomości

### Streszczenie projektu

Celem projektu było stworzenie modelu regresyjnego do predykcji cen sprzedaży nieruchomości na podstawie zbioru danych zawierającego szczegółowe informacje o ich cechach. Wykorzystano głęboką sieć neuronową zaimplementowaną w bibliotece PyTorch. Kluczowe etapy projektu obejmowały wczytanie danych, ich dogłębne przetworzenie (w tym imputację brakujących wartości, kodowanie zmiennych kategorycznych oraz inżynierię cech), budowę i trening modelu sieci neuronowej, a także jego ewaluację i optymalizację hiperparametrów. Ostateczny model osiągnął współczynnik determinacji R² na poziomie około 0.855 na zbiorze walidacyjnym, co wskazuje na wysoką zdolność predykcyjną.

### 1. Wprowadzenie

Predykcja cen nieruchomości jest istotnym problemem w dziedzinie analizy danych i uczenia maszynowego, mającym zastosowanie zarówno w sektorze prywatnym (np. dla agencji nieruchomości, kupujących, sprzedających), jak i publicznym. Złożoność tego zadania wynika z dużej liczby czynników wpływających na cenę oraz ich wzajemnych interakcji. Projekt ten skupia się na zastosowaniu nowoczesnych technik uczenia głębokiego do rozwiązania tego problemu.

### 2. Przygotowanie i przetwarzanie danych

Dane wejściowe składały się ze zbioru treningowego (`train.csv`) zawierającego cechy nieruchomości wraz z ich cenami sprzedaży oraz zbioru testowego (`test.csv`) z cechami, dla których należało przewidzieć ceny.

#### 2.1. Wstępna analiza i czyszczenie

*   **Wczytanie danych:** Dane zostały wczytane przy użyciu biblioteki Pandas.
*   **Identyfikatory:** Kolumna `Id` została usunięta ze zbiorów danych używanych do treningu, a identyfikatory ze zbioru testowego zostały zachowane do generowania pliku z predykcjami.

#### 2.2. Inżynieria cech (feature engineering)

W celu zwiększenia mocy predykcyjnej modelu, stworzono szereg nowych cech na podstawie istniejących:
*   **Wiek domu i remontu:**
    *   `HouseAge = YrSold - YearBuilt`
    *   `RemodAge = YrSold - YearRemodAdd` (z korektą, jeśli nie było remontu)
    *   `IsRemodeled`: Cechą binarną wskazującą, czy dom był remontowany.
*   **Agregacja powierzchni:**
    *   `TotalSF = GrLivArea + TotalBsmtSF` (całkowita powierzchnia)
    *   `TotalFinishedSF = GrLivArea + BsmtFinSF1 + BsmtFinSF2` (całkowita powierzchnia wykończona)
    *   `TotalPorchSF`: Suma powierzchni różnych typów ganków/werand.
*   **Agregacja łazienek:**
    *   `TotalBath`: Sumaryczna liczba łazienek (pełne liczone jako 1, połówkowe jako 0.5).
*   **Cechy binarne dla udogodnień:**
    *   `HasPool`, `HasFireplace`, `HasGarage`: Wskazujące na obecność basenu, kominka, garażu.
*   **Stosunki powierzchni:**
    *   `BsmtFinToTotalBsmt_Ratio`: Stosunek wykończonej powierzchni piwnicy do jej całkowitej powierzchni.
    *   `LotArea_x_GrLivArea_Ratio`: Stosunek powierzchni mieszkalnej do powierzchni działki.
*   **Połączone wskaźniki jakości/kondycji:**
    *   `OverallGrade = OverallQual * OverallCond`
    *   `GarageGrade = GarageQual * GarageCond` (po wcześniejszym zmapowaniu jakości na liczby)
    *   `ExterGrade = ExterQual * ExterCond` (po wcześniejszym zmapowaniu jakości na liczby)

    *Ważne: Podczas tworzenia tych cech, zbiory treningowy i testowy zostały tymczasowo połączone, aby zapewnić spójność obliczeń (np. dla cech zależnych od `YrSold`). Następnie zostały ponownie rozdzielone. W tej iteracji kluczowe okazało się **nieusuwanie oryginalnych cech** po stworzeniu nowych, zagregowanych.*

#### 2.3. Transformacja zmiennej docelowej

*   Zmienna docelowa `SalePrice` charakteryzowała się skośnym rozkładem. Aby przybliżyć jej rozkład do normalnego i ustabilizować wariancję, zastosowano transformację logarytmiczną: `SalePrice = np.log1p(SalePrice)`. Predykcje modelu są następnie transformowane z powrotem za pomocą `np.expm1()`.

#### 2.4. Obsługa brakujących wartości (imputacja)

Brakujące wartości zostały uzupełnione w następujący sposób, na połączonym zbiorze danych (cechy treningowe i testowe):
*   **`LotFrontage`:** Medianą pogrupowaną według `Neighborhood`, a pozostałe globalną medianą.
*   **Kolumny kategoryczne, gdzie `NA` ma znaczenie** (np. `Alley`, `BsmtQual`, `FireplaceQu`): Uzupełnione stringiem `'None'`.
*   **Pozostałe kolumny numeryczne:** Medianą danej kolumny.
*   **Pozostałe kolumny kategoryczne:** Modą (najczęściej występującą wartością) danej kolumny.
*   **Ostateczne czyszczenie:** Wszelkie pozostałe (nieliczne) wartości `NA` zostały wypełnione zerem.

#### 2.5. Kodowanie zmiennych kategorycznych

*   **Kodowanie porządkowe (ordinal encoding):** Zmienne kategoryczne, których wartości mają naturalną kolejność (np. oceny jakości od `Po` - słaba do `Ex` - doskonała), zostały zmapowane na wartości liczbowe (np. 0-5). Mapowania zostały zdefiniowane na podstawie pliku `data_description.txt`.
*   **Kodowanie one-hot (one-hot encoding):** Pozostałe zmienne kategoryczne nominalne (w tym `MSSubClass`, która mimo bycia liczbą, reprezentuje kategorie) zostały przekształcone za pomocą techniki one-hot encoding (`pd.get_dummies`), tworząc nowe binarne kolumny dla każdej unikalnej wartości kategorii. Ustawiono `dtype=int` dla nowo tworzonych kolumn.

#### 2.6. Podział na zbiory treningowy i walidacyjny

Przetworzony zbiór treningowy (cechy `X` i transformowana cena `y`) został podzielony na zbiór treningowy (używany bezpośrednio do uczenia modelu) i zbiór walidacyjny (używany do monitorowania procesu uczenia, early stopping i oceny generalizacji modelu). Użyto podziału 80/20 (`validation_size = 0.2`) z ustalonym `random_state` dla odtwarzalności.

#### 2.7. Skalowanie cech

Wszystkie cechy numeryczne w zbiorach `X_train`, `X_val` i `X_test` zostały przeskalowane za pomocą `StandardScaler` ze biblioteki `scikit-learn`. Skaler został dopasowany (`fit_transform`) tylko na zbiorze `X_train`, a następnie użyty do transformacji (`transform`) zbiorów `X_val` i `X_test`, aby uniknąć wycieku informacji ze zbioru walidacyjnego i testowego do procesu treningowego.

### 3. Budowa i trening modelu sieci neuronowej

#### 3.1. Architektura modelu

Zastosowano wielowarstwową sieć neuronową typu perceptron (MLP) z następującą architekturą (konfiguracja, która dała najlepsze wyniki):
*   **Warstwy ukryte:** Cztery warstwy ukryte z malejącą liczbą neuronów: 512, 256, 128, 64.
*   **Funkcja aktywacji:** ReLU (`nn.ReLU`) po każdej warstwie ukrytej.
*   **Normalizacja batchowa (`nn.BatchNorm1d`):** Dodana po każdej warstwie liniowej (przed aktywacją).
*   **Dropout (`nn.Dropout`):** Zastosowany po każdej warstwie ReLU ze współczynnikiem `dropout_rate = 0.3`.
*   **Warstwa wyjściowa:** Pojedynczy neuron z aktywacją liniową.

#### 3.2. Proces treningu

*   **Funkcja straty:** `Mean Squared Error` (`nn.MSELoss`).
*   **Optymalizator:** `Adam` (`optim.Adam`) z początkowym współczynnikiem uczenia (`learning_rate`) ustawionym na `0.003` oraz regularyzacją L2 (`weight_decay`) o wartości `1e-4`.
*   **Rozmiar batcha (`batch_size`):** 32.
*   **Liczba epok (`epochs`):** 2000 (maksymalna).
*   **Early stopping:** Cierpliwość (`early_stopping_patience`) ustawiona na 100 epok.
*   **Scheduler współczynnika uczenia (`ReduceLROnPlateau`):** Cierpliwość (`scheduler_patience`) ustawiona na 25 epok, współczynnik redukcji (`scheduler_factor`) na 0.3, minimalny `learning_rate` (`min_lr`) na `1e-7`.
*   **Przetwarzanie na urządzeniu:** CPU (zgodnie z logami).

#### 3.3. Zapis modelu

Najlepsza wersja modelu była zapisywana do pliku `best_house_price_model.pth` i wczytywana po treningu.

### 4. Ewaluacja modelu

Model był ewaluowany przy użyciu metryk MSE, RMSE, R² oraz RMSLE na oryginalnej skali cen.
Ostateczne metryki dla najlepszej konfiguracji (z feature engineeringiem, gdzie oryginalne cechy nie zostały usunięte) wyniosły około:
*   **Zbiór treningowy:** R² ≈ 0.917, RMSLE ≈ 0.095
*   **Zbiór walidacyjny:** R² ≈ 0.855, RMSLE ≈ 0.141

Niewielka różnica między metrykami treningowymi a walidacyjnymi oraz wysokie wartości R² i niskie RMSLE na zbiorze walidacyjnym wskazują na dobrze wytrenowany i generalizujący model.

### 5. Generowanie predykcji

Po zakończeniu treningu i wczytaniu najlepszego modelu, wygenerowano predykcje dla zbioru testowego. Przetworzone cechy zbioru testowego zostały podane do modelu, a uzyskane zlogarytmowane predykcje zostały przekształcone z powrotem do oryginalnej skali cen za pomocą `np.expm1()`. Wynikowe predykcje zostały zapisane do pliku CSV.

### 6. Podsumowanie i wnioski

Projekt zakończył się sukcesem, tworząc model sieci neuronowej zdolny do predykcji cen nieruchomości z wysoką dokładnością (R² walidacyjne ~0.855). Kluczowe dla osiągnięcia tego wyniku były:
*   Staranne przetwarzanie danych, w tym imputacja i odpowiednie kodowanie kategorii.
*   Efektywna inżynieria cech, w szczególności dodanie nowych, zagregowanych informacji przy jednoczesnym zachowaniu większości oryginalnych cech.
*   Zastosowanie transformacji logarytmicznej zmiennej docelowej.
*   Dobór odpowiedniej architektury sieci neuronowej z mechanizmami regularyzacji (Dropout, L2) i stabilizacji (BatchNorm).
*   Iteracyjne dostrajanie hiperparametrów, które wykazało, że wyższy początkowy `learning_rate` (0.003) w połączeniu ze schedulerem był korzystny.

Model wykazuje dobrą zdolność generalizacji, co czyni go użytecznym narzędziem do szacowania wartości nieruchomości.

---

