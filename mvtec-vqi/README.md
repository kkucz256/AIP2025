# Minimalny MVP systemu wizualnej inspekcji jakości w czasie rzeczywistym dla MVTec AD

## Cel projektu
Repozytorium dostarcza kompletny pipeline do pobrania danych MVTec AD, trenowania modeli PaDiM i CAE, ewaluacji metryk anomalii oraz uruchomienia aplikacji webowej łączącej FastAPI i Gradio z obsługą kamery użytkownika.

## Wymagania
*   **Python 3.11**
*   **Konto Kaggle** z wygenerowanym tokenem `kaggle.json` (szczegóły poniżej)
*   **CPU** z obsługą AVX; **GPU CUDA** opcjonalnie (zalecane dla treningu)
*   **Docker i Docker Compose** (opcjonalnie, do uruchamiania w kontenerach)

## Instalacja i konfiguracja środowiska

### 1. Klonowanie repozytorium (jeśli jeszcze tego nie zrobiłeś)
```bash
git clone <adres_repozytorium>
cd mvtec-vqi
```

### 2. Konfiguracja środowiska wirtualnego i instalacja zależności
Zaleca się użycie środowiska wirtualnego, aby uniknąć konfliktów zależności.

```bash
# Utwórz środowisko wirtualne
python -m venv .venv

# Aktywuj środowisko wirtualne (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Zainstaluj wymagane biblioteki
pip install -r requirements.txt
```

### 3. Konfiguracja Kaggle API
Aby pobrać dane MVTec AD, potrzebujesz tokenu API z Kaggle.

1.  **Pobierz `kaggle.json`**:
    *   Zaloguj się na swoje konto na [Kaggle](https://www.kaggle.com).
    *   Przejdź do ustawień swojego profilu (kliknij swoje zdjęcie profilowe -> **"Account"**).
    *   W sekcji "API" kliknij przycisk **"Create New API Token"**. Spowoduje to pobranie pliku `kaggle.json`.

2.  **Umieść `kaggle.json` w folderze domowym (Windows)**:
    *   Utwórz folder `.kaggle` w swoim katalogu domowym: `C:\Users\TwojaNazwaUżytkownika\.kaggle\`
    *   Przenieś pobrany plik `kaggle.json` do tego folderu.
    *   Ostateczna ścieżka powinna wyglądać tak: `C:\Users\TwojaNazwaUżytkownika\.kaggle\kaggle.json`

### 4. Pobieranie i przygotowanie danych
Po skonfigurowaniu Kaggle API, możesz pobrać i przygotować dane:

```bash
# Pobierz dane MVTec AD
python -m scripts.download_mvtec --config configs/default.yaml

# Przygotuj pliki podziału danych (dla kategorii 'bottle' jako przykład)
python -m scripts.prepare_splits --category bottle --config configs/default.yaml
```

## Trening modeli

Projekt umożliwia trening dwóch typów modeli do detekcji anomalii: PaDiM i CAE.

*   **Trening PaDiM (dla kategorii 'bottle')**:
    ```bash
    python -m src.train.fit_padim --category bottle --config configs/default.yaml
    ```

*   **Trening CAE (dla kategorii 'bottle', 30 epok)**:
    ```bash
    python -m src.train.train_cae --category bottle --epochs 30 --config configs/default.yaml
    ```

Wytrenowane modele zostaną zapisane w katalogu `artifacts/`.

## Ewaluacja modeli

Po wytrenowaniu modeli możesz ocenić ich wydajność:

```bash
# Ewaluacja modelu PaDiM (backend 'padim_resnet50')
python -m src.eval.evaluate --backend padim_resnet50 --category bottle --config configs/default.yaml
```

Raporty ewaluacji i wizualizacje zostaną zapisane w katalogu `reports/`.

## Uruchamianie aplikacji

Aplikacja webowa łączy backend FastAPI z interfejsem Gradio, umożliwiając wizualną inspekcję jakości.

### 1. Uruchamianie w VS Code (zalecane)

Najwygodniejszym sposobem jest użycie konfiguracji `launch.json`:

1.  **Utwórz/edytuj plik `.vscode/launch.json`**:
W głównym katalogu projektu utwórz folder `.vscode`, a w nim plik `launch.json`. Wklej do niego poniższą zawartość:

    ```json
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Run Gradio App",
                "type": "python",
                "request": "launch",
                "module": "src.app.ui_gradio",
                "args": [
                    "--config",
                    "configs/default.yaml"
                ],
                "console": "integratedTerminal",
                "justMyCode": true
            }
        ]
    }
    ```

2.  **Uruchom aplikację**:
    *   W VS Code przejdź do widoku "Run and Debug" (ikona robaka po lewej).
    *   Upewnij się, że wybrana jest konfiguracja **"Run Gradio App"**.
    *   Kliknij zieloną strzałkę "Start Debugging" (lub naciśnij klawisz **F5**).

### 2. Uruchamianie z terminala

Jeśli wolisz uruchomić aplikację bezpośrednio z terminala (po aktywacji środowiska wirtualnego):

```bash
python -m src.app.ui_gradio --config configs/default.yaml
```

### 3. Dostęp do interfejsu

Po uruchomieniu aplikacji, w terminalu zobaczysz adresy URL. Aby uzyskać dostęp do interfejsu Gradio, otwórz w przeglądarce:

*   **Interfejs Gradio (aplikacja wizualna):** [http://localhost:7860](http://localhost:7860)
*   **API FastAPI (backend):** [http://localhost:8000](http://localhost:8000)

## Struktura projektu i moduły

Projekt jest zorganizowany w następujący sposób:

*   `configs/`: Zawiera pliki konfiguracyjne YAML (np. `default.yaml`) dla różnych aspektów projektu.
*   `data/`: Przechowuje surowe dane MVTec AD oraz metadane i pliki podziału danych (`splits/`).
*   `scripts/`: Skrypty narzędziowe do automatyzacji zadań:
    *   `download_mvtec.py`: Pobiera dane MVTec AD z Kaggle.
    *   `prepare_splits.py`: Przygotowuje pliki podziału danych treningowych, walidacyjnych i testowych.
*   `src/`: Główny kod źródłowy projektu, podzielony na moduły:
    *   `src/app/`: Zawiera logikę aplikacji webowej.
        *   `api.py`: Implementuje backend FastAPI, udostępniając endpoint `/infer` do detekcji anomalii.
        *   `ui_gradio.py`: Tworzy interfejs użytkownika Gradio, który komunikuje się z API FastAPI.
    *   `src/data/`: Moduły związane z przetwarzaniem danych.
        *   `mvtec.py`: Prawdopodobnie zawiera klasy lub funkcje do ładowania i transformacji danych MVTec AD.
    *   `src/eval/`: Moduły do ewaluacji modeli.
        *   `evaluate.py`: Skrypt do obliczania metryk anomalii (np. AUROC) i generowania raportów.
    *   `src/infer/`: Moduły do wnioskowania (inferencji).
        *   `predictor.py`: Kapsułkuje logikę ładowania modeli i wykonywania detekcji anomalii.
    *   `src/models/`: Implementacje modeli uczenia maszynowego.
        *   `cae.py`: Konwolucyjny Autoenkoder (CAE).
        *   `padim.py`: Model PaDiM (Patch Distribution Modeling).
    *   `src/train/`: Moduły do treningu modeli.
        *   `fit_padim.py`: Skrypt do treningu modelu PaDiM.
        *   `train_cae.py`: Skrypt do treningu modelu CAE.
    *   `src/utils/`: Pomocnicze funkcje i narzędzia.
        *   `common.py`: Ogólne funkcje pomocnicze (np. ładowanie konfiguracji, ustawianie seeda).
        *   `metrics.py`: Implementacje metryk oceny.
        *   `transforms.py`: Transformacje danych obrazowych.
        *   `viz.py`: Funkcje do wizualizacji (np. generowanie map cieplnych, nakładanie obrazów, kodowanie/dekodowanie base64).
*   `artifacts/`: Katalog, w którym przechowywane są wytrenowane wagi i statystyki modeli.
*   `reports/`: Katalog na raporty ewaluacji i wizualizacje wyników.

## Docker

Projekt można również uruchomić za pomocą Dockera i Docker Compose, co zapewnia izolowane środowisko.

```bash
docker compose up --build
```
Usługa udostępnia API na porcie 8000 oraz interfejs Gradio na porcie 7860.

