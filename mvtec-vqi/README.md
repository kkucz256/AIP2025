Minimalny MVP systemu wizualnej inspekcji jakości w czasie rzeczywistym dla MVTec AD.

## Cel projektu
Repozytorium dostarcza kompletny pipeline do pobrania danych MVTec AD, trenowania modeli PaDiM i CAE, ewaluacji metryk anomalii oraz uruchomienia aplikacji webowej łączącej FastAPI i Gradio z obsługą kamery użytkownika.

## Wymagania
- Python 3.11
- Konto Kaggle z wygenerowanym tokenem `~/.kaggle/kaggle.json`
- CPU z obsługą AVX; GPU CUDA opcjonalnie
- Docker i Docker Compose (opcjonalnie)

## Instalacja
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Konfiguracja danych
1. Utwórz katalog `~/.kaggle` i zapisz tam `kaggle.json` z ustawieniami API.
2. Pobierz dane MVTec AD:
```bash
python scripts/download_mvtec.py --config configs/default.yaml
```
3. Przygotuj pliki splitów (domyślnie kategoria bottle):
```bash
python scripts/prepare_splits.py --category bottle --config configs/default.yaml
```

## Trening
- PaDiM:
```bash
python -m src.train.fit_padim --category bottle --config configs/default.yaml
```
- CAE:
```bash
python -m src.train.train_cae --category bottle --epochs 30 --config configs/default.yaml
```

## Ewaluacja
```bash
python -m src.eval.evaluate --backend padim_resnet50 --category bottle --config configs/default.yaml
```

## Aplikacja
Uruchom interfejs webowy (FastAPI + Gradio + kamera/WebRTC):
```bash
python -m src.app.ui_gradio --config configs/default.yaml
```

## Docker
```bash
docker compose up --build
```
Usługa udostępnia API na porcie 8000 oraz interfejs Gradio na porcie 7860.

## Struktura katalogów
```
configs/         # Konfiguracje YAML
data/            # Dane surowe i metadane
scripts/         # Skrypty narzędziowe
src/             # Kod źródłowy modeli, treningu, API i UI
artifacts/       # Wagi i statystyki modeli
reports/         # Raporty ewaluacji i wizualizacje
```

## Makefile
Najważniejsze cele:
```bash
make setup
make data
make splits
make train_padim
make train_cae
make eval
make app
```


 - mvtec-vqi/README.md:1 opisuje cele, zależności, komendy CLI/Docker oraz
    układ katalogów w 260 słowach, zgodnie z wytycznymi.
  - mvtec-vqi/configs/default.yaml:1 i mvtec-vqi/scripts/download_mvtec.py:1/
    prepare_splits.py:1 dostarczają konfigurację domyślną, pobieranie danych z
    Kaggle i generowanie splitów.
  - mvtec-vqi/src/models/padim.py:10 oraz mvtec-vqi/src/models/cae.py:9
    implementują odpowiednio backend PaDiM (ResNet-50 + statystyki Gaussa) i
    konwolucyjny autoenkoder z metryką MSE+SSIM.
  - mvtec-vqi/src/train/fit_padim.py:1 i mvtec-vqi/src/train/train_cae.py:1
    uruchamiają trening backendów zgodnie z CLI, zapisując artefakty do
    artifacts/<backend>/<category>.
  - mvtec-vqi/src/eval/evaluate.py:1 liczy AUROC (image/pixel) i Dice (Otsu,
    percentyl) oraz zapisuje raporty i wizualizacje w reports/<backend>/
    <category>.
  - mvtec-vqi/src/infer/predictor.py:14 kapsułkuje wczytywanie modeli,
    normalizację i generowanie map cieplnych do późniejszego wykorzystania przez
    API/UI.
  - mvtec-vqi/src/app/api.py:15 i mvtec-vqi/src/app/ui_gradio.py:14 integrują
    FastAPI (endpoint /infer) z Gradio 5 (kamera WebRTC, upload plików, overlay
    z progiem, FPS/score) przy uruchamianiu wspólnego serwera.
  - mvtec-vqi/Makefile:1, mvtec-vqi/Dockerfile:1 i mvtec-vqi/docker-
    compose.yml:1 dostarczają podstawowe komendy, obraz CPU (Python 3.11 slim +
    OpenCV deps) oraz usługę Docker Compose z mapowaniem danych i artefaktów.