# Terminalowy system wizualnej inspekcji jakości (MVTec AD)

## Cel projektu
Repozytorium dostarcza kompletny pipeline do pobrania danych MVTec AD, trenowania modeli PaDiM i CAE, ewaluacji metryk anomalii oraz terminalowego trybu pracy w czasie rzeczywistym. Aplikacja działa bez warstwy webowej — wszystkie akcje wykonujesz z poziomu menu sterowanego strzałkami i Enterem.

## Wymagania
- Python **3.11**
- Konto Kaggle z wygenerowanym tokenem `kaggle.json`
- CPU z obsługą AVX; opcjonalnie GPU CUDA/MPS (zalecane dla treningu i szybkiej inferencji)
- Docker i Docker Compose (opcjonalnie, do izolowanego uruchomienia CLI)

## Instalacja i konfiguracja środowiska

1. Klonowanie repozytorium
   ```bash
   git clone <adres_repozytorium>
   cd mvtec-vqi
   ```
2. Wirtualne środowisko i zależności
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   pip install -r requirements.txt
   ```
3. Konfiguracja Kaggle API
   - Pobierz `kaggle.json` z ustawień konta Kaggle (sekcja API).
   - Umieść plik w `C:\Users\<TwojaNazwaUżytkownika>\.kaggle\kaggle.json`.
4. Pobieranie i przygotowanie danych
   ```bash
   python -m scripts.download_mvtec --config configs/default.yaml
   python -m scripts.prepare_splits --category bottle --config configs/default.yaml
   ```

## Trening modeli
- PaDiM (np. kategoria `bottle`)
  ```bash
  python -m src.train.fit_padim --category bottle --config configs/default.yaml
  ```
- CAE (np. 30 epok)
  ```bash
  python -m src.train.train_cae --category bottle --epochs 30 --config configs/default.yaml
  ```
Wytrenowane modele lądują w `artifacts/`.

## Ewaluacja
```bash
python -m src.eval.evaluate --backend padim_resnet50 --category bottle --config configs/default.yaml
```
Raporty i wizualizacje zapisywane są w `reports/`.

## Tryb terminalowy w czasie rzeczywistym
Uruchom interaktywną aplikację:
```bash
python -m src.app.cli --config configs/default.yaml
```
Sterowanie strzałkami i Enterem. Menu główne:
- Wybór modelu (PaDiM lub CAE) — aktualny wybór oznaczony `(current)`.
- Wybór zdjęcia z `data/mvtec_ad/<kategoria>/...` — lista obrazów w podkatalogach produktów, paginacja po 15 pozycji, z opcją cofnięcia.
- Parametry analizy — próg klasyfikacji, percentyl mapy anomalii, wygładzanie PaDiM (kernel/sigma), przełącznik zapisu wizualizacji oraz zmiana urządzenia.
- Uruchom analizę — wynik z klasyfikacją OK/DEFEKT, zapis overlay/heatmap w `reports/overlays_cli` (jeśli włączone, pliki są nadpisywane dla danego zdjęcia).
- Wyjście — kończy działanie programu.

Każde podmenu ma przycisk „Cofnij”, a w głównym menu znajduje się „Wyjście”. Aplikacja działa w pętli, dopóki nie wybierzesz zakończenia.

Domyślne nastawy dla kategorii `bottle` (możesz je zmienić w `configs/default.yaml` → `infer.backend_defaults` albo w menu „Parametry analizy”):
- PaDiM: `score_percentile=0.9`, próg ≈ 11.5 (skala surowych odległości; wysokie wartości są spodziewane nawet dla dobrych zdjęć).
- CAE: `score_percentile=1.0` (maksimum mapy rekonstrukcji), próg ≈ 0.14 (bardziej konserwatywny, aby ograniczyć fałszywe alarmy na „good”).

## Struktura projektu
- `configs/` — pliki konfiguracyjne (m.in. nowe sekcje `infer` dla progu, percentyla i ścieżek overlay).
- `data/` — dane MVTec AD.
- `scripts/` — narzędzia (`download_mvtec.py`, `prepare_splits.py`).
- `src/app/cli.py` — terminalowy interfejs czasu rzeczywistego.
- `src/utils/terminal.py` — obsługa klawiatury (strzałki/Enter) i renderowanie menu w konsoli.
- `src/infer/predictor.py` — ładowanie modeli i inferencja (obsługa percentyla, opcjonalne AMP dla CAE).
- `src/models/` — implementacje PaDiM i CAE (CAE z AMP w treningu).
- `src/train/` — skrypty treningowe.
- `src/eval/` — ewaluacja modeli.
- `reports/`, `artifacts/` — wyniki i wytrenowane wagi/statystyki.

## Docker
Uruchomienie w kontenerze (interaktywny terminal, brak otwieranych portów):
```bash
docker compose run --rm app
```
Katalogi `data`, `artifacts`, `reports` i `configs` są podmontowane do kontenera.

## Jak działają algorytmy i aktualne wyniki
**Przepływ:** obraz → transformacja (rozmiar 256) → model → mapa anomalii → wynik liczbowy → progowanie OK/DEFEKT → (opcjonalnie) zapis `overlay.png` i `heatmap.png`.

**PaDiM (ResNet50):**
- Ekstrahuje cechy z warstw `layer1-3` ResNet50 (ImageNet), losuje podzbiór kanałów, uczy rozkład Gaussa (średnia + macierz kowariancji) na obrazach „good”.
- W inferencji oblicza odległość Mahalanobisa dla każdego piksela, wygładza Gauss’em, normalizuje mapę; wynik = percentyl (domyślnie 0.9) lub maksimum.
- Szybki, odporny na szum; dobre na „bottle”. Domyślny próg ≈ 11.5 (surowa skala Mahalanobisa), percentyl 0.9.

**CAE (Convolutional Autoencoder):**
- Uczy rekonstrukcji obrazów „good”; mapa anomalii = ważona suma błędu MSE i (1 - SSIM) bez normalizacji per-obraz.
- Wynik = wybrany percentyl mapy (domyślnie 1.0 = maksimum) – wysokie wartości wskazują defekty.
- Lepszy dla złożonych wzorców, wolniejszy; obecnie próg ≈ 0.16 dla „bottle”.

**Aktualne wyniki (kategoria bottle, konfiguracja domyślna):**
- PaDiM: image AUROC 0.979, pixel AUROC 0.949, Dice (Otsu) 0.48.
- CAE: image AUROC ~0.897, pixel AUROC ~0.836, Dice (Otsu) ~0.27 (próg sugerowany ≈0.14).

**Teoria działania projektu:**
- PaDiM wykorzystuje statystykę cech: jeśli bieżące cechy odstają od rozkładu wyuczonych „dobrych” próbek, rośnie odległość Mahalanobisa → defekt.
- CAE minimalizuje błąd rekonstrukcji; fragmenty niepodobne do danych uczących (tylko „good”) są rekonstruowane słabo → wysoki błąd → defekt.
- Progowanie wyników (per obraz) pozwala na decyzję OK/DEFEKT; percentyl mapy kontroluje czułość (większy percentyl = bardziej konserwatywny).

## Lepsze trenowanie CAE (bardziej dokładny model)
- W `configs/default.yaml` podniesiono pojemność modelu (base_channels=48, latent_channels=384), wydłużono trening do 50 epok, pozostawiono lekkie `weight_decay=1e-5`, a wynik liczony jest z maksimum mapy (percentyl 1.0). Augmentacje są domyślnie wyłączone, żeby lepiej odwzorować realne dane bottle.
- Trenowanie z nowymi ustawieniami (nadpisze `artifacts/cae/<kategoria>/model.pt`):
  ```bash
  python -m src.train.train_cae --category bottle --config configs/default.yaml --epochs 50
  ```
- Po treningu zaktualizuj próg/percentyl dla CAE na podstawie walidacji (menu Parametry lub `infer.backend_defaults`), aby zoptymalizować decyzję OK/DEFEKT.
