# Terminalowy system wizualnej inspekcji jakości (MVTec AD)

Projekt dostarcza terminalową aplikację do wykrywania anomalii na obrazach, bazując na zbiorze danych MVTec AD.

## Uruchomienie aplikacji

Upewnij się, że masz zainstalowane wszystkie zależności z `requirements.txt`:
```bash
pip install -r requirements.txt
```
Oraz dataset (https://www.kaggle.com/datasets/ipythonx/mvtec-ad?resource=download) umieszczony w folderze `data/mvtec_ad`

### Trening modeli

*   **PaDiM** (np. kategoria `bottle`):
    ```bash
    python3 -m src.train.fit_padim --category bottle --config configs/default.yaml
    ```
*   **CAE**:
    ```bash
    python3 -m src.train.train_cae --category bottle --config configs/default.yaml
    ```
Wytrenowane modele zapisywane są w katalogu `artifacts/`.

### Metryki modeli
    python3 -m src.eval.evaluate --backend cae --category bottle
    

### Uruchomienie trybu terminalowego

Aby uruchomić interaktywną aplikację w terminalu:
```bash
python3 -m src.app.cli --config configs/default.yaml
```
Aplikacja pozwala na wybór modelu, obrazu, konfigurację parametrów analizy oraz uruchomienie inspekcji.