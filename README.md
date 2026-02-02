# Posner EEG Analysis

Python analysis pipeline for Posner spatial cueing task (RT + cue validity).

## Structure

- **`analysis.ipynb`** — анализ RT Posnera (CSV PsychoPy): загрузка, статистика, графики
- **`erp_analysis.ipynb`** — анализ ERP (Spike2 .smr): загрузка Neo/MNE, артефакты, evoked, пики, статистика
- **`src/`** — модули RT:
  - `constants.py`, `data.py`, `statistics.py`, `plots.py`
- **`src/erp/`** — модули ERP:
  - `constants.py`, `io_spike2.py`, `raw_mne.py`, `events.py`, `epochs_mne.py`
  - `artifacts.py` (артефакты очные, odrzucanie), `erp.py` (evoked, wykresy), `peaks.py`, `stats.py`
- **`data/`** — CSV PsychoPy (Posner) oraz plik .smr (Spike2) dla ERP. **Dane nie są w repozytorium** — należy włożyć własne pliki do `data/`. Ścieżki w pierwszej komórce notatnika.
- **`results/`** — tabele CSV i wykresy PNG z analizy RT i ERP (tworzone automatycznie).

## Results

- **Efekt Posnera**: ~22.8 ms (p < 0.001), trafna vs nietrafna wskazówka
- **Retencja danych**: po odrzuceniu prób z ruchami oczu i błędami

## Stack

Python 3.10+ | pandas | scipy | matplotlib | seaborn | statsmodels | MNE | neo | Jupyter (zob. `requirements.txt`)

## Usage

**Dane nie są w repozytorium** — umieść własny plik CSV (RT) i/lub .smr (ERP) w katalogu `data/`.

```bash
pip install -r requirements.txt
# RT: włóż plik CSV eksperymentu Posnera do data/
# ERP: włóż plik .smr (Spike2) do data/
jupyter notebook analysis.ipynb      # analiza RT
jupyter notebook erp_analysis.ipynb # analiza ERP
```

Uruchom wszystkie komórki z katalogu repozytorium (working directory = root repo), żeby ścieżki `data/...` i `src` działały.
