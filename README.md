# Posner EEG Analysis

Python analysis pipeline for Posner spatial cueing task with 19-channel EEG.

## Results
- **Posner effect**: 22.8 ms (p < 0.001)
- **P3 enhancement** for invalid cues (attention reorienting)
- **Data retention**: 90.8% after artifact rejection

## Stack
Python 3.11 | MNE-Python | scipy | pandas | seaborn

## Usage
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_preprocessing.ipynb
```

See protocol PDF for full methodology.
