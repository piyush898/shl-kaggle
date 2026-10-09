# Grammar Scoring Engine for Spoken Audio

Predicts a continuous grammar score in **[0, 5]** (MOS Likert rubric) from a 45–60 s `.wav` recording.

**Metrics:** RMSE (lower is better) and Pearson correlation (higher is better).
**Baseline:** predicting the mean label (~3.3) gives RMSE ≈ 1.24.

---

## Approach

Grammar quality shows up in *what* is said (syntax, sentence completeness) and *how* it is said (hesitations, repairs, fluency). The pipeline combines both views.

1. **Acoustic view (strongest signal, CV Pearson ≈ 0.89)**
   Mean- and std-pooled hidden states from every layer of the Whisper encoder (`small`, `medium`), using two 30 s windows per clip. Each layer is probed by cross-validation and the best few are kept. Later versions also test self-supervised encoders (WavLM, HuBERT).
2. **Text view**
   Whisper ASR transcript, then:
   - surface statistics (length, vocabulary richness, sentence length, fillers, repetitions)
   - ASR confidence (average log-probability, no-speech probability, compression ratio)
   - GPT-2 perplexity
   - CoLA grammatical-acceptability score (RoBERTa)
   - MiniLM sentence embedding
3. **Models**
   Ridge, SVR (nested hyper-parameter search) and gradient boosting, one family per feature view plus one on all features combined. Out-of-fold predictions are blended with a non-negative linear stacker, and the final output is clipped to [0, 5].

**Evaluation:** 5-fold cross-validated RMSE / Pearson, with hyper-parameter search nested inside each fold. The RMSE on the full training data after the final fit is also reported, as the assessment requires. It is optimistic by nature.

---

## Repository contents

| File | Description |
|---|---|
| `Dataset_Final/grammar_scoring_engine (2).ipynb` | First version: Whisper `small`, text features, MiniLM + acoustic embeddings, stacking |
| `Dataset_Final/grammar_scoring_engine (3).ipynb` | Layer-wise Whisper encoder probing, CoLA scores, nested SVR search |
| `Dataset_Final/grammar_scoring_engine (4).ipynb` | Same pipeline as (3), saved with outputs |
| `Dataset_Final/grammar_scoring_engine (5).ipynb` | Multiple acoustic views (Whisper small/medium, WavLM, HuBERT) |
| `Dataset_Final/grammar_scoring_engine (6).ipynb` | Latest version: verbatim ASR prompting (keeps fillers / false starts), Whisper encoders only |
| `Dataset_Final/final_models.pkl` | Saved final models |
| `Dataset_Final/oof_cache_v2.pkl` | Cached out-of-fold predictions |
| `Dataset_Final/sample_submission.csv` | Submission format |

> The training and test audio (`train/`, `test/`) and `train.csv` / `test.csv` are **not** included in this repository.

---

## How to run

### 1. Requirements
- Python 3.9+
- [`ffmpeg`](https://ffmpeg.org/) (macOS: `brew install ffmpeg`)
- Internet access on first run to download the Whisper, GPT-2, RoBERTa and MiniLM weights
- A GPU helps (CUDA, or Apple `mps`), but the notebooks also run on CPU, just slower

The first notebook cell installs the Python packages:

```bash
pip install openai-whisper sentence-transformers librosa soundfile scikit-learn scipy seaborn
```

### 2. Add the data
Place the competition data in a folder containing `train.csv`, `test.csv`, `sample_submission.csv`, `train/` and `test/`, then edit the config cell in the notebook:

```python
DATA_ROOT = "/path/to/Dataset_Final"
```

### 3. Run
Open a notebook (use `(6)` for the latest) and run all cells.

- Set `QUICK_TEST = True` first to smoke-test the pipeline on 30 train / 10 test files.
- Heavy steps (ASR, embeddings) are cached in `./working`, so re-runs are fast.
- Full feature extraction can take from tens of minutes to a few hours, depending on hardware and model size.

The final cell writes `submission.csv` into `./working`.

---

## Notes
- Random seed is fixed (`SEED = 42`).
- `VERBATIM = True` in notebook (6) prompts Whisper to keep fillers; compare CV against `False` before relying on it.
- Labels cluster around 2–5, with a block of `0` scores (likely empty or unintelligible recordings) and few 1–1.5 scores.
