# Anomaly Detection on ECG Data (plus Machine Learning coursework)

The main project in this repository is **`ML11tamam.ipynb`**: an autoencoder that detects abnormal heart rhythms in ECG recordings. The other notebooks are exercises from my Machine Learning course (listed below).

## Anomaly detection on ECG5000

### Problem
Find abnormal ECG signals without training on them. The model learns only what a normal heartbeat looks like, then flags any signal it cannot rebuild well.

### Data
[ECG5000](http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv) (the notebook downloads it): 4,998 heartbeats, 140 time steps each, labelled normal or abnormal. Split 80/20 into train and test (`random_state=21`), scaled to 0–1.

### Method
1. Train a dense autoencoder (140 → 64 → 32 → 16 → 8 → back to 140, with dropout and batch normalisation) **on normal heartbeats only**, using mean absolute error (MAE) as the loss.
2. Measure the reconstruction error on the normal training heartbeats. Set the threshold at mean + 1 standard deviation (**MAE = 0.0375**).
3. Label a heartbeat abnormal when its reconstruction error is above the threshold.

### Results (1,000 test heartbeats)

| Class | Precision | Recall | F1 | Count |
|---|---|---|---|---|
| Abnormal | 0.90 | 0.90 | 0.90 | 440 |
| Normal | 0.92 | 0.92 | 0.92 | 560 |
| **Accuracy** | | | **0.91** | 1,000 |

The notebook also shows plots of normal and abnormal heartbeats next to their reconstructions, and histograms of the reconstruction error.

### Also in the same notebook
- **Image compression:** a dense autoencoder that compresses Fashion-MNIST images (28×28 = 784 values) into 64 numbers and rebuilds them.
- **Image denoising:** a convolutional autoencoder that removes added random noise from Fashion-MNIST images.

## Coursework notebooks

| Notebook | Topic |
|---|---|
| `01_bayes.ipynb` | Bayesian decision theory, loss functions and the likelihood ratio |
| `02_bayes_classifiersession.ipynb` | Bayesian classifier from scratch, with priors |
| `03_bayes_recommender.ipynb` | Simple Bayes recommender |
| `04_gmm_em.ipynb` | Expectation–maximisation for a Gaussian mixture model |
| `05_introKeras.ipynb` | Introduction to Keras: regression and binary classification |
| `LLMbyexample.ipynb` | Summarising Wikipedia articles so they fit a language model's context window (NLTK, cosine similarity) |

## How to run

```bash
git clone https://github.com/Amirhatamian/Machine-Learning-.git
cd Machine-Learning-
pip install tensorflow scikit-learn pandas numpy matplotlib jupyter
jupyter notebook ML11tamam.ipynb
```

Or open any notebook in Google Colab with the badge at the top of the notebook.

## Tools

Python, TensorFlow / Keras, scikit-learn, pandas, NumPy, Matplotlib
