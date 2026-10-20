# Predictive Maintenance: Gaussian NB, Kernel NB and AdaBoost (Streamlit app)

Predicts machine failure from temperature, speed, torque and tool-wear sensors.
Models: Gaussian Naive Bayes, Kernel (KDE) Naive Bayes and AdaBoost, 80/20 split, with Accuracy,
Confusion Matrix heat map, Type I / Type II error, Precision, Recall and F1.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface (purple / green / red / yellow theme) |
| `ml_core.py` | Models, training, threshold tuning, metrics |
| `requirements.txt` | Libraries Streamlit Cloud installs |
| `.streamlit/config.toml` | Theme colours (hidden folder, **upload it too**) |
| `ai4i2020.csv` | **You add this** (download from UCI / Kaggle) |

## App tabs
- 🔮 **Predict**: sensor inputs, quick presets, probability gauge, verdict from all 3 models, sensor health check
- 📊 **Evaluation**: all required metrics, heat map, threshold sweep for the selected model
- ⚔️ **Model Comparison**: side-by-side table, charts and heat maps for the 3 classifiers, AdaBoost feature importance
- 🧠 **How It Works**: pipeline, model formulas, Type I / Type II explanation, metric cheat-sheet
- 🗂️ **Data**: preview, sensor guide, class balance, correlation and distributions

## Run locally (optional test)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Upload `app.py`, `ml_core.py`, `requirements.txt`, `.streamlit/config.toml` and `ai4i2020.csv` to the root of a public GitHub repo (keep the `.streamlit` folder name exactly).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file path `app.py` → **Deploy**.

If the app is already deployed, just push the updated files to GitHub and it redeploys automatically.
