import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import ml_core as core

st.set_page_config(page_title="Predictive Maintenance | ML Dashboard", page_icon="🛠️", layout="wide")

DATA_PATH = "ai4i2020.csv"

# ---------- colour palette: purple / green / red / yellow ----------
PURPLE, GREEN, RED, YELLOW = "#7C3AED", "#10B981", "#EF4444", "#F59E0B"
MODEL_META = {
    "Gaussian Naive Bayes":     {"icon": "🔔", "color": PURPLE, "short": "Gaussian NB"},
    "Kernel (KDE) Naive Bayes": {"icon": "🌊", "color": GREEN,  "short": "Kernel NB"},
    "AdaBoost Classifier":      {"icon": "🚀", "color": YELLOW, "short": "AdaBoost"},
}
SENSOR_ICONS = ["🌡️", "🔥", "⚙️", "🔩", "🛠️"]
SENSOR_INFO = [("K", "Temperature of the air around the machine"),
               ("K", "Temperature inside the manufacturing process"),
               ("rpm", "Rotational speed of the tool"),
               ("Nm", "Torque (turning force) applied by the tool"),
               ("min", "Minutes the tool has already been used")]

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#5B21B6 0%,#7C3AED 45%,#10B981 100%);border-radius:22px;padding:28px 34px;color:#fff;box-shadow:0 10px 30px rgba(124,58,237,.35);margin-bottom:18px}
.hero-title{font-size:2.2rem;font-weight:800;line-height:1.15}
.hero-sub{margin:8px 0 14px 0;font-size:1.05rem;opacity:.95}
.pill{display:inline-block;padding:4px 13px;border-radius:999px;font-size:.82rem;font-weight:600;margin:2px 6px 2px 0;background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.5)}
.kpi{display:flex;align-items:center;gap:12px;background:#fff;border-radius:14px;padding:12px 16px;box-shadow:0 2px 12px rgba(124,58,237,.13);min-height:84px}
.kpi-icon{font-size:1.9rem}
.kpi-label{font-size:.72rem;color:#6B7280;font-weight:700;text-transform:uppercase;letter-spacing:.05em}
.kpi-value{font-size:1.5rem;font-weight:800;line-height:1.15}
.kpi-sub{font-size:.74rem;color:#9CA3AF}
.verdict{border-radius:18px;padding:18px 24px;color:#fff;font-size:1.05rem;box-shadow:0 6px 18px rgba(0,0,0,.15)}
.verdict b{font-size:1.5rem}
.gauge-wrap{margin:26px 4px 6px 4px}
.gauge{position:relative;height:18px;border-radius:999px;background:linear-gradient(90deg,#10B981 0%,#F59E0B 50%,#EF4444 100%)}
.g-pin{position:absolute;top:-7px;width:30px;height:30px;margin-left:-15px;border-radius:50%;background:#fff;border:5px solid #4C1D95;box-shadow:0 2px 8px rgba(0,0,0,.35)}
.g-thr{position:absolute;top:-8px;height:34px;width:3px;background:#1F2937}
.g-labels{display:flex;justify-content:space-between;font-size:.78rem;color:#6B7280;margin-top:10px}
.mcard{background:#fff;border-radius:14px;padding:12px 16px;box-shadow:0 2px 12px rgba(0,0,0,.08);text-align:center}
.mcard .mname{font-weight:700;font-size:.95rem}
.mcard .mprob{font-size:1.6rem;font-weight:800}
.badge{display:inline-block;padding:2px 12px;border-radius:999px;color:#fff;font-size:.78rem;font-weight:700}
.box{border-radius:14px;padding:14px 18px;margin-bottom:8px}
.steps{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:6px 0 14px 0}
.step{background:#F5F3FF;border:2px solid #7C3AED;border-radius:14px;padding:10px 14px;font-weight:700;font-size:.9rem;color:#4C1D95;text-align:center}
.arrow{font-size:1.3rem;color:#10B981;font-weight:800}
h3{color:#5B21B6}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#F5F3FF 0%,#ECFDF5 100%)}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{background:#F5F3FF;border-radius:12px 12px 0 0;padding:10px 18px;font-weight:700}
.stTabs [aria-selected="true"]{background:#7C3AED}
.stTabs [aria-selected="true"] p{color:#fff}
</style>
""", unsafe_allow_html=True)


# ---------- small HTML helpers ----------
def kpi(icon, label, value, color, sub=""):
    return (f'<div class="kpi" style="border-left:6px solid {color}"><div class="kpi-icon">{icon}</div>'
            f'<div><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color}">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div></div>')


def badge(text, color):
    return f'<span class="badge" style="background:{color}">{text}</span>'


def gauge(p, thr):
    return (f'<div class="gauge-wrap"><div class="gauge"><div class="g-thr" style="left:{thr * 100:.1f}%"></div>'
            f'<div class="g-pin" style="left:{p * 100:.1f}%"></div></div>'
            f'<div class="g-labels"><span>0% · Safe</span><span>▲ black line = threshold {thr:.2f}</span>'
            f'<span>100% · Failure</span></div></div>')


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def heatmap(m, title):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    labels = np.array([[f"TN\n{m['TN']}", f"FP (Type I)\n{m['FP']}"],
                       [f"FN (Type II)\n{m['FN']}", f"TP\n{m['TP']}"]])
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Purples", cbar=False, ax=ax, linewidths=2, linecolor="white",
                xticklabels=["Pred Normal", "Pred Failure"], yticklabels=["Actual Normal", "Actual Failure"])
    ax.set_title(title, fontsize=10, color="#5B21B6", fontweight="bold")
    fig.tight_layout()
    return fig


def sweep_table(y, proba, thresholds):
    rows = []
    for t in thresholds:
        pred = proba >= t
        fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tp = int(((pred == 1) & (y == 1)).sum())
        rows.append({"Threshold": round(float(t), 2),
                     "Type I error": fp / (fp + tn) if fp + tn else 0.0,
                     "Type II error": fn / (fn + tp) if fn + tp else 0.0,
                     "Precision": tp / (tp + fp) if tp + fp else 0.0,
                     "Recall": tp / (tp + fn) if tp + fn else 0.0})
    return pd.DataFrame(rows).set_index("Threshold")


# ---------- data + training (cached so the cloud app stays fast) ----------
@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource(show_spinner="🧠 Training Gaussian NB, Kernel NB and AdaBoost...")
def get_artifacts(_df, data_key, bandwidth):
    return core.train_all(_df, bandwidth)


@st.cache_data(show_spinner=False)
def get_probas(_art, data_key, bandwidth):
    return {n: m.predict_proba(_art["Xte_s"])[:, 1] for n, m in _art["models"].items()}


# ---------- hero banner ----------
st.markdown("""
<div class="hero">
<div class="hero-title">🛠️ Machine Failure Prediction Dashboard</div>
<div class="hero-sub">Predictive maintenance from temperature 🌡️, speed ⚙️, torque 🔩 and tool-wear 🛠️ sensors, comparing three classifiers</div>
<span class="pill">📦 AI4I 2020 dataset</span><span class="pill">🧪 3 classifiers</span><span class="pill">🎯 Type I / Type II analysis</span><span class="pill">☁️ Streamlit Cloud</span>
</div>
""", unsafe_allow_html=True)

# ---------- load data ----------
if os.path.exists(DATA_PATH):
    df, data_key = load_data(DATA_PATH), "repo_csv"
else:
    up = st.file_uploader("📁 `ai4i2020.csv` not found in the repo. Upload it here:", type="csv")
    if up is None:
        st.stop()
    df, data_key = pd.read_csv(up), f"upload_{up.name}_{up.size}"

try:
    core.find_columns(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---------- sidebar ----------
st.sidebar.markdown("## 🎛️ Control Panel")
model_name = st.sidebar.selectbox("🧠 Classifier", list(MODEL_META.keys()),
                                  format_func=lambda n: f"{MODEL_META[n]['icon']} {n}")
with st.sidebar.expander("🌊 Kernel NB settings"):
    bandwidth = st.slider("Kernel bandwidth (h)", 0.1, 1.0, 0.3, 0.05,
                          help="Small h = wiggly density, large h = smoother density")
art = get_artifacts(df, data_key, bandwidth)
probas = get_probas(art, data_key, bandwidth)

thr_mode = st.sidebar.radio("🎯 Decision threshold", ["Default (0.50)", "Tuned (recall-focused)", "Manual"])
manual_thr = 0.30
if thr_mode == "Manual":
    manual_thr = st.sidebar.slider("P(Failure) cutoff", 0.01, 0.99, 0.30, 0.01)


def model_threshold(name):
    if thr_mode.startswith("Default"):
        return 0.5
    if thr_mode.startswith("Tuned"):
        return art["thresholds"][name]
    return manual_thr


threshold = model_threshold(model_name)
if thr_mode.startswith("Tuned"):
    st.sidebar.info(f"Threshold chosen by 5-fold CV on training data (F2-score): **{threshold:.2f}**")
st.sidebar.caption(f"✂️ Train/Test split: 80/20 → {art['n_train']} / {art['n_test']} rows")
st.sidebar.markdown(
    f'<div class="box" style="background:#FEF3C7;border-left:6px solid {YELLOW}"><b>🟡 Type I</b> = false alarm<br>'
    f'<small>healthy machine flagged as failing</small></div>'
    f'<div class="box" style="background:#FEE2E2;border-left:6px solid {RED}"><b>🔴 Type II</b> = missed failure<br>'
    f'<small>real failure not detected (costly!)</small></div>', unsafe_allow_html=True)

# ---------- KPI strip ----------
feats, target = art["feats"], art["target"]
k = st.columns(4)
k[0].markdown(kpi("📦", "Dataset rows", f"{len(df):,}", PURPLE, "AI4I 2020 (UCI)"), unsafe_allow_html=True)
k[1].markdown(kpi("🧰", "Sensor features", f"{len(feats)}", GREEN, "temp · speed · torque · wear"), unsafe_allow_html=True)
k[2].markdown(kpi("⚠️", "Failure rate", f"{art['fail_rate'] * 100:.2f}%", RED, "highly imbalanced"), unsafe_allow_html=True)
k[3].markdown(kpi("✂️", "Train / Test", f"{art['n_train']} / {art['n_test']}", YELLOW, "80% / 20% stratified"), unsafe_allow_html=True)
st.write("")

tab_pred, tab_eval, tab_cmp, tab_how, tab_data = st.tabs(
    ["🔮 Predict", "📊 Evaluation", "⚔️ Model Comparison", "🧠 How It Works", "🗂️ Data"])

# ======================= Predict tab =======================
with tab_pred:
    st.markdown("### 🎚️ Enter live sensor readings")
    normal_med = df[df[target] == 0][feats].median().values
    fail_med = df[df[target] == 1][feats].median().values
    for f, v in zip(feats, df[feats].median().values):
        st.session_state.setdefault(f"in_{f}", float(v))

    def set_inputs(values):
        for f, v in zip(feats, values):
            st.session_state[f"in_{f}"] = float(v)

    def set_random():
        set_inputs(df[feats].sample(1).iloc[0].values)

    b = st.columns([1.2, 1.2, 1.2, 3])
    b[0].button("✅ Typical normal machine", on_click=set_inputs, args=(normal_med,))
    b[1].button("⚠️ Typical failing machine", on_click=set_inputs, args=(fail_med,))
    b[2].button("🎲 Random dataset row", on_click=set_random)

    cols = st.columns(5)
    vals = []
    for i, (c, f) in enumerate(zip(cols, feats)):
        lo, hi = float(df[f].min()), float(df[f].max())
        vals.append(c.number_input(f"{SENSOR_ICONS[i]} {f}", min_value=lo, max_value=hi,
                                   step=(hi - lo) / 200, key=f"in_{f}", format="%.2f"))

    res = {n: core.predict_one(art, n, vals, model_threshold(n)) for n in MODEL_META}
    p_sel, label = res[model_name]

    left, right = st.columns([1, 1.6])
    if label:
        left.markdown(f'<div class="verdict" style="background:linear-gradient(135deg,#B91C1C,{RED})">'
                      f'🚨 <b>FAILURE predicted</b><br>P(Failure) = {p_sel:.3f} ≥ threshold {threshold:.2f}<br>'
                      f'🔧 Schedule maintenance</div>', unsafe_allow_html=True)
    else:
        left.markdown(f'<div class="verdict" style="background:linear-gradient(135deg,#047857,{GREEN})">'
                      f'✅ <b>Normal operation</b><br>P(Failure) = {p_sel:.3f} &lt; threshold {threshold:.2f}<br>'
                      f'👍 No action needed</div>', unsafe_allow_html=True)
    right.markdown(f"**{MODEL_META[model_name]['icon']} {model_name}: failure probability {p_sel * 100:.1f}%**")
    right.markdown(gauge(p_sel, threshold), unsafe_allow_html=True)

    st.markdown("### 🗳️ What do all three models say?")
    mc = st.columns(3)
    for c, n in zip(mc, MODEL_META):
        p, l = res[n]
        c.markdown(f'<div class="mcard" style="border-top:6px solid {MODEL_META[n]["color"]}">'
                   f'<div class="mname">{MODEL_META[n]["icon"]} {n}</div>'
                   f'<div class="mprob" style="color:{MODEL_META[n]["color"]}">{p * 100:.1f}%</div>'
                   f'{badge("FAILURE", RED) if l else badge("NORMAL", GREEN)}'
                   f'<div class="kpi-sub">threshold {model_threshold(n):.2f}</div></div>', unsafe_allow_html=True)
    votes = sum(l for _, l in res.values())
    st.info(f"🗳️ **Consensus:** {votes} of 3 models predict failure.")

    st.markdown("### 🩺 Sensor health check (vs. normal machines)")
    dn, dfail = df[df[target] == 0][feats], df[df[target] == 1][feats]
    rows, worst = [], (None, 0.0)
    for i, f in enumerate(feats):
        z = (vals[i] - dn[f].mean()) / (dn[f].std() or 1.0)
        status = "🟢 Normal" if abs(z) < 1 else ("🟡 Unusual" if abs(z) < 2 else "🔴 Extreme")
        if abs(z) > abs(worst[1]):
            worst = (f, z)
        rows.append({"Sensor": f"{SENSOR_ICONS[i]} {f}", "Your value": round(float(vals[i]), 2),
                     "Normal avg": round(float(dn[f].mean()), 2), "Failure avg": round(float(dfail[f].mean()), 2),
                     "Deviation (σ)": round(float(z), 2), "Status": status})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Most unusual reading: **{worst[0]}** ({worst[1]:+.2f}σ from the normal-machine average)")

# ======================= Evaluation tab =======================
with tab_eval:
    meta = MODEL_META[model_name]
    st.markdown(f"### {meta['icon']} {model_name} &nbsp;|&nbsp; threshold = {threshold:.2f} &nbsp;|&nbsp; test set (20%)")
    proba = probas[model_name]
    m = core.evaluate(art["y_te"], (proba >= threshold).astype(int))

    k = st.columns(6)
    k[0].markdown(kpi("🎯", "Accuracy", f"{m['Accuracy']:.3f}", PURPLE, "overall correct"), unsafe_allow_html=True)
    k[1].markdown(kpi("🔍", "Precision", f"{m['Precision']:.3f}", GREEN, "alarms that were real"), unsafe_allow_html=True)
    k[2].markdown(kpi("📡", "Recall", f"{m['Recall']:.3f}", PURPLE, "failures caught"), unsafe_allow_html=True)
    k[3].markdown(kpi("⚖️", "F1-score", f"{m['F1']:.3f}", GREEN, "precision × recall"), unsafe_allow_html=True)
    k[4].markdown(kpi("🟡", "Type I error", f"{m['TypeI']:.3f}", YELLOW, "FP/(FP+TN) false alarms"), unsafe_allow_html=True)
    k[5].markdown(kpi("🔴", "Type II error", f"{m['TypeII']:.3f}", RED, "FN/(FN+TP) missed"), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2)
    with left:
        st.markdown("#### 🔥 Confusion Matrix Heat Map")
        show(heatmap(m, f"{meta['short']} (threshold {threshold:.2f})"))
    with right:
        st.markdown("#### 📋 Classification report")
        st.code(m["report"])
        st.markdown(
            f'<div class="box" style="background:#FEF3C7;border-left:6px solid {YELLOW}">🟡 <b>Type I (false alarms):</b> '
            f'{m["FP"]} healthy machines flagged as failing</div>'
            f'<div class="box" style="background:#FEE2E2;border-left:6px solid {RED}">🔴 <b>Type II (missed failures):</b> '
            f'{m["FN"]} real failures not detected</div>'
            f'<div class="box" style="background:#EDE9FE;border-left:6px solid {PURPLE}">↕️ Lowering the threshold '
            f'<b>reduces Type II</b> but <b>increases Type I</b>.</div>', unsafe_allow_html=True)

    st.markdown("#### 🎚️ Threshold sweep")
    sw = sweep_table(art["y_te"], proba, np.linspace(0.05, 0.95, 91))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(sw.index, sw["Type I error"], color=YELLOW, lw=2.5, label="Type I (false alarm)")
    ax.plot(sw.index, sw["Type II error"], color=RED, lw=2.5, label="Type II (missed failure)")
    ax.axvline(threshold, color=PURPLE, ls="--", lw=2, label=f"current threshold {threshold:.2f}")
    ax.set_xlabel("Decision threshold on P(Failure)"); ax.set_ylabel("Error rate"); ax.legend()
    ax.grid(alpha=.3); fig.tight_layout()
    show(fig)
    st.dataframe(sw.iloc[::5].style.format("{:.3f}").background_gradient(cmap="Purples"))

# ======================= Comparison tab =======================
with tab_cmp:
    st.markdown("### ⚔️ Gaussian NB vs Kernel NB vs AdaBoost")
    st.caption(f"Each model evaluated on the same 20% test set. Threshold mode: **{thr_mode}**")
    rows, mets = [], {}
    for n in MODEL_META:
        t = model_threshold(n)
        mm = core.evaluate(art["y_te"], (probas[n] >= t).astype(int))
        mets[n] = mm
        rows.append({"Model": f"{MODEL_META[n]['icon']} {n}", "Threshold": round(t, 2),
                     "Accuracy": mm["Accuracy"], "Precision": mm["Precision"], "Recall": mm["Recall"],
                     "F1-score": mm["F1"], "Type I error": mm["TypeI"], "Type II error": mm["TypeII"],
                     "FP": mm["FP"], "FN": mm["FN"]})
    cmp_df = pd.DataFrame(rows).set_index("Model")

    best_f1 = max(mets, key=lambda n: mets[n]["F1"])
    best_t2 = min(mets, key=lambda n: mets[n]["TypeII"])
    best_t1 = min(mets, key=lambda n: mets[n]["TypeI"])
    w = st.columns(3)
    w[0].markdown(kpi("🏆", "Best F1-score", MODEL_META[best_f1]["short"], PURPLE, f"F1 = {mets[best_f1]['F1']:.3f}"), unsafe_allow_html=True)
    w[1].markdown(kpi("🛡️", "Fewest missed failures", MODEL_META[best_t2]["short"], RED, f"Type II = {mets[best_t2]['TypeII']:.3f}"), unsafe_allow_html=True)
    w[2].markdown(kpi("🔔", "Fewest false alarms", MODEL_META[best_t1]["short"], YELLOW, f"Type I = {mets[best_t1]['TypeI']:.3f}"), unsafe_allow_html=True)
    st.write("")

    st.dataframe(cmp_df.style.format({c: "{:.3f}" for c in cmp_df.columns if c not in ("FP", "FN")})
                 .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-score"], color="#D1FAE5")
                 .highlight_min(subset=["Type I error", "Type II error"], color="#D1FAE5"))

    ch1, ch2 = st.columns(2)
    names = [MODEL_META[n]["short"] for n in MODEL_META]
    x = np.arange(len(names))
    with ch1:
        st.markdown("#### 📈 Accuracy · Precision · Recall · F1")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for j, (key, col) in enumerate([("Accuracy", PURPLE), ("Precision", GREEN), ("Recall", YELLOW), ("F1", RED)]):
            ax.bar(x + (j - 1.5) * 0.2, [mets[n][key] for n in MODEL_META], 0.2, label=key, color=col)
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.05); ax.legend(ncol=4, fontsize=8)
        ax.grid(axis="y", alpha=.3); fig.tight_layout(); show(fig)
    with ch2:
        st.markdown("#### 🟡 Type I vs 🔴 Type II error")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(x - 0.2, [mets[n]["TypeI"] for n in MODEL_META], 0.4, label="Type I (false alarm)", color=YELLOW)
        ax.bar(x + 0.2, [mets[n]["TypeII"] for n in MODEL_META], 0.4, label="Type II (missed failure)", color=RED)
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Error rate"); ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.3); fig.tight_layout(); show(fig)

    st.markdown("#### 🔥 Confusion-matrix heat maps")
    hc = st.columns(3)
    for c, n in zip(hc, MODEL_META):
        with c:
            show(heatmap(mets[n], f"{MODEL_META[n]['short']} (thr {model_threshold(n):.2f})"))

    st.markdown("#### 🚀 What does AdaBoost rely on most?")
    imp = pd.Series(art["models"]["AdaBoost Classifier"].feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(imp.index, imp.values, color=PURPLE)
    ax.set_xlabel("Feature importance"); ax.grid(axis="x", alpha=.3); fig.tight_layout(); show(fig)

# ======================= How it works tab =======================
with tab_how:
    st.markdown("### 🧭 Project pipeline")
    st.markdown('<div class="steps"><div class="step">📥 Load CSV</div><span class="arrow">➜</span>'
                '<div class="step">🔎 Explore data</div><span class="arrow">➜</span>'
                '<div class="step">✂️ 80/20 split</div><span class="arrow">➜</span>'
                '<div class="step">📏 Scale features</div><span class="arrow">➜</span>'
                '<div class="step">🧠 Train 3 models</div><span class="arrow">➜</span>'
                '<div class="step">🎯 Tune threshold</div><span class="arrow">➜</span>'
                '<div class="step">📊 Evaluate</div></div>', unsafe_allow_html=True)

    st.markdown("### 🧠 The three classifiers")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="box" style="background:#EDE9FE;border-top:6px solid {PURPLE}"><b>🔔 Gaussian Naive Bayes</b><br>'
                    'Assumes each sensor follows a bell curve within each class and that sensors are independent. '
                    'Fast, simple, interpretable baseline.</div>', unsafe_allow_html=True)
        st.latex(r"P(c\mid x)\propto P(c)\prod_i \mathcal{N}(x_i;\mu_{c,i},\sigma_{c,i}^2)")
    with c2:
        st.markdown(f'<div class="box" style="background:#D1FAE5;border-top:6px solid {GREEN}"><b>🌊 Kernel (KDE) Naive Bayes</b><br>'
                    'Replaces the bell curve with a kernel density estimate, so it can learn skewed, non-Gaussian '
                    'sensor distributions (the kernel idea for Naive Bayes).</div>', unsafe_allow_html=True)
        st.latex(r"\hat p(x_i\mid c)=\frac{1}{n_c h}\sum_j K\!\left(\frac{x_i-x_{ij}}{h}\right)")
    with c3:
        st.markdown(f'<div class="box" style="background:#FEF3C7;border-top:6px solid {YELLOW}"><b>🚀 AdaBoost</b><br>'
                    'Trains many tiny decision trees in sequence; each one focuses on the samples the previous ones got '
                    'wrong. No independence assumption between sensors.</div>', unsafe_allow_html=True)
        st.latex(r"F(x)=\mathrm{sign}\Big(\sum_t \alpha_t h_t(x)\Big)")

    st.markdown("### 🟡 Type I vs 🔴 Type II error")
    t1, t2 = st.columns(2)
    t1.markdown(f'<div class="box" style="background:#FEF3C7;border-left:6px solid {YELLOW}"><b>🟡 Type I error (False Positive)</b><br>'
                'Model says <i>failure</i>, machine is healthy.<br>💸 Cost: unnecessary inspection / downtime.<br>'
                '<b>Rate = FP / (FP + TN)</b></div>', unsafe_allow_html=True)
    t2.markdown(f'<div class="box" style="background:#FEE2E2;border-left:6px solid {RED}"><b>🔴 Type II error (False Negative)</b><br>'
                'Model says <i>normal</i>, machine actually fails.<br>💥 Cost: breakdown, lost production, safety risk.<br>'
                '<b>Rate = FN / (FN + TP) = 1 − Recall</b></div>', unsafe_allow_html=True)

    st.markdown("### 📐 Metric cheat-sheet")
    st.dataframe(pd.DataFrame({
        "Metric": ["🎯 Accuracy", "🔍 Precision", "📡 Recall", "⚖️ F1-score"],
        "Formula": ["(TP+TN)/(all)", "TP/(TP+FP)", "TP/(TP+FN)", "2·P·R/(P+R)"],
        "Plain meaning": ["Overall share of correct predictions (misleading when failures are rare)",
                          "When the model raises an alarm, how often is it right?",
                          "Of all real failures, how many did we catch?",
                          "Balance between precision and recall"]}), hide_index=True)

# ======================= Data tab =======================
with tab_data:
    st.markdown("### 🗂️ Dataset preview")
    st.dataframe(df.head(50))

    st.markdown("### 🧰 Sensor guide")
    st.dataframe(pd.DataFrame({"Icon": SENSOR_ICONS, "Sensor": feats,
                               "Unit": [u for u, _ in SENSOR_INFO],
                               "What it measures": [d for _, d in SENSOR_INFO]}), hide_index=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### 🍩 Class balance")
        counts = df[target].value_counts().reindex([0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        ax.pie(counts.values, labels=["Normal", "Failure"], colors=[GREEN, RED], autopct="%1.1f%%", startangle=90,
               wedgeprops=dict(width=0.45, edgecolor="white"))
        fig.tight_layout(); show(fig)
        st.caption("Failures are rare, so accuracy alone is misleading. Always check Type I / Type II error.")
    with d2:
        st.markdown("#### 🔗 Correlation heat map")
        fig, ax = plt.subplots(figsize=(5.2, 4))
        sns.heatmap(df[feats + [target]].corr(), annot=True, fmt=".2f", cmap="PRGn", center=0, ax=ax)
        fig.tight_layout(); show(fig)

    st.markdown("#### 📉 Sensor distributions: normal 🟢 vs failure 🔴")
    plot_df = df[feats].copy()
    plot_df["Status"] = df[target].map({0: "Normal", 1: "Failure"})
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))
    for ax, f in zip(axes, feats):
        sns.kdeplot(data=plot_df, x=f, hue="Status", common_norm=False, fill=True,
                    palette={"Normal": GREEN, "Failure": RED}, ax=ax)
        ax.set_title(f, fontsize=9)
    fig.tight_layout(); show(fig)

st.markdown("---")
st.caption("🛠️ Predictive Maintenance · Gaussian NB · Kernel NB · AdaBoost · Built with Streamlit & scikit-learn")
