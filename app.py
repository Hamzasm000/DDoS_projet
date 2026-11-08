import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="DDoS Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Détection & Prédiction des attaques DDoS")

DATA_PATH = "dataset.csv"
LABEL_COL = "label"


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def train_model(df):
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="DDoS"),
        "Recall": recall_score(y_test, y_pred, pos_label="DDoS"),
        "F1": f1_score(y_test, y_pred, pos_label="DDoS"),
    }
    cm = confusion_matrix(y_test, y_pred, labels=["Normal", "DDoS"])
    return model, metrics, cm


# ---------- Data ----------
df = load_data(DATA_PATH)
features = [c for c in df.columns if c != LABEL_COL]

st.header("📂 Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Lignes", f"{df.shape[0]:,}")
col2.metric("Colonnes (features)", len(features))
col3.metric("Taux DDoS", f"{(df[LABEL_COL] == 'DDoS').mean():.1%}")

with st.expander("Aperçu des données"):
    st.dataframe(df.head(50), use_container_width=True)

with st.expander("Statistiques descriptives"):
    st.dataframe(df[features].describe().T, use_container_width=True)

st.subheader("Répartition des classes")
st.bar_chart(df[LABEL_COL].value_counts())

# ---------- Model ----------
model, metrics, cm = train_model(df)

st.header("📊 Métriques du modèle")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
m2.metric("Precision", f"{metrics['Precision']:.3f}")
m3.metric("Recall", f"{metrics['Recall']:.3f}")
m4.metric("F1", f"{metrics['F1']:.3f}")

st.subheader("Matrice de confusion")
st.dataframe(
    pd.DataFrame(cm, index=["Réel Normal", "Réel DDoS"],
                 columns=["Prédit Normal", "Prédit DDoS"]),
    use_container_width=True,
)

# ---------- Prediction ----------
st.header("🔮 Prédiction")
tab1, tab2 = st.tabs(["Saisie manuelle", "Fichier CSV"])

with tab1:
    inputs = {}
    cols = st.columns(3)
    for i, feat in enumerate(features):
        lo, hi = float(df[feat].min()), float(df[feat].max())
        default = float(df[feat].median())
        inputs[feat] = cols[i % 3].number_input(
            feat, min_value=lo, max_value=hi if hi > lo else lo + 1.0, value=default
        )
    if st.button("Prédire", type="primary"):
        sample = pd.DataFrame([inputs])
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]
        conf = proba[list(model.classes_).index(pred)]
        if pred == "DDoS":
            st.error(f"⚠️ Prédiction : **DDoS** (confiance {conf:.1%})")
        else:
            st.success(f"✅ Prédiction : **Normal** (confiance {conf:.1%})")

with tab2:
    up = st.file_uploader("Uploader un CSV avec les mêmes features", type="csv")
    if up is not None:
        new_df = pd.read_csv(up)
        missing = [f for f in features if f not in new_df.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
        else:
            new_df["prediction"] = model.predict(new_df[features])
            st.dataframe(new_df, use_container_width=True)
            st.download_button(
                "Télécharger les prédictions",
                new_df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )
