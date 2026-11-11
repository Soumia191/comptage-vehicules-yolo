"""Interface Streamlit de MaintAI : explorer les données, puis demander une prédiction à l'API.

Lancement (depuis la racine du projet) : python -m streamlit run ui/app.py
"""
import httpx
import pandas as pd
import streamlit as st

from maintai.client import APIError, MaintaiClient
from maintai.config import FEATURES, get_settings

st.set_page_config(page_title="MaintAI", page_icon="🔧")
st.title("MaintAI · prédiction de panne")


@st.cache_data
def charger(path) -> pd.DataFrame:
    # Lu une seule fois : Streamlit relance tout le script à chaque clic
    return pd.read_csv(path)


with st.sidebar:
    url = st.text_input("URL de l'API", "http://localhost:8000")
    cle = st.text_input("Clé API", type="password")

onglet_donnees, onglet_prediction = st.tabs(["Données", "Prédiction"])

# Sprint 1 : explorer les données
with onglet_donnees:
    df = charger(get_settings().data_path)

    # 1. Trois indicateurs côte à côte (le premier, le nombre de relevés, est lu par le test)
    c1, c2, c3 = st.columns(3)
    c1.metric("Relevés", len(df))
    # failure vaut 0 ou 1 : sa moyenne est la part de pannes
    c2.metric("Taux de panne", f"{df.failure.mean():.1%}")
    c3.metric("Température moyenne", f"{df.temperature_c.mean():.1f} °C")

    # 2. Le choix d'une variable
    variable = st.selectbox("Variable", FEATURES)

    # 3. Le taux de panne par tranche de cette variable
    tranches = pd.cut(df[variable], bins=10)
    taux = df.groupby(tranches, observed=True).failure.mean()
    # Le milieu de chaque tranche, gardé en nombre (en texte, les barres seraient mal triées)
    taux.index = [round(i.mid, 2) for i in taux.index]
    taux.index.name = variable
    st.subheader(f"Taux de panne par tranche de {variable}")
    st.bar_chart(taux, x_label=variable, y_label="taux de panne")

    # 4. Le nuage de points température / vibration, coloré selon la panne
    st.subheader("Température et vibration")
    nuage = df.assign(panne=df.failure.map({0: "non", 1: "oui"}))
    st.scatter_chart(nuage, x="temperature_c", y="vibration_mm_s", color="panne")


# Sprint 2 : demander une prédiction à l'API
with onglet_prediction:
    with st.form("releve"):
        releve = {
            "temperature_c": st.slider("Température (°C)", -40.0, 150.0, 72.0),
            "vibration_mm_s": st.slider("Vibration (mm/s)", 0.0, 50.0, 6.0),
            "load_pct": st.slider("Charge (%)", 0.0, 100.0, 85.0),
            "runtime_hours": st.number_input("Heures de fonctionnement", 0, 100_000, 15000),
        }
        envoye = st.form_submit_button("Prédire")

    if envoye:  # True seulement pendant la relance provoquée par le bouton « Prédire »
        api = MaintaiClient(url, api_key=cle)
        try:
            resultat = api.predict(releve)
        except APIError as e:
            # L'API a répondu, mais a refusé : 401 (clé), 422 (relevé invalide), 503 (modèle absent)
            st.warning(f"L'API a refusé la requête (erreur {e.status_code}) : {e.detail}")
        except httpx.HTTPError:
            # Pas de réponse du tout : uvicorn arrêté, mauvaise URL ou mauvais port
            st.error(f"API injoignable à {url} : vérifiez qu'uvicorn est lancé, l'URL et le port.")
        else:
            pct = round(resultat["failure_probability"] * 100)
            st.metric("Probabilité de panne", f"{pct} %")
            seuil = round(resultat["threshold"] * 100)
            if resultat["alert"]:
                st.error(f"ALERTE : risque de panne au-dessus du seuil de {seuil} %. "
                         "Planifiez une intervention.")
            else:
                st.success(f"Fonctionnement normal : risque sous le seuil de {seuil} %.")
            st.caption(f"Modèle entraîné le {resultat['model_version']}")
