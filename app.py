import streamlit as st
import pandas as pd
import joblib

modele = joblib.load("modele_detection_billets.joblib")

st.title("Détection de faux billets 💶")
st.write(
    "Importez un fichier CSV contenant les caractéristiques géométriques "
    "des billets afin d'obtenir automatiquement leur classification."
)

fichier = st.file_uploader("Importer un fichier CSV")

if fichier is not None:
    billets = pd.read_csv(fichier)

    X = billets[
        [
            "diagonal",
            "height_left",
            "height_right",
            "margin_low",
            "margin_up",
            "length"
        ]
    ]

    predictions = modele.predict(X)

    billets["prediction"] = [
    "Authentique" if prediction else "Faux"
    for prediction in predictions
]

st.subheader("Résultats de l'analyse")
st.dataframe(billets, use_container_width=True)

nb_authentiques = sum(predictions)
nb_faux = len(predictions) - nb_authentiques

col1, col2 = st.columns(2)

col1.metric("Billets authentiques", nb_authentiques)
col2.metric("Faux billets", nb_faux)