import argparse
import pandas as pd
import joblib


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Les six caractéristiques géométriques utilisées
# par le modèle pour réaliser ses prédictions.
FEATURES = [
    "diagonal",
    "height_left",
    "height_right",
    "margin_low",
    "margin_up",
    "length"
]


# Chargement du pipeline entraîné dans le notebook.
#
# Ce pipeline contient :
# SimpleImputer → StandardScaler → LogisticRegression
#
# Il n'est donc pas nécessaire de refaire manuellement
# le preprocessing dans ce script.
modele = joblib.load(
    "modele_detection_billets.joblib"
)


# ---------------------------------------------------------
# PRÉDICTION D'UN SEUL BILLET
# ---------------------------------------------------------

def predire_billet(
    diagonal,
    height_left,
    height_right,
    margin_low,
    margin_up,
    length
):
    """
    Prédit si un billet est authentique ou faux
    à partir de ses six dimensions géométriques.

    Retour :
    True  = authentique
    False = faux
    """

    # Transformation des six valeurs reçues en DataFrame.
    #
    # Le modèle a été entraîné avec ces noms de colonnes :
    # on conserve donc exactement la même structure.
    billet = pd.DataFrame(
        [[
            diagonal,
            height_left,
            height_right,
            margin_low,
            margin_up,
            length
        ]],
        columns=FEATURES
    )

    # Le pipeline applique automatiquement :
    # imputation → standardisation → prédiction.
    prediction = modele.predict(billet)[0]

    # Conversion en bool Python afin d'obtenir
    # l'output binaire demandé.
    return bool(prediction)


# ---------------------------------------------------------
# PRÉDICTION D'UN FICHIER CSV
# ---------------------------------------------------------

def predire_fichier(chemin_csv):
    """
    Prédit la classe de plusieurs billets
    contenus dans un fichier CSV.

    Retour :
    True  = authentique
    False = faux
    """

    # Chargement du fichier fourni.
    billets = pd.read_csv(chemin_csv)

    # Sélection uniquement des six variables géométriques.
    #
    # Une éventuelle colonne "id" n'est pas utilisée
    # par l'algorithme.
    X = billets[FEATURES]

    # Prédiction de tous les billets du fichier.
    predictions = modele.predict(X)

    return predictions


# ---------------------------------------------------------
# INTERFACE DU SCRIPT
# ---------------------------------------------------------

if __name__ == "__main__":

    # argparse permet de fournir les informations
    # directement depuis le terminal.
    parser = argparse.ArgumentParser(
        description="Détection automatique de faux billets."
    )


    # Première possibilité :
    # fournir le chemin vers un fichier CSV.
    parser.add_argument(
        "--csv",
        type=str,
        help="Chemin vers un fichier CSV contenant plusieurs billets."
    )


    # Deuxième possibilité :
    # fournir directement les six dimensions d'un billet.
    parser.add_argument(
        "--billet",
        nargs=6,
        type=float,
        metavar=(
            "DIAGONAL",
            "HEIGHT_LEFT",
            "HEIGHT_RIGHT",
            "MARGIN_LOW",
            "MARGIN_UP",
            "LENGTH"
        ),
        help="Six dimensions géométriques d'un billet."
    )


    # Lecture des arguments fournis par l'utilisateur.
    args = parser.parse_args()


    # CAS 1 : un fichier CSV a été fourni.
    if args.csv:

        predictions = predire_fichier(
            args.csv
        )

        # Un output binaire est affiché
        # pour chaque billet.
        for prediction in predictions:
            print(bool(prediction))


    # CAS 2 : les dimensions d'un billet ont été fournies.
    elif args.billet:

        prediction = predire_billet(
            *args.billet
        )

        print(prediction)


    # Aucun des deux inputs attendus n'a été fourni.
    else:

        parser.error(
            "Fournissez soit --csv, soit --billet."
        )
