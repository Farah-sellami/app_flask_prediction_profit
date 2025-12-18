from flask import Blueprint, render_template, request
import pandas as pd
import joblib

main = Blueprint("main", __name__)

# charger les objets ML
reg = joblib.load("models/reg.pkl")
ct = joblib.load("models/ct.pkl")
noms = joblib.load("models/noms.pkl")

@main.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        rd = float(request.form["rd_spend"])
        admin = float(request.form["administration"])
        marketing = float(request.form["marketing_spend"])
        state = request.form["state"]

        # recréer EXACTEMENT la ligne comme dans le training
        ligne = [rd, admin, marketing, state]

        ligne_df = pd.DataFrame(
            [ligne],
            columns=['R&D Spend', 'Administration', 'Marketing Spend', 'State']
        )

        # transformations identiques au training
        ligne_df = ct.transform(ligne_df)
        ligne_df = pd.DataFrame(ligne_df, columns=noms)

        # supprimer la colonne redondante
        ligne_df = ligne_df.iloc[:, 1:]

        # prédiction
        prediction = reg.predict(ligne_df)[0]

    return render_template("index.html", prediction=prediction)
