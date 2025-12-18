from flask import Blueprint, render_template, request
import pandas as pd
import joblib

main = Blueprint("main", __name__)

# charger les modèles
reg = joblib.load("models/reg.pkl")        # modèle complet
reg1 = joblib.load("models/reg1.pkl")      # R&D seulement
ct = joblib.load("models/ct.pkl")
noms = joblib.load("models/noms.pkl")

@main.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        action = request.form["action"]
        rd = float(request.form["rd_spend"])

        # ---------- CAS 1 : R&D seulement ----------
        if action == "rd":
            X = pd.DataFrame([[rd]], columns=['R&D Spend'])
            prediction = reg1.predict(X)[0]

        # ---------- CAS 2 : modèle complet ----------
        else:
            admin = float(request.form["administration"])
            marketing = float(request.form["marketing_spend"])
            state = request.form["state"]

            ligne = [rd, admin, marketing, state]

            ligne_df = pd.DataFrame(
                [ligne],
                columns=['R&D Spend', 'Administration', 'Marketing Spend', 'State']
            )

            ligne_df = ct.transform(ligne_df)
            ligne_df = pd.DataFrame(ligne_df, columns=noms)
            ligne_df = ligne_df.iloc[:, 1:]

            prediction = reg.predict(ligne_df)[0]

    return render_template("index.html", prediction=prediction)
