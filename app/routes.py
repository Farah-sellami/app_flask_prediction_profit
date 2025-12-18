from flask import Blueprint, render_template, request, redirect, url_for, session
import pandas as pd
import joblib

main = Blueprint("main", __name__)

# ===== USERS (simple demo) =====
USERS = {
    "admin": "1234",
    "rahma": "rahma123"
}

# ===== Charger les modèles =====
reg = joblib.load("models/reg.pkl")
reg1 = joblib.load("models/reg1.pkl")
ct = joblib.load("models/ct.pkl")
noms = joblib.load("models/noms.pkl")


# ================= LOGIN =================
@main.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in USERS and USERS[username] == password:
            session["user"] = username
            return redirect(url_for("main.index"))
        else:
            error = "Identifiants incorrects"

    return render_template("login.html", error=error)


# ================= LOGOUT =================
@main.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("main.login"))


# ================= PAGE PRINCIPALE =================
@main.route("/", methods=["GET", "POST"])
def index():
    if "user" not in session:
        return redirect(url_for("main.login"))

    prediction = None

    if request.method == "POST":
        mode = request.form.get("mode")   # full ou rd
        rd = float(request.form["rd_spend"])

        # ---------- CAS 1 : R&D seulement ----------
        if mode == "rd":
            X = pd.DataFrame([[rd]], columns=['R&D Spend'])
            prediction = reg1.predict(X)[0]

        # ---- Modèle complet ----
        else:
            admin = float(request.form["administration"])
            marketing = float(request.form["marketing_spend"])
            state = request.form["state"]

            ligne_df = pd.DataFrame(
                [[rd, admin, marketing, state]],
                columns=['R&D Spend', 'Administration', 'Marketing Spend', 'State']
            )

            ligne_df = ct.transform(ligne_df)
            ligne_df = pd.DataFrame(ligne_df, columns=noms)
            ligne_df = ligne_df.iloc[:, 1:]

            prediction = reg.predict(ligne_df)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        user=session["user"]
    )
