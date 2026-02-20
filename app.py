from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

app = Flask(__name__)

# Load Data
calories = pd.read_csv("calories.csv")
exercise_data = pd.read_csv("exercise.csv")

calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

# Preprocess
calories_data["Gender"] = calories_data["Gender"].map({
    'male': 0,
    'female': 1
})

calories_data = calories_data.apply(pd.to_numeric)

X = calories_data.drop(columns=['User_ID', 'Calories'])
Y = calories_data['Calories']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, Y_train)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        gender = request.form["gender"]
        age = float(request.form["age"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        duration = float(request.form["duration"])
        heart_rate = float(request.form["heart_rate"])
        body_temp = float(request.form["body_temp"])

        gender_val = 0 if gender == "male" else 1

        user_input = pd.DataFrame(
            [[gender_val, age, height, weight,
              duration, heart_rate, body_temp]],
            columns=['Gender', 'Age', 'Height', 'Weight',
                     'Duration', 'Heart_Rate', 'Body_Temp']
        )

        prediction = round(model.predict(user_input)[0], 2)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)