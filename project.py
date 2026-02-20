import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# ==============================
# LOAD DATA
# ==============================

calories = pd.read_csv("calories.csv")
exercise_data = pd.read_csv("exercise.csv")

# Combine datasets
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

# ==============================
# PREPROCESSING
# ==============================

# Convert Gender properly
calories_data["Gender"] = calories_data["Gender"].map({
    'male': 0,
    'female': 1
})

# Convert all columns to numeric (important for XGBoost)
calories_data = calories_data.apply(pd.to_numeric)

# Debug check (optional)
print("Data Types:\n", calories_data.dtypes)

# ==============================
# FEATURE & TARGET SPLIT
# ==============================

X = calories_data.drop(columns=['User_ID', 'Calories'])
Y = calories_data['Calories']

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# ==============================
# MODEL TRAINING
# ==============================

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, Y_train)

# Model Evaluation
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Training R2 Score:", r2_score(Y_train, train_pred))
print("Testing R2 Score:", r2_score(Y_test, test_pred))


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_calories_burnt():
    try:
        gender = gender_var.get()
        age = float(age_entry.get())
        height = float(height_entry.get())
        weight = float(weight_entry.get())
        duration = float(duration_entry.get())
        heart_rate = float(heart_rate_entry.get())
        body_temp = float(body_temp_entry.get())

        # Convert gender
        gender_val = 0 if gender == "male" else 1

        user_input = pd.DataFrame(
            [[gender_val, age, height, weight, duration, heart_rate, body_temp]],
            columns=['Gender', 'Age', 'Height', 'Weight',
                     'Duration', 'Heart_Rate', 'Body_Temp']
        )

        prediction = model.predict(user_input)

        messagebox.showinfo(
            "Prediction",
            f"Estimated Calories Burnt: {prediction[0]:.2f} calories"
        )

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# ==============================
# GUI DESIGN
# ==============================

root = tk.Tk()
root.title("Calories Burned Prediction")
root.geometry("400x400")

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

# Gender Dropdown
tk.Label(frame, text="Gender").grid(row=0, column=0, pady=5)
gender_var = tk.StringVar(value="male")
tk.OptionMenu(frame, gender_var, "male", "female").grid(row=0, column=1)

# Age
tk.Label(frame, text="Age").grid(row=1, column=0, pady=5)
age_entry = tk.Entry(frame)
age_entry.grid(row=1, column=1)

# Height
tk.Label(frame, text="Height (cm)").grid(row=2, column=0, pady=5)
height_entry = tk.Entry(frame)
height_entry.grid(row=2, column=1)

# Weight
tk.Label(frame, text="Weight (kg)").grid(row=3, column=0, pady=5)
weight_entry = tk.Entry(frame)
weight_entry.grid(row=3, column=1)

# Duration
tk.Label(frame, text="Duration (minutes)").grid(row=4, column=0, pady=5)
duration_entry = tk.Entry(frame)
duration_entry.grid(row=4, column=1)

# Heart Rate
tk.Label(frame, text="Heart Rate (bpm)").grid(row=5, column=0, pady=5)
heart_rate_entry = tk.Entry(frame)
heart_rate_entry.grid(row=5, column=1)

# Body Temp
tk.Label(frame, text="Body Temp (Celsius)").grid(row=6, column=0, pady=5)
body_temp_entry = tk.Entry(frame)
body_temp_entry.grid(row=6, column=1)

# Predict Button
tk.Button(
    frame,
    text="Predict Calories",
    command=predict_calories_burnt,
    bg="green",
    fg="white"
).grid(row=7, columnspan=2, pady=15)

root.mainloop()