import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Data Collection & Processing
calories = pd.read_csv("C:/Users/vivek reddy/OneDrive/Desktop/content/calories.csv")  # Corrected path
exercise_data = pd.read_csv("C:/Users/vivek reddy/OneDrive/Desktop/content/exercise.csv")  # Corrected path

# Combine the two DataFrames
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

# Preprocessing: Converting categorical data to numerical
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# Separating Features and Target
X = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)
Y = calories_data['Calories']

# Splitting the data into Training and Test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training with XGBoost Regressor
model = XGBRegressor()
model.fit(X_train, Y_train)

# Function to take user input and make a prediction
def predict_calories_burnt():
    try:
        # Get user inputs from Tkinter Entry widgets
        gender = gender_var.get().strip().lower()
        age = float(age_entry.get())
        height = float(height_entry.get())
        weight = float(weight_entry.get())
        duration = float(duration_entry.get())
        heart_rate = float(heart_rate_entry.get())
        body_temp = float(body_temp_entry.get())

        # Convert gender to numerical value
        if gender == "male":
            gender = 0
        elif gender == "female":
            gender = 1
        else:
            raise ValueError("Invalid gender input. Please enter either 'male' or 'female'.")
        
        # Create the user input DataFrame with the correct column names (as used during training)
        user_input = pd.DataFrame([[gender, age, height, weight, duration, heart_rate, body_temp]],
                                  columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

        # Predicting the number of calories burned
        prediction = model.predict(user_input)

        # Show the prediction result in a message box
        messagebox.showinfo("Prediction", f"Calories Burnt: {prediction[0]:.2f} calories")
    
    except ValueError as e:
        messagebox.showerror("Input Error", f"Error: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Set up the main window
root = tk.Tk()
root.title("Calories Burned Prediction")

# Create a frame for the input fields
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Gender input (dropdown)
gender_label = tk.Label(frame, text="Gender (male/female):")
gender_label.grid(row=0, column=0, sticky="e", pady=5)
gender_var = tk.StringVar()
gender_entry = tk.Entry(frame, textvariable=gender_var)
gender_entry.grid(row=0, column=1, pady=5)

# Age input
age_label = tk.Label(frame, text="Age (in years):")
age_label.grid(row=1, column=0, sticky="e", pady=5)
age_entry = tk.Entry(frame)
age_entry.grid(row=1, column=1, pady=5)

# Height input
height_label = tk.Label(frame, text="Height (in cm):")
height_label.grid(row=2, column=0, sticky="e", pady=5)
height_entry = tk.Entry(frame)
height_entry.grid(row=2, column=1, pady=5)

# Weight input
weight_label = tk.Label(frame, text="Weight (in kg):")
weight_label.grid(row=3, column=0, sticky="e", pady=5)
weight_entry = tk.Entry(frame)
weight_entry.grid(row=3, column=1, pady=5)

# Duration input
duration_label = tk.Label(frame, text="Exercise Duration (in minutes):")
duration_label.grid(row=4, column=0, sticky="e", pady=5)
duration_entry = tk.Entry(frame)
duration_entry.grid(row=4, column=1, pady=5)

# Heart Rate input
heart_rate_label = tk.Label(frame, text="Heart Rate (beats per minute):")
heart_rate_label.grid(row=5, column=0, sticky="e", pady=5)
heart_rate_entry = tk.Entry(frame)
heart_rate_entry.grid(row=5, column=1, pady=5)

# Body Temperature input
body_temp_label = tk.Label(frame, text="Body Temperature (in Celsius):")
body_temp_label.grid(row=6, column=0, sticky="e", pady=5)
body_temp_entry = tk.Entry(frame)
body_temp_entry.grid(row=6, column=1, pady=5)

# Submit button to predict calories
submit_button = tk.Button(frame, text="Predict Calories Burnt", command=predict_calories_burnt)
submit_button.grid(row=7, columnspan=2, pady=10)

# Run the Tkinter event loop
root.mainloop()
