# Calorie Burn Prediction Using XGBoost

This is a Python-based application that predicts the number of calories burned during exercise based on various input parameters such as gender, age, height, weight, exercise duration, heart rate, and body temperature. The application uses the **XGBoost** machine learning model for prediction and the **Tkinter** library for a user-friendly graphical interface.

## Features
- **User-Friendly Interface**: The application has an intuitive GUI built with Tkinter where users can input their personal details and exercise data.
- **Real-time Prediction**: It calculates the number of calories burned instantly based on the user's input.
- **Machine Learning Model**: Uses **XGBoost Regressor** to predict the number of calories burned based on past training data.

## Technologies Used
- **Python**: Main programming language for backend processing and logic.
- **Tkinter**: For building the graphical user interface (GUI).
- **XGBoost**: A machine learning model for regression tasks used to predict the calories burned.
- **Pandas**: For data manipulation and cleaning.
- **Scikit-learn**: For splitting data into training and test sets.

## How It Works

1. **Data Collection & Processing**:
   - The project loads exercise data and calories data using **Pandas**.
   - The data is preprocessed, including converting categorical data (gender) to numeric values.
   - The features (independent variables) and target (dependent variable) are separated.

2. **Model Training**:
   - The **XGBoost Regressor** model is trained using the exercise data.
   - The model is then used to make predictions about the number of calories burned during a workout.

3. **User Input**:
   - Users input details such as gender, age, height, weight, exercise duration, heart rate, and body temperature into a Tkinter form.
   - The form uses **Tkinter Entry widgets** for input fields.

4. **Prediction**:
   - The model takes the input data and predicts the calories burned during exercise.
   - The prediction result is displayed in a message box.

5. **Error Handling**:
   - If the user enters invalid data, error messages are displayed to guide the user.

## Setup & Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/calorie-burn-prediction.git
