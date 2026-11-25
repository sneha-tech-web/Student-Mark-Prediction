import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders for generated results
os.makedirs("outputs", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# 1. Load the dataset
data = pd.read_csv("Student_Performance.csv")

# 2. Select input features and target
X = data[["StudyHours", "Attendance", "PreviousScore"]]
y = data["Marks"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict marks for test data
y_pred = model.predict(X_test)

# 6. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

with open("outputs/evaluation_results.txt", "w") as file:
    file.write("STUDENT MARKS PREDICTION - EVALUATION RESULTS\\n")
    file.write(f"Mean Absolute Error (MAE): {mae:.2f}\\n")
    file.write(f"Mean Squared Error (MSE): {mse:.2f}\\n")
    file.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\\n")
    file.write(f"R2 Score: {r2:.4f}\\n")

# 7. Save actual and predicted marks
results = pd.DataFrame({
    "ActualMarks": y_test.values.round(2),
    "PredictedMarks": y_pred.round(2)
})
results.to_csv("outputs/predictions.csv", index=False)

# 8. Save model coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_.round(4)
})
coefficients.to_csv("outputs/model_coefficients.csv", index=False)

# 9. Graph: study hours vs marks
plt.figure(figsize=(7, 5))
plt.scatter(data["StudyHours"], data["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Student Marks")
plt.tight_layout()
plt.savefig("visualizations/study_hours_vs_marks.png")
plt.close()

# 10. Graph: actual vs predicted marks
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.plot([0, 100], [0, 100], linestyle="--")
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Student Marks")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("visualizations/actual_vs_predicted.png")
plt.close()

# 11. Example prediction for a new student
new_student = pd.DataFrame({
    "StudyHours": [5.0],
    "Attendance": [85.0],
    "PreviousScore": [70.0]
})
predicted_mark = model.predict(new_student)[0]
print(f"Predicted marks for the sample student: {predicted_mark:.2f}")

print("Project completed. Check the outputs and visualizations folders.")
print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")
