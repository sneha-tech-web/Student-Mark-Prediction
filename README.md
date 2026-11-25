# Student Marks Prediction 

## 1. Project description
This beginner-friendly project predicts a student's marks using study hours,
attendance percentage, and previous exam score.

## 2. Algorithm
Linear Regression (a supervised machine-learning regression algorithm).

## 3. Tools and libraries
- Python
- Pandas
- Matplotlib
- Scikit-learn

## 4. Dataset
`Student_Performance.csv` contains illustrative sample data with these columns:
- `StudyHours`: hours studied
- `Attendance`: attendance percentage
- `PreviousScore`: previous exam score
- `Marks`: target marks to predict

**Dataset note:** This is a synthetic demonstration dataset created for learning.
It is not collected from a real school or university and should not be presented
as real student records.

## 5. Project workflow
1. Read the CSV dataset.
2. Select study hours, attendance, and previous score as input features.
3. Select marks as the target.
4. Split the data into training and testing sets.
5. Train a Linear Regression model.
6. Predict marks for test records.
7. Evaluate predictions using MAE, MSE, RMSE, and R-squared.
8. Save prediction results and graphs.

## 6. Installation
```bash
pip install -r requirements.txt
```

## 7. Run
Keep the CSV and Python file in the same project folder, then run:
```bash
python Student_Marks.py
```

## 8. Generated files
The program creates:
- `outputs/predictions.csv`
- `outputs/evaluation_results.txt`
- `outputs/model_coefficients.csv`
- `visualizations/study_hours_vs_marks.png`
- `visualizations/actual_vs_predicted.png`

## 9. Example prediction
The program also predicts marks for an example student with 5 study hours,
85% attendance, and a previous score of 70.

## 10. Conclusion
The project demonstrates how Linear Regression can estimate marks from
academic input features. Predictions are estimates, not guaranteed exam results.
