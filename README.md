# SAD Predictive System

A Streamlit prototype for the Systems Analysis and Design laboratory examination. It follows the notebook tasks for data understanding, cleaning/feature engineering, exploratory data analysis, Decision Tree modeling, and a dashboard/prediction prototype.

## Project structure

```text
SAD_Predictive_System/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── facility_reservations.csv
└── documentation/
    └── deployment_report.md
```

## Requirements
- Python 3.10 or newer
- The supplied CSV at `data/facility_reservations.csv`

## Run locally
1. Extract the ZIP file.
2. Open a terminal in the `SAD_Predictive_System` folder.
3. Install packages: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`
5. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Features
- Dataset overview and data-quality checks.
- Reservation counts by facility type.
- Cancellation/no-show count and percentage by user category.
- Outcome counts and percentages by lead-time category.
- Decision Tree model using an 80/20 train-test split, with accuracy, weighted precision/recall/F1, confusion matrix, classification report, and top-three aggregated predictor importance.
- Prediction form using candidate predictors available at reservation submission time.
- Leakage checks that exclude `Usage_Status`, `Actual_Attendance`, and `Cancellation_Days_Before` from model inputs.

## Target definition
The app derives `Reservation_Outcome` from `Usage_Status`: `Used` becomes `Completed`; `Cancelled` and `No-Show` become `Not Completed`. Other unrecognized statuses are excluded from model training and evaluation.

## Important limitation
This is an academic prototype. Predictions are not confirmed outcomes and must not automatically approve or reject reservations. Validate the dataset, model, privacy safeguards, and results with relevant campus stakeholders before real deployment.
