# Deployment Report — SAD Predictive System

## 1. Project overview
The Campus Facility Reservation and Monitoring System prototype uses historical reservation records to explore facility usage and predict `Reservation_Outcome` with a Decision Tree classifier. It supports the laboratory workflow: data understanding, data cleaning/feature engineering, EDA, predictive modeling, and dashboard/decision support.

## 2. Included files
- `app.py` — Streamlit dashboard, preprocessing, EDA, model training/evaluation, feature importance, prediction form, and data-quality checks.
- `requirements.txt` — Python dependencies.
- `README.md` — installation and run instructions.
- `data/facility_reservations.csv` — supplied reservation dataset.
- `documentation/deployment_report.md` — this report.

## 3. Technology and dependencies
Python 3.10+, Streamlit, pandas, scikit-learn, and Matplotlib. Install dependencies using `pip install -r requirements.txt`.

## 4. Target and modeling approach
The app derives `Reservation_Outcome` from `Usage_Status`: `Used` is mapped to `Completed`, while `Cancelled` and `No-Show`/`Noshow` are mapped to `Not Completed`. The Decision Tree uses the notebook's candidate predictors when present: `User_Category`, `Facility_Type`, `Facility_ID`, `Purpose`, `Lead_Time_Days`, `Day_Of_Week`, `Participants`, and `Duration_Hours`. The split is 80% training and 20% testing with `random_state=42`; the tree has `max_depth=5` and class balancing.

## 5. Deployment procedure
1. Extract the complete `SAD_Predictive_System` folder.
2. Confirm `data/facility_reservations.csv` exists.
3. Open a terminal in the project folder.
4. Run `pip install -r requirements.txt`.
5. Run `streamlit run app.py`.
6. Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## 6. Validation and checks
The application verifies the presence of required columns and recognized target values before training. It shows missing-value and duplicate-row checks, model metrics, a confusion matrix, classification report, and aggregated predictor importance. Predictions use only candidate reservation-time fields; `Usage_Status`, `Actual_Attendance`, and `Cancellation_Days_Before` are excluded from model inputs to reduce leakage.

## 7. Limitations and responsible use
- Historical data may be incomplete, imbalanced, or unrepresentative.
- A held-out test score does not guarantee future performance.
- Prediction probabilities and labels are estimates, not confirmed outcomes.
- Do not automatically approve or reject reservations based on model output. Human review and existing reservation policies must remain in place.
- Before real deployment, confirm data permissions, privacy protections, model validity, accessibility, and monitoring procedures with campus stakeholders.

## 8. Conclusion
The project folder includes the application, requirements, README, supplied CSV, and deployment report. It is intended for local academic demonstration after dependencies are installed; production use requires further testing and governance.
