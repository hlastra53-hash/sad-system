 
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "facility_reservations.csv"
TARGET = "Reservation_Outcome"

PREDICTORS = [
    "User_Category",
    "Facility_Type",
    "Facility_ID",
    "Purpose",
    "Lead_Time_Days",
    "Day_Of_Week",
    "Participants",
    "Duration_Hours",
]

st.set_page_config(
    page_title="SAD Predictive System",
    page_icon="🏫",
    layout="wide",
)

st.title("Campus Facility Reservation and Monitoring System")
st.caption(
    "SAD Data Mining Project · Exploratory analysis, "
    "Decision Tree prediction, and decision support"
)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Keep facility_reservations.csv inside the data folder."
        )

    raw = pd.read_csv(DATA_PATH)

    required = [
        "Reservation_ID",
        "User_Category",
        "Facility_Type",
        "Facility_ID",
        "Purpose",
        "Submission_Date",
        "Reservation_Date",
        "Lead_Time_Days",
        "Day_Of_Week",
        "Participants",
        "Duration_Hours",
        "Usage_Status",
    ]

    missing = [column for column in required if column not in raw.columns]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )

    df = raw.copy()

    categorical_columns = [
        "User_Category",
        "Facility_Type",
        "Facility_ID",
        "Purpose",
        "Day_Of_Week",
        "Usage_Status",
    ]

    for column in categorical_columns:
        series = df[column].astype(object)
        series = series.map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )
        series = series.replace("", np.nan)
        df[column] = series.map(
            lambda value: value.title()
            if isinstance(value, str)
            else value
        )

    for column in ["Submission_Date", "Reservation_Date"]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    calculated_lead_time = (
        df["Reservation_Date"] - df["Submission_Date"]
    ).dt.days

    df["Lead_Time_Days"] = pd.to_numeric(
        df["Lead_Time_Days"],
        errors="coerce",
    )

    df["Lead_Time_Days"] = df["Lead_Time_Days"].fillna(
        calculated_lead_time
    )

    for column in ["Participants", "Duration_Hours"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["Lead_Time_Category"] = pd.cut(
        df["Lead_Time_Days"],
        bins=[
            -float("inf"),
            2,
            7,
            14,
            float("inf"),
        ],
        labels=[
            "Last Minute",
            "Short",
            "Medium",
            "Early",
        ],
    )

    status = df["Usage_Status"].map(
        lambda value: value.casefold()
        if isinstance(value, str)
        else np.nan
    )

    df[TARGET] = status.map(
        {
            "used": "Completed",
            "cancelled": "Not Completed",
            "no-show": "Not Completed",
            "noshow": "Not Completed",
        }
    )

    model_df = df[
        df[TARGET].isin(["Completed", "Not Completed"])
    ].copy()

    if model_df.empty:
        raise ValueError(
            "No recognized Usage_Status values found. "
            "Expected Used, Cancelled, or No-Show."
        )

    return raw, df, model_df


try:
    raw, df, model_df = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()


available_predictors = [
    column for column in PREDICTORS
    if column in model_df.columns
]

X = model_df[available_predictors].copy()
y = model_df[TARGET].astype(str)

if len(model_df) < 5 or y.nunique() < 2:
    st.error(
        "At least five recognized records and two outcome "
        "classes are needed to train the model."
    )
    st.stop()


numeric_cols = X.select_dtypes(
    include="number"
).columns.tolist()

cat_cols = [
    column for column in X.columns
    if column not in numeric_cols
]

# Fix missing categorical values before model training.
for column in cat_cols:
    X[column] = X[column].astype(object)
    X[column] = X[column].where(
        pd.notna(X[column]),
        np.nan,
    )


transformers = []

if numeric_cols:
    transformers.append(
        (
            "numeric",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    )
                ]
            ),
            numeric_cols,
        )
    )

if cat_cols:
    transformers.append(
        (
            "categorical",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                    ),
                ]
            ),
            cat_cols,
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=5,
                class_weight="balanced",
            ),
        ),
    ]
)

counts = y.value_counts()

stratify = (
    y
    if (
        y.nunique() >= 2
        and counts.min() >= 2
        and len(y) * 0.2 >= y.nunique()
    )
    else None
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify,
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)


# Dashboard metrics
c1, c2, c3, c4 = st.columns(4)

c1.metric("Dataset records", f"{len(raw):,}")
c2.metric("Recognized outcomes", f"{len(model_df):,}")
c3.metric(
    "Facility types",
    str(df["Facility_Type"].nunique()),
)
c4.metric("Test accuracy", f"{acc:.1%}")

st.info(
    "Outcome mapping: Used → Completed; "
    "Cancelled/No-Show → Not Completed. "
    "Predictions support review and must not automatically "
    "approve or reject reservations."
)


# Part III: Exploratory Data Analysis
st.header("Part III · Exploratory Data Analysis")

left, right = st.columns(2)

with left:
    st.subheader("Reservations by facility type")

    facility_counts = (
        df["Facility_Type"]
        .dropna()
        .value_counts()
        .sort_values(ascending=False)
    )

    st.dataframe(
        facility_counts.rename_axis(
            "Facility_Type"
        ).reset_index(
            name="Reservation_Count"
        ),
        hide_index=True,
        use_container_width=True,
    )

    fig, ax = plt.subplots()
    facility_counts.sort_values().plot(
        kind="barh",
        ax=ax,
    )
    ax.set_xlabel("Reservation count")
    ax.set_ylabel("Facility type")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with right:
    st.subheader(
        "Cancellation / no-show by user category"
    )

    normalized_status = df["Usage_Status"].map(
        lambda value: value.casefold()
        if isinstance(value, str)
        else np.nan
    )

    recognized = df[
        normalized_status.isin(
            ["used", "cancelled", "no-show", "noshow"]
        )
    ].copy()

    recognized["Not_Completed"] = (
        normalized_status.loc[recognized.index].isin(
            ["cancelled", "no-show", "noshow"]
        )
    )

    rates = recognized.groupby(
        "User_Category",
        dropna=False,
    ).agg(
        Eligible_Reservations=("Usage_Status", "size"),
        Not_Completed_Count=("Not_Completed", "sum"),
    )

    rates["Not_Completed_Percent"] = (
        rates["Not_Completed_Count"]
        / rates["Eligible_Reservations"]
        * 100
    ).round(2)

    st.dataframe(
        rates.reset_index(),
        hide_index=True,
        use_container_width=True,
    )

    fig, ax = plt.subplots()
    rates["Not_Completed_Percent"].sort_values().plot(
        kind="barh",
        ax=ax,
    )
    ax.set_xlabel("Cancellation/no-show (%)")
    ax.set_ylabel("User category")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.subheader(
    "Reservation outcome by lead-time category"
)

outcomes = df[
    df[TARGET].isin(["Completed", "Not Completed"])
].copy()

counts_by_lead = pd.crosstab(
    outcomes["Lead_Time_Category"],
    outcomes[TARGET],
).reindex(
    ["Last Minute", "Short", "Medium", "Early"]
)

left, right = st.columns(2)

with left:
    st.markdown("**Counts**")
    st.bar_chart(counts_by_lead)

with right:
    st.markdown("**Percentages within each category**")

    pct = (
        counts_by_lead
        .div(
            counts_by_lead.sum(axis=1).replace(0, np.nan),
            axis=0,
        )
        .fillna(0)
        * 100
    )

    st.bar_chart(pct)


# Part IV: Decision Tree Model
st.header("Part IV · Decision Tree Model")

metric_cols = st.columns(4)

metric_cols[0].metric("Accuracy", f"{acc:.3f}")

metric_cols[1].metric(
    "Weighted precision",
    f"{precision_score(y_test, pred, average='weighted', zero_division=0):.3f}",
)

metric_cols[2].metric(
    "Weighted recall",
    f"{recall_score(y_test, pred, average='weighted', zero_division=0):.3f}",
)

metric_cols[3].metric(
    "Weighted F1-score",
    f"{f1_score(y_test, pred, average='weighted', zero_division=0):.3f}",
)


left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")

    labels = list(
        model.named_steps["classifier"].classes_
    )

    cm = confusion_matrix(
        y_test,
        pred,
        labels=labels,
    )

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    ).plot(
        ax=ax,
        colorbar=False,
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with right:
    st.subheader("Classification report")

    report = classification_report(
        y_test,
        pred,
        output_dict=True,
        zero_division=0,
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True,
    )


prep = model.named_steps["preprocessor"]
tree = model.named_steps["classifier"]

encoded_names = prep.get_feature_names_out()

importance = pd.DataFrame(
    {
        "Encoded_Feature": encoded_names,
        "Importance": tree.feature_importances_,
    }
)


def map_feature(name):
    name = name.split("__", 1)[-1]

    for column in sorted(
        available_predictors,
        key=len,
        reverse=True,
    ):
        if name == column or name.startswith(column + "_"):
            return column

    return name


importance["Predictor"] = (
    importance["Encoded_Feature"].map(map_feature)
)

predictor_importance = (
    importance.groupby("Predictor")["Importance"]
    .sum()
    .sort_values(ascending=False)
)

st.subheader("Top three influential predictors")

top3 = (
    predictor_importance.head(3)
    .rename("Feature_Importance")
    .reset_index()
)

st.dataframe(
    top3,
    hide_index=True,
    use_container_width=True,
)

fig, ax = plt.subplots()

predictor_importance.head(3).sort_values().plot(
    kind="barh",
    ax=ax,
)

ax.set_xlabel("Aggregated feature importance")
ax.set_ylabel("Predictor")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


with st.expander("View Decision Tree diagram"):
    fig, ax = plt.subplots(figsize=(18, 8))

    plot_tree(
        tree,
        feature_names=encoded_names,
        class_names=[
            str(value) for value in tree.classes_
        ],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
        ax=ax,
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# Part V: Predict a reservation outcome
st.header("Part V · Predict a reservation outcome")

st.write(
    "Enter details available when submitting a reservation. "
    "Actual attendance and cancellation timing are excluded "
    "to reduce data leakage."
)

with st.form("prediction_form"):
    values = {}
    cols = st.columns(2)

    for i, column in enumerate(available_predictors):
        slot = cols[i % 2]

        if column in cat_cols:
            choices = sorted(
                model_df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ) or ["Unknown"]

            values[column] = slot.selectbox(
                column.replace("_", " "),
                choices,
            )

        else:
            series = pd.to_numeric(
                model_df[column],
                errors="coerce",
            ).dropna()

            default = (
                float(series.median())
                if not series.empty
                else 0.0
            )

            lo = (
                float(series.min())
                if not series.empty
                else 0.0
            )

            hi = (
                float(series.max())
                if not series.empty
                else max(1.0, default)
            )

            if lo == hi:
                hi = lo + 1

            values[column] = slot.number_input(
                column.replace("_", " "),
                min_value=lo,
                max_value=hi,
                value=min(max(default, lo), hi),
                step=(
                    1.0
                    if column in [
                        "Lead_Time_Days",
                        "Participants",
                    ]
                    else 0.5
                ),
            )

    submitted = st.form_submit_button(
        "Predict reservation outcome",
        type="primary",
    )


if submitted:
    sample = pd.DataFrame(
        [values],
        columns=available_predictors,
    )

    for column in numeric_cols:
        sample[column] = pd.to_numeric(
            sample[column],
            errors="coerce",
        )

    for column in cat_cols:
        sample[column] = sample[column].astype(object)
        sample[column] = sample[column].where(
            pd.notna(sample[column]),
            np.nan,
        )

    prediction = model.predict(sample)[0]

    st.success(
        f"Predicted outcome: **{prediction}**"
    )

    if hasattr(
        model.named_steps["classifier"],
        "predict_proba",
    ):
        probabilities = model.predict_proba(sample)[0]

        probability_df = pd.DataFrame(
            {
                "Outcome": (
                    model.named_steps["classifier"].classes_
                ),
                "Estimated probability": probabilities,
            }
        )

        probability_df["Estimated probability"] = (
            probability_df["Estimated probability"].map(
                lambda value: f"{value:.1%}"
            )
        )

        st.dataframe(
            probability_df,
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "This is an estimate based on historical patterns, "
        "not a confirmed result or a reason to automatically "
        "reject a reservation."
    )


with st.expander("Data quality and leakage checks"):
    st.write("Original dataset shape:", raw.shape)

    st.write(
        "Exact duplicate rows:",
        int(raw.duplicated().sum()),
    )

    st.dataframe(
        raw.isna().sum()
        .rename("Missing values")
        .to_frame()
    )

    forbidden = [
        "Usage_Status",
        "Actual_Attendance",
        "Cancellation_Days_Before",
        "Reservation_Outcome",
    ]

    st.write(
        "Predictors used:",
        available_predictors,
    )

    st.write(
        "Forbidden outcome/post-event fields in predictors:",
        [
            column
            for column in available_predictors
            if column in forbidden
        ],
    )

    st.caption(
        "Model uses an 80/20 train-test split with "
        "random_state=42 and a Decision Tree (max_depth=5). "
        "Results are educational estimates; validate with "
        "additional data before real use."
    )


st.divider()

st.caption(
    "SAD Predictive System · Academic prototype · "
    "Human review is required for decisions."
)
