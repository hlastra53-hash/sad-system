    "Weighted precision", f"{precision_score(y_test, pred, average='weighted', zero_division=0):.3f}"
)
metric_cols[2].metric(
    "Weighted recall", f"{recall_score(y_test, pred, average='weighted', zero_division=0):.3f}"
)
metric_cols[3].metric(
    "Weighted F1-score", f"{f1_score(y_test, pred, average='weighted', zero_division=0):.3f}"
)

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    labels = list(model.named_steps["classifier"].classes_)
    cm = confusion_matrix(y_test, pred, labels=labels)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax, colorbar=False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

prep = model.named_steps["preprocessor"]
tree = model.named_steps["classifier"]
encoded_names = prep.get_feature_names_out()
importance = pd.DataFrame({
    "Encoded_Feature": encoded_names,
    "Importance": tree.feature_importances_,
})


def map_feature(name):
    name = name.split("__", 1)[-1]
    for column in sorted(available_predictors, key=len, reverse=True):
        if name == column or name.startswith(column + "_"):
            return column
    return name


importance["Predictor"] = importance["Encoded_Feature"].map(map_feature)
predictor_importance = importance.groupby("Predictor")["Importance"].sum().sort_values(ascending=False)
st.subheader("Top three influential predictors")
top3 = predictor_importance.head(3).rename("Feature_Importance").reset_index()
st.dataframe(top3, hide_index=True, use_container_width=True)
fig, ax = plt.subplots()
predictor_importance.head(3).sort_values().plot(kind="barh", ax=ax)
ax.set_xlabel("Aggregated feature importance")
ax.set_ylabel("Predictor")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("View Decision Tree diagram"):
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(
        tree, feature_names=encoded_names,
        class_names=[str(value) for value in tree.classes_],
        filled=True, rounded=True, max_depth=3, fontsize=7, ax=ax,
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.header("Part V · Predict a reservation outcome")
st.write(
    "Enter details available when submitting a reservation. Actual attendance and cancellation timing "
    "are excluded to reduce data leakage."
)
with st.form("prediction_form"):
    values = {}
    cols = st.columns(2)
    for i, column in enumerate(available_predictors):
        slot = cols[i % 2]
        if column in cat_cols:
            choices = sorted(model_df[column].dropna().astype(str).unique().tolist()) or ["Unknown"]
            values[column] = slot.selectbox(column.replace("_", " "), choices)
        else:
            series = pd.to_numeric(model_df[column], errors="coerce").dropna()
            default = float(series.median()) if not series.empty else 0.0
            lo = float(series.min()) if not series.empty else 0.0
            hi = float(series.max()) if not series.empty else max(1.0, default)
            if lo == hi:
                hi = lo + 1
            values[column] = slot.number_input(
                column.replace("_", " "), min_value=lo, max_value=hi,
                value=min(max(default, lo), hi),
                step=1.0 if column in ["Lead_Time_Days", "Participants"] else 0.5,
            )
    submitted = st.form_submit_button("Predict reservation outcome", type="primary")

if submitted:
    sample = pd.DataFrame([values], columns=available_predictors)
    for column in numeric_cols:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    for column in cat_cols:
        sample[column] = sample[column].astype(object).where(pd.notna(sample[column]), np.nan)
    prediction = model.predict(sample)[0]
    st.success(f"Predicted outcome: **{prediction}**")
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        probability_df = pd.DataFrame({
            "Outcome": model.named_steps["classifier"].classes_,
            "Estimated probability": probabilities,
        })
        probability_df["Estimated probability"] = probability_df["Estimated probability"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(probability_df, hide_index=True, use_container_width=True)
    st.caption(
        "This is an estimate based on historical patterns, not a confirmed result or a reason "
        "to automatically reject a reservation."
    )

with st.expander("Data quality and leakage checks"):
    st.write("Original dataset shape:", raw.shape)
    st.write("Exact duplicate rows:", int(raw.duplicated().sum()))
    st.dataframe(raw.isna().sum().rename("Missing values").to_frame())
    forbidden = ["Usage_Status", "Actual_Attendance", "Cancellation_Days_Before", "Reservation_Outcome"]
    st.write("Predictors used:", available_predictors)
    st.write("Forbidden outcome/post-event fields in predictors:", [
        column for column in available_predictors if column in forbidden
    ])
    st.caption(
        "Model uses an 80/20 train-test split with random_state=42 and a Decision Tree (max_depth=5). "
        "Results are educational estimates; validate with additional data before real use."
    )
