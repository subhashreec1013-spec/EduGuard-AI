import os
import tempfile
from datetime import datetime

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

APP_TITLE = "EduGuard AI"

# Exact 32-feature order used when training on student-mat.csv (G3 excluded).
FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2"
]

CATEGORIES = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
    "romantic": ["yes", "no"],
}

NUMERIC_RANGES = {
    "age": (15, 22), "Medu": (0, 4), "Fedu": (0, 4),
    "traveltime": (1, 4), "studytime": (1, 4), "failures": (0, 4),
    "famrel": (1, 5), "freetime": (1, 5), "goout": (1, 5),
    "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
    "absences": (0, 100), "G1": (0, 20), "G2": (0, 20)
}

# Published project evaluation values from the held-out test set.
TEST_METRICS = pd.DataFrame([
    ["Logistic Regression", 0.886, 0.815, 0.846, 0.830],
    ["Decision Tree",       0.873, 0.786, 0.846, 0.815],
], columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])

CV_METRICS = pd.DataFrame([
    ["Logistic Regression", 0.912, 0.037, 0.823, 0.933, 0.874, 0.055],
    ["Decision Tree",       0.905, 0.020, 0.850, 0.876, 0.857, 0.032],
    ["Rule-Based Baseline", 0.924, 0.026, 0.843, 0.952, 0.892, 0.037],
], columns=["Model", "Accuracy mean", "Accuracy SD", "Precision mean",
            "Recall mean", "F1 mean", "F1 SD"])

CONFUSION = {
    "Logistic Regression": pd.DataFrame([[48, 5], [4, 22]],
        index=["Actual not at risk", "Actual at risk"],
        columns=["Predicted not at risk", "Predicted at risk"]),
    "Decision Tree": pd.DataFrame([[47, 6], [4, 22]],
        index=["Actual not at risk", "Actual at risk"],
        columns=["Predicted not at risk", "Predicted at risk"]),
}

def load_model():
    candidates = [
        os.getenv("MODEL_PATH", ""),
        "eduguard_model.joblib",
        "eduguard_model (1).joblib",
        "eduguard_model.pkl",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            loaded = joblib.load(path)
            expected = list(getattr(loaded, "feature_names_in_", FEATURES))
            if expected != FEATURES:
                # Fail loudly rather than silently feeding columns in the wrong order.
                raise ValueError(
                    "The saved model feature order does not match this app. "
                    f"Expected app schema: {FEATURES}; model schema: {expected}"
                )
            return loaded, path
    raise FileNotFoundError(
        "Model file not found. Add eduguard_model.joblib to the repository root "
        "or set the MODEL_PATH environment variable."
    )

model, MODEL_FILE = load_model()

def make_input_components():
    components = []
    for feature in FEATURES:
        if feature in CATEGORIES:
            components.append(gr.Dropdown(
                choices=CATEGORIES[feature],
                value=CATEGORIES[feature][0],
                label=feature,
                interactive=True
            ))
        else:
            low, high = NUMERIC_RANGES[feature]
            default = {
                "age": 17, "Medu": 2, "Fedu": 2, "traveltime": 1,
                "studytime": 2, "failures": 0, "famrel": 4,
                "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2,
                "health": 3, "absences": 4, "G1": 10, "G2": 10
            }[feature]
            components.append(gr.Number(
                value=default, label=feature, precision=0,
                minimum=low, maximum=high, interactive=True
            ))
    return components

def validate_values(values):
    if len(values) != len(FEATURES):
        raise gr.Error("The assessment form is incomplete. Refresh and try again.")
    record = {}
    for feature, value in zip(FEATURES, values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise gr.Error(f"Please enter a value for {feature}.")
        if feature in NUMERIC_RANGES:
            low, high = NUMERIC_RANGES[feature]
            number = float(value)
            if not number.is_integer():
                raise gr.Error(f"{feature} must be a whole number.")
            if not low <= number <= high:
                raise gr.Error(f"{feature} must be between {low} and {high}.")
            record[feature] = int(number)
        else:
            if str(value) not in CATEGORIES[feature]:
                raise gr.Error(f"Invalid selection for {feature}.")
            record[feature] = str(value)
    return record

def supportive_guidance(record, risk):
    tips = []
    if record["G2"] < 10:
        tips.append("Review recent subject performance and arrange targeted academic support.")
    if record["G1"] < 10:
        tips.append("Discuss earlier assessment difficulties and agree on a manageable study plan.")
    if record["absences"] >= 10:
        tips.append("Review attendance barriers and consider a supportive attendance follow-up.")
    if record["failures"] > 0:
        tips.append("Review previously unsuccessful subjects and identify specific learning gaps.")
    if record["studytime"] <= 1:
        tips.append("Discuss a realistic weekly study routine with the student.")
    if not tips:
        tips.append("Continue regular academic check-ins and monitor progress in upcoming assessments.")
    if risk == 1:
        tips.insert(0, "Arrange a teacher-led review before deciding on any intervention.")
    return tips

def assess(*values):
    record = validate_values(values)
    row = pd.DataFrame([record], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    label = "AT RISK" if prediction == 1 else "NOT AT RISK"
    tips = supportive_guidance(record, prediction)

    # Keep session history only; no student record is written to disk.
    history_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Prediction": label,
        "Estimated risk probability": round(probability, 4),
        **record
    }
    summary = f"""
### {label}
**Estimated risk probability:** {probability:.1%}

**Academic review suggestions**
""" + "\n".join(f"- {tip}" for tip in tips) + """

*This is a decision-support estimate based on the supplied dataset. A teacher must review the student's circumstances; it is not a diagnosis or a final academic decision.*
"""
    insights = (
        f"**Grade indicators:** G1 = {record['G1']}/20 · G2 = {record['G2']}/20  \n"
        f"**Attendance indicator:** {record['absences']} recorded absences  \n"
        f"**Previous failures:** {record['failures']}  \n\n"
        + "\n".join(f"- {tip}" for tip in tips)
    )
    return summary, insights, history_row

def add_history(history, *values):
    result, insights, row = assess(*values)
    history = (history or []) + [row]
    return result, insights, history, pd.DataFrame(history)

def reset_form():
    vals = []
    for feature in FEATURES:
        if feature in CATEGORIES:
            vals.append(CATEGORIES[feature][0])
        else:
            defaults = {
                "age":17,"Medu":2,"Fedu":2,"traveltime":1,"studytime":2,
                "failures":0,"famrel":4,"freetime":3,"goout":3,"Dalc":1,
                "Walc":2,"health":3,"absences":4,"G1":10,"G2":10
            }
            vals.append(defaults[feature])
    return vals

def make_distribution():
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(["Not at risk", "At risk"], [265, 130])
    ax.set_ylabel("Student records")
    ax.set_title("Academic-risk distribution")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def make_confusion_plot(model_name):
    matrix = CONFUSION[model_name].values
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.imshow(matrix)
    ax.set_xticks([0, 1], ["Not at risk", "At risk"])
    ax.set_yticks([0, 1], ["Not at risk", "At risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — confusion matrix")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=13)
    fig.tight_layout()
    return fig

def export_history(history):
    if not history:
        raise gr.Error("No assessments are available to export in this session.")
    path = os.path.join(tempfile.gettempdir(), "eduguard_assessment_history.csv")
    pd.DataFrame(history).to_csv(path, index=False)
    return path

CSS = """
:root { --brand:#3157d5; }
.gradio-container { max-width: 1440px !important; margin: auto !important; }
#header { padding: 22px 26px; border-radius: 18px; background: linear-gradient(115deg,#101d3d,#243f83); color: white; margin-bottom: 18px; }
#header h1 { margin:0; font-size:30px; font-weight:750; color:white; }
#header p { margin:6px 0 0; color:#dbe5ff; }
.kpi { background:white; border:1px solid #e6eaf2; border-radius:15px; padding:16px; min-height:90px; }
.kpi-label { color:#697586; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
.kpi-value { color:#15294b; font-size:27px; font-weight:750; margin-top:7px; }
"""

with gr.Blocks(
    title=APP_TITLE,
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=CSS
) as demo:
    history_state = gr.State([])

    gr.HTML("""
    <div id="header">
      <h1>EduGuard AI</h1>
      <p>Student Academic Risk Assessment · Teacher decision support</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("Dashboard"):
            gr.Markdown("## Overview")
            with gr.Row():
                gr.HTML('<div class="kpi"><div class="kpi-label">Dataset records</div><div class="kpi-value">395</div></div>')
                gr.HTML('<div class="kpi"><div class="kpi-label">Logistic Regression accuracy</div><div class="kpi-value">88.6%</div></div>')
                gr.HTML('<div class="kpi"><div class="kpi-label">Logistic Regression F1</div><div class="kpi-value">83.0%</div></div>')
            with gr.Row():
                gr.Plot(value=make_distribution(), label="Risk distribution")
                gr.Dataframe(value=TEST_METRICS, label="Held-out test-set performance", interactive=False)
            gr.Markdown("**Recent assessments — current session only**")
            history_table = gr.Dataframe(headers=["Timestamp", "Prediction", "Estimated risk probability"], interactive=False)

        with gr.Tab("Student Assessment"):
            gr.Markdown("## Student assessment")
            gr.Markdown("Complete the 32 input fields. G3 is excluded because it defines the target label.")
            inputs = []
            for start in range(0, len(FEATURES), 4):
                with gr.Row():
                    for feature in FEATURES[start:start+4]:
                        idx = FEATURES.index(feature)
                        if feature in CATEGORIES:
                            comp = gr.Dropdown(CATEGORIES[feature], value=CATEGORIES[feature][0], label=feature)
                        else:
                            lo, hi = NUMERIC_RANGES[feature]
                            defaults = {"age":17,"Medu":2,"Fedu":2,"traveltime":1,"studytime":2,"failures":0,"famrel":4,"freetime":3,"goout":3,"Dalc":1,"Walc":2,"health":3,"absences":4,"G1":10,"G2":10}
                            comp = gr.Number(value=defaults[feature], label=feature, precision=0, minimum=lo, maximum=hi)
                        inputs.append(comp)
            with gr.Row():
                submit = gr.Button("Generate assessment", variant="primary")
                reset = gr.Button("Reset form")
            result = gr.Markdown()
            insight_result = gr.Markdown()
            submit.click(add_history, inputs=[history_state] + inputs,
                         outputs=[result, insight_result, history_state, history_table])
            reset.click(lambda: (None, None, [], pd.DataFrame()), outputs=[result, insight_result, history_state, history_table])
            reset.click(reset_form, outputs=inputs)

        with gr.Tab("Academic Insights"):
            gr.Markdown("## Academic indicators")
            gr.Markdown("""
            - **Grades:** G1 and G2 are prior-period academic indicators.
            - **Attendance:** Absences can be reviewed alongside academic performance.
            - **Learning history:** Previous failures and study-time indicators may help guide a teacher discussion.

            Suggestions are generated from entered indicators. They are supportive prompts, not causal explanations of the model prediction.
            """)

        with gr.Tab("Model Evaluation"):
            gr.Markdown("## Model performance")
            gr.Dataframe(value=TEST_METRICS, interactive=False, label="Held-out test-set metrics")
            gr.Markdown("### Five-fold cross-validation")
            gr.Dataframe(value=CV_METRICS, interactive=False)
            with gr.Row():
                cm_model = gr.Dropdown(["Logistic Regression", "Decision Tree"], value="Logistic Regression", label="Select model")
                cm_plot = gr.Plot(value=make_confusion_plot("Logistic Regression"))
            cm_model.change(make_confusion_plot, inputs=cm_model, outputs=cm_plot)
            gr.Markdown("""
            **Methodology:** UCI Student Performance dataset; 395 records; 32 input features; target is `G3 < 10`; stratified 80/20 train-test split; preprocessing uses imputation, scaling and one-hot encoding; models are Logistic Regression and Decision Tree.

            **Limitations:** The dataset comes from two Portuguese schools. Results may not generalise to other institutions. Since G2 is an input, this is an assessment using second-period grade information, not a prediction made before G2 is available. Predictions are intended for human review only.
            """)

        with gr.Tab("Reports"):
            gr.Markdown("## Assessment reports")
            gr.Markdown("Export assessment records created during this running session.")
            export_btn = gr.Button("Download assessment history", variant="primary")
            export_file = gr.File(label="CSV report")
            export_btn.click(export_history, inputs=history_state, outputs=export_file)

    gr.Markdown(
        "EduGuard AI · Experimental academic decision-support tool. "
        "Use predictions to prompt supportive review, never as the sole basis for decisions."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
