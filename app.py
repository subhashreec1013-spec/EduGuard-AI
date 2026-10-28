
import os
import joblib
import pandas as pd
import gradio as gr

MODEL_PATH = "eduguard_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "eduguard_model.joblib is missing from the repository."
    )

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["feature_columns"]

if len(FEATURES) != 32:
    raise ValueError(f"Expected 32 features, got {len(FEATURES)}")

NUMERIC_FEATURES = {
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout", "Dalc",
    "Walc", "health", "absences", "G1", "G2"
}

def assess(*values):
    try:
        record = dict(zip(FEATURES, values))
        student = pd.DataFrame([record], columns=FEATURES)

        for col in NUMERIC_FEATURES.intersection(FEATURES):
            student[col] = pd.to_numeric(student[col], errors="raise")

        if student.isnull().any().any():
            return "Please complete every field."

        prediction = int(model.predict(student)[0])
        probability = float(model.predict_proba(student)[0][1])

        label = "AT RISK" if prediction == 1 else "NOT AT RISK"
        return (
            f"## {label}\n\n"
            f"**Estimated risk probability:** {probability:.1%}\n\n"
            "This experimental result is a support indicator, not a "
            "definitive judgment. Please review the student's circumstances."
        )
    except Exception as exc:
        return f"Unable to assess these inputs: {exc}"

with gr.Blocks(title="EduGuard AI", theme=gr.themes.Soft()) as app:
    gr.Markdown("# EduGuard AI\nStudent Academic Risk Assessment")
    gr.Markdown(
        "Enter student information to estimate the likelihood of a final "
        "mathematics grade below 10/20."
    )

    inputs = []
    for feature in FEATURES:
        if feature in NUMERIC_FEATURES:
            field = gr.Number(label=feature)
        else:
            field = gr.Textbox(label=feature)
        inputs.append(field)

    submit = gr.Button("Generate assessment", variant="primary")
    output = gr.Markdown()

    submit.click(assess, inputs=inputs, outputs=output)

    gr.Markdown(
        "**Limitations:** This model was trained on historical data from "
        "two Portuguese schools. It may not generalize to other populations."
    )

app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
