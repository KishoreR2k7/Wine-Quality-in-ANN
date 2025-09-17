import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model

model = load_model("wine_ann_model.h5",compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    feature_names = pickle.load(f)

quality_map = {0: "Low", 1: "Medium", 2: "High"}

def predict_wine_quality(*inputs):
    x = np.array(inputs).reshape(1, -1)
    
    x_scaled = scaler.transform(x)
    
    y_prob = model.predict(x_scaled)
    
    y_class = np.argmax(y_prob, axis=1)[0]
    
    return quality_map[y_class], {quality_map[i]: float(y_prob[0][i]) for i in range(3)}

inputs = [gr.Number(label=feature) for feature in feature_names]

outputs = [
    gr.Textbox(label="Predicted Wine Quality"),
    gr.Label(label="Class Probabilities")
]

demo = gr.Interface(
    fn=predict_wine_quality,
    inputs=inputs,
    outputs=outputs,
    title="Wine Quality Prediction ANN",
    description="Enter numeric values of wine features to predict wine quality."
)

if __name__ == "__main__":
    demo.launch(share=True)
