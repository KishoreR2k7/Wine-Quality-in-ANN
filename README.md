# 🍷 Wine Quality Prediction using Artificial Neural Networks (ANN)

## 📌 Overview
This project predicts the quality of wine (red/white) using an Artificial Neural Network (ANN).  
The dataset contains physicochemical properties of wine, and the ANN is trained to classify wine quality into categories.

---

## 📊 Dataset
- **Source**: UCI Machine Learning Repository – Wine Quality Dataset  
- **Features**:  
  - Fixed acidity  
  - Volatile acidity  
  - Citric acid  
  - Residual sugar  
  - Chlorides  
  - Free sulfur dioxide  
  - Total sulfur dioxide  
  - Density  
  - pH  
  - Sulphates  
  - Alcohol  
- **Target**: Wine Quality (integer values 3–8)

---

## 🛠️ Preprocessing
- Handled missing values using **SimpleImputer**  
- Scaled features using **StandardScaler**  
- Converted quality into categorical classes:  
  - **0 → Poor (3–4)**  
  - **1 → Average (5–6)**  
  - **2 → Good (7–8)**  
- Balanced dataset using **SMOTE**  

---

## 🧠 Model Architecture (ANN)
- Input layer: Number of neurons = features (11)  
- Hidden Layers:  
  - Dense(128, activation='relu')  
  - Dropout(0.3)  
  - Dense(64, activation='relu')  
- Output Layer:  
  - Dense(3, activation='softmax')  
- Optimizer: **Adam**  
- Loss: **categorical_crossentropy**  
- Metrics: **accuracy**

---

## 📈 Results
- Training accuracy: ~**XX%**  
- Test accuracy: ~**XX%**  
- Confusion matrix & classification report included in notebooks.  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/KishoreR2k7/Wine-Quality-in-ANN/edit/main/README.md
   cd Wine-Quality-in-ANN
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run Jupyter Notebook or Python script:

bash
Copy code
jupyter notebook
or

bash
Copy code
python train_model.py
📂 Project Structure
graphql
Copy code
Wine-Quality-in-ANN/
│── data/                 # Dataset files
│── notebooks/            # Jupyter notebooks
│── models/               # Saved ANN models (.h5, .pkl)
│── train_model.py        # Training script
│── app.py                # Gradio/Streamlit app
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
🎯 Future Work
Hyperparameter tuning

Compare ANN with Random Forest, XGBoost, etc.

Deployment with Flask/Streamlit

🙌 Acknowledgments
UCI Machine Learning Repository for dataset

TensorFlow & Scikit-learn for model building

Gradio/Streamlit for deployment
