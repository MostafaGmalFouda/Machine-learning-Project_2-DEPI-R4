# Machine-learning-Project_2-DEPI-R4 
---

# 🚀 Finding Donors for CharityML
## End-to-End Machine Learning Pipeline & Interactive Dashboard**

---

###  📋 Project Overview
This project is part of the **Machine Learning Engineer Nanodegree**. The objective is to help the non-profit organization **CharityML** identify potential donors. By analyzing the 1994 U.S. Census data, we predict whether an individual's income exceeds **$50,000** per year, enabling more efficient and targeted fundraising campaigns.

---

###  🖥️ Interactive Dash Application
I have built a comprehensive **Interactive Dashboard** using **Plotly Dash** and **Bootstrap** to make the model results accessible and actionable for non-technical users.

**Dashboard Sections:**

- 📊 Data Visualization:
  - **-** Interactive charts (Pie charts, Bar plots, Histograms) showing income distribution across demographics like **Education, Age, and Occupation**.
  - **-** Real-time statistics of the census dataset.

- 🤖 Income Prediction Engine:
  - **-** **"Predict Now" Feature:** Users input personal profile data (Age, Workclass, Education, etc.).
  - **-** **Live Inference:** The app scales the input and runs it through trained models to provide an instant prediction with a **Confidence Score**.

- 📈 Model Comparison Dashboard:
  - **-** Dynamic comparison of metrics: **Accuracy, Precision, Recall, and F1-Score**.
  - **-** Visual benchmarking to identify and deploy the **"Best Model"**.

---

### 🧪 Project Workflow 

#### 1. Data Exploration & Preprocessing
- **-** **Log-Transformation:** Applied to highly skewed features like `capital-gain` and `capital-loss`.
- **-** **Scaling:** Used `MinMaxScaler` to normalize numerical features for equal treatment by algorithms.
- **-** **Encoding:** Performed **One-Hot Encoding** for categorical variables and binary encoding for the target label.

####  2. Machine Learning Models
I evaluated four supervised learning algorithms to find the most efficient predictor:
- **-** **Logistic Regression** (Baseline)
- **-** **Random Forest** (Ensemble Method)
- **-** **Gradient Boosting**
- **-** **XGBoost** (Optimized Model)

#### 3. Model Evaluation & Tuning
- **-** **Metric:** Used **F-beta score ($\beta = 0.5$)**, prioritizing **Precision** over Recall to minimize wasted resources on low-income individuals.
- **-** **Optimization:** Used `GridSearchCV` and `RandomizedSearchCV` to fine-tune **XGBoost** hyperparameters.

---

### 📊 Key Results
- **-- Top 5 Features:** The most critical factors were **marital-status**, **education-num**, and **capital-gain**.
- **-- Final Model:** The optimized **XGBoost** model achieved:
  - **-** **Accuracy:** 87.15%
  - **-** **F-score:** 0.7538
  - **-** *Significantly outperforming the naive predictor.*

---

###  🛠️ Tech Stack
- **-** **Language:** Python 3.x
- **-** **ML Libraries:** Scikit-learn, XGBoost
- **-** **Web Framework:** Plotly Dash, Dash Bootstrap Components
- **-** **Data Analysis:** Pandas, NumPy
- **-** **Visualization:** Plotly, Matplotlib, Seaborn

---

### 📂 Project Structure
```plaintext
├── data/                # Raw and processed census data
├── models/              # Saved .pkl models, scaler, and feature lists
├── utils/               # Helper scripts for model loading and saving
├── app.py               # Main Dash application code
├── visuals.py           # Custom visualization functions
├── Finding_Donors.ipynb  # Analysis and training notebook
└── README.md            # Project documentation
```

---

###  🏃 How to Run
1. **-** Clone the repository.
2. **-** Install dependencies: `pip install -r requirements.txt`.
3. **-** Run the dashboard: `python app.py`.
4. **-** Access the app at: `http://127.0.0.1:8050/`.

