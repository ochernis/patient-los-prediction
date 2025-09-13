# Patient Length of Stay Prediction

This repository contains a neural network model built with **TensorFlow/Keras** to predict patient **Length of Stay (LOS)** in hospitals.  
The project was completed as part of the *Fundamentals of Neural Networks* lab for HealthTech Innovations.  

## 📂 Contents
- `nn_fundamentals_lab.ipynb` — Jupyter Notebook with full workflow (EDA → preprocessing → model training → evaluation).  
- `patient_los.csv` — Dataset with demographic, admission, and clinical variables.  

## 🔍 Project Overview
Accurate prediction of patient LOS helps hospitals improve:
- **Bed management**  
- **Staff scheduling**  
- **Financial planning**

We trained a feed-forward neural network on patient demographics, admission details, and clinical indicators. Three optimizers (**SGD, RMSprop, Adam**) were compared to evaluate learning efficiency and predictive accuracy.  

## ⚙️ Methods
1. **Data exploration (EDA):** distribution of LOS, correlation analysis, visualization by admission type, department, and age.  
2. **Preprocessing:** standard scaling of numeric features, one-hot encoding for categorical features.  
3. **Modeling:** neural network (64 → 32 → 1 architecture, ReLU activations).  
4. **Optimization:** trained with SGD, RMSprop, and Adam; evaluated with MSE and MAE.  
5. **Evaluation:** selected the best model and tested on hold-out data; analyzed error distribution and subgroup performance.  

## 📊 Results
- Best model: **RMSprop optimizer**  
- **MAE ≈ 1.7 days**, **RMSE ≈ 2.4 days** on the test set  
- ~**70% of predictions within ±2 days** of actual LOS  
- Errors consistent across age groups (no systemic bias)  
- Largest errors occur for long and complex stays (multiple procedures)  

## 💡 Insights for Stakeholders
- Predictions are reliable for short-to-medium stays.  
- For long/complex hospitalizations, add **+2–3 days buffer** or manual review.  
- Next steps: enrich dataset with clinical codes, ICU flags, prior admissions; explore ensembles and quantile regression for confidence intervals.  

## 🚀 How to Run
1. Clone this repo:  
   ```bash
   git clone https://github.com/<your-username>/patient-los-prediction.git

2.	Install dependencies:
   pip install -r requirements.txt

3.	Open the notebook:
   jupyter notebook nn_fundamentals_lab.ipynb
