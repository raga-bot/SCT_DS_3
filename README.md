# Task 3 – Decision Tree Classifier: Bank Marketing Dataset

## 🎯 Objective
Build a machine learning model to predict whether a customer will subscribe to a term deposit based on their demographic and behavioral data.

## 📁 Dataset
- **Source:** UCI Machine Learning Repository  
- **Link:** [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  
- **Size:** Approximately 45,000 records with features including age, job, marital status, contact type, campaign info, and previous outcomes.

## 🔍 Features Used
- Age, Job, Marital Status, Education
- Contact type, Month, Duration, Campaign, Previous outcome
- And other relevant demographic and behavioral attributes

## 🧠 Model Details
- **Algorithm:** Decision Tree Classifier
- **Library:** scikit-learn
- **Preprocessing:** One-hot encoding for categorical variables, train-test split (80/20)
- **Hyperparameters:** max_depth=5, random_state=42

## 📊 Performance Metrics

| Metric       | Class 0 (No) | Class 1 (Yes) |
|--------------|--------------|---------------|
| Precision    | 91%          | 65%           |
| Recall       | 97%          | 33%           |
| F1-Score     | 94%          | 44%           |
| Accuracy     | **89.7%** overall accuracy           |

### Confusion Matrix

|               | Predicted No | Predicted Yes |
|---------------|--------------|---------------|
| Actual No     | 7753         | 199           |
| Actual Yes    | 729          | 362           |

## 🔎 Analysis
- The model performs well in predicting customers who **do not subscribe** (Class 0), with high precision and recall.
- Performance on predicting customers who **subscribe** (Class 1) is lower, especially recall (33%), meaning many buyers are missed.
- This is typical in **imbalanced datasets** where the "Yes" class is much smaller.

## 🛠️ Future Work
- Use techniques like **class weighting** or **SMOTE oversampling** to address imbalance.
- Experiment with other algorithms like **Random Forests** or **Gradient Boosting** for improved accuracy.
- Tune hyperparameters using **GridSearchCV**.

## 📈 Visualization
Includes decision tree visualization to understand feature splits influencing predictions.

## 🛠️ Tools Used
- Python, pandas, scikit-learn, matplotlib, seaborn

## 📂 Files in this repo
- `Bank Marketing Decision Tree.ipynb` — Jupyter Notebook with full code  
- `bank-full.csv` — Dataset (download separately due to size)

---

## 👩‍💻 Author
RagaVarshini BS

---

Feel free to explore the notebook and reach out for any questions or collaboration opportunities!
