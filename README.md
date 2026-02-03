# 📱 SMS Spam Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
The **Spam_Sms_Detection** project is a Machine Learning-based classifier designed to filter out unsolicited and unwanted messages (Spam) from legitimate ones (Ham). Using Natural Language Processing (NLP) techniques, this model analyzes the textual patterns of messages to predict their category with high precision.



---

## 🚀 Features
* **Data Preprocessing:** Tokenization, stop-word removal, and stemming/lemmatization using NLTK.
* **Vectorization:** Implementation of TF-IDF or CountVectorizer to convert text into numerical data.
* **Model Variety:** Comparison between multiple algorithms (e.g., Naive Bayes, Logistic Regression, and SVM).
* **Performance Metrics:** Detailed evaluation using Accuracy, Precision, Recall, and F1-Score.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** * `Pandas` & `NumPy` (Data Manipulation)
    * `NLTK` (Natural Language Processing)
    * `Scikit-Learn` (Machine Learning)
    * `Matplotlib` & `Seaborn` (Visualization)

---

## 📂 Project Structure
```text
├── data/               # Dataset files (e.g., spam.csv)
├── notebooks/          # Jupyter notebooks for EDA and Model Training
├── models/             # Saved model files (.pkl or .h5)
├── src/                # Source code for preprocessing and prediction
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
```

---
## Installation & usage
    1. Clone the repository:

git clone [https://github.com/shu7620/Spam_Sms_Detection.git](https://github.com/shu7620/Spam_Sms_Detection.git)
cd Spam_Sms_Detection
    
    2. Install Dependencies:

 pip install -r requirements.txt
    
    3. Run the Project:

       1. If using a Jupyter notebook:

          jupyter notebook notebooks/spam_classifier.ipynb

       2. If using a script:

          python src/main.py
