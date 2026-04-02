# 🛡 SpamGuard — End-to-End Machine Learning Spam Detection System

SpamGuard is a full-stack Machine Learning web application that classifies messages as **Spam 🚫** or **Safe 📥** using Natural Language Processing (NLP) techniques and a **Naive Bayes classifier**.

This project demonstrates a complete ML pipeline — from **data preprocessing → model training → prediction → web deployment → storage**.

---

# 🚀 Features

* 🔍 Real-time message classification
* 📥 Inbox (Safe Messages)
* 🚫 Spam Folder (Detected Spam)
* 🧠 NLP-based text cleaning using NLTK
* ⚡ FastAPI backend (high performance)
* 💾 MongoDB integration for storage
* 📊 Model training pipeline
* 🌐 Clean UI with HTML, CSS, JavaScript

---

# 🧠 Machine Learning Workflow

## 1. Text Preprocessing (NLP using NLTK)

Raw text is cleaned and transformed before feeding into the model.

### Steps:

* Convert text to lowercase
* Remove special characters & punctuation
* Tokenization (splitting into words)
* Remove stopwords (e.g., "is", "the", "and")
* Stemming using PorterStemmer

### Why NLTK?

* Simple and powerful NLP library
* Best for preprocessing tasks
* Lightweight and beginner-friendly

---

## 2. Feature Extraction

Text is converted into numerical vectors using:

* Bag of Words (BoW)
* TF-IDF (Term Frequency - Inverse Document Frequency)

---

## 3. Model: Naive Bayes (Multinomial)

### Why Naive Bayes is Best for Spam Detection?

* Works extremely well for text classification
* Handles high-dimensional sparse data
* Fast and efficient
* Performs well even with smaller datasets

### Core Idea:

Uses probability to classify messages:

* P(Spam | Message)
* P(Safe | Message)

---

## 📊 Dataset

* SMS Spam Collection Dataset (commonly from Kaggle)
* Two labels:

  * **ham** → Safe message
  * **spam** → Unwanted message

### Example:

| Message                                   | Label |
| ----------------------------------------- | ----- |
| "Congratulations! You won a free iPhone!" | Spam  |
| "Hey, let's meet tomorrow"                | Safe  |

---

# 🏗 Project Structure

```
spamguard/
│
├── app.py                     # FastAPI application
├── requirements.txt
├── setup.py
│
├── src/
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │
│   ├── utils/
│   │   └── helpers.py
│
├── templates/
│   ├── index.html
│   ├── prediction.html
│   ├── inbox.html
│   └── spam.html
│
├── static/
│   └── css/
│       └── style.css
│
├── artifacts/
│   ├── model.pkl
│   ├── vectorizer.pkl
│
└── README.md
```

---

# 🛠 Tech Stack

### Backend:

* FastAPI
* Uvicorn

### Machine Learning:

* scikit-learn
* NLTK
* NumPy
* Pandas

### Database:

* MongoDB (using pymongo)

### Frontend:

* HTML
* CSS
* JavaScript

---

# 💾 MongoDB Integration

MongoDB is used to store:

* Classified messages
* Spam and safe categories

### Why MongoDB?

* Flexible NoSQL database
* Easy integration with Python
* Scalable for production systems

---

# ⚙️ Installation Guide

## 1. Clone Repository

```
git clone https://github.com/your-username/spamguard.git
cd spamguard
```

---

## 2. Create Virtual Environment

```
python -m venv venv
```

### Activate:

**Windows**

```
venv\Scripts\activate
```

**Linux/Mac**

```
source venv/bin/activate
```

---

## 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## 4. Install Project (Editable Mode)

```
pip install -e .
```

---

# 📦 Requirements

Key libraries used:

* fastapi>=0.100.0
* uvicorn>=0.23.0
* nltk
* numpy
* pandas>=2.1.0
* pymongo>=4.6.0
* python-dotenv
* jinja2
* imblearn
* matplotlib
* seaborn
* xgboost>=2.0.0
* boto3>=1.28.0
* aiofiles
* dill

---

# ▶️ Running the Application

```
uvicorn app:app --reload
```

---

# 🌐 Access the App

```
http://127.0.0.1:8000
```

---

# 🔄 Model Training

To retrain the model:

```
python src/pipeline/train_pipeline.py
```

---

# 📈 Why Not Other Models?

| Model               | Reason                  |
| ------------------- | ----------------------- |
| Logistic Regression | Slower for text data    |
| SVM                 | High computation cost   |
| XGBoost             | Overkill for simple NLP |
| Deep Learning       | Needs large dataset     |

👉 **Naive Bayes = Best balance of speed + performance**

---

# 🔮 Future Improvements

* 🔐 User authentication system
* ☁️ Cloud deployment (AWS / Render)
* 📊 Model monitoring (Evidently AI)
* 🤖 Deep learning integration
* 📱 Mobile app version

---

# 👨‍💻 Author

**Ayush Ghadai**

---

# ⭐ Support

If you like this project:

* Give it a ⭐ on GitHub
* Share with others

---

# 🧾 License

This project is for educational purposes.
