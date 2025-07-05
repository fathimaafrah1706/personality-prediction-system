import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sqlite3
from datetime import datetime

# Download NLTK resources if needed
nltk.download('punkt')
nltk.download('stopwords')

# Dummy data for example
texts = [
    "I love working in teams and enjoy social gatherings.",
    "I am very organized and always plan ahead.",
    "I like to explore new ideas and experiences.",
    "I am very cooperative and value others' opinions.",
    "I often feel anxious in stressful situations."
]
labels = ["Extroversion", "Conscientiousness", "Openness", "Agreeableness", "Neuroticism"]

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Create/connect database
conn = sqlite3.connect("predictions.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_text TEXT,
    predicted_trait TEXT,
    created_at TEXT
)
''')
conn.commit()

# User input
resume_text = input("Enter resume text: ")

# Preprocess
resume_cleaned = re.sub(r'\W+', ' ', resume_text.lower())
X_input = vectorizer.transform([resume_cleaned])

# Predict
prediction = model.predict(X_input)[0]

print("Predicted Personality Trait:", prediction)

# Insert into DB
cursor.execute(
    "INSERT INTO predictions (resume_text, predicted_trait, created_at) VALUES (?, ?, ?)",
    (resume_text, prediction, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
)
conn.commit()
conn.close()
