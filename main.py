import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
texts = [
    "I love working with teams and enjoy social interactions.",
    "I prefer working alone and focusing deeply on tasks.",
    "I am organized and always meet my deadlines.",
    "I like trying new experiences and taking risks."
]
labels = [
    "Extroversion",
    "Introversion",
    "Conscientiousness",
    "Openness"
]
nltk.download('punkt')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)
test_text = "I enjoy meeting new people and collaborating on projects."
X_test = vectorizer.transform([test_text])
prediction = model.predict(X_test)
print("Predicted Personality Trait:", prediction[0])
