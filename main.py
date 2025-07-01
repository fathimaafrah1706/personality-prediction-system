import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
nltk.download('stopwords')
data = {
    'resume_text': [
        "Managed a team and led meetings, organized events, strong communication",
        "Worked independently on projects, data analysis and report generation",
        "Enjoys team activities and public speaking, leads group tasks",
        "Highly focused on timelines, loves planning and structure",
        "Developed emotional intelligence, helped others and volunteered regularly"
    ],
    'personality_type': [
        "Extrovert",
        "Conscientious",
        "Extrovert",
        "Conscientious",
        "Agreeable"
    ]
}
df = pd.DataFrame(data)
stop_words = stopwords.words('english')
model = make_pipeline(
    TfidfVectorizer(stop_words=stop_words),
    MultinomialNB()
)
model.fit(df['resume_text'], df['personality_type'])
print("\n--- Personality Prediction System ---")
resume_input = input("Paste your resume text here: ")
prediction = model.predict([resume_input])
print("Predicted Personality Type:", prediction[0])
