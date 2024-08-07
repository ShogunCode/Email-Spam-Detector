import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import random

def load_data():
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
    data = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
    data['label'] = data.label.map({'ham': 0, 'spam': 1})
    return data

def load_model():
    model = joblib.load('spam_detector.pkl')
    vectorizer = joblib.load('spam_vectorizer.pkl')
    return model, vectorizer

def predict(text):
    model, vectorizer = load_model()
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    return prediction[0]  # 1 for spam, 0 for ham

def generate_spam():
    spam_phrases = [
        "Congratulations! You've won a prize!",
        "Claim your free gift now!",
        "Limited time offer, act now!",
        "You have been selected for a special offer.",
        "Win big prizes today! Click here!",
        "Urgent: Your account needs verification.",
        "Free entry into our sweepstakes.",
        "Get your exclusive deal today!"
    ]
    spam_message = random.choice(spam_phrases)
    return spam_message

if __name__ == "__main__":
    for _ in range(10):
        message = generate_spam()
        is_spam = predict(message)
        print(f"Message: {message} - Label: {'Spam' if is_spam == 1 else 'Not Spam'}")

data = load_data()
spam_examples = data[data['label'] == 1].sample(5)
print(spam_examples)
