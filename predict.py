import joblib
import sys

# load the model and vectorizer 
def load_model():
    model = joblib.load('spam_detector.pkl')
    vectorizer = joblib.load('spam_vectorizer.pkl')
    return model, vectorizer

# predict the if text is spam or not
def predict(text):
    model, vectorizer = load_model()
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    return "Spam" if prediction[0] == 1 else "Not Spam"

if __name__ == "__main__":
    text = sys.argv[1]
    prediction = predict(text)
    print(f"Prediction: {prediction}")