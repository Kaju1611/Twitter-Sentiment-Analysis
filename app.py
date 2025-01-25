from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an index.html file in the templates folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form
        user_input = request.form['user_input']

        # Preprocess the input
        processed_input = preprocess_text(user_input)  # Replace with actual preprocessing logic

        # Transform input to numeric format using the vectorizer
        tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))  # Load your TF-IDF vectorizer
        processed_input_vectorized = tfidf_vectorizer.transform([processed_input]).toarray()

        # Make a prediction
        prediction = loaded_model.predict(processed_input_vectorized)

        # Map prediction to sentiment
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

        return render_template('index.html', prediction_text=f'Tweet Sentiment: {sentiment}')
    except Exception as e:
        return jsonify({'error': str(e)})



# Helper function for preprocessing (replace with your actual preprocessing logic)
def preprocess_text(text):
    # Example preprocessing: lowercasing, stripping, etc.
    return text.lower().strip()

if __name__ == '__main__':
    app.run(debug=True)
