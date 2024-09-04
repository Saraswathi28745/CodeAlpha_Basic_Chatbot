import nltk
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
nltk.download('punkt_tab')
nltk.download('wordnet')

# Load your dataset
with open(r'C:\Users\Hello\Desktop\intents.json') as file:
    data = json.load(file)

# Prepare training data
patterns = []
responses = []
classes = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = word_tokenize(pattern.lower())  # Tokenize using 'punkt'
        patterns.append(' '.join([lemmatizer.lemmatize(token) for token in tokens]))
        responses.append(intent['responses'][0])  # Choose the first response for simplicity
        classes.append(intent['tag'])

# Vectorize the patterns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = np.array(classes)

# Train a classifier
classifier = LinearSVC()
classifier.fit(X, y)

# Define a function to get responses
def chatbot_response(text):
    text_tokens = word_tokenize(text.lower())
    text_vector = vectorizer.transform([' '.join([lemmatizer.lemmatize(token) for token in text_tokens])])
    prediction = classifier.predict(text_vector)
    tag = prediction[0]
    
    for intent in data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    return "I'm sorry, I don't understand."

# Chat with the bot
print("Chatbot: Hi! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")
