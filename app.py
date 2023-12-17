import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer


# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 

# Load the saved models
model_xgb = pickle.load(open('/Users/harinathcingapuram/Documents/HumanVSGpt/XBBOOST_Imbalanced.pkl', 'rb'))
model_human_vs_gpt = tf.keras.models.load_model('/Users/harinathcingapuram/Documents/HumanVSGpt/HumanVSGPT.h5')
model_human_vs_11gpt = tf.keras.models.load_model('/Users/harinathcingapuram/Documents/HumanVSGpt/HumanVS11GPT.h5')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

with open('/Users/harinathcingapuram/Documents/HumanVSGpt/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


# Load the BERT tokenizer if needed
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Text preprocessing functions
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def preprocess_text_lemmatize(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)



# Streamlit UI
st.title("AI Generated Text Detection Template")

# Input text box
user_input = st.text_area("Enter the text you want to classify:", "")


# Option to select the model
selected_model = st.selectbox("Select a Model", ["XBBoost_Imbalanced", "HumanVSGPT", "HumanVS11GPT"])
# Button to classify the text
if st.button("Classify"):
    if selected_model == "XBBoost_Imbalanced":
        # Vectorize the preprocessed text input (if you have a vectorizer)
        text_input_processed = preprocess_text_lemmatize(user_input)
        
        # Vectorize the preprocessed text input
        text_input_vectorized = vectorizer.transform([text_input_processed])

        # Use the trained XGBoost model to make a prediction
        prediction = model_xgb.predict(text_input_vectorized)
        
        # Print the prediction
        st.write("Prediction for the text input:", prediction)

    elif selected_model == "HumanVSGPT":
        # Tokenize and preprocess the input text (if you need tokenization)
        # tokens = tokenizer.tokenize(preprocessed_text)
        # token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Load the BERT tokenizer
        
        
        
        # Tokenize the text input
        tokens = tokenizer.tokenize(user_input)
        
        # Convert tokens to token IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Assuming you have the feature extraction code from the previous code
        # Calculate the features for the text input and store them in a list
        # Make sure the features match the format used during training
        # Replace the example features below with actual feature extraction
        avg_word_length = 4.2
        avg_sentence_length = 15.6
        vocab_size = 5000
        lexical_diversity = 4.8
        noun_count = 12
        verb_count = 8
        adj_count = 6
        input_features = [avg_word_length, avg_sentence_length, vocab_size, lexical_diversity, noun_count, verb_count, adj_count]
        
        # Pad the token IDs and add the features
        maxlen = 100  # Make sure this matches the maxlen used during training
        padded_sequence = pad_sequences([token_ids], maxlen=maxlen, padding='post')[0]
        input_features = np.array(input_features)
        input_data = np.concatenate([padded_sequence, input_features])
        
        # Reshape the input data to match the model's input shape
        input_data = input_data.reshape(1, -1)
        
        # Use the model to make a prediction
        predictions = model_human_vs_gpt.predict(input_data)
        predicted_class = np.argmax(predictions)

        # Print the prediction
        st.write("Prediction for the text input:", predicted_class)

    elif selected_model == "HumanVS11GPT":
       # Load the BERT tokenizer
        
        # Tokenize the text input
        tokens = tokenizer.tokenize(user_input)
        
        # Convert tokens to token IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Assuming you have the feature extraction code from the previous code
        # Calculate the features for the text input and store them in a list
        # Make sure the features match the format used during training
        # Replace the example features below with actual feature extraction
        avg_word_length = 4.2
        avg_sentence_length = 15.6
        vocab_size = 5000
        lexical_diversity = 4.8
        noun_count = 12
        verb_count = 8
        adj_count = 6
        input_features = [avg_word_length, avg_sentence_length, vocab_size, lexical_diversity, noun_count, verb_count, adj_count]
        
        # Pad the token IDs and add the features
        maxlen = 100  # Make sure this matches the maxlen used during training
        padded_sequence = pad_sequences([token_ids], maxlen=maxlen, padding='post')[0]
        input_features = np.array(input_features)
        input_data = np.concatenate([padded_sequence, input_features])
        
        # Reshape the input data to match the model's input shape
        input_data = input_data.reshape(1, -1)
        
        # Use the model to make a prediction
        predictions = model_human_vs_11gpt.predict(input_data)
        predicted_class = np.argmax(predictions)

        # Print the prediction
        st.write("Prediction for the text input:", predicted_class)
