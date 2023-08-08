import os
import pandas as pd
import json
import nltk

import zipfile
import shutil

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import spacy
from spacy.lang.en import English

from collections import Counter

from nltk.sentiment import SentimentIntensityAnalyzer

## Specify the path to the folder containing the JSON files
#folder_path = 'Tweets_Corpus/06-2017'

## List all JSON files in the folder
#json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]

# Construct the NLP pipeline
stop_words = set(stopwords.words('english'))
stop_words.update(['|', '&', '!', '@', '#', '$', '%', '*', '(', ')', '-', '_', "'", ";", ":", ".", ",", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"])
porter_stemmer = PorterStemmer()
word_net_lemmatizer = WordNetLemmatizer()

# Load the English language model for SpaCy
nlp = spacy.load('en_core_web_sm')

# Initialize the SentimentIntentsityAnalyzer
analyzer = SentimentIntensityAnalyzer()

## Process each JSON file
#for json_file in json_files:
#    file_path = os.path.join(folder_path, json_file)

def process_json_file(json_file_path):
    # Read the JSON file and load the data
    with open(json_file_path, 'r', encoding='utf-8') as file:
        tweets_data = json.load(file)

    # Initialize lists to store all the data
    data = []

    for tweet in tweets_data:     
        # Extract relevant information for each tweet
        username = tweet['screen_name']
        # Convert the 'time_of_tweet' to datetime format
        time_of_tweet = pd.to_datetime(tweet['time'])
        # Extract the date and time of day separately
        date = time_of_tweet.date()
        time_of_day = time_of_tweet.time()
        full_tweet_text = tweet['text']

        # Calculate statistics for the aggregated tweets
        num_chars = len(full_tweet_text)
        num_words = len(full_tweet_text.split())
        num_sents = len(full_tweet_text.split('.'))
        unique_words = set(word.lower() for word in full_tweet_text.split())
        num_vocab = len(unique_words)

        # Clean the text using the NLP pipeline
        cleaned_text = [
            word_net_lemmatizer.lemmatize(porter_stemmer.stem(word.lower()))
            for word in nltk.word_tokenize(full_tweet_text)
            if word.lower() not in stop_words
        ]

        # Convert the cleaned_text list to a single string
        cleaned_text_str = " ".join(cleaned_text)
        
        # Perform Named Entity Recognition (NER) using SpaCy
        doc = nlp(cleaned_text_str)
        ner_entities = [ent.text for ent in doc.ents]

        # Perform sentiment analysis for each entity
        entity_sentiment_scores = []
        for entity in ner_entities:
            sentiment_scores = analyzer.polarity_scores(entity)
            entity_sentiment_scores.append(sentiment_scores['compound'])

        # Append the data to the list (each tweet's data is appended separately)
        data.append([username, date, time_of_day, full_tweet_text, cleaned_text_str, num_chars, num_words, num_sents, num_vocab, ner_entities, entity_sentiment_scores])    

        # Next Create a second Dataframe that the NER_entities and entity_sentiment_scores.
        # Then wide merge this every time the Loop runs to create a filled dataframe with all NER_Entities and Sentiment Scores from all tweets during the day
        # Then Remerge with the data Dataframe based on the cleaned_text_str. 

    # Create a DataFrame from the collected data
    columns = ['username', 'date', 'time', 'full_tweet_text', 'cleaned_text', 'num_chars', 'num_words', 'num_sents', 'num_vocab', 'ner_entities', 'entity_sentiment_scores']
    Daily_Tweets = pd.DataFrame(data, columns=columns)

    # Calculate the number of tweets each person publishes per day
    num_tweets = Daily_Tweets['username'].value_counts().to_dict()

    # Add the count as a new column 'num_tweets' in the Daily_Tweets DataFrame
    Daily_Tweets['num_tweets'] = Daily_Tweets['username'].map(num_tweets)

    # Create a DataFrame from the collected data
    updated_columns = ['username', 'date', 'time', 'full_tweet_text', 'cleaned_text', 'num_tweets', 'num_chars', 'num_words', 'num_sents', 'num_vocab', 'ner_entities', 'entity_sentiment_scores']
    Daily_Tweets = Daily_Tweets.loc[:, updated_columns]

    return Daily_Tweets

# Specify the paths for input and output zip folders
input_zip_folder = 'Tweets_Corpus'
output_zip_folder = 'Processed_Tweets'

# List all zip files in the input folder
zip_files = [file for file in os.listdir(input_zip_folder) if file.endswith('.zip')]

    # Export the result DataFrame to a CSV file with the filename_NLP.csv format
    #filename_nlp = os.path.splitext(json_file)[0] + '_NLP.csv'
    #Daily_Tweets.to_csv(filename_nlp, index=False)

    #print(f"Data successfully exported to {filename_nlp}")

# Process each zip file
for zip_file in zip_files:
    zip_file_path = os.path.join(input_zip_folder, zip_file)

    # Create a folder with the same name as the zip file (without the extension)
    output_folder_name = os.path.splitext(zip_file)[0]
    output_folder_path = os.path.join(output_zip_folder, output_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)

    # Extract all files from the zip folder
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder_path)

    # Process each JSON file in the extracted folder (including subfolders)
    for root, dirs, files in os.walk(output_folder_path):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                Daily_Tweets = process_json_file(json_file_path)

                # Export the result DataFrame to a CSV file with the filename_NLP.csv format
                filename_nlp = os.path.splitext(file)[0] + '_NLP.csv'
                csv_output_path = os.path.join(output_folder_path, filename_nlp)
                Daily_Tweets.to_csv(csv_output_path, index=False)

                # Add a print statement for each exported file
                print(f"Processed data exported to {csv_output_path}")

    # Create a new zip file with the processed CSV files
    with zipfile.ZipFile(os.path.join(output_zip_folder, f"{output_folder_name}_processed.zip"), 'w') as zip_out:
        for root, dirs, files in os.walk(output_folder_path):
            for file in files:
                zip_out.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_folder_path))

    # Remove the intermediate folder containing CSV files
    shutil.rmtree(output_folder_path)

print("Data processing and zip export completed.")
