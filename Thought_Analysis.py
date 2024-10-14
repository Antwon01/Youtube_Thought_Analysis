import os
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import csv


# YouTube API Credentials
YOUTUBE_API_KEY = 'AIzaSyBMSiEOiUKaTTKMfMVKJ2TfgNVAA2gOWMk'  
# Other Configuration
YOUTUBE_VIDEO_ID = 'bn0Kh9c4Zv4'  # Replace with the target YouTube video ID
SEARCH_KEYWORD = 'great'  # (Optional) Keyword to filter comments
LANGUAGE = 'en'  # Language for YouTube comments
MAX_RESULTS = 1000  # Total number of comments to fetch
DATA_CSV_PATH = 'youtube_sentiment_data.csv'


def collect_youtube_comments(youtube, video_id, max_results=1000):
    print(f"Collecting YouTube comments for video ID: {video_id}")
    comments = []
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=100,  # Maximum allowed per request
        order='relevance',  # Can be 'relevance' or 'time'
        # If SEARCH_KEYWORD is used, apply here
    )
    response = request.execute()

    while request is not None and len(comments) < max_results:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            timestamp = item['snippet']['topLevelComment']['snippet']['publishedAt']
            comments.append((comment, timestamp))
            if len(comments) >= max_results:
                break
        request = youtube.commentThreads().list_next(request, response)
        if request:
            response = request.execute()
        else:
            break
    print(f"Collected {len(comments)} comments.")
    return comments


def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Range from -1 (negative) to 1 (positive)


def initialize_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['comment', 'sentiment', 'timestamp'])
        print(f"Created new CSV file at {csv_path}")
    else:
        print(f"CSV file already exists at {csv_path}")

def append_to_csv(csv_path, data):
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for entry in data:
            writer.writerow(entry)
    print(f"Appended {len(data)} records to {csv_path}")


def visualize_sentiment_trends(csv_path):
    print(f"Visualizing sentiment trends from {csv_path}")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_sentiment['date'], daily_sentiment['sentiment'], marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment')
    plt.title('Sentiment Trend Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('youtube_sentiment_trend.png')
    plt.show()
    print("Sentiment trend visualization saved as 'youtube_sentiment_trend.png'")


def main():
    # Initialize data storage
    initialize_csv(DATA_CSV_PATH)

    # Set up YouTube API
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    # Collect comments
    comments = collect_youtube_comments(youtube, YOUTUBE_VIDEO_ID, max_results=MAX_RESULTS)

    # Process and analyze comments
    processed_data = []
    for comment, timestamp in comments:
        if SEARCH_KEYWORD:
            if SEARCH_KEYWORD.lower() not in comment.lower():
                continue  # Skip comments that do not contain the keyword
        cleaned = clean_text(comment)
        if not cleaned:
            continue  # Skip empty comments after cleaning
        sentiment = analyze_sentiment(cleaned)
        processed_data.append([cleaned, sentiment, timestamp])

    print(f"Processed {len(processed_data)} comments for sentiment analysis.")

    # Append data to CSV
    append_to_csv(DATA_CSV_PATH, processed_data)

    # Visualize sentiment trends
    visualize_sentiment_trends(DATA_CSV_PATH)

if __name__ == "__main__":
    main()
