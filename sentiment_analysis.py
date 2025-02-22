from transformers import pipeline

# Load Pre-trained Sentiment Model
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]

# Test
if __name__ == "__main__":
    text = "I am really happy with the service!"
    sentiment = analyze_sentiment(text)
    print("Sentiment:", sentiment)
