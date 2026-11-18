# Customer Review Sentiment & Topic Dashboard

An interactive NLP dashboard that analyzes customer reviews to identify
sentiment, discussion topics, common complaints, and satisfaction factors.

## Features

- Upload customer-review CSV files
- Automatic review-column detection
- Sentiment classification
- Topic analysis
- Interactive filters
- Review search and exploration
- Downloadable analysis results

## Technologies

- Python
- Streamlit
- Pandas
- Plotly
- NLTK / VADER
- Hugging Face Transformers
- Scikit-learn
- LDA / BERTopic

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

The project uses customer product reviews containing review text, ratings,
product categories, and review dates.

## Project Workflow

1. Data collection and exploration
2. Text preprocessing
3. Sentiment analysis
4. Topic modeling
5. Model evaluation
6. Interactive dashboard development
7. Deployment

## Dashboard

The dashboard includes:

- Overview metrics
- Sentiment distribution
- Sentiment trends
- Topic frequency
- Topic-sentiment analysis
- Customer complaints
- Positive feedback keywords
- Review Explorer
- CSV download
