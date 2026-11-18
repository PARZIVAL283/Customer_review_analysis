# Customer Review Analysis

A Python project for exploring customer feedback through sentiment analysis, keyword insights, and topic-oriented review exploration. It includes an interactive **Streamlit dashboard** and a **Jupyter notebook** for the underlying analysis workflow.

## Features

- Upload CSV files through the Streamlit interface
- Automatically detect likely review-text columns
- Map review text, rating, category, and date columns
- Classify sentiment as **Positive**, **Neutral**, or **Negative**
- Filter reviews by sentiment, product category, topic, and search text
- Explore sentiment and rating distributions
- Inspect topic frequency and topic-by-sentiment summaries
- Display frequently mentioned positive and negative keywords
- Preview and download filtered results as CSV

## Project Structure

```text
.
├── app.py                              # Streamlit dashboard
├── customer_review_analysis.ipynb      # Analysis notebook
├── data/
│   ├── sample_reviews.csv              # Small example dataset
│   └── evaluated_reviews_stage5.csv    # Evaluation dataset
├── requirements.txt                    # Python dependencies
└── README.md
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/PARZIVAL283/Customer_review_analysis.git
cd Customer_review_analysis
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`, that you can open in a browser.

## Using the Dashboard

1. Upload a CSV file containing customer-review text.
2. Confirm or change the detected **review text** column.
3. Optionally map the rating, product category, and review date columns.
4. Click **Analyze Reviews**.
5. Use the sidebar to filter by sentiment, category, topic, or text.
6. Explore the Overview, Sentiment Analysis, Topic Analysis, Customer Insights, and Review Explorer tabs.
7. Download the filtered, analyzed dataset from the Review Explorer tab.

A small ready-to-use example is available at [`data/sample_reviews.csv`](data/sample_reviews.csv).

## Input Format

The dashboard requires at least one text column containing review content. Column names such as `review_text`, `review`, `customer_review`, `comment`, `feedback`, `text`, or `body` are detected automatically.

Optional columns improve the dashboard experience:

| Column type | Example column | Purpose |
|---|---|---|
| Review text | `review_text` | Text sent to the analysis pipeline |
| Rating | `rating` | Rating distribution by sentiment |
| Category | `product_category` | Product-category filtering |
| Date | `review_date` | Temporal metadata; requires a parseable date format |

When supplied CSV data already contains `transformer_sentiment` or `vader_sentiment`, the dashboard prefers that column over its built-in classification. It also recognizes `topic_label` or `topic_id` when present.

## Analysis Notebook

Open [`customer_review_analysis.ipynb`](customer_review_analysis.ipynb) in Jupyter, VS Code, Google Colab, or another notebook environment to explore the analysis workflow and experiment with the included datasets.

## Dependencies

- [Streamlit](https://streamlit.io/) — interactive dashboard
- [Pandas](https://pandas.pydata.org/) — CSV loading and data manipulation
- [Plotly](https://plotly.com/python/) — interactive charts
- [NLTK](https://www.nltk.org/) — text-processing ecosystem
- [Transformers](https://huggingface.co/docs/transformers/) and [PyTorch](https://pytorch.org/) — transformer-based analysis support

## Method and Limitations

The dashboard's default sentiment mode uses a simple lexicon-based baseline: it compares counts of predefined positive and negative words. This makes the app fast and easy to run, but it may misclassify negation, sarcasm, mixed reviews, misspellings, and domain-specific language.

The dashboard is intended as a practical exploratory tool rather than a production sentiment system. Results should be reviewed in context, and more robust performance will generally require a larger labeled dataset and a validated model.


## Author
Sourodip Das Gupta

Data Science / NLP / AI Enthusiast Student

Email: sourodipdasgupta283@gmail.com

linkedIn: www.linkedin.com/in/sourodip-das-gupta-3b97a62a6


## Contributing

Contributions are welcome. For example, useful improvements include:

- Replacing or augmenting the keyword baseline with a validated sentiment model
- Adding evaluation metrics and labeled test data
- Improving topic modeling and text preprocessing
- Adding automated tests
- Improving CSV validation and date handling
