import io

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Customer Review Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Customer Review Sentiment Dashboard")
st.write(
    "Upload a customer-review CSV file to analyze sentiment, "
    "explore reviews, and download the results."
)

SENTIMENT_COLORS = {
    "Positive": "#2E8B57",
    "Neutral": "#F4A261",
    "Negative": "#D64545"
}

POSITIVE_WORDS = {
    "good", "great", "excellent", "love", "loved", "amazing",
    "best", "perfect", "happy", "satisfied", "recommend",
    "awesome", "fantastic", "wonderful", "useful", "reliable",
    "fast", "quick", "affordable", "value"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "worst", "poor", "disappointed",
    "disappointing", "hate", "broken", "defective", "useless",
    "waste", "refund", "return", "slow", "cheap", "damaged",
    "problem", "issue", "complaint", "late", "delay"
}


@st.cache_data
def load_csv(file):
    """Load and cache an uploaded CSV file."""
    return pd.read_csv(file)


def clean_text(text):
    """Clean review text for simple keyword analysis."""
    text = str(text).lower()

    text = "".join(
        character if character.isalpha() or character.isspace()
        else " "
        for character in text
    )

    return " ".join(text.split())


def predict_sentiment(text):
    """Classify sentiment using a simple keyword-based baseline."""
    cleaned_text = clean_text(text)
    words = set(cleaned_text.split())

    positive_count = len(words & POSITIVE_WORDS)
    negative_count = len(words & NEGATIVE_WORDS)

    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"


def detect_review_column(dataframe):
    """Find the most likely review-text column."""
    preferred_names = [
        "review_text",
        "review",
        "customer_review",
        "comment",
        "feedback",
        "text",
        "body"
    ]

    for preferred_name in preferred_names:
        for column in dataframe.columns:
            if preferred_name in column.lower():
                return column

    object_columns = dataframe.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return object_columns[0] if object_columns else None


def create_sentiment_chart(data):
    """Create a sentiment distribution pie chart."""
    sentiment_counts = (
        data["sentiment"]
        .value_counts()
        .reindex(
            ["Positive", "Neutral", "Negative"],
            fill_value=0
        )
        .reset_index()
    )

    sentiment_counts.columns = ["Sentiment", "Reviews"]

    fig = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Reviews",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        hole=0.45,
        title="Sentiment Distribution"
    )

    fig.update_traces(textinfo="percent+label")

    return fig


def create_rating_chart(data):
    """Create a rating distribution chart if ratings exist."""
    if "rating" not in data.columns:
        return None

    rating_data = data.dropna(subset=["rating"])

    if rating_data.empty:
        return None

    fig = px.histogram(
        rating_data,
        x="rating",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        barmode="group",
        title="Rating Distribution by Sentiment",
        labels={
            "rating": "Rating",
            "count": "Number of Reviews"
        }
    )

    return fig


def convert_to_csv(dataframe):
    """Convert a DataFrame to CSV bytes for download."""
    buffer = io.StringIO()
    dataframe.to_csv(buffer, index=False)

    return buffer.getvalue().encode("utf-8")


uploaded_file = st.file_uploader(
    "Upload a customer review CSV file",
    type=["csv"],
    key="review_csv_uploader"
)

if uploaded_file is None:
    st.info(
        "Upload a CSV file to begin. "
        "The file must contain at least one review-text column."
    )
    st.stop()

try:
    raw_data = load_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

if raw_data.empty:
    st.error("The uploaded CSV file is empty.")
    st.stop()

review_column = detect_review_column(raw_data)

if review_column is None:
    st.error(
        "Could not detect a review-text column. "
        "Please ensure the CSV contains a text column."
    )
    st.stop()

st.success(
    f"Loaded {len(raw_data):,} reviews from "
    f"`{uploaded_file.name}`."
)

st.subheader("Column Mapping")

review_column = st.selectbox(
    "Review text column",
    options=raw_data.columns.tolist(),
    index=raw_data.columns.get_loc(review_column),
    key="review_text_column_selector"
)

rating_column = st.selectbox(
    "Rating column",
    options=["Not available"] + raw_data.columns.tolist(),
    index=0,
    key="rating_column_selector"
)

category_column = st.selectbox(
    "Product category column",
    options=["Not available"] + raw_data.columns.tolist(),
    index=0,
    key="category_column_selector"
)

date_column = st.selectbox(
    "Review date column",
    options=["Not available"] + raw_data.columns.tolist(),
    index=0,
    key="date_column_selector"
)

if st.button(
    "Analyze Reviews",
    type="primary",
    key="analyze_reviews_button"
):
    with st.spinner("Analyzing reviews..."):
        data = raw_data.copy()

        data["review_text"] = data[review_column].astype(str)

        if "transformer_sentiment" in data.columns:
            data["sentiment"] = data["transformer_sentiment"]
        elif "vader_sentiment" in data.columns:
            data["sentiment"] = data["vader_sentiment"]
        else:
            data["sentiment"] = data["review_text"].apply(predict_sentiment)


        if "topic_label" in data.columns:
            data["topic"] = data["topic_label"]

        elif "topic_id" in data.columns:
            data["topic"] = data["topic_id"].astype(str)

        else:
            data["topic"] = "Not analyzed"



        if rating_column != "Not available":
            data["rating"] = pd.to_numeric(
                data[rating_column],
                errors="coerce"
            )
        else:
            data["rating"] = pd.NA

        if category_column != "Not available":
            data["product_category"] = (
                data[category_column].astype(str)
            )
        else:
            data["product_category"] = "Unknown"

        if date_column != "Not available":
            data["review_date"] = pd.to_datetime(
                data[date_column],
                errors="coerce"
            )
        else:
            data["review_date"] = pd.NaT
            if "topic_label" in data.columns:
                 data["topic"] = data["topic_label"]
            elif "topic_id" in data.columns:
                data["topic"] = data["topic_id"].astype(str)
            else:
                data["topic"] = "Not analyzed"

    st.session_state["review_data"] = data

if "review_data" not in st.session_state:
    st.info("Select columns and click **Analyze Reviews**.")
    st.stop()

data = st.session_state["review_data"]
if "topic" not in data.columns:
    data["topic"] = "Not analyzed"


st.sidebar.header("Filters")

sentiment_filter = st.sidebar.multiselect(
    "Sentiment",
    options=["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"],
    key="sentiment_filter"
)

data["product_category"] = (
    data["product_category"]
    .astype(str)
    .fillna("Unknown")
    .replace("nan", "Unknown")
)

category_options = sorted(
    data["product_category"].unique().tolist()
)

category_filter = st.sidebar.multiselect(
    "Product category",
    options=category_options,
    default=category_options,
    key="category_filter"
)

topic_options = sorted(
    data["topic"].unique().tolist()
)

topic_filter = st.sidebar.multiselect(
    "Topic",
    options=topic_options,
    default=topic_options,
    key="topic_filter"
)

search_query = st.sidebar.text_input(
    "Search reviews",
    placeholder="For example: battery, delivery, price",
    key="review_search_input"
)

filtered_data = data.loc[
    data["sentiment"].isin(sentiment_filter)
    & data["product_category"].isin(category_filter)
    & data["topic"].isin(topic_filter)
].copy()

if search_query:
    filtered_data = filtered_data.loc[
        filtered_data["review_text"]
        .str.contains(search_query, case=False, na=False)
    ]

if filtered_data.empty:
    st.warning(
        "No reviews match the selected filters. "
        "Try changing the filters."
    )
    st.stop()

overview_tab, sentiment_tab, topic_tab, insights_tab, explorer_tab = st.tabs(
    [
        "Overview",
        "Sentiment Analysis",
        "Topic Analysis",
        "Customer Insights",
        "Review Explorer"
    ]
)
with overview_tab:
    total_reviews = len(filtered_data)

    positive_reviews = int(
        (filtered_data["sentiment"] == "Positive").sum()
    )

    neutral_reviews = int(
        (filtered_data["sentiment"] == "Neutral").sum()
    )

    negative_reviews = int(
        (filtered_data["sentiment"] == "Negative").sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Reviews",
        f"{total_reviews:,}"
    )

    col2.metric(
        "Positive",
        f"{positive_reviews:,}"
    )

    col3.metric(
        "Neutral",
        f"{neutral_reviews:,}"
    )

    col4.metric(
        "Negative",
        f"{negative_reviews:,}"
    )

    st.plotly_chart(
        create_sentiment_chart(filtered_data),
        use_container_width=True,
        key="overview_sentiment_chart"
    )

with sentiment_tab:
    st.plotly_chart(
        create_sentiment_chart(filtered_data),
        use_container_width=True,
        key="sentiment_analysis_chart"
    )

    rating_chart = create_rating_chart(filtered_data)

    if rating_chart:
        st.plotly_chart(
            rating_chart,
            use_container_width=True,
            key="sentiment_rating_chart"
        )

with topic_tab:
    st.subheader("Most Frequently Discussed Topics")

    topic_counts = (
        filtered_data["topic"]
        .value_counts()
        .reset_index()
    )

    topic_counts.columns = ["Topic", "Reviews"]

    topic_chart = px.bar(
        topic_counts,
        x="Reviews",
        y="Topic",
        orientation="h",
        color="Reviews",
        color_continuous_scale="Teal",
        title="Topic Frequency"
    )

    topic_chart.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=False
    )

    st.plotly_chart(
        topic_chart,
        use_container_width=True,
        key="topic_frequency_chart"
    )

    st.subheader("Topic Distribution by Sentiment")

    topic_sentiment = (
        filtered_data
        .groupby(["topic", "sentiment"])
        .size()
        .reset_index(name="Reviews")
    )

    topic_sentiment_chart = px.bar(
        topic_sentiment,
        x="Reviews",
        y="topic",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        orientation="h",
        barmode="stack",
        title="Topic Distribution by Sentiment"
    )

    topic_sentiment_chart.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        topic_sentiment_chart,
        use_container_width=True,
        key="topic_sentiment_chart"
    )

    selected_topic = st.selectbox(
        "Select a topic to inspect",
        options=sorted(filtered_data["topic"].unique()),
        key="topic_inspection_selector"
    )

    topic_reviews = filtered_data.loc[
        filtered_data["topic"] == selected_topic
    ]

    st.subheader(f"Representative Reviews: {selected_topic}")

    st.dataframe(
        topic_reviews[[
            "review_text",
            "sentiment",
            "product_category"
        ]].head(10),
        use_container_width=True,
        key="topic_representative_reviews"
    )
with insights_tab:
    st.subheader("Frequently Mentioned Positive Aspects")

    positive_reviews = filtered_data.loc[
        filtered_data["sentiment"] == "Positive",
        "review_text"
    ]

    if not positive_reviews.empty:
        positive_text = " ".join(
            positive_reviews.astype(str)
        )

        positive_words = clean_text(positive_text).split()

        stop_words = {
            "the", "and", "for", "with", "this", "that", "was",
            "are", "have", "has", "had", "not", "but", "you",
            "your", "from", "they", "them", "its", "will", "would",
            "can", "could", "very", "just", "about", "there",
            "their", "when", "what", "which", "been", "were",
            "more", "than", "into", "after", "before", "because",
            "also", "only", "over", "some", "all", "one", "get",
            "got", "out", "up", "down", "so", "if", "at", "on",
            "in", "to", "of", "is", "it", "as", "be", "by", "or",
            "an", "a", "i", "my", "me", "we", "us", "do", "did",
            "does"
        }

        positive_keywords = pd.Series(
            [
                word for word in positive_words
                if len(word) > 2 and word not in stop_words
            ]
        ).value_counts().head(15).reset_index()

        positive_keywords.columns = ["Keyword", "Frequency"]

        positive_chart = px.bar(
            positive_keywords,
            x="Frequency",
            y="Keyword",
            orientation="h",
            color="Frequency",
            color_continuous_scale="Greens",
            title="Top Keywords in Positive Reviews"
        )

        positive_chart.update_layout(
            yaxis={"categoryorder": "total ascending"},
            showlegend=False
        )

        st.plotly_chart(
            positive_chart,
            use_container_width=True,
            key="positive_keywords_chart"
        )

    st.subheader("Common Customer Complaints")

    negative_reviews = filtered_data.loc[
        filtered_data["sentiment"] == "Negative",
        "review_text"
    ]

    if not negative_reviews.empty:
        negative_text = " ".join(
            negative_reviews.astype(str)
        )

        negative_words = clean_text(negative_text).split()

        negative_keywords = pd.Series(
            [
                word for word in negative_words
                if len(word) > 2 and word not in stop_words
            ]
        ).value_counts().head(15).reset_index()

        negative_keywords.columns = ["Keyword", "Frequency"]

        negative_chart = px.bar(
            negative_keywords,
            x="Frequency",
            y="Keyword",
            orientation="h",
            color="Frequency",
            color_continuous_scale="Reds",
            title="Top Keywords in Negative Reviews"
        )

        negative_chart.update_layout(
            yaxis={"categoryorder": "total ascending"},
            showlegend=False
        )

        st.plotly_chart(
            negative_chart,
            use_container_width=True,
            key="negative_keywords_chart"
        )

with explorer_tab:
    st.subheader("Review Explorer")

    display_columns = [
        "review_text",
        "sentiment",
        "product_category"
    ]

    if "rating" in filtered_data.columns:
        display_columns.append("rating")

    if "review_date" in filtered_data.columns:
        display_columns.append("review_date")

    st.dataframe(
        filtered_data[display_columns],
        use_container_width=True,
        height=500,
        key="review_explorer_table"
    )

    csv_data = convert_to_csv(filtered_data)

    st.download_button(
        label="Download Filtered Results as CSV",
        data=csv_data,
        file_name="customer_review_analysis.csv",
        mime="text/csv",
        key="download_filtered_results"
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Customer Review Sentiment Dashboard**\n\n"
    "Built with Python, Streamlit, Pandas, and Plotly."
)