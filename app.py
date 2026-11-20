import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

st.set_page_config(
    page_title="Customer Sentiment Intelligence",
    page_icon="🧠",
    layout="wide"
)

BASE = Path(__file__).parent
analyzer = SentimentIntensityAnalyzer()

ISSUES = {
    "Delivery": ["delivery", "late", "shipping", "arrived", "package"],
    "Product Quality": ["broken", "damaged", "defective", "quality"],
    "App Experience": ["app", "crash", "bug", "interface", "checkout"],
    "Payment": ["payment", "refund", "charged", "billing"],
    "Customer Support": ["support", "service", "response", "unhelpful"]
}

ACTIONS = {
    "Delivery": "Review shipping delays and delivery reliability.",
    "Product Quality": "Investigate defects and strengthen quality checks.",
    "App Experience": "Prioritize usability and application stability.",
    "Payment": "Audit payment failures and refund workflows.",
    "Customer Support": "Review response times and support processes."
}

def analyze(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score

def detect_issues(text):
    words = str(text).lower()
    found = [
        name for name, terms in ISSUES.items()
        if any(term in words for term in terms)
    ]
    return ", ".join(found) if found else "Other"

def prepare(df):
    required = ["Date", "Product", "Review", "Rating"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Review", "Product"])
    df = df[df["Review"].astype(str).str.strip().ne("")]
    df = df[df["Rating"].between(1, 5)]

    results = df["Review"].apply(analyze)
    df["Sentiment"] = results.apply(lambda x: x[0])
    df["Sentiment Score"] = results.apply(lambda x: x[1])
    df["Issue Topics"] = df["Review"].apply(detect_issues)
    return df

with st.sidebar:
    st.title("🧠 Customer Intelligence")
    st.caption("AI-Powered Feedback Analytics | V2")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Review Explorer", "Business Insights",
         "About the Developer"]
    )

    st.divider()
    uploaded = st.file_uploader(
        "Upload customer reviews",
        type=["csv", "xlsx"]
    )
    st.caption("Required: Date, Product, Review, Rating")
    st.caption("English-language sentiment analysis.")
    st.caption("Built-in data is simulated.")

try:
    if uploaded is None:
        raw = pd.read_csv(BASE / "data" / "reviews.csv")
    elif uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_excel(uploaded)
    df = prepare(raw)
except Exception as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

if df.empty:
    st.warning("No valid reviews found.")
    st.stop()

if page == "Dashboard":
    st.title("Customer Sentiment Intelligence")
    st.caption("NLP-Based Customer Feedback & Business Analytics")

    products = sorted(df["Product"].astype(str).unique())
    selected = st.multiselect(
        "Filter by product", products, default=products
    )
    view = df[df["Product"].isin(selected)]

    if view.empty:
        st.info("Please select at least one product.")
        st.stop()

    positive = (view["Sentiment"] == "Positive").mean() * 100
    negative = (view["Sentiment"] == "Negative").sum()

    a, b, c, d = st.columns(4)
    a.metric("Total Reviews", len(view))
    b.metric("Positive Share", f"{positive:.1f}%")
    c.metric("Negative Reviews", int(negative))
    d.metric("Average Rating", f"{view['Rating'].mean():.2f}/5")

    left, right = st.columns(2)

    with left:
        counts = view["Sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = px.pie(
            counts, names="Sentiment", values="Count",
            hole=0.45, title="AI Sentiment Distribution"
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        grouped = (
            view.groupby(["Product", "Sentiment"])
            .size().reset_index(name="Count")
        )
        fig = px.bar(
            grouped, x="Product", y="Count",
            color="Sentiment", barmode="group",
            title="Sentiment by Product"
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Monthly Feedback Trend")
    dated = view.dropna(subset=["Date"]).copy()
    if not dated.empty:
        dated["Month"] = dated["Date"].dt.to_period("M").astype(str)
        trend = (
            dated.groupby("Month").size()
            .reset_index(name="Reviews")
        )
        st.plotly_chart(
            px.line(trend, x="Month", y="Reviews", markers=True),
            width="stretch"
        )

    st.info(
        "Sentiment is predicted from English review text using VADER. "
        "Results may be inaccurate for sarcasm, context or mixed opinions."
    )

elif page == "Review Explorer":
    st.title("Review Explorer")
    choice = st.selectbox(
        "Filter sentiment",
        ["All", "Positive", "Neutral", "Negative"]
    )
    view = df if choice == "All" else df[df["Sentiment"] == choice]
    st.dataframe(view, width="stretch", hide_index=True)

    st.download_button(
        "Download Analyzed Reviews (CSV)",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name="sentiment_analysis_v2.csv",
        mime="text/csv"
    )

elif page == "Business Insights":
    st.title("Business Insights")
    st.caption("Rule-based recommendations from negative review topics")

    negatives = df[df["Sentiment"] == "Negative"]
    st.metric("Negative Reviews", len(negatives))

    if negatives.empty:
        st.success("No negative reviews identified by the model.")
    else:
        rows = []
        for _, review in negatives.iterrows():
            for topic in review["Issue Topics"].split(", "):
                rows.append({"Topic": topic, "Review": review["Review"]})

        issues_df = pd.DataFrame(rows)
        counts = (
            issues_df["Topic"].value_counts()
            .reset_index()
        )
        counts.columns = ["Topic", "Mentions"]

        st.plotly_chart(
            px.bar(
                counts, x="Topic", y="Mentions",
                title="Negative Feedback Topics"
            ),
            width="stretch"
        )

        st.subheader("Recommended Actions")
        for _, row in counts.iterrows():
            topic = row["Topic"]
            recommendation = ACTIONS.get(
                topic,
                "Manually review these comments to identify root causes."
            )
            st.write(
                f"**{topic}** ({row['Mentions']} mentions): "
                f"{recommendation}"
            )

        st.subheader("Supporting Reviews")
        st.dataframe(issues_df, width="stretch", hide_index=True)

    st.caption(
        "Topics use keyword matching; recommendations are "
        "rule-based, not LLM-generated. Topic mentions may overlap."
    )

else:
    st.title("About the Developer")
    st.subheader("Hongyu Pan")

    st.write(
        "International Business postgraduate student with "
        "an interest in AI applications, business analytics, "
        "supply chain operations and digital transformation."
    )

    st.subheader("Portfolio Projects")
    st.markdown(
        "**01 — Smart Inventory Analytics**"
    )
    st.link_button(
        "Explore Inventory Dashboard",
        "https://smart-inventory-analytics.streamlit.app"
    )

    st.markdown(
        "**02 — Customer Sentiment Intelligence**"
    )
    st.write(
        "NLP-powered customer review analysis with "
        "interactive dashboards and rule-based business insights."
    )

    st.subheader("Connect")
    st.link_button(
        "GitHub Profile",
        "https://github.com/sharkie-204218"
    )

    st.caption(
        "Contact email and downloadable resume will be added "
        "after verifying the approved public files."
    )

st.divider()
st.caption(
    "Portfolio demonstration | Simulated data | "
    "VADER NLP sentiment analysis | Hongyu Pan"
)
