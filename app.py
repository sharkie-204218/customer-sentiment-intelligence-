import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Customer Sentiment Intelligence",
    page_icon="🧠",
    layout="wide"
)

DATA_FILE = Path(__file__).parent / "data" / "reviews.csv"

@st.cache_data
def load_sample():
    return pd.read_csv(DATA_FILE)

def classify_rating(rating):
    if rating >= 4:
        return "Positive"
    if rating <= 2:
        return "Negative"
    return "Neutral"

def prepare_data(df):
    required = ["Date", "Product", "Review", "Rating"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=required)
    df = df[df["Rating"].between(1, 5)]
    df["Sentiment"] = df["Rating"].apply(classify_rating)
    return df

with st.sidebar:
    st.title("🧠 Customer Intelligence")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Review Explorer", "About the Developer"]
    )
    st.divider()
    uploaded = st.file_uploader(
        "Upload customer reviews",
        type=["csv", "xlsx"]
    )
    st.caption("Required: Date, Product, Review, Rating")
    st.caption("Demo uses simulated customer reviews.")

try:
    if uploaded is None:
        raw = load_sample()
    elif uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
    else:
        raw = pd.read_excel(uploaded)
    df = prepare_data(raw)
except Exception as exc:
    st.error(f"Unable to load data: {exc}")
    st.stop()

if df.empty:
    st.warning("No valid reviews found.")
    st.stop()

if page == "Dashboard":
    st.title("Customer Sentiment Intelligence")
    st.caption("Customer Feedback & Business Insights Dashboard")

    products = sorted(df["Product"].astype(str).unique())
    selected = st.multiselect(
        "Filter by product",
        products,
        default=products
    )
    filtered = df[df["Product"].isin(selected)]

    if filtered.empty:
        st.info("Select at least one product.")
        st.stop()

    positive = (filtered["Sentiment"] == "Positive").mean() * 100
    negative = (filtered["Sentiment"] == "Negative").sum()

    a, b, c, d = st.columns(4)
    a.metric("Total Reviews", len(filtered))
    b.metric("Positive Share", f"{positive:.1f}%")
    c.metric("Negative Reviews", int(negative))
    d.metric("Average Rating", f"{filtered['Rating'].mean():.2f}/5")

    left, right = st.columns(2)

    with left:
        counts = filtered["Sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = px.pie(
            counts, names="Sentiment", values="Count",
            title="Sentiment Distribution",
            hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        summary = (
            filtered.groupby(["Product", "Sentiment"])
            .size().reset_index(name="Count")
        )
        fig = px.bar(
            summary, x="Product", y="Count",
            color="Sentiment",
            title="Sentiment by Product",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Review Volume")
    monthly = filtered.dropna(subset=["Date"]).copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
    trend = monthly.groupby("Month").size().reset_index(name="Reviews")
    st.plotly_chart(
        px.line(trend, x="Month", y="Reviews", markers=True),
        use_container_width=True
    )

    st.subheader("Business Insights")
    if negative:
        issues = (
            filtered[filtered["Sentiment"] == "Negative"]
            .groupby("Product").size().sort_values(ascending=False)
        )
        st.warning(
            f"Review negative feedback for {issues.index[0]} first: "
            f"{issues.iloc[0]} negative reviews."
        )
    else:
        st.success("No negative ratings in the selected dataset.")

    st.caption(
        "V1 sentiment labels are derived from ratings, not an NLP model. "
        "Business recommendations are rule-based."
    )

elif page == "Review Explorer":
    st.title("Review Explorer")
    sentiment = st.selectbox(
        "Sentiment", ["All", "Positive", "Neutral", "Negative"]
    )
    view = df if sentiment == "All" else df[df["Sentiment"] == sentiment]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Analyzed Reviews",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name="analyzed_reviews.csv",
        mime="text/csv"
    )

else:
    st.title("About the Developer")
    st.subheader("Hongyu Pan")
    st.write(
        "International Business postgraduate student interested in "
        "business analytics, AI applications, and digital transformation."
    )
    st.write(
        "I develop practical data products that turn business "
        "information into actionable insights."
    )
    st.markdown(
        "[Smart Inventory Analytics]"
        "(https://smart-inventory-analytics.streamlit.app)"
    )
    st.markdown(
        "[GitHub Portfolio]"
        "(https://github.com/sharkie-204218)"
    )
    st.info(
        "Contact email and public resume will be added after "
        "verifying the approved public contact details."
    )
