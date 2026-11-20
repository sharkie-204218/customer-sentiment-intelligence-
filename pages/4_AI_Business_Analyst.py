
import pandas as pd
import streamlit as st
from ai_analyst import generate_business_report

st.set_page_config(
    page_title="AI Business Analyst",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Business Decision Assistant")
st.caption(
    "Turn customer feedback into actionable business insights."
)

st.info(
    "Portfolio demonstration. Use simulated or "
    "anonymized customer feedback only."
)

uploaded = st.file_uploader(
    "Upload customer feedback",
    type=["csv", "xlsx"]
)

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        required = {"Date", "Product", "Review", "Rating"}

        if not required.issubset(df.columns):
            st.error(
                "Required columns: Date, Product, Review, Rating"
            )
            st.stop()

        st.success(f"Loaded {len(df)} customer reviews.")

        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("AI Business Analysis")

        st.caption(
            "The AI analyzes up to 100 reviews per request. "
            "API usage may incur charges."
        )

        if st.button(
            "Generate AI Business Report",
            type="primary"
        ):
            with st.spinner("Analyzing customer feedback..."):
                report, error = generate_business_report(df)

            if error:
                st.error(error)
            else:
                st.session_state["v3_report"] = report

        if "v3_report" in st.session_state:
            report = st.session_state["v3_report"]

            st.markdown(report)

            st.download_button(
                "Download Management Report",
                data=report,
                file_name="AI_Business_Report.md",
                mime="text/markdown"
            )

    except Exception as exc:
        st.error(
            f"Unable to read the uploaded file: "
            f"{type(exc).__name__}"
        )

else:
    st.warning(
        "Upload a CSV or Excel file to start analysis."
    )
    st.markdown(
        """
        **Required data format**

        | Date | Product | Review | Rating |
        |---|---|---|---|
        | 2026-11-01 | Mobile App | App crashes often | 1 |
        | 2026-11-02 | Delivery | Very fast delivery | 5 |
        """
    )
