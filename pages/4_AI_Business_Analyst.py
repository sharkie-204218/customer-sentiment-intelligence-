
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

      
        st.subheader("Automated Business Intelligence Report")
        st.caption("Free rule-based analytics. No API key required.")

        if st.button("Generate Business Report", type="primary"):
            reviews = df["Review"].fillna("").astype(str)
            ratings = pd.to_numeric(df["Rating"], errors="coerce")

            total = len(df)
            average_rating = ratings.mean()
            negative = int((ratings <= 2).sum())
            positive = int((ratings >= 4).sum())
            neutral = int((ratings == 3).sum())

            issues = {
                "Delivery delays": [
                    "late", "delay", "slow delivery",
                    "物流慢", "延迟", "配送慢", "送货慢"
                ],
                "App stability": [
                    "crash", "bug", "freeze", "error",
                    "闪退", "崩溃", "卡顿", "故障"
                ],
                "Customer support": [
                    "no response", "unhelpful", "rude",
                    "客服慢", "没人回复", "态度差"
                ],
                "Payment issues": [
                    "payment failed", "checkout", "refund",
                    "支付失败", "付款失败", "退款"
                ]
            }

            issue_counts = {}
            for issue, keywords in issues.items():
                issue_counts[issue] = sum(
                    any(word in review.lower() for word in keywords)
                    for review in reviews
                )

            ranked_issues = sorted(
                issue_counts.items(),
                key=lambda item: item[1],
                reverse=True
            )

            recommendations = {
                "Delivery delays": "Review delivery SLA performance and improve shipment tracking.",
                "App stability": "Prioritize bug fixes and monitor crash frequency.",
                "Customer support": "Improve response time and customer service training.",
                "Payment issues": "Investigate checkout failures and streamline refund processes."
            }

            lines = [
                "# Customer Feedback Business Intelligence Report",
                "",
                "## Executive Summary",
                f"- Total reviews: {total}",
                f"- Average rating: {average_rating:.2f}/5",
                f"- Positive ratings (4-5): {positive}",
                f"- Neutral ratings (3): {neutral}",
                f"- Negative ratings (1-2): {negative}",
                "",
                "## Key Issues and Recommendations"
            ]

            for issue, count in ranked_issues:
                if count > 0:
                    lines.extend([
                        f"### {issue}",
                        f"- Reviews mentioning this issue: {count}",
                        f"- Recommendation: {recommendations[issue]}",
                        ""
                    ])

            if not any(issue_counts.values()):
                lines.append(
                    "No predefined issue keywords were detected."
                )

            lines.extend([
                "## Methodology",
                "This report uses rating statistics and keyword matching.",
                "Issue counts may overlap. Results are descriptive, not causal.",
                "No external AI API was used."
            ])

            report = "\n".join(lines)
            st.session_state["v3_report"] = report

        if "v3_report" in st.session_state:
            report = st.session_state["v3_report"]
            st.markdown(report)

            st.download_button(
                "Download Management Report",
                data=report,
                file_name="Business_Intelligence_Report.md",
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
