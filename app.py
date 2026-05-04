import os
import re
from itertools import combinations
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px
import networkx as nx


# ============================================================
# App Configuration
# ============================================================

st.set_page_config(
    page_title="Food Delivery Pattern Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom Styling
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #0f172a 0%, #111827 45%, #f8fafc 45%, #f8fafc 100%);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero-card {
        padding: 2.4rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #ff6b35 0%, #f97316 35%, #ef4444 100%);
        color: white;
        box-shadow: 0 25px 60px rgba(239, 68, 68, 0.32);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 3.1rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        line-height: 1.05;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        opacity: 0.96;
        max-width: 920px;
        line-height: 1.7;
    }

    .mini-card {
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        min-height: 112px;
    }

    .metric-title {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 1.72rem;
        color: #0f172a;
        font-weight: 800;
        margin-top: 0.3rem;
        word-break: break-word;
    }

    .metric-note {
        font-size: 0.84rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    .insight-box {
        padding: 1rem 1.2rem;
        border-left: 5px solid #f97316;
        background: #fff7ed;
        color: #431407;
        border-radius: 14px;
        margin: 0.7rem 0 1rem 0;
        line-height: 1.6;
    }

    .warning-box {
        padding: 1rem 1.2rem;
        border-left: 5px solid #ef4444;
        background: #fef2f2;
        color: #450a0a;
        border-radius: 14px;
        margin: 0.7rem 0 1rem 0;
        line-height: 1.6;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 16px;
        padding: 0 18px;
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #0f172a;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background: #f97316 !important;
        color: white !important;
    }

    .footer {
        color: #64748b;
        text-align: center;
        font-size: 0.9rem;
        padding: 2rem 0 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Paths
# ============================================================

def find_base_dir():
    """
    Supports running the app from:
    1. Project root: Food-Delivery-Pattern-Analysis/
    2. data folder: Food-Delivery-Pattern-Analysis/data/
    """
    if os.path.exists(os.path.join("data", "output")):
        return os.path.join("data", "output")
    if os.path.exists("output"):
        return "output"
    if os.path.exists(os.path.join("..", "data", "output")):
        return os.path.join("..", "data", "output")
    raise FileNotFoundError("Could not find data/output or output directory.")


BASE_DIR = find_base_dir()
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
TRANSACTIONS_DIR = os.path.join(BASE_DIR, "transactions")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

TRANSACTIONS_PATH = os.path.join(TRANSACTIONS_DIR, "transactions.csv")
MEALS_PATH = os.path.join(PROCESSED_DIR, "meals_cleaned.csv")
REVIEWS_PATH = os.path.join(PROCESSED_DIR, "reviews.csv")
REVIEWS_SENTIMENT_PATH = os.path.join(PROCESSED_DIR, "reviews_with_sentiment.csv")
ASSOCIATION_RULES_PATH = os.path.join(PROCESSED_DIR, "association_rules.csv")


# ============================================================
# Data Loading
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=True)
def load_all_data():
    transactions_df = load_csv(TRANSACTIONS_PATH)
    meals_df = load_csv(MEALS_PATH)
    reviews_df = load_csv(REVIEWS_PATH)
    sentiment_df = load_csv(REVIEWS_SENTIMENT_PATH)
    rules_df = load_csv(ASSOCIATION_RULES_PATH)

    if not transactions_df.empty and "order_date" in transactions_df.columns:
        transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"], errors="coerce")

    return transactions_df, meals_df, reviews_df, sentiment_df, rules_df


transactions_df, meals_df, reviews_df, sentiment_df, rules_df = load_all_data()

if transactions_df.empty:
    st.error("transactions.csv was not found. Please run the data pipeline first.")
    st.stop()


# ============================================================
# Utilities
# ============================================================

def split_items(series):
    items = []
    for value in series.dropna():
        items.extend([x.strip() for x in str(value).split(",") if x.strip()])
    return items


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def clean_rule_text(rule):
    rule = str(rule)
    rule = rule.replace("frozenset", "")
    rule = rule.replace("{", "").replace("}", "")
    rule = rule.replace("(", "").replace(")", "")
    rule = rule.replace("'", "").replace('"', "")
    rule = re.sub(r"\s+", " ", rule).strip()
    return rule


def prepare_rules(df):
    if df.empty:
        return df

    out = df.copy()

    rename_map = {
        "Support": "support",
        "Confidence": "confidence",
        "Lift": "lift"
    }
    out = out.rename(columns=rename_map)

    if "rule" not in out.columns:
        if "antecedents_str" in out.columns and "consequents_str" in out.columns:
            out["rule"] = out["antecedents_str"].astype(str) + " → " + out["consequents_str"].astype(str)
        elif "antecedents" in out.columns and "consequents" in out.columns:
            out["rule"] = (
                out["antecedents"].astype(str).apply(clean_rule_text)
                + " → "
                + out["consequents"].astype(str).apply(clean_rule_text)
            )

    for col in ["support", "confidence", "lift"]:
        if col in out.columns:
            out[col] = safe_numeric(out[col])

    if {"lift", "confidence", "support"}.issubset(out.columns):
        out["reliability_score"] = out["lift"] * out["confidence"] * out["support"]

    if "rule" in out.columns:
        out["rule"] = out["rule"].astype(str)

    return out


@st.cache_data(show_spinner=False)
def build_pagerank_from_transactions(transactions_df):
    """
    Rebuilds PageRank directly from transactions.csv using the same logic
    as Task4_Visualization_Reporting.ipynb.
    This avoids mismatches caused by stale pagerank_results.csv files.
    """
    meal_lookup = {}

    for _, row in transactions_df.iterrows():
        meal_ids = [x.strip() for x in str(row["items"]).split(",") if x.strip()]
        meal_names = [x.strip() for x in str(row["item_names"]).split(",") if x.strip()]
        meal_categories = [x.strip() for x in str(row["item_categories"]).split(",") if x.strip()]

        for i, meal_id in enumerate(meal_ids):
            if meal_id not in meal_lookup:
                meal_lookup[meal_id] = {
                    "meal_name": meal_names[i] if i < len(meal_names) else meal_id,
                    "category": meal_categories[i] if i < len(meal_categories) else "Unknown"
                }

    graph = nx.Graph()

    for meal_id, info in meal_lookup.items():
        graph.add_node(
            meal_id,
            meal_name=info["meal_name"],
            category=info["category"]
        )

    edge_counter = Counter()

    for _, row in transactions_df.iterrows():
        meal_ids = [x.strip() for x in str(row["items"]).split(",") if x.strip()]

        if len(meal_ids) >= 2:
            for a, b in combinations(sorted(meal_ids), 2):
                edge_counter[(a, b)] += 1

    for (a, b), weight in edge_counter.items():
        graph.add_edge(a, b, weight=weight)

    pagerank_scores = nx.pagerank(
        graph,
        alpha=0.85,
        weight="weight",
        max_iter=200
    )

    pagerank_results = []

    for meal_id, score in pagerank_scores.items():
        info = meal_lookup.get(meal_id, {})

        pagerank_results.append({
            "meal_id": meal_id,
            "meal_name": info.get("meal_name", meal_id),
            "category": info.get("category", "Unknown"),
            "pagerank_score": score,
            "degree": graph.degree(meal_id),
            "weighted_degree": sum(
                edge_data["weight"]
                for _, _, edge_data in graph.edges(meal_id, data=True)
            )
        })

    pagerank_df = pd.DataFrame(pagerank_results)
    pagerank_df = pagerank_df.sort_values("pagerank_score", ascending=False).reset_index(drop=True)
    pagerank_df["rank"] = pagerank_df.index + 1

    graph_info = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "is_connected": nx.is_connected(graph) if graph.number_of_nodes() > 0 else False
    }

    return pagerank_df, graph_info


def prepare_sentiment():
    if sentiment_df.empty or meals_df.empty:
        return pd.DataFrame()

    cols = [c for c in ["meal_id", "meal_name", "category"] if c in meals_df.columns]

    if "meal_id" not in cols:
        return pd.DataFrame()

    meal_info = meals_df[cols].copy()

    merged = sentiment_df.merge(
        meal_info,
        on="meal_id",
        how="left"
    )

    return merged


def build_recommendation_table(pagerank_df, sentiment_df, meals_df):
    if pagerank_df.empty or sentiment_df.empty or meals_df.empty:
        return pd.DataFrame()

    sentiment_merged = prepare_sentiment()

    if sentiment_merged.empty:
        return pd.DataFrame()

    sentiment_summary = (
        sentiment_merged
        .dropna(subset=["meal_id"])
        .groupby("meal_id")
        .agg(
            avg_sentiment_score=("sentiment_score", "mean"),
            review_count=("review_id", "count")
        )
        .reset_index()
    )

    rec_df = pagerank_df.merge(
        sentiment_summary,
        on="meal_id",
        how="left"
    )

    rec_df["avg_sentiment_score"] = rec_df["avg_sentiment_score"].fillna(0)
    rec_df["review_count"] = rec_df["review_count"].fillna(0)

    pr_min = rec_df["pagerank_score"].min()
    pr_max = rec_df["pagerank_score"].max()

    if pr_max == pr_min:
        rec_df["pagerank_norm"] = 0
    else:
        rec_df["pagerank_norm"] = (
            (rec_df["pagerank_score"] - pr_min) /
            (pr_max - pr_min)
        )

    rec_df["sentiment_norm"] = (rec_df["avg_sentiment_score"] + 1) / 2

    rec_df["recommendation_score"] = (
        0.7 * rec_df["pagerank_norm"] +
        0.3 * rec_df["sentiment_norm"]
    )

    return rec_df.sort_values("recommendation_score", ascending=False).reset_index(drop=True)


def metric_card(title, value, note=""):
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def make_plotly_layout(fig, height=480):
    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family="Inter, Arial", size=13),
        title=dict(font=dict(size=20, family="Inter", color="#0f172a")),
        margin=dict(l=20, r=20, t=70, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(15,23,42,0.08)",
            borderwidth=1
        )
    )
    return fig


def show_notebook_figure(title, filename):
    """Display a saved notebook figure from the figures directory."""
    fig_path = os.path.join(FIGURES_DIR, filename)

    st.subheader(title)

    if os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)
    else:
        st.warning(f"Figure not found: {filename}")


rules_df = prepare_rules(rules_df)
pagerank_df, pagerank_graph_info = build_pagerank_from_transactions(transactions_df)
sentiment_merged_df = prepare_sentiment()
recommendation_df = build_recommendation_table(pagerank_df, sentiment_df, meals_df)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.markdown("## Food Delivery")
st.sidebar.markdown("### Pattern Analysis")

with st.sidebar.expander("Data Status", expanded=True):
    st.write(f"Base directory: `{BASE_DIR}`")
    st.write(f"Transactions: {'Available' if not transactions_df.empty else 'Missing'}")
    st.write(f"Meals: {'Available' if not meals_df.empty else 'Missing'}")
    st.write(f"Rules: {'Available' if not rules_df.empty else 'Missing'}")
    st.write("PageRank: Rebuilt from transactions")
    st.write(f"Sentiment: {'Available' if not sentiment_df.empty else 'Missing'}")

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "EDA Dashboard",
        "Association Rules",
        "PageRank Network",
        "Sentiment Intelligence",
        "Recommendation Engine",
        "Notebook Figures"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data Mining and Web Mining Project")


# ============================================================
# Hero
# ============================================================

def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# Page: Executive Overview
# ============================================================

if page == "Executive Overview":
    hero(
        "Food Delivery Pattern Analysis",
        "An interactive analytics dashboard combining transaction mining, PageRank network analysis, and sentiment intelligence to support smarter meal recommendations."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Orders", f"{len(transactions_df):,}", "Food delivery transactions")
    with col2:
        metric_card("Customers", f"{transactions_df['customer_id'].nunique():,}", "Unique customers")
    with col3:
        metric_card("Restaurants", f"{transactions_df['restaurant'].nunique():,}", "Restaurant partners")
    with col4:
        metric_card("Avg Rating", f"{transactions_df['order_rating'].mean():.2f}/5", "Order satisfaction")

    st.markdown("### Project Flow")

    flow_cols = st.columns(5)
    flow_items = [
        ("1", "Data Collection", "Meals, reviews, orders"),
        ("2", "Transactions", "Items ordered together"),
        ("3", "Association Rules", "Meal combo patterns"),
        ("4", "PageRank", "Influential meals"),
        ("5", "Recommendation", "PageRank + sentiment")
    ]

    for col, (num, title, desc) in zip(flow_cols, flow_items):
        with col:
            st.markdown(
                f"""
                <div class="mini-card" style="min-height: 145px;">
                    <div style="font-size: 2rem; font-weight: 800; color:#f97316;">{num}</div>
                    <div style="font-size: 1.05rem; font-weight: 800; color:#0f172a;">{title}</div>
                    <div style="font-size: 0.9rem; color:#64748b; margin-top:0.4rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Key Results")

    result_cols = st.columns(4)

    with result_cols[0]:
        metric_card("Association Rules", f"{len(rules_df):,}" if not rules_df.empty else "N/A", "Generated rules")
    with result_cols[1]:
        metric_card("Top PageRank Meal", pagerank_df.iloc[0]["meal_name"] if not pagerank_df.empty else "N/A", "Most influential")
    with result_cols[2]:
        metric_card("Reviews Analyzed", f"{len(sentiment_df):,}" if not sentiment_df.empty else "N/A", "Sentiment dataset")
    with result_cols[3]:
        metric_card("Top Recommendation", recommendation_df.iloc[0]["meal_name"] if not recommendation_df.empty else "N/A", "Final model output")

    st.markdown(
        """
        <div class="insight-box">
        <b>Executive Insight:</b> The final system does not rely on popularity alone. It combines co-ordering behavior,
        network influence, and customer satisfaction to recommend meals that are both influential and positively perceived.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Preview Raw Transactions"):
        st.dataframe(transactions_df.head(50), use_container_width=True)


# ============================================================
# Page: EDA Dashboard
# ============================================================

elif page == "EDA Dashboard":
    hero(
        "Exploratory Data Analysis",
        "Understand customer ordering behavior, category demand, order timing, payment preferences, ratings, and top-performing meals."
    )

    all_meals = split_items(transactions_df["item_names"])
    meal_counts = pd.Series(all_meals).value_counts().reset_index()
    meal_counts.columns = ["Meal", "Count"]

    all_categories = split_items(transactions_df["item_categories"])
    category_counts = pd.Series(all_categories).value_counts().reset_index()
    category_counts.columns = ["Category", "Count"]

    tab1, tab2, tab3 = st.tabs(["Popularity", "Time Patterns", "Orders and Ratings"])

    with tab1:
        col1, col2 = st.columns([1.15, 0.85])

        with col1:
            fig = px.bar(
                meal_counts.head(20).sort_values("Count"),
                x="Count",
                y="Meal",
                orientation="h",
                color="Count",
                color_continuous_scale="Oranges",
                title="Top 20 Most Ordered Meals"
            )
            fig = make_plotly_layout(fig, 650)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(
                category_counts,
                values="Count",
                names="Category",
                hole=0.48,
                title="Category Demand Share",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig = make_plotly_layout(fig, 650)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            hour_counts = transactions_df["order_hour"].value_counts().sort_index().reset_index()
            hour_counts.columns = ["Hour", "Orders"]

            fig = px.line(
                hour_counts,
                x="Hour",
                y="Orders",
                markers=True,
                title="Orders by Hour of Day"
            )
            fig.update_traces(line=dict(width=4), marker=dict(size=9))
            fig = make_plotly_layout(fig, 480)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_counts = transactions_df["order_day"].value_counts().reindex(day_order).reset_index()
            day_counts.columns = ["Day", "Orders"]

            fig = px.bar(
                day_counts,
                x="Day",
                y="Orders",
                color="Orders",
                color_continuous_scale="Blues",
                title="Orders by Day of Week"
            )
            fig = make_plotly_layout(fig, 480)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                transactions_df,
                x="order_rating",
                nbins=16,
                marginal="box",
                title="Order Rating Distribution",
                color_discrete_sequence=["#f97316"]
            )
            fig.add_vline(
                x=transactions_df["order_rating"].mean(),
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Mean {transactions_df['order_rating'].mean():.2f}"
            )
            fig = make_plotly_layout(fig, 520)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            payment_counts = transactions_df["payment_method"].value_counts().reset_index()
            payment_counts.columns = ["Payment Method", "Orders"]

            fig = px.bar(
                payment_counts,
                x="Payment Method",
                y="Orders",
                color="Payment Method",
                title="Payment Method Distribution"
            )
            fig = make_plotly_layout(fig, 520)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Notebook Figures")

    show_notebook_figure("EDA Dashboard from Notebook", "01_eda_dashboard.png")
    show_notebook_figure("Top Ordered Meals from Notebook", "02_top15_ordered_meals.png")


# ============================================================
# Page: Association Rules
# ============================================================

elif page == "Association Rules":
    hero(
        "Association Rule Mining",
        "Discover meal combinations that appear together more often than expected using support, confidence, lift, and a reliability score."
    )

    if rules_df.empty:
        st.markdown(
            """
            <div class="warning-box">
            association_rules.csv was not found. Run Task4_Visualization_Reporting.ipynb first to generate association rules.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card("Rules", f"{len(rules_df):,}", "Total generated")
        with col2:
            metric_card("Max Lift", f"{rules_df['lift'].max():.2f}", "Strongest lift")
        with col3:
            metric_card("Avg Confidence", f"{rules_df['confidence'].mean():.2f}", "Mean rule confidence")
        with col4:
            metric_card("Avg Support", f"{rules_df['support'].mean():.4f}", "Mean rule support")

        st.markdown(
            """
            <div class="warning-box">
            <b>Important:</b> Some high-lift rules have low support. They should be interpreted as exploratory patterns,
            not final business decisions.
            </div>
            """,
            unsafe_allow_html=True
        )

        tab1, tab2, tab3 = st.tabs(["Top Rules", "Rule Explorer", "Quality Map"])

        with tab1:
            metric_choice = st.selectbox(
                "Rank rules by",
                ["lift", "reliability_score", "confidence", "support"],
                index=0
            )

            top_rules = rules_df.sort_values(metric_choice, ascending=False).head(15).copy()

            fig = px.bar(
                top_rules.sort_values(metric_choice),
                x=metric_choice,
                y="rule",
                orientation="h",
                color=metric_choice,
                color_continuous_scale="Turbo",
                title=f"Top 15 Association Rules by {metric_choice.replace('_', ' ').title()}"
            )
            fig = make_plotly_layout(fig, 720)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                top_rules[["rule", "support", "confidence", "lift", "reliability_score"]],
                use_container_width=True
            )

        with tab2:
            meals = sorted(set(split_items(transactions_df["item_names"])))
            selected_meal = st.selectbox("Choose a meal to explore related rules", meals)

            filtered_rules = rules_df[
                rules_df["rule"].str.contains(selected_meal, case=False, na=False)
            ].sort_values(["lift", "confidence"], ascending=False)

            st.metric("Matching Rules", f"{len(filtered_rules):,}")

            if filtered_rules.empty:
                st.info("No rules found for this meal.")
            else:
                st.dataframe(
                    filtered_rules[["rule", "support", "confidence", "lift", "reliability_score"]].head(50),
                    use_container_width=True
                )

        with tab3:
            fig = px.scatter(
                rules_df,
                x="support",
                y="confidence",
                size="lift",
                color="lift",
                hover_data=["rule", "reliability_score"],
                color_continuous_scale="Viridis",
                title="Support vs Confidence — Bubble Size = Lift"
            )
            fig = make_plotly_layout(fig, 650)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.header("Notebook Figures")

        show_notebook_figure("Top Association Rules by Lift", "05_top10_association_rules_by_lift.png")
        show_notebook_figure("Association Rules Scatter Plot", "06_association_rules_scatter.png")
        show_notebook_figure("Association Rule Network", "07_association_rules_network.png")
        show_notebook_figure("Top Reliable Association Rules", "08_top10_reliable_association_rules.png")


# ============================================================
# Page: PageRank Network
# ============================================================

elif page == "PageRank Network":
    hero(
        "PageRank Network Analysis",
        "Rank meals based on influence inside the co-ordering graph. Influential meals are highly connected to other important meals."
    )

    if pagerank_df.empty:
        st.markdown(
            """
            <div class="warning-box">
            PageRank could not be calculated from transactions.csv.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        top_meal = pagerank_df.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card("Meal Nodes", f"{pagerank_graph_info['nodes']:,}", "Graph nodes")
        with col2:
            metric_card("Graph Edges", f"{pagerank_graph_info['edges']:,}", "Co-ordering pairs")
        with col3:
            metric_card("Top Meal", top_meal["meal_name"], top_meal["category"])
        with col4:
            metric_card("Top Score", f"{top_meal['pagerank_score']:.5f}", "PageRank")

        st.markdown(
            f"""
            <div class="insight-box">
            <b>Consistency Check:</b> PageRank is rebuilt directly from transactions.csv, so it matches the Task 4 notebook logic.
            Top meal should be <b>{top_meal['meal_name']}</b> with score <b>{top_meal['pagerank_score']:.5f}</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

        tab1, tab2, tab3 = st.tabs(["Ranking", "Category Analysis", "Influence Map"])

        with tab1:
            top_n = st.slider("Number of meals to show", 10, 50, 20)

            top_df = pagerank_df.head(top_n).copy()

            fig = px.bar(
                top_df.sort_values("pagerank_score"),
                x="pagerank_score",
                y="meal_name",
                color="category",
                orientation="h",
                title=f"Top {top_n} Influential Meals by PageRank"
            )
            fig = make_plotly_layout(fig, 760)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                top_df[["rank", "meal_name", "category", "pagerank_score", "degree", "weighted_degree"]],
                use_container_width=True
            )

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                avg_category_pr = (
                    pagerank_df
                    .groupby("category")["pagerank_score"]
                    .mean()
                    .sort_values(ascending=False)
                    .reset_index()
                )

                fig = px.bar(
                    avg_category_pr,
                    x="category",
                    y="pagerank_score",
                    color="pagerank_score",
                    color_continuous_scale="Oranges",
                    title="Average PageRank by Category"
                )
                fig = make_plotly_layout(fig, 520)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.box(
                    pagerank_df,
                    x="category",
                    y="pagerank_score",
                    color="category",
                    title="PageRank Distribution by Category"
                )
                fig = make_plotly_layout(fig, 520)
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = px.scatter(
                pagerank_df,
                x="degree",
                y="pagerank_score",
                color="category",
                size="weighted_degree",
                hover_data=["meal_name", "rank"],
                title="PageRank Score vs Meal Connectivity"
            )
            fig = make_plotly_layout(fig, 650)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.header("Notebook Figures")

        show_notebook_figure("Top PageRank Meals", "09_top20_pagerank_meals.png")
        show_notebook_figure("PageRank by Category", "10_pagerank_by_category.png")
        show_notebook_figure("PageRank vs Connectivity", "11_pagerank_vs_degree.png")
        show_notebook_figure("PageRank Network Graph", "12_pagerank_network_graph.png")


# ============================================================
# Page: Sentiment Intelligence
# ============================================================

elif page == "Sentiment Intelligence":
    hero(
        "Sentiment Intelligence",
        "Analyze customer review sentiment and connect customer satisfaction to categories, meals, and recommendation quality."
    )

    if sentiment_df.empty:
        st.markdown(
            """
            <div class="warning-box">
            reviews_with_sentiment.csv was not found. Run BERT_Algorithm.ipynb first.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card("Reviews", f"{len(sentiment_df):,}", "Analyzed customer reviews")
        with col2:
            metric_card("Main Sentiment", sentiment_df["sentiment"].mode()[0], "Most frequent label")
        with col3:
            metric_card("Avg Score", f"{sentiment_df['sentiment_score'].mean():.2f}", "Binary sentiment average")

        tab1, tab2, tab3 = st.tabs(["Overview", "Meal Sentiment", "Category Sentiment"])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                sentiment_counts = sentiment_df["sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]

                fig = px.pie(
                    sentiment_counts,
                    names="Sentiment",
                    values="Count",
                    hole=0.5,
                    title="Review Sentiment Share",
                    color="Sentiment",
                    color_discrete_map={
                        "POSITIVE": "#22c55e",
                        "NEGATIVE": "#ef4444",
                        "NEUTRAL": "#64748b"
                    }
                )
                fig = make_plotly_layout(fig, 520)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(
                    sentiment_df,
                    x="sentiment_score",
                    nbins=10,
                    title="Sentiment Score Distribution",
                    color_discrete_sequence=["#f97316"]
                )
                fig = make_plotly_layout(fig, 520)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if sentiment_merged_df.empty:
                st.info("Meal metadata could not be merged with sentiment data.")
            else:
                meal_sentiment = (
                    sentiment_merged_df
                    .dropna(subset=["meal_name"])
                    .groupby(["meal_name", "category"])
                    .agg(
                        avg_sentiment_score=("sentiment_score", "mean"),
                        review_count=("review_id", "count")
                    )
                    .reset_index()
                )

                max_review_count = int(meal_sentiment["review_count"].max())
                min_reviews = st.slider("Minimum review count", 5, max_review_count, 5)

                top_meals = (
                    meal_sentiment[meal_sentiment["review_count"] >= min_reviews]
                    .sort_values(["avg_sentiment_score", "review_count"], ascending=False)
                    .head(20)
                )

                fig = px.bar(
                    top_meals.sort_values("avg_sentiment_score"),
                    x="avg_sentiment_score",
                    y="meal_name",
                    color="category",
                    orientation="h",
                    hover_data=["review_count"],
                    title="Top Meals by Average Sentiment Score"
                )
                fig = make_plotly_layout(fig, 720)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(top_meals, use_container_width=True)

        with tab3:
            if sentiment_merged_df.empty:
                st.info("Category sentiment could not be calculated.")
            else:
                category_sentiment = (
                    sentiment_merged_df
                    .dropna(subset=["category"])
                    .groupby("category")
                    .agg(
                        avg_sentiment_score=("sentiment_score", "mean"),
                        review_count=("review_id", "count")
                    )
                    .sort_values("avg_sentiment_score", ascending=False)
                    .reset_index()
                )

                fig = px.bar(
                    category_sentiment.sort_values("avg_sentiment_score"),
                    x="avg_sentiment_score",
                    y="category",
                    orientation="h",
                    color="avg_sentiment_score",
                    color_continuous_scale="RdYlGn",
                    title="Average Sentiment Score by Category"
                )
                fig = make_plotly_layout(fig, 560)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(category_sentiment, use_container_width=True)

        st.markdown("---")
        st.header("Notebook Figures")

        show_notebook_figure("Sentiment Distribution", "13_sentiment_distribution.png")
        show_notebook_figure("Sentiment Score Distribution", "14_sentiment_score_distribution.png")
        show_notebook_figure("Average Sentiment by Category", "15_avg_sentiment_by_category.png")
        show_notebook_figure("Top Meals by Sentiment", "16_top15_meals_by_sentiment.png")


# ============================================================
# Page: Recommendation Engine
# ============================================================

elif page == "Recommendation Engine":
    hero(
        "Final Recommendation Engine",
        "A polished recommendation layer that combines network influence from PageRank with customer satisfaction from sentiment analysis."
    )

    if recommendation_df.empty:
        st.markdown(
            """
            <div class="warning-box">
            Recommendation table could not be built. Make sure reviews_with_sentiment.csv exists.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="insight-box">
            <b>Model Formula:</b> Final Recommendation Score = 0.7 × Normalized PageRank + 0.3 × Normalized Sentiment
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card("Top Recommended", recommendation_df.iloc[0]["meal_name"], recommendation_df.iloc[0]["category"])
        with col2:
            metric_card("Final Score", f"{recommendation_df.iloc[0]['recommendation_score']:.2f}", "Best combined score")
        with col3:
            metric_card("Sentiment", f"{recommendation_df.iloc[0]['avg_sentiment_score']:.2f}", "Average review score")

        tab1, tab2, tab3 = st.tabs(["Top Recommendations", "Interactive Filter", "Combo Builder"])

        with tab1:
            top_n = st.slider("Number of recommendations", 10, 50, 15)
            top_recs = recommendation_df.head(top_n).copy()

            fig = px.bar(
                top_recs.sort_values("recommendation_score"),
                x="recommendation_score",
                y="meal_name",
                color="category",
                orientation="h",
                hover_data=["pagerank_score", "avg_sentiment_score", "review_count"],
                title=f"Top {top_n} Recommended Meals"
            )
            fig = make_plotly_layout(fig, 750)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                top_recs[
                    [
                        "meal_name",
                        "category",
                        "pagerank_score",
                        "avg_sentiment_score",
                        "review_count",
                        "recommendation_score"
                    ]
                ],
                use_container_width=True
            )

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                categories = ["All"] + sorted(recommendation_df["category"].dropna().unique().tolist())
                selected_category = st.selectbox("Category", categories)

            with col2:
                min_sentiment = st.slider("Minimum sentiment score", -1.0, 1.0, 0.0, 0.1)

            filtered = recommendation_df.copy()

            if selected_category != "All":
                filtered = filtered[filtered["category"] == selected_category]

            filtered = filtered[filtered["avg_sentiment_score"] >= min_sentiment]

            st.metric("Matching Meals", f"{len(filtered):,}")

            st.dataframe(
                filtered[
                    [
                        "meal_name",
                        "category",
                        "pagerank_score",
                        "avg_sentiment_score",
                        "review_count",
                        "recommendation_score"
                    ]
                ].head(50),
                use_container_width=True
            )

        with tab3:
            st.write("Choose a meal and discover potential combo recommendations from association rules.")

            meals = sorted(set(split_items(transactions_df["item_names"])))
            selected_meal = st.selectbox("Starter meal", meals, key="combo_meal")

            if rules_df.empty:
                st.info("Association rules are not available.")
            else:
                combo_rules = rules_df[
                    rules_df["rule"].str.contains(selected_meal, case=False, na=False)
                ].sort_values(["reliability_score", "lift"], ascending=False)

                if combo_rules.empty:
                    st.warning("No combo rules found for this meal.")
                else:
                    st.dataframe(
                        combo_rules[["rule", "support", "confidence", "lift", "reliability_score"]].head(15),
                        use_container_width=True
                    )

        st.markdown("---")
        st.header("Notebook Figure")

        show_notebook_figure("Final Recommendation Score", "17_final_recommendation_score.png")


# ============================================================
# Page: Notebook Figures
# ============================================================

elif page == "Notebook Figures":
    hero(
        "Notebook Figures",
        "This section displays the original figures generated by the analysis notebook. These figures match the report outputs."
    )

    figure_groups = {
        "Exploratory Data Analysis": [
            "01_eda_dashboard.png",
            "02_top15_ordered_meals.png",
            "03_daily_orders_over_time.png",
            "04_temporal_patterns.png"
        ],
        "Association Rule Mining": [
            "05_top10_association_rules_by_lift.png",
            "06_association_rules_scatter.png",
            "07_association_rules_network.png",
            "08_top10_reliable_association_rules.png"
        ],
        "PageRank Analysis": [
            "09_top20_pagerank_meals.png",
            "10_pagerank_by_category.png",
            "11_pagerank_vs_degree.png",
            "12_pagerank_network_graph.png"
        ],
        "Sentiment Analysis": [
            "13_sentiment_distribution.png",
            "14_sentiment_score_distribution.png",
            "15_avg_sentiment_by_category.png",
            "16_top15_meals_by_sentiment.png"
        ],
        "Recommendation System": [
            "17_final_recommendation_score.png"
        ]
    }

    for section, files in figure_groups.items():
        st.header(section)

        for file in files:
            title = file.replace("_", " ").replace(".png", "").title()
            show_notebook_figure(title, file)

        st.markdown("---")


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer">
    Food Delivery Pattern Analysis | Association Rules | PageRank | Sentiment | Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)
