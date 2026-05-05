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


    .premium-panel {
        padding: 1.35rem;
        border-radius: 24px;
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
        margin-bottom: 1.1rem;
    }

    .meal-hero {
        padding: 1.8rem;
        border-radius: 26px;
        background:
            radial-gradient(circle at top left, rgba(249,115,22,0.24), transparent 36%),
            linear-gradient(135deg, #0f172a 0%, #1e293b 52%, #111827 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 25px 60px rgba(15,23,42,0.25);
        margin: 1rem 0 1.2rem 0;
    }

    .meal-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
    }

    .meal-subtitle {
        color: #cbd5e1;
        margin-top: 0.45rem;
        font-size: 1rem;
        line-height: 1.6;
    }

    .pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.78rem;
        margin-right: 0.35rem;
        margin-top: 0.75rem;
        border: 1px solid rgba(255,255,255,0.15);
    }

    .pill-positive { background: rgba(34,197,94,0.18); color: #86efac; }
    .pill-negative { background: rgba(239,68,68,0.18); color: #fca5a5; }
    .pill-mixed { background: rgba(249,115,22,0.18); color: #fdba74; }
    .pill-neutral { background: rgba(148,163,184,0.18); color: #cbd5e1; }

    .rec-card {
        padding: 1.05rem 1.15rem;
        border-radius: 18px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 10px 26px rgba(15,23,42,0.06);
        margin-bottom: 0.7rem;
    }

    .rec-title {
        font-weight: 800;
        font-size: 1.05rem;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }

    .rec-meta {
        color: #64748b;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .score-line {
        height: 9px;
        width: 100%;
        border-radius: 999px;
        background: #e2e8f0;
        overflow: hidden;
        margin-top: 0.65rem;
    }

    .score-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #f97316, #ef4444);
    }

    .pro-note {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.7;
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



def normalize_text(value):
    """Normalize text for search and matching."""
    return str(value).strip().lower()


def find_best_meal_match(query, meal_names):
    """Return the best meal match from a typed query."""
    query_norm = normalize_text(query)
    if not query_norm:
        return None

    exact = [m for m in meal_names if normalize_text(m) == query_norm]
    if exact:
        return exact[0]

    starts = [m for m in meal_names if normalize_text(m).startswith(query_norm)]
    if starts:
        return starts[0]

    contains = [m for m in meal_names if query_norm in normalize_text(m)]
    if contains:
        return contains[0]

    return None


def get_meal_order_counts():
    """Return order counts per meal name based on transactions.csv."""
    all_meals = split_items(transactions_df["item_names"])
    counts = pd.Series(all_meals).value_counts().reset_index()
    counts.columns = ["meal_name", "order_count"]
    counts["popularity_rank"] = counts["order_count"].rank(method="dense", ascending=False).astype(int)
    return counts


def get_meal_sentiment_summary():
    """Return sentiment summary aggregated by meal name and category using reviews_with_sentiment.csv."""
    if sentiment_merged_df.empty:
        return pd.DataFrame(columns=["meal_name", "category", "avg_sentiment_score", "review_count", "sentiment_label"])

    summary = (
        sentiment_merged_df
        .dropna(subset=["meal_name"])
        .groupby(["meal_name", "category"])
        .agg(
            avg_sentiment_score=("sentiment_score", "mean"),
            review_count=("review_id", "count")
        )
        .reset_index()
    )

    def label(score):
        if score >= 0.25:
            return "Positive"
        if score <= -0.25:
            return "Negative"
        return "Mixed"

    summary["sentiment_label"] = summary["avg_sentiment_score"].apply(label)
    return summary


def get_pagerank_summary_by_meal():
    """Aggregate PageRank results by meal name."""
    if pagerank_df.empty:
        return pd.DataFrame(columns=["meal_name", "category", "pagerank_score", "degree", "weighted_degree", "pagerank_rank"])

    summary = (
        pagerank_df
        .groupby(["meal_name", "category"])
        .agg(
            pagerank_score=("pagerank_score", "max"),
            degree=("degree", "max"),
            weighted_degree=("weighted_degree", "max")
        )
        .sort_values("pagerank_score", ascending=False)
        .reset_index()
    )
    summary["pagerank_rank"] = summary["pagerank_score"].rank(method="dense", ascending=False).astype(int)
    return summary


def get_recommendation_summary_by_meal():
    """Aggregate final recommendation scores by meal name."""
    if recommendation_df.empty:
        return pd.DataFrame(columns=["meal_name", "category", "recommendation_score", "avg_sentiment_score", "review_count"])

    summary = (
        recommendation_df
        .groupby(["meal_name", "category"])
        .agg(
            recommendation_score=("recommendation_score", "max"),
            pagerank_score=("pagerank_score", "max"),
            avg_sentiment_score=("avg_sentiment_score", "mean"),
            review_count=("review_count", "sum")
        )
        .sort_values("recommendation_score", ascending=False)
        .reset_index()
    )
    summary["recommendation_rank"] = summary["recommendation_score"].rank(method="dense", ascending=False).astype(int)
    return summary


def get_meal_profile_table():
    """Build a unified meal profile table using project data only."""
    order_counts = get_meal_order_counts()
    sentiment_summary = get_meal_sentiment_summary()
    pagerank_summary = get_pagerank_summary_by_meal()
    recommendation_summary = get_recommendation_summary_by_meal()

    profile = order_counts.merge(sentiment_summary, on="meal_name", how="left")

    profile = profile.merge(
        pagerank_summary[["meal_name", "pagerank_score", "degree", "weighted_degree", "pagerank_rank"]],
        on="meal_name",
        how="left"
    )

    profile = profile.merge(
        recommendation_summary[["meal_name", "recommendation_score", "recommendation_rank"]],
        on="meal_name",
        how="left"
    )

    profile["category"] = profile["category"].fillna("Unknown")
    profile["avg_sentiment_score"] = profile["avg_sentiment_score"].fillna(0)
    profile["review_count"] = profile["review_count"].fillna(0)
    profile["sentiment_label"] = profile["sentiment_label"].fillna("No Reviews")
    profile["pagerank_score"] = profile["pagerank_score"].fillna(0)
    profile["degree"] = profile["degree"].fillna(0)
    profile["weighted_degree"] = profile["weighted_degree"].fillna(0)
    profile["recommendation_score"] = profile["recommendation_score"].fillna(0)

    return profile.sort_values("recommendation_score", ascending=False).reset_index(drop=True)


def get_co_ordered_meals(selected_meal, top_n=10):
    """Find meals most frequently ordered together with the selected meal."""
    counter = Counter()

    for _, row in transactions_df.iterrows():
        items = [x.strip() for x in str(row["item_names"]).split(",") if x.strip()]
        if selected_meal in items:
            for item in items:
                if item != selected_meal:
                    counter[item] += 1

    if not counter:
        return pd.DataFrame(columns=["Recommended Meal", "Co-order Count"])

    return pd.DataFrame(counter.most_common(top_n), columns=["Recommended Meal", "Co-order Count"])


def parse_meal_list(text_value):
    """Parse a comma-separated meal list from association rules."""
    if pd.isna(text_value):
        return []
    return [x.strip() for x in str(text_value).split(",") if x.strip()]


def get_rule_based_recommendations(selected_meal, top_n=10):
    """Recommend meals from association rules where selected meal appears in the antecedent side."""
    if rules_df.empty:
        return pd.DataFrame(columns=["Recommended Meal", "Rule", "support", "confidence", "lift", "reliability_score"])

    recs = []

    for _, row in rules_df.iterrows():
        antecedents_text = row.get("antecedents_str", "")
        consequents_text = row.get("consequents_str", "")

        if not antecedents_text or str(antecedents_text) == "nan":
            rule_text = str(row.get("rule", ""))
            if "→" in rule_text:
                antecedents_text = rule_text.split("→")[0].strip()
                consequents_text = rule_text.split("→")[1].strip()

        antecedents = parse_meal_list(antecedents_text)
        consequents = parse_meal_list(consequents_text)

        if selected_meal in antecedents:
            for meal in consequents:
                if meal != selected_meal:
                    recs.append({
                        "Recommended Meal": meal,
                        "Rule": row.get("rule", ""),
                        "support": row.get("support", 0),
                        "confidence": row.get("confidence", 0),
                        "lift": row.get("lift", 0),
                        "reliability_score": row.get("reliability_score", 0)
                    })

    if not recs:
        return pd.DataFrame(columns=["Recommended Meal", "Rule", "support", "confidence", "lift", "reliability_score"])

    out = pd.DataFrame(recs)
    out = (
        out.sort_values(["reliability_score", "lift", "confidence"], ascending=False)
        .drop_duplicates(subset=["Recommended Meal"])
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def get_hybrid_recommendations(selected_meal, top_n=10):
    """Recommend high-scoring meals from the same category, then fallback to global top recommendations."""
    rec_summary = get_recommendation_summary_by_meal()
    profile = get_meal_profile_table()

    selected_rows = profile[profile["meal_name"] == selected_meal]
    if selected_rows.empty or rec_summary.empty:
        return pd.DataFrame()

    selected_category = selected_rows.iloc[0]["category"]

    same_category = rec_summary[
        (rec_summary["category"] == selected_category) &
        (rec_summary["meal_name"] != selected_meal)
    ].copy()

    fallback = rec_summary[rec_summary["meal_name"] != selected_meal].copy()
    combined = pd.concat([same_category, fallback], ignore_index=True)
    combined = combined.drop_duplicates(subset=["meal_name"])

    return combined.sort_values("recommendation_score", ascending=False).head(top_n).reset_index(drop=True)


def sentiment_badge(label):
    """Return a clean sentiment explanation for display."""
    if label == "Positive":
        return "Positive customer feedback"
    if label == "Negative":
        return "Negative customer feedback"
    if label == "Mixed":
        return "Mixed customer feedback"
    return "No sentiment data available"


def format_percent(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def render_recommendation_cards(df, title_col, score_col=None, meta_cols=None, limit=5):
    """Render premium recommendation cards."""
    if df.empty:
        st.info("No recommendations available for this selection.")
        return

    meta_cols = meta_cols or []

    for _, row in df.head(limit).iterrows():
        title = row.get(title_col, "Recommended Meal")
        score_html = ""

        if score_col and score_col in row:
            try:
                score = float(row[score_col])
                width = max(0, min(score, 1)) * 100
                score_html = (
                    f'<div class="score-line">'
                    f'<div class="score-fill" style="width:{width:.1f}%"></div>'
                    f'</div>'
                )
            except Exception:
                score_html = ""

        meta_parts = []
        for col in meta_cols:
            if col in row:
                value = row[col]
                if isinstance(value, float):
                    value = f"{value:.4f}"
                meta_parts.append(f"{col.replace('_', ' ').title()}: {value}")

        meta_text = " | ".join(meta_parts)

        card_html = f"""
        <div class="rec-card">
            <div class="rec-title">{title}</div>
            <div class="rec-meta">{meta_text}</div>
            {score_html}
        </div>
        """

        st.markdown(card_html, unsafe_allow_html=True)


def sentiment_class(label):
    if label == "Positive":
        return "pill-positive"
    if label == "Negative":
        return "pill-negative"
    if label == "Mixed":
        return "pill-mixed"
    return "pill-neutral"


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
        "Meal Explorer",
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
# Page: Meal Explorer
# ============================================================

elif page == "Meal Explorer":
    hero(
        "Meal Intelligence Center",
        "A professional meal-level decision panel that combines sentiment, order behavior, association rules, PageRank influence, and final recommendation scoring."
    )

    meal_profile = get_meal_profile_table()

    if meal_profile.empty:
        st.markdown(
            """
            <div class="warning-box">
            Meal profile could not be built. Please check transactions, meals, sentiment, and association rule files.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        all_meals = sorted(meal_profile["meal_name"].dropna().unique().tolist())

        st.markdown('<div class="premium-panel">', unsafe_allow_html=True)
        st.markdown("### Search and Select Meal")

        col_search, col_select, col_topn = st.columns([1.35, 1.55, 0.75])

        with col_search:
            typed_meal = st.text_input(
                "Type a meal name",
                placeholder="Example: Mint Lemonade, Pepperoni Pizza, Iced Latte"
            )

        matched_meal = find_best_meal_match(typed_meal, all_meals) if typed_meal.strip() else None
        default_meal = matched_meal if matched_meal else ("Mint Lemonade" if "Mint Lemonade" in all_meals else all_meals[0])

        with col_select:
            selected_meal = st.selectbox(
                "Confirmed meal",
                all_meals,
                index=all_meals.index(default_meal)
            )

        with col_topn:
            top_n = st.slider("Results", 5, 20, 10)

        if typed_meal.strip() and matched_meal is None:
            st.warning("No exact or partial meal match was found. Please choose a meal from the dropdown.")
        elif typed_meal.strip() and matched_meal:
            st.success(f"Matched meal: {matched_meal}")

        st.markdown('</div>', unsafe_allow_html=True)

        selected_info = meal_profile[meal_profile["meal_name"] == selected_meal].iloc[0]
        sentiment_label = selected_info["sentiment_label"]
        pill_class = sentiment_class(sentiment_label)

        st.markdown(
            f"""
            <div class="meal-hero">
                <div class="meal-title">{selected_meal}</div>
                <div class="meal-subtitle">
                    Category: <b>{selected_info['category']}</b> |
                    Orders: <b>{int(selected_info['order_count']):,}</b> |
                    Reviews: <b>{int(selected_info['review_count']):,}</b> |
                    PageRank: <b>{selected_info['pagerank_score']:.5f}</b> |
                    Recommendation Score: <b>{selected_info['recommendation_score']:.2f}</b>
                </div>
                <span class="pill {pill_class}">{sentiment_label} Sentiment</span>
                <span class="pill pill-neutral">Popularity Rank #{int(selected_info['popularity_rank'])}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            metric_card("Sentiment", selected_info["sentiment_label"], sentiment_badge(selected_info["sentiment_label"]))
        with col2:
            metric_card("Avg Sentiment", f"{selected_info['avg_sentiment_score']:.2f}", "Range: -1 to 1")
        with col3:
            metric_card("Order Count", f"{int(selected_info['order_count']):,}", "Transactions")
        with col4:
            metric_card("PageRank", f"{selected_info['pagerank_score']:.5f}", f"Rank: {int(selected_info['pagerank_rank']) if not pd.isna(selected_info.get('pagerank_rank', None)) else 'N/A'}")
        with col5:
            metric_card("Final Score", f"{selected_info['recommendation_score']:.2f}", "Recommendation strength")

        st.markdown(
            f"""
            <div class="insight-box">
            <b>Decision Insight:</b> <b>{selected_meal}</b> is classified as <b>{selected_info['sentiment_label']}</b>
            based on its average customer sentiment score. The recommendation system combines this sentiment with
            PageRank influence from the co-ordering network, making the final score more balanced than popularity alone.
            </div>
            """,
            unsafe_allow_html=True
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview",
            "Sentiment Evidence",
            "Co-order Patterns",
            "Association Recommendations",
            "Final Recommendations"
        ])

        with tab1:
            st.markdown("### Meal Performance Overview")

            overview_df = pd.DataFrame({
                "Metric": [
                    "Meal Name",
                    "Category",
                    "Sentiment Label",
                    "Average Sentiment Score",
                    "Review Count",
                    "Order Count",
                    "Popularity Rank",
                    "PageRank Score",
                    "PageRank Degree",
                    "Weighted Degree",
                    "Final Recommendation Score"
                ],
                "Value": [
                    selected_meal,
                    selected_info["category"],
                    selected_info["sentiment_label"],
                    f"{selected_info['avg_sentiment_score']:.3f}",
                    f"{int(selected_info['review_count']):,}",
                    f"{int(selected_info['order_count']):,}",
                    f"{int(selected_info['popularity_rank'])}",
                    f"{selected_info['pagerank_score']:.6f}",
                    f"{int(selected_info['degree'])}",
                    f"{int(selected_info['weighted_degree'])}",
                    f"{selected_info['recommendation_score']:.3f}"
                ]
            })

            st.dataframe(overview_df, use_container_width=True, hide_index=True)

            profile_slice = pd.DataFrame({
                "Score Component": ["Sentiment", "PageRank", "Recommendation"],
                "Value": [
                    (selected_info["avg_sentiment_score"] + 1) / 2,
                    selected_info["pagerank_score"] / max(meal_profile["pagerank_score"].max(), 1e-9),
                    selected_info["recommendation_score"]
                ]
            })

            fig = px.bar(
                profile_slice,
                x="Score Component",
                y="Value",
                color="Score Component",
                title=f"Normalized Score Components for {selected_meal}",
                text=profile_slice["Value"].round(3)
            )
            fig.update_yaxes(range=[0, 1])
            fig = make_plotly_layout(fig, 440)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### Sentiment Evidence")

            sentiment_row = meal_profile[meal_profile["meal_name"] == selected_meal].copy()

            fig = px.bar(
                sentiment_row,
                x="meal_name",
                y="avg_sentiment_score",
                color="sentiment_label",
                hover_data=["review_count", "category"],
                title=f"Average Sentiment Score for {selected_meal}",
                color_discrete_map={
                    "Positive": "#22c55e",
                    "Negative": "#ef4444",
                    "Mixed": "#f97316",
                    "No Reviews": "#64748b"
                }
            )
            fig.update_yaxes(range=[-1, 1])
            fig = make_plotly_layout(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

            if sentiment_merged_df.empty:
                st.info("No detailed sentiment data is available.")
            else:
                sample_reviews = sentiment_merged_df[
                    sentiment_merged_df["meal_name"] == selected_meal
                ][["review_id", "review_text", "sentiment", "sentiment_score"]].head(10)

                if sample_reviews.empty:
                    st.info("No review text found for this meal.")
                else:
                    st.subheader("Review Samples")
                    for _, review in sample_reviews.iterrows():
                        review_label = str(review["sentiment"]).title()
                        score = review["sentiment_score"]
                        text = str(review["review_text"])
                        st.markdown(
                            f"""
                            <div class="rec-card">
                                <div class="rec-title">{review_label} | Score: {score}</div>
                                <div class="rec-meta">{text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        with tab3:
            st.markdown("### Co-order Patterns")

            co_ordered = get_co_ordered_meals(selected_meal, top_n=top_n)

            if co_ordered.empty:
                st.info("No co-ordered meals were found for this selection.")
            else:
                fig = px.bar(
                    co_ordered.sort_values("Co-order Count"),
                    x="Co-order Count",
                    y="Recommended Meal",
                    orientation="h",
                    color="Co-order Count",
                    color_continuous_scale="Oranges",
                    title=f"Meals Most Frequently Ordered with {selected_meal}"
                )
                fig = make_plotly_layout(fig, 540)
                st.plotly_chart(fig, use_container_width=True)

                render_recommendation_cards(
                    co_ordered,
                    title_col="Recommended Meal",
                    score_col=None,
                    meta_cols=["Co-order Count"],
                    limit=top_n
                )

        with tab4:
            st.markdown("### Association Rule Recommendations")

            rule_recs = get_rule_based_recommendations(selected_meal, top_n=top_n)

            if rule_recs.empty:
                st.info("No direct association rules were found for this meal. Use co-ordered or final recommendations instead.")
            else:
                fig = px.bar(
                    rule_recs.sort_values("confidence"),
                    x="confidence",
                    y="Recommended Meal",
                    orientation="h",
                    color="lift",
                    color_continuous_scale="Turbo",
                    hover_data=["support", "lift", "reliability_score"],
                    title=f"Association Rule Recommendations for {selected_meal}"
                )
                fig = make_plotly_layout(fig, 560)
                st.plotly_chart(fig, use_container_width=True)

                render_recommendation_cards(
                    rule_recs,
                    title_col="Recommended Meal",
                    score_col="confidence",
                    meta_cols=["confidence", "lift", "support", "reliability_score"],
                    limit=top_n
                )

                with st.expander("Detailed Association Rule Table"):
                    st.dataframe(
                        rule_recs[["Recommended Meal", "support", "confidence", "lift", "reliability_score", "Rule"]],
                        use_container_width=True
                    )

        with tab5:
            st.markdown("### Final Recommendation Output")

            hybrid_recs = get_hybrid_recommendations(selected_meal, top_n=top_n)

            if hybrid_recs.empty:
                st.info("No final recommendations are available.")
            else:
                fig = px.bar(
                    hybrid_recs.sort_values("recommendation_score"),
                    x="recommendation_score",
                    y="meal_name",
                    color="category",
                    orientation="h",
                    hover_data=["pagerank_score", "avg_sentiment_score", "review_count"],
                    title=f"Final Recommendations Related to {selected_meal}"
                )
                fig = make_plotly_layout(fig, 580)
                st.plotly_chart(fig, use_container_width=True)

                cards_df = hybrid_recs.rename(columns={"meal_name": "Recommended Meal"})
                render_recommendation_cards(
                    cards_df,
                    title_col="Recommended Meal",
                    score_col="recommendation_score",
                    meta_cols=["category", "recommendation_score", "pagerank_score", "avg_sentiment_score", "review_count"],
                    limit=top_n
                )

                with st.expander("Detailed Final Recommendation Table"):
                    st.dataframe(
                        hybrid_recs[
                            [
                                "meal_name",
                                "category",
                                "recommendation_score",
                                "pagerank_score",
                                "avg_sentiment_score",
                                "review_count"
                            ]
                        ],
                        use_container_width=True
                    )

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
