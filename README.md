# Food Delivery Pattern Analysis

A Data Mining and Web Mining project that analyzes food delivery ordering behavior using transaction mining, graph-based ranking, and sentiment analysis.

The project aims to discover popular meals, frequently co-ordered meal combinations, influential meals in a co-ordering network, and customer sentiment patterns. These insights are then combined to support a smarter food recommendation system.

---

## Project Objectives

The main objectives of this project are to:

- Collect and preprocess food delivery-style data.
- Build a transaction dataset where each row represents meals ordered together.
- Apply association rule mining to discover frequent meal combinations.
- Use PageRank and link analysis to identify influential meals in the co-ordering network.
- Apply BERT-based sentiment analysis on customer reviews.
- Create visualizations and reporting outputs to explain the results clearly.
- Combine PageRank and sentiment insights into a final recommendation score.

---

## Team Tasks

### Person 1 — Data Collection & Preprocessing

Responsible for collecting and preparing the dataset.

Main responsibilities:

- Collect meal/product data from public food sources.
- Generate realistic food delivery-style data when direct scraping is limited.
- Extract meal names, categories, prices, reviews, and order records.
- Clean missing values, duplicates, and formatting issues.
- Build the transaction dataset.
- Generate one-hot encoded transaction data for association rule mining.

Main file:

```text
scraping.py
```

Main outputs:

```text
data/output/processed/meals_cleaned.csv
data/output/processed/reviews.csv
data/output/transactions/transactions.csv
data/output/transactions/transactions_ohe.csv
```

---

### Person 2 — Association Rule Mining

Responsible for discovering frequently ordered meal combinations.

Main responsibilities:

- Apply Apriori / FP-Growth on the transaction dataset.
- Discover frequent itemsets.
- Generate association rules.
- Evaluate rules using support, confidence, and lift.
- Identify meal combinations useful for bundle offers and recommendations.

Main file:

```text
Apriory.ipynb
```

Main generated outputs:

```text
data/output/processed/association_rules.csv
data/output/processed/top20_association_rules.csv
```

---

### Person 3 — Link Analysis / PageRank

Responsible for graph-based analysis of meal co-ordering relationships.

Main responsibilities:

- Build a graph where each node represents a meal.
- Add edges between meals that appear together in the same order.
- Use edge weights to represent co-ordering frequency.
- Apply PageRank to rank influential meals.
- Apply HITS analysis for hub and authority insights.
- Visualize the meal co-ordering network.

Main file:

```text
Task3_PageRank_HITS.ipynb
```

Main generated outputs:

```text
data/output/processed/pagerank_results.csv
data/output/processed/top20_pagerank.csv
```

---

### Person 4 — Visualization & Reporting

Responsible for building final visualizations and summarizing the results.

Main responsibilities:

- Create exploratory data analysis visualizations.
- Show popular meals and categories.
- Visualize association rules.
- Visualize PageRank and co-ordering networks.
- Visualize sentiment analysis results.
- Combine PageRank and sentiment into a final recommendation score.
- Prepare report-ready charts and final insights.

Main file:

```text
Task4_Visualization_Reporting.ipynb
```

Main generated figures:

```text
data/output/figures/01_eda_dashboard.png
data/output/figures/02_top15_ordered_meals.png
data/output/figures/03_daily_orders_over_time.png
data/output/figures/04_temporal_patterns.png
data/output/figures/05_top10_association_rules_by_lift.png
data/output/figures/06_association_rules_scatter.png
data/output/figures/07_association_rules_network.png
data/output/figures/08_top10_reliable_association_rules.png
data/output/figures/09_top20_pagerank_meals.png
data/output/figures/10_pagerank_by_category.png
data/output/figures/11_pagerank_vs_degree.png
data/output/figures/12_pagerank_network_graph.png
data/output/figures/13_sentiment_distribution.png
data/output/figures/14_sentiment_score_distribution.png
data/output/figures/15_avg_sentiment_by_category.png
data/output/figures/16_top15_meals_by_sentiment.png
data/output/figures/17_final_recommendation_score.png
```

---

### Person 5 — BERT-based Sentiment Analysis

Responsible for analyzing customer reviews using a BERT-based sentiment model.

Main responsibilities:

- Use a pre-trained sentiment model on customer reviews.
- Fine-tune a BERT classifier.
- Classify reviews into positive and negative sentiment.
- Generate sentiment scores.
- Save sentiment output for recommendation enhancement.

Main file:

```text
BERT_Algorithm.ipynb
```

Main output:

```text
data/output/processed/reviews_with_sentiment.csv
```

---

## Dataset Description

The project uses food delivery-style data consisting of meals, customer reviews, and order transactions.

### Important Note About Data Source

Due to scraping restrictions and anti-bot protection on real food delivery platforms, the project uses a public food data source and supplements it with realistic synthetic food delivery data.

The dataset includes:

- Meal names
- Categories
- Restaurants
- Prices
- Ratings
- Reviews
- Order records
- Items ordered together
- Delivery information
- Payment methods

---

## Main Data Files

```text
data/output/processed/meals_cleaned.csv
```

Contains cleaned meal information such as meal ID, meal name, category, restaurant, price, rating, and other attributes.

```text
data/output/processed/reviews.csv
```

Contains generated customer reviews with ratings and review text.

```text
data/output/processed/reviews_with_sentiment.csv
```

Contains review sentiment predictions and sentiment scores.

```text
data/output/transactions/transactions.csv
```

Contains order-level transaction data. Each row represents one order, and the item_names column contains meals ordered together.

```text
data/output/transactions/transactions_ohe.csv
```

Contains one-hot encoded transaction data for association rule mining.

---

## Project Structure

```text
Food-Delivery-Pattern-Analysis/
|
├── README.md
├── scraping.py
├── Apriory.ipynb
├── Task3_PageRank_HITS.ipynb
├── BERT_Algorithm.ipynb
├── Task4_Visualization_Reporting.ipynb
|
└── data/
    └── output/
        ├── processed/
        │   ├── meals_cleaned.csv
        │   ├── meals_cleaned.json
        │   ├── reviews.csv
        │   ├── reviews_with_sentiment.csv
        │   ├── association_rules.csv
        │   ├── top20_association_rules.csv
        │   ├── pagerank_results.csv
        │   └── top20_pagerank.csv
        |
        ├── transactions/
        │   ├── transactions.csv
        │   └── transactions_ohe.csv
        |
        └── figures/
            ├── 01_eda_dashboard.png
            ├── 02_top15_ordered_meals.png
            ├── 05_top10_association_rules_by_lift.png
            ├── 07_association_rules_network.png
            ├── 09_top20_pagerank_meals.png
            ├── 12_pagerank_network_graph.png
            ├── 13_sentiment_distribution.png
            └── 17_final_recommendation_score.png
```

---

## Methods Used

### 1. Data Collection and Preprocessing

The data pipeline collects food product information, cleans it, removes duplicates, handles missing values, and generates realistic food delivery transactions and reviews.

The final transaction dataset is structured so that each row represents one customer order.

Example:

```text
Order ID: ORD00001
Items Ordered Together: Veggie Supreme Pizza, Umm Ali
Categories: Pizza, Desserts
```

---

### 2. Association Rule Mining

Association rule mining is used to discover meals that are frequently ordered together.

Metrics used:

- Support: How often a meal combination appears in all transactions.
- Confidence: How likely the consequent meal is ordered when the antecedent meal is ordered.
- Lift: How much more likely the meals are ordered together compared to random chance.

Example rule:

```text
Truffle Burger, Club Sandwich -> Spaghetti Bolognese, Iced Latte
```

This means customers who ordered the first set of meals also tended to order the second set.

A reliability score was also calculated:

```text
Reliability Score = Lift x Confidence x Support
```

This helps reduce overreliance on high-lift rules with very low support.

---

### 3. PageRank / Link Analysis

A co-ordering graph was built where:

- Each node represents a meal.
- Each edge represents two meals ordered together.
- Edge weight represents the number of times the meals appeared together.

PageRank was then applied to identify influential meals in the ordering network.

Graph summary:

```text
Nodes: 355 meals
Edges: 5,890 co-ordering relationships
Most influential meal: Mint Lemonade
```

PageRank helps identify meals that are not only popular, but also connected to other important meals.

---

### 4. Sentiment Analysis

BERT-based sentiment analysis was applied to customer review text.

The sentiment output contains:

- Positive sentiment
- Negative sentiment
- Sentiment score

The sentiment analysis helps ensure that the recommendation system does not only recommend popular meals, but also meals that customers liked.

---

### 5. Final Recommendation Score

The final recommendation score combines:

- PageRank influence
- Sentiment score

Formula concept:

```text
Final Recommendation Score = 0.7 x Normalized PageRank + 0.3 x Normalized Sentiment
```

This produces recommendations that balance network influence and customer satisfaction.

Top recommended meal:

```text
Mint Lemonade
```

---

## Key Results

### Dataset Overview

```text
Total orders analyzed: 2,000
Unique customers: 487
Unique restaurants: 20
Average items per order: 2.83
Average order rating: 3.76 / 5
```

### Association Rule Mining

```text
Total association rules generated: 620
Highest-lift rule:
Truffle Burger, Club Sandwich -> Spaghetti Bolognese, Iced Latte

Highest lift score: 200.00
```

### PageRank Analysis

```text
Meals represented as graph nodes: 355
Co-ordering relationships represented as edges: 5,890
Most influential meal: Mint Lemonade
PageRank score: 0.00782
```

### Sentiment Analysis

```text
Reviews analyzed: 1,835
Most frequent sentiment: POSITIVE
Highest average sentiment category: Salads
```

### Final Recommendation

```text
Top recommended meal: Mint Lemonade
Final recommendation score: 1.00
```

---

## Main Visualizations

The project includes several report-ready visualizations:

- EDA dashboard
- Top ordered meals
- Orders over time
- Orders by hour and day
- Association rules by lift
- Association rule network graph
- Reliable association rules
- Top PageRank meals
- PageRank by category
- PageRank vs degree
- Co-ordering network graph
- Sentiment distribution
- Average sentiment by category
- Top meals by sentiment
- Final recommendation score

All generated charts are saved in:

```text
data/output/figures/
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Karim405/Food-Delivery-Pattern-Analysis.git
cd Food-Delivery-Pattern-Analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn networkx mlxtend scikit-learn transformers torch beautifulsoup4 requests jupyter
```

If a requirements.txt file is added, use:

```bash
pip install -r requirements.txt
```

### 4. Run the data pipeline

```bash
python scraping.py
```

This generates the main processed datasets.

### 5. Run the notebooks

Recommended order:

```text
1. Apriory.ipynb
2. Task3_PageRank_HITS.ipynb
3. BERT_Algorithm.ipynb
4. Task4_Visualization_Reporting.ipynb
```

The final notebook for visual reporting is:

```text
Task4_Visualization_Reporting.ipynb
```

---

## Requirements

Main Python libraries used:

```text
pandas
numpy
matplotlib
seaborn
networkx
mlxtend
scikit-learn
transformers
torch
beautifulsoup4
requests
jupyter
```

---

## Limitations

This project has some limitations:

1. The data is not scraped directly from Talabat or Uber Eats due to scraping restrictions and anti-bot protection.
2. The dataset includes synthetic food delivery transactions and reviews.
3. Some association rules have very low support, so high-lift rules should be treated as exploratory patterns.
4. Sentiment analysis output is binary positive/negative, with no neutral class in the final output.
5. PageRank is calculated at the meal ID level, so identical meal names from different restaurants may appear as separate nodes.
6. Recommendation scores are based on available transaction and sentiment data, not real-time user behavior.

---

## Conclusion

This project demonstrates how multiple data mining techniques can be combined to build a food recommendation system.

Association rule mining identifies meals that are frequently ordered together. PageRank identifies influential meals in the co-ordering network. Sentiment analysis captures customer satisfaction from reviews. Finally, the recommendation score combines meal influence and sentiment to generate stronger recommendations.

The final system provides a more balanced recommendation approach than relying only on meal popularity or order frequency.

---

## Authors

Data Mining & Web Mining Project

Team roles:

```text
Person 1: Data Collection & Preprocessing
Person 2: Association Rule Mining
Person 3: Link Analysis / PageRank
Person 4: Visualization & Reporting
Person 5: BERT-based Sentiment Analysis
```
