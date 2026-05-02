"""
=============================================================================
Food Delivery Data Scraper
Project: Data Mining & Web Mining - Person 1 (Data Collection)
Description: Scrapes meal data, categories, prices, and reviews from
             food delivery platforms. Falls back to synthetic realistic
             data if scraping is blocked (anti-bot protection).
=============================================================================
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import time
import random
import re
import os
import logging
from datetime import datetime, timedelta
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── Output Directories ───────────────────────────────────────────────────────
os.makedirs("output/raw", exist_ok=True)
os.makedirs("output/processed", exist_ok=True)
os.makedirs("output/transactions", exist_ok=True)


# =============================================================================
# SECTION 1: HTTP SESSION WITH ROTATION
# =============================================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
]

def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    })
    return session

def polite_get(session, url, delay=(2, 5), retries=3):
    for attempt in range(retries):
        time.sleep(random.uniform(*delay))
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            if attempt == retries - 1:
                return None

# =============================================================================
# SECTION 2: REAL SCRAPER — OpenFoodFacts (Open API, No Login Required)
# =============================================================================

def scrape_openfoodfacts(max_pages=10):
    """
    Scrapes real food product data from OpenFoodFacts public API.
    Returns a list of meal/product dicts.
    """
    log.info("=== Scraping OpenFoodFacts API ===")
    session = get_session()
    meals = []

    categories_to_fetch = [
        "pizzas", "burgers", "sandwiches", "salads",
        "pasta", "sushi", "desserts", "beverages",
        "wraps", "soups"
    ]

    for category in categories_to_fetch:
        for page in range(1, max_pages + 1):
            url = (
                f"https://world.openfoodfacts.org/cgi/search.pl"
                f"?search_terms=&tagtype_0=categories&tag_contains_0=contains"
                f"&tag_0={category}&page_size=50&page={page}&action=process&json=1"
            )
            resp = polite_get(session, url, delay=(1, 3))
            if not resp:
                break

            data = resp.json()
            products = data.get("products", [])
            if not products:
                break

            for p in products:
                name = p.get("product_name", "").strip()
                if (
                        not name
                        or len(name) < 4
                        or any(x in name.lower() for x in ["pack", "sauce", "powder", "frozen", "chips", "biscuit"])
                    ):
                        continue

                meals.append({
                    "meal_id": p.get("id", f"R{len(meals)+1:05d}"),
                    "meal_name": name,
                    "category": category.capitalize(),
                    "restaurant": random.choice(RESTAURANTS),
                    "brand": p.get("brands", "Unknown"),
                    "price_egp": round(random.uniform(25, 150), 0),
                    "description": p.get("ingredients_text", "")[:200],
                    "nutrition_grade": p.get("nutrition_grade_fr", ""),
                    "energy_kcal": p.get("nutriments", {}).get("energy-kcal_100g", None),
                    "fat_g": p.get("nutriments", {}).get("fat_100g", None),
                    "carbs_g": p.get("nutriments", {}).get("carbohydrates_100g", None),
                    "protein_g": p.get("nutriments", {}).get("proteins_100g", None),
                    "image_url": p.get("image_url", ""),
                    "rating": round(random.uniform(3.2, 4.8), 1),
                    "num_ratings": random.randint(20, 1500),
                    "prep_time_min": random.randint(15, 50),
                    "is_available": True,
                    "source": "OpenFoodFacts",
                })

            log.info(f"  {category} page {page}: fetched {len(products)} items")

        log.info(f"Category '{category}' done. Total so far: {len(meals)}")

    log.info(f"OpenFoodFacts scraping complete: {len(meals)} raw meals")
    return meals


# =============================================================================
# SECTION 3: SYNTHETIC REALISTIC DATA GENERATOR
# (Used when real scraping is blocked or supplemented for full dataset)
# =============================================================================

MEAL_TEMPLATES = {
    "Burgers": [
        ("Classic Beef Burger", 45, "Juicy beef patty with lettuce, tomato, and special sauce"),
        ("Double Smash Burger", 65, "Two smashed beef patties with American cheese"),
        ("Crispy Chicken Burger", 55, "Crispy fried chicken fillet with coleslaw"),
        ("BBQ Bacon Burger", 70, "Beef patty topped with crispy bacon and BBQ sauce"),
        ("Mushroom Swiss Burger", 60, "Beef patty with sautéed mushrooms and Swiss cheese"),
        ("Spicy Jalapeño Burger", 58, "Fiery beef burger with jalapeños and sriracha mayo"),
        ("Veggie Garden Burger", 50, "Plant-based patty with avocado and fresh greens"),
        ("Truffle Burger", 85, "Premium beef with truffle mayo and caramelized onions"),
    ],
    "Pizza": [
        ("Margherita Pizza", 80, "Classic tomato sauce, mozzarella, fresh basil"),
        ("Pepperoni Pizza", 95, "Loaded with spicy pepperoni and mozzarella"),
        ("BBQ Chicken Pizza", 100, "Grilled chicken, BBQ sauce, red onions, mozzarella"),
        ("Four Cheese Pizza", 110, "Mozzarella, cheddar, parmesan, and gorgonzola"),
        ("Meat Lovers Pizza", 120, "Beef, chicken, sausage, and pepperoni"),
        ("Veggie Supreme Pizza", 90, "Bell peppers, olives, mushrooms, onions"),
        ("Hawaiian Pizza", 95, "Ham, pineapple, mozzarella on tomato base"),
        ("Truffle Mushroom Pizza", 125, "Wild mushrooms, truffle oil, parmesan, arugula"),
    ],
    "Shawarma": [
        ("Chicken Shawarma Wrap", 35, "Marinated chicken with garlic sauce and pickles"),
        ("Meat Shawarma Wrap", 40, "Mixed meat with tahini and fresh vegetables"),
        ("Falafel Wrap", 30, "Crispy falafel with hummus and salad"),
        ("Shawarma Plate", 55, "Shawarma served with rice, salad, and sauce"),
        ("Mixed Grill Wrap", 65, "Chicken and meat shawarma combined"),
        ("Cheese Shawarma", 42, "Chicken shawarma with melted cheese"),
    ],
    "Sushi": [
        ("Salmon Nigiri (2pcs)", 45, "Fresh Atlantic salmon over seasoned rice"),
        ("California Roll (8pcs)", 60, "Crab, avocado, cucumber, sesame"),
        ("Dragon Roll (8pcs)", 90, "Shrimp tempura inside, avocado on top"),
        ("Spicy Tuna Roll (8pcs)", 75, "Spicy tuna with cucumber and sriracha"),
        ("Rainbow Roll (8pcs)", 95, "California roll topped with assorted sashimi"),
        ("Veggie Roll (8pcs)", 50, "Avocado, cucumber, pickled radish"),
        ("Sashimi Platter (12pcs)", 120, "Chef's selection of fresh fish sashimi"),
    ],
    "Pasta": [
        ("Spaghetti Bolognese", 70, "Slow-cooked beef ragù with spaghetti"),
        ("Penne Arrabiata", 60, "Spicy tomato sauce with garlic and chili"),
        ("Fettuccine Alfredo", 75, "Creamy parmesan sauce with fettuccine"),
        ("Pasta Carbonara", 80, "Eggs, pancetta, parmesan, black pepper"),
        ("Pesto Pasta", 68, "Fresh basil pesto with cherry tomatoes"),
        ("Seafood Linguine", 95, "Shrimp, mussels, calamari in white wine sauce"),
        ("Lasagna", 85, "Layered pasta with beef ragù and béchamel"),
    ],
    "Salads": [
        ("Caesar Salad", 45, "Romaine, croutons, parmesan, Caesar dressing"),
        ("Greek Salad", 40, "Tomatoes, cucumber, olives, feta, oregano"),
        ("Fattoush Salad", 35, "Fresh veggies, crispy bread, pomegranate dressing"),
        ("Quinoa Power Bowl", 60, "Quinoa, roasted veggies, avocado, tahini"),
        ("Grilled Chicken Salad", 65, "Mixed greens with grilled chicken strips"),
        ("Caprese Salad", 50, "Fresh tomatoes, mozzarella, basil, balsamic"),
    ],
    "Desserts": [
        ("Chocolate Lava Cake", 40, "Warm chocolate cake with molten center"),
        ("Cheesecake Slice", 35, "New York style with berry compote"),
        ("Crème Brûlée", 42, "Classic French vanilla custard with caramel top"),
        ("Tiramisu", 38, "Italian dessert with mascarpone and espresso"),
        ("Kunafa", 30, "Traditional cheese dessert with orange blossom syrup"),
        ("Umm Ali", 28, "Egyptian bread pudding with nuts and cream"),
        ("Waffles & Ice Cream", 45, "Belgian waffle with vanilla ice cream"),
        ("Mango Panna Cotta", 38, "Silky Italian dessert with mango coulis"),
    ],
    "Beverages": [
        ("Fresh Orange Juice", 20, "Freshly squeezed oranges"),
        ("Mango Smoothie", 25, "Blended mango with milk and honey"),
        ("Iced Latte", 28, "Espresso over ice with cold milk"),
        ("Mint Lemonade", 18, "Fresh lemon juice with mint and sugar"),
        ("Watermelon Juice", 20, "Freshly blended watermelon"),
        ("Avocado Smoothie", 32, "Creamy avocado blended with milk"),
        ("Strawberry Milkshake", 30, "Thick strawberry shake with vanilla ice cream"),
        ("Hot Chocolate", 22, "Rich Belgian chocolate with steamed milk"),
    ],
    "Sandwiches": [
        ("Club Sandwich", 45, "Triple-decker with chicken, bacon, lettuce, tomato"),
        ("Grilled Cheese", 30, "Melted mixed cheese on toasted sourdough"),
        ("Tuna Melt", 40, "Tuna salad with melted cheddar on rye"),
        ("BLT Sandwich", 38, "Bacon, lettuce, tomato on toasted white bread"),
        ("Egg & Avocado Toast", 35, "Poached egg on avocado smash, sourdough"),
        ("Pastrami Sandwich", 55, "Hot pastrami with mustard and pickles on rye"),
    ],
    "Soups": [
        ("Tomato Basil Soup", 30, "Creamy roasted tomato with fresh basil"),
        ("French Onion Soup", 38, "Caramelized onion with gruyère crouton"),
        ("Chicken Noodle Soup", 35, "Classic chicken broth with vegetables"),
        ("Lentil Soup", 28, "Egyptian red lentil with cumin and lemon"),
        ("Mushroom Cream Soup", 40, "Mixed mushroom with cream and thyme"),
        ("Seafood Chowder", 55, "Creamy chowder with shrimp, fish, potatoes"),
    ],
}

REVIEW_TEMPLATES = {
    "positive": [
        "Amazing food! Delivered hot and fresh. Will definitely order again.",
        "Best {meal} I've ever had. The flavor was incredible!",
        "Absolutely delicious. Generous portions and great value for money.",
        "Fresh ingredients and perfectly cooked. Highly recommended!",
        "Loved every bite. The packaging was also excellent.",
        "Super fast delivery and the food was outstanding. 5 stars!",
        "The quality is consistently great. My go-to order every week.",
        "Perfect seasoning and amazing presentation. Worth every penny.",
        "Exceeded my expectations. This is exactly what comfort food should taste like.",
        "Fantastic taste and quick delivery. Family loved it!",
    ],
    "neutral": [
        "Good food overall but delivery took longer than expected.",
        "Decent taste, nothing extraordinary but satisfying.",
        "The {meal} was okay. Expected a bit more flavor.",
        "Average portion size for the price. Food was warm but not hot.",
        "Pretty good, I've had better but wouldn't say no to ordering again.",
        "Food arrived on time but was slightly cold. Taste was fine.",
    ],
    "negative": [
        "Disappointed with the quality. Not what I expected.",
        "The {meal} was cold on arrival. Very frustrating.",
        "Portion was too small for the price. Won't order again.",
        "The taste was bland and the delivery was late.",
        "Missing items from my order. Customer service was unhelpful.",
        "Packaging was damaged and food quality was below average.",
    ],
}

RESTAURANTS = [
    "The Grill House", "Burger Factory", "Pizza Palace", "Shawarma Station",
    "Sushi World", "Pasta Corner", "Fresh & Fit", "Desert Rose Kitchen",
    "The Hungry Bear", "Flavor Street", "Golden Fork", "Urban Bites",
    "Mediterranean Nights", "El Classico", "Mama's Kitchen", "Street Eats",
    "The Foodie Hub", "Royal Kitchen", "Crave It", "Daily Bites",
]

def generate_synthetic_meals(n=2000):
    """Generate realistic synthetic meal dataset."""
    log.info(f"Generating {n} synthetic meals...")
    meals = []
    meal_id = 1

    for category, items in MEAL_TEMPLATES.items():
        for meal_name, base_price, description in items:
            for restaurant in random.sample(RESTAURANTS, k=random.randint(3, 8)):
                price_variation = round(base_price * random.uniform(0.85, 1.25), 0)
                meals.append({
                    "meal_id": f"M{meal_id:04d}",
                    "meal_name": meal_name,
                    "category": category,
                    "restaurant": restaurant,
                    "price_egp": float(price_variation),
                    "description": description,
                    "rating": round(random.uniform(3.0, 5.0), 1),
                    "num_ratings": random.randint(10, 2000),
                    "prep_time_min": random.randint(10, 45),
                    "calories": random.randint(200, 900),
                    "is_available": random.choices([True, False], weights=[90, 10])[0],
                    "source": "synthetic",
                })
                meal_id += 1
                if meal_id > n:
                    return meals[:n]

    return meals[:n]


def generate_reviews(meals_df, reviews_per_meal=5):
    """Generate synthetic customer reviews for meals."""
    log.info("Generating customer reviews...")
    reviews = []
    sentiments = ["positive", "positive", "positive", "neutral", "negative"]

    for _, meal in meals_df.iterrows():
        for i in range(reviews_per_meal):
            sentiment = random.choice(sentiments)
            template = random.choice(REVIEW_TEMPLATES[sentiment])
            review_text = template.replace("{meal}", meal["meal_name"])

            review_date = datetime.now() - timedelta(days=random.randint(1, 365))
            reviews.append({
                "review_id": f"R{len(reviews)+1:06d}",
                "meal_id": meal["meal_id"],
                "meal_name": meal["meal_name"],
                "restaurant": meal.get("restaurant", "Unknown"),
                "rating": {
                    "positive": random.uniform(4.0, 5.0),
                    "neutral": random.uniform(2.5, 3.9),
                    "negative": random.uniform(1.0, 2.4),
                }[sentiment],
                "review_text": review_text,
                "sentiment_label": sentiment,
                "review_date": review_date.strftime("%Y-%m-%d"),
                "helpful_votes": random.randint(0, 50),
            })

    reviews_df = pd.DataFrame(reviews)
    reviews_df["rating"] = reviews_df["rating"].round(1)
    return reviews_df


# =============================================================================
# SECTION 4: TRANSACTION DATASET BUILDER
# =============================================================================

def build_transactions(meals_df, n_transactions=5000):
    """
    Build a transaction dataset where each row represents
    items ordered together in one order session.
    """
    log.info(f"Building {n_transactions} transaction records...")

    # Weight meals by rating and num_ratings for realistic popularity
    if "rating" in meals_df.columns and "num_ratings" in meals_df.columns:
        weights = (meals_df["rating"] * meals_df["num_ratings"]).fillna(1)
        weights = weights / weights.sum()
    else:
        weights = None

    meal_names = meals_df["meal_id"].tolist()
    transactions = []

    for order_id in range(1, n_transactions + 1):
        # Typical order: 1-5 items
        n_items = random.choices([1, 2, 3, 4, 5], weights=[10, 35, 30, 15, 10])[0]
        n_items = min(n_items, len(meal_names))

        if weights is not None:
            chosen = meals_df.sample(n=n_items, weights=weights.values, replace=False)
        else:
            chosen = meals_df.sample(n=n_items, replace=False)

        order_date = datetime.now() - timedelta(days=random.randint(0, 180))
        hour = random.choices(
            range(24),
            weights=[1,1,1,1,1,1,1,2,5,8,10,12,12,10,8,7,10,12,12,10,8,5,3,2]
        )[0]

        transactions.append({
            "order_id": f"ORD{order_id:05d}",
            "customer_id": f"CUST{random.randint(1, 500):04d}",
            "restaurant": chosen.iloc[0].get("restaurant", "Unknown") if "restaurant" in chosen.columns else "Unknown",
            "items": ",".join(chosen["meal_id"].tolist()),
            "item_names": ",".join(chosen["meal_name"].tolist()),
            "item_categories": ",".join(chosen["category"].tolist()),
            "total_price": round(chosen["price_egp"].sum() if "price_egp" in chosen.columns else 0, 2),
            "order_date": order_date.strftime("%Y-%m-%d"),
            "order_hour": hour,
            "order_day": order_date.strftime("%A"),
            "n_items": n_items,
            "payment_method": random.choice(["Cash", "Credit Card", "Wallet"]),
            "delivery_time_min": random.randint(20, 60),
            "order_rating": round(random.uniform(2.5, 5.0), 1),
        })

    return pd.DataFrame(transactions)


def build_mlxtend_format(transactions_df, meals_df):
    """
    Create one-hot encoded transaction matrix for use with
    mlxtend (Apriori / FP-Growth) — Person 2's input.
    """
    log.info("Creating one-hot encoded transaction matrix for Association Mining...")

    all_meals = meals_df["meal_id"].tolist()
    rows = []

    for _, row in transactions_df.iterrows():
        items_in_order = set(row["items"].split(","))
        encoded = {meal: (meal in items_in_order) for meal in all_meals}
        rows.append(encoded)

    ohe_df = pd.DataFrame(rows, dtype=bool)
    return ohe_df


# =============================================================================
# SECTION 5: DATA CLEANING & PREPROCESSING
# =============================================================================

def clean_meals(meals_list):
    """Clean and deduplicate meals data."""
    log.info("Cleaning meals data...")
    df = pd.DataFrame(meals_list)

    # Drop rows with missing meal names
    df = df.dropna(subset=["meal_name"])
    df["meal_name"] = df["meal_name"].str.strip().str.title()

    # Remove duplicates based on meal_name + restaurant
    if "restaurant" in df.columns:
        df = df.drop_duplicates(subset=["meal_name", "restaurant"])
    else:
        df = df.drop_duplicates(subset=["meal_name"])

    # Fill missing prices
    if "price_egp" in df.columns:
        median_price = df["price_egp"].median()
        df["price_egp"] = df["price_egp"].fillna(median_price)

    # Ensure meal_id is string
    df["meal_id"] = df["meal_id"].astype(str)

    # Add scrape timestamp
    df["scraped_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.info(f"Cleaned meals: {len(df)} records")
    return df.reset_index(drop=True)


# =============================================================================
# SECTION 6: MAIN PIPELINE
# =============================================================================

def run_pipeline():
    log.info("=" * 60)
    log.info("FOOD DELIVERY DATA COLLECTION PIPELINE — STARTING")
    log.info("=" * 60)

    # ── Step 1: Collect Data ──────────────────────────────────────
    meals_list = []

    # Try real scraping first
    try:
        real_meals = scrape_openfoodfacts(max_pages=5)
        meals_list.extend(real_meals)
        log.info(f"Real meals scraped: {len(real_meals)}")
    except Exception as e:
        log.warning(f"Real scraping failed: {e}. Using synthetic data only.")

    # Add synthetic data (always include for richness)
    synthetic_meals = generate_synthetic_meals(n=500)
    meals_list.extend(synthetic_meals)
    log.info(f"Total raw meals: {len(meals_list)}")

    # ── Step 2: Clean & Preprocess ────────────────────────────────
    meals_df = clean_meals(meals_list)

    # ── Step 3: Generate Reviews ──────────────────────────────────
    # Only use synthetic-source meals for reviews (they have restaurant info)
    synthetic_df = meals_df[meals_df["source"] == "synthetic"].reset_index(drop=True)
    if len(synthetic_df) == 0:
        synthetic_df = meals_df.head(200)

    reviews_df = generate_reviews(synthetic_df, reviews_per_meal=5)

    # ── Step 4: Build Transactions ────────────────────────────────
    transactions_df = build_transactions(synthetic_df, n_transactions=2000)
    ohe_df = build_mlxtend_format(transactions_df, synthetic_df)

    # ── Step 5: Save All Outputs ──────────────────────────────────
    log.info("Saving all datasets...")

    # Raw meals (all sources)
    meals_df.to_csv("output/raw/meals_raw.csv", index=False)
    log.info("  Saved: output/raw/meals_raw.csv")

    # Processed meals
    meals_df.to_csv("output/processed/meals_cleaned.csv", index=False)
    meals_df.to_json("output/processed/meals_cleaned.json", orient="records", indent=2)
    log.info("  Saved: output/processed/meals_cleaned.csv + .json")

    # Reviews
    reviews_df.to_csv("output/processed/reviews.csv", index=False)
    log.info("  Saved: output/processed/reviews.csv")

    # Transactions
    transactions_df.to_csv("output/transactions/transactions.csv", index=False)
    log.info("  Saved: output/transactions/transactions.csv")

    # One-hot encoded (for Apriori / FP-Growth)
    ohe_df.to_csv("output/transactions/transactions_ohe.csv", index=False)
    log.info("  Saved: output/transactions/transactions_ohe.csv")

    # ── Step 6: Summary Report ────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Total Meals Collected    : {len(meals_df)}")
    print(f"  Unique Categories        : {meals_df['category'].nunique()}")
    print(f"  Total Reviews            : {len(reviews_df)}")
    print(f"  Total Transactions       : {len(transactions_df)}")
    print(f"  OHE Matrix Shape         : {ohe_df.shape}")
    print("\n  Category Distribution:")
    print(meals_df["category"].value_counts().to_string())
    print("\n  Top 10 Meals by Category:")
    if "rating" in meals_df.columns:
        top = meals_df.nlargest(10, "rating")[["meal_name", "category", "rating"]]
        print(top.to_string(index=False))
    print("\n  Files saved in ./output/")
    print("=" * 60)

    return meals_df, reviews_df, transactions_df, ohe_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    meals_df, reviews_df, transactions_df, ohe_df = run_pipeline()