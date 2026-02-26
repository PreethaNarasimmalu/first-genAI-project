"""Create a realistic sample restaurant dataset and rebuild parquet + ChromaDB.

Run this once to populate the system with working data:
    python -m scripts.create_sample_data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.indexing.indexer import index_dataframe

CLEAN_PARQUET = Path(__file__).resolve().parents[1] / "data" / "clean" / "restaurants.parquet"

# ---------------------------------------------------------------------------
# Sample data — realistic Zomato-style Bangalore restaurants
# Each dict represents one row (one listing per meal_type).
# ---------------------------------------------------------------------------

RESTAURANTS = [
    # ── Koramangala ─────────────────────────────────────────────────────────
    {"name": "Mainland China", "location": "Koramangala 5th Block", "cuisines": ["Chinese", "Thai", "Asian"],
     "approx_cost": 1200, "rate": 4.2, "votes": 3800, "rest_type": "Casual Dining",
     "online_order": True, "book_table": True, "dish_liked": "Dimsums, Kung Pao Chicken",
     "meal_type": "Buffet", "city": "Bangalore"},

    {"name": "Mainland China", "location": "Koramangala 5th Block", "cuisines": ["Chinese", "Thai", "Asian"],
     "approx_cost": 1200, "rate": 4.2, "votes": 3800, "rest_type": "Casual Dining",
     "online_order": True, "book_table": True, "dish_liked": "Dimsums, Kung Pao Chicken",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Empire Restaurant", "location": "Koramangala 1st Block", "cuisines": ["North Indian", "Biryani", "Kebabs"],
     "approx_cost": 400, "rate": 4.1, "votes": 12000, "rest_type": "Quick Bites",
     "online_order": True, "book_table": False, "dish_liked": "Chicken Biryani, Kebabs",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Empire Restaurant", "location": "Koramangala 1st Block", "cuisines": ["North Indian", "Biryani", "Kebabs"],
     "approx_cost": 400, "rate": 4.1, "votes": 12000, "rest_type": "Quick Bites",
     "online_order": True, "book_table": False, "dish_liked": "Chicken Biryani, Kebabs",
     "meal_type": "Delivery", "city": "Bangalore"},

    {"name": "Truffles", "location": "Koramangala 7th Block", "cuisines": ["American", "Continental", "Burgers"],
     "approx_cost": 600, "rate": 4.5, "votes": 8500, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Chicken Drifter, Truffle Burger",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Hole in the Wall Cafe", "location": "Koramangala 4th Block", "cuisines": ["Cafe", "Continental", "Pasta"],
     "approx_cost": 700, "rate": 4.3, "votes": 3200, "rest_type": "Café",
     "online_order": False, "book_table": False, "dish_liked": "Pasta, Coffee, Sandwiches",
     "meal_type": "Cafes", "city": "Bangalore"},

    {"name": "Chutney Chang", "location": "Koramangala 3rd Block", "cuisines": ["Chinese", "Asian", "Street Food"],
     "approx_cost": 300, "rate": 3.8, "votes": 1500, "rest_type": "Quick Bites",
     "online_order": True, "book_table": False, "dish_liked": "Momos, Chowmein",
     "meal_type": "Delivery", "city": "Bangalore"},

    {"name": "Fatty Bao", "location": "Koramangala 5th Block", "cuisines": ["Asian", "Chinese", "Japanese"],
     "approx_cost": 1400, "rate": 4.4, "votes": 4100, "rest_type": "Casual Dining",
     "online_order": False, "book_table": True, "dish_liked": "Bao, Ramen, Gyoza",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Big Brewsky", "location": "Koramangala", "cuisines": ["North Indian", "Continental", "Italian"],
     "approx_cost": 1600, "rate": 4.0, "votes": 5200, "rest_type": "Bar",
     "online_order": False, "book_table": True, "dish_liked": "Pizza, Grills, Cocktails",
     "meal_type": "Pubs and bars", "city": "Bangalore"},

    {"name": "The Permit Room", "location": "Koramangala", "cuisines": ["South Indian", "Kerala"],
     "approx_cost": 800, "rate": 4.6, "votes": 2800, "rest_type": "Casual Dining",
     "online_order": False, "book_table": True, "dish_liked": "Appam, Stew, Puttu",
     "meal_type": "Dine-out", "city": "Bangalore"},

    # ── Indiranagar ─────────────────────────────────────────────────────────
    {"name": "Toit Brewpub", "location": "Indiranagar", "cuisines": ["Continental", "Pizzas", "Burgers"],
     "approx_cost": 1500, "rate": 4.6, "votes": 14000, "rest_type": "Microbrewery",
     "online_order": False, "book_table": True, "dish_liked": "Pizzas, Craft Beer, Nachos",
     "meal_type": "Pubs and bars", "city": "Bangalore"},

    {"name": "1947 - The Restaurant", "location": "Indiranagar", "cuisines": ["North Indian", "Mughlai", "Biryani"],
     "approx_cost": 1000, "rate": 4.3, "votes": 3600, "rest_type": "Casual Dining",
     "online_order": True, "book_table": True, "dish_liked": "Dal Makhani, Biryani, Naan",
     "meal_type": "Buffet", "city": "Bangalore"},

    {"name": "1947 - The Restaurant", "location": "Indiranagar", "cuisines": ["North Indian", "Mughlai", "Biryani"],
     "approx_cost": 1000, "rate": 4.3, "votes": 3600, "rest_type": "Casual Dining",
     "online_order": True, "book_table": True, "dish_liked": "Dal Makhani, Biryani, Naan",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Brahmin's Coffee Bar", "location": "Indiranagar", "cuisines": ["South Indian", "Breakfast"],
     "approx_cost": 100, "rate": 4.4, "votes": 6700, "rest_type": "Quick Bites",
     "online_order": False, "book_table": False, "dish_liked": "Idli, Vada, Filter Coffee",
     "meal_type": "Cafes", "city": "Bangalore"},

    {"name": "The Black Pearl", "location": "Indiranagar", "cuisines": ["Seafood", "Coastal", "Goan"],
     "approx_cost": 1300, "rate": 4.1, "votes": 2400, "rest_type": "Casual Dining",
     "online_order": True, "book_table": True, "dish_liked": "Fish Curry, Prawn Masala",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Fenny's Lounge & Kitchen", "location": "Indiranagar", "cuisines": ["Goan", "Coastal", "Italian"],
     "approx_cost": 1100, "rate": 4.2, "votes": 1800, "rest_type": "Lounge",
     "online_order": False, "book_table": True, "dish_liked": "Prawn Balchao, Fish & Chips",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Kanti Sweets", "location": "Indiranagar", "cuisines": ["Desserts", "Sweets", "Mithai"],
     "approx_cost": 200, "rate": 4.5, "votes": 4300, "rest_type": "Sweet Shop",
     "online_order": True, "book_table": False, "dish_liked": "Mysore Pak, Halwa",
     "meal_type": "Desserts", "city": "Bangalore"},

    {"name": "Bob's Bar", "location": "Indiranagar", "cuisines": ["Continental", "Chinese", "Thai"],
     "approx_cost": 900, "rate": 3.9, "votes": 2100, "rest_type": "Bar",
     "online_order": False, "book_table": False, "dish_liked": "Cocktails, Grilled Fish",
     "meal_type": "Pubs and bars", "city": "Bangalore"},

    # ── BTM Layout ───────────────────────────────────────────────────────────
    {"name": "Nagarjuna Restaurant", "location": "BTM Layout", "cuisines": ["South Indian", "Andhra", "Biryani"],
     "approx_cost": 350, "rate": 4.0, "votes": 9800, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Biryani, Gongura Mutton, Rasam",
     "meal_type": "Buffet", "city": "Bangalore"},

    {"name": "Nagarjuna Restaurant", "location": "BTM Layout", "cuisines": ["South Indian", "Andhra", "Biryani"],
     "approx_cost": 350, "rate": 4.0, "votes": 9800, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Biryani, Gongura Mutton, Rasam",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Pizza Hut", "location": "BTM Layout", "cuisines": ["Pizzas", "Italian", "Fast Food"],
     "approx_cost": 700, "rate": 3.7, "votes": 5600, "rest_type": "Quick Bites",
     "online_order": True, "book_table": False, "dish_liked": "Pepperoni Pizza, Garlic Bread",
     "meal_type": "Delivery", "city": "Bangalore"},

    {"name": "Meghana Foods", "location": "BTM Layout", "cuisines": ["Biryani", "Andhra", "North Indian"],
     "approx_cost": 400, "rate": 4.2, "votes": 7800, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Chicken Biryani, Mutton Biryani",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Meghana Foods", "location": "BTM Layout", "cuisines": ["Biryani", "Andhra", "North Indian"],
     "approx_cost": 400, "rate": 4.2, "votes": 7800, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Chicken Biryani, Mutton Biryani",
     "meal_type": "Delivery", "city": "Bangalore"},

    {"name": "Hotel Decent", "location": "BTM Layout", "cuisines": ["South Indian", "North Indian"],
     "approx_cost": 250, "rate": 3.5, "votes": 2100, "rest_type": "Quick Bites",
     "online_order": False, "book_table": False, "dish_liked": "Dosa, Meals",
     "meal_type": "Dine-out", "city": "Bangalore"},

    # ── Whitefield ───────────────────────────────────────────────────────────
    {"name": "Grasshopper", "location": "Whitefield", "cuisines": ["Continental", "Italian", "Mediterranean"],
     "approx_cost": 2000, "rate": 4.7, "votes": 3100, "rest_type": "Fine Dining",
     "online_order": False, "book_table": True, "dish_liked": "Chef's Tasting Menu, Risotto",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "The Fatty Gut", "location": "Whitefield", "cuisines": ["American", "Burgers", "Fast Food"],
     "approx_cost": 500, "rate": 4.1, "votes": 1900, "rest_type": "Quick Bites",
     "online_order": True, "book_table": False, "dish_liked": "Smash Burger, Loaded Fries",
     "meal_type": "Delivery", "city": "Bangalore"},

    {"name": "Royal Afghan", "location": "Whitefield", "cuisines": ["Mughlai", "Arabian", "North Indian"],
     "approx_cost": 1500, "rate": 4.4, "votes": 2700, "rest_type": "Fine Dining",
     "online_order": False, "book_table": True, "dish_liked": "Raan, Seekh Kebab, Biryani",
     "meal_type": "Buffet", "city": "Bangalore"},

    {"name": "Royal Afghan", "location": "Whitefield", "cuisines": ["Mughlai", "Arabian", "North Indian"],
     "approx_cost": 1500, "rate": 4.4, "votes": 2700, "rest_type": "Fine Dining",
     "online_order": False, "book_table": True, "dish_liked": "Raan, Seekh Kebab, Biryani",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Barbeque Nation", "location": "Whitefield", "cuisines": ["North Indian", "Barbecue", "Continental"],
     "approx_cost": 1200, "rate": 4.3, "votes": 8900, "rest_type": "Casual Dining",
     "online_order": False, "book_table": True, "dish_liked": "Grilled Chicken, Peri Peri Fish",
     "meal_type": "Buffet", "city": "Bangalore"},

    {"name": "Barbeque Nation", "location": "Whitefield", "cuisines": ["North Indian", "Barbecue", "Continental"],
     "approx_cost": 1200, "rate": 4.3, "votes": 8900, "rest_type": "Casual Dining",
     "online_order": False, "book_table": True, "dish_liked": "Grilled Chicken, Peri Peri Fish",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Spice Garden", "location": "Whitefield", "cuisines": ["South Indian", "North Indian", "Chinese"],
     "approx_cost": 600, "rate": 3.9, "votes": 3400, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Dosa, Naan, Fried Rice",
     "meal_type": "Delivery", "city": "Bangalore"},

    # ── Jayanagar ────────────────────────────────────────────────────────────
    {"name": "Airlines Hotel", "location": "Jayanagar", "cuisines": ["South Indian", "Breakfast"],
     "approx_cost": 200, "rate": 4.5, "votes": 8900, "rest_type": "Quick Bites",
     "online_order": False, "book_table": False, "dish_liked": "Masala Dosa, Filter Coffee, Vada",
     "meal_type": "Cafes", "city": "Bangalore"},

    {"name": "Udupi Palace", "location": "Jayanagar", "cuisines": ["South Indian", "Udupi"],
     "approx_cost": 300, "rate": 4.2, "votes": 5600, "rest_type": "Quick Bites",
     "online_order": True, "book_table": False, "dish_liked": "Thali, Dosa, Meals",
     "meal_type": "Buffet", "city": "Bangalore"},

    {"name": "Udupi Palace", "location": "Jayanagar", "cuisines": ["South Indian", "Udupi"],
     "approx_cost": 300, "rate": 4.2, "votes": 5600, "rest_type": "Quick Bites",
     "online_order": True, "book_table": False, "dish_liked": "Thali, Dosa, Meals",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "The Tangerine", "location": "Jayanagar", "cuisines": ["Continental", "Chinese", "North Indian"],
     "approx_cost": 1100, "rate": 4.0, "votes": 2300, "rest_type": "Casual Dining",
     "online_order": True, "book_table": True, "dish_liked": "Butter Chicken, Pasta",
     "meal_type": "Buffet", "city": "Bangalore"},

    {"name": "The Tangerine", "location": "Jayanagar", "cuisines": ["Continental", "Chinese", "North Indian"],
     "approx_cost": 1100, "rate": 4.0, "votes": 2300, "rest_type": "Casual Dining",
     "online_order": True, "book_table": True, "dish_liked": "Butter Chicken, Pasta",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Hotel Sagar", "location": "Jayanagar", "cuisines": ["South Indian", "North Indian"],
     "approx_cost": 200, "rate": 3.8, "votes": 3100, "rest_type": "Quick Bites",
     "online_order": False, "book_table": False, "dish_liked": "Dosa, Idli, Poori",
     "meal_type": "Dine-out", "city": "Bangalore"},

    # ── Malleshwaram ─────────────────────────────────────────────────────────
    {"name": "Vidhyarthi Bhavan", "location": "Malleshwaram", "cuisines": ["South Indian"],
     "approx_cost": 100, "rate": 4.6, "votes": 10200, "rest_type": "Quick Bites",
     "online_order": False, "book_table": False, "dish_liked": "Masala Dosa, Bisi Bele Bath",
     "meal_type": "Cafes", "city": "Bangalore"},

    {"name": "Sri Udupi Park", "location": "Malleshwaram", "cuisines": ["South Indian", "Udupi"],
     "approx_cost": 150, "rate": 4.1, "votes": 3800, "rest_type": "Quick Bites",
     "online_order": False, "book_table": False, "dish_liked": "Idli, Dosa, Coffee",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Kadambam", "location": "Malleshwaram", "cuisines": ["South Indian", "Tamil", "Chettinad"],
     "approx_cost": 400, "rate": 4.3, "votes": 4200, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Chettinad Chicken, Appam",
     "meal_type": "Buffet", "city": "Bangalore"},

    {"name": "Kadambam", "location": "Malleshwaram", "cuisines": ["South Indian", "Tamil", "Chettinad"],
     "approx_cost": 400, "rate": 4.3, "votes": 4200, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Chettinad Chicken, Appam",
     "meal_type": "Dine-out", "city": "Bangalore"},

    # ── Banaswadi ────────────────────────────────────────────────────────────
    {"name": "Paradise Biryani", "location": "Banaswadi", "cuisines": ["Biryani", "Hyderabadi", "Andhra"],
     "approx_cost": 450, "rate": 4.1, "votes": 6700, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Chicken Biryani, Haleem",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Paradise Biryani", "location": "Banaswadi", "cuisines": ["Biryani", "Hyderabadi", "Andhra"],
     "approx_cost": 450, "rate": 4.1, "votes": 6700, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Chicken Biryani, Haleem",
     "meal_type": "Delivery", "city": "Bangalore"},

    {"name": "Chilis Restaurant", "location": "Banaswadi", "cuisines": ["North Indian", "Chinese", "Continental"],
     "approx_cost": 600, "rate": 3.7, "votes": 1400, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Butter Chicken, Fried Rice",
     "meal_type": "Dine-out", "city": "Bangalore"},

    # ── HSR Layout ───────────────────────────────────────────────────────────
    {"name": "Socials", "location": "HSR Layout", "cuisines": ["Continental", "Pizzas", "North Indian"],
     "approx_cost": 1100, "rate": 4.2, "votes": 7800, "rest_type": "Bar",
     "online_order": True, "book_table": True, "dish_liked": "Nachos, Pizza, Cocktails",
     "meal_type": "Pubs and bars", "city": "Bangalore"},

    {"name": "Hammered", "location": "HSR Layout", "cuisines": ["Continental", "Finger Food"],
     "approx_cost": 1000, "rate": 4.0, "votes": 3200, "rest_type": "Bar",
     "online_order": False, "book_table": False, "dish_liked": "Wings, Beer, Sliders",
     "meal_type": "Pubs and bars", "city": "Bangalore"},

    {"name": "The Biryani Life", "location": "HSR Layout", "cuisines": ["Biryani", "North Indian"],
     "approx_cost": 350, "rate": 3.9, "votes": 4500, "rest_type": "Quick Bites",
     "online_order": True, "book_table": False, "dish_liked": "Biryani, Raita",
     "meal_type": "Delivery", "city": "Bangalore"},

    {"name": "Chinese Hut", "location": "HSR Layout", "cuisines": ["Chinese", "Asian", "Tibetan"],
     "approx_cost": 400, "rate": 4.0, "votes": 2600, "rest_type": "Quick Bites",
     "online_order": True, "book_table": False, "dish_liked": "Dim Sum, Thukpa, Momos",
     "meal_type": "Buffet", "city": "Bangalore"},

    {"name": "Chinese Hut", "location": "HSR Layout", "cuisines": ["Chinese", "Asian", "Tibetan"],
     "approx_cost": 400, "rate": 4.0, "votes": 2600, "rest_type": "Quick Bites",
     "online_order": True, "book_table": False, "dish_liked": "Dim Sum, Thukpa, Momos",
     "meal_type": "Dine-out", "city": "Bangalore"},

    # ── MG Road ──────────────────────────────────────────────────────────────
    {"name": "The Humming Tree", "location": "MG Road", "cuisines": ["Continental", "Cafe"],
     "approx_cost": 800, "rate": 4.1, "votes": 2900, "rest_type": "Café",
     "online_order": False, "book_table": False, "dish_liked": "Sandwiches, Coffee, Desserts",
     "meal_type": "Cafes", "city": "Bangalore"},

    {"name": "Sunny's", "location": "MG Road", "cuisines": ["Italian", "Continental", "Mediterranean"],
     "approx_cost": 1800, "rate": 4.3, "votes": 3700, "rest_type": "Fine Dining",
     "online_order": False, "book_table": True, "dish_liked": "Pasta, Tiramisu, Risotto",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "The Tandoor", "location": "MG Road", "cuisines": ["North Indian", "Mughlai", "Tandoor"],
     "approx_cost": 1100, "rate": 4.4, "votes": 5100, "rest_type": "Casual Dining",
     "online_order": True, "book_table": True, "dish_liked": "Tandoori Chicken, Naan, Biryani",
     "meal_type": "Buffet", "city": "Bangalore"},

    {"name": "The Tandoor", "location": "MG Road", "cuisines": ["North Indian", "Mughlai", "Tandoor"],
     "approx_cost": 1100, "rate": 4.4, "votes": 5100, "rest_type": "Casual Dining",
     "online_order": True, "book_table": True, "dish_liked": "Tandoori Chicken, Naan, Biryani",
     "meal_type": "Dine-out", "city": "Bangalore"},

    # ── Electronic City ──────────────────────────────────────────────────────
    {"name": "Onesta", "location": "Electronic City", "cuisines": ["Pizzas", "Italian", "Pasta"],
     "approx_cost": 600, "rate": 4.3, "votes": 4900, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "BBQ Chicken Pizza, Garlic Bread",
     "meal_type": "Dine-out", "city": "Bangalore"},

    {"name": "Onesta", "location": "Electronic City", "cuisines": ["Pizzas", "Italian", "Pasta"],
     "approx_cost": 600, "rate": 4.3, "votes": 4900, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "BBQ Chicken Pizza, Garlic Bread",
     "meal_type": "Delivery", "city": "Bangalore"},

    {"name": "Nandhini Hotel", "location": "Electronic City", "cuisines": ["South Indian", "Andhra"],
     "approx_cost": 300, "rate": 4.0, "votes": 6800, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Biryani, Meals",
     "meal_type": "Buffet", "city": "Bangalore"},

    {"name": "Nandhini Hotel", "location": "Electronic City", "cuisines": ["South Indian", "Andhra"],
     "approx_cost": 300, "rate": 4.0, "votes": 6800, "rest_type": "Casual Dining",
     "online_order": True, "book_table": False, "dish_liked": "Biryani, Meals",
     "meal_type": "Dine-out", "city": "Bangalore"},
]


def main() -> None:
    df = pd.DataFrame(RESTAURANTS)

    # Flag near-duplicates (same name + location) — keep first occurrence.
    # This mirrors the preprocessing pipeline but we keep ALL rows so that
    # each meal_type listing is indexed and the meal_type post-filter works.
    df["is_near_duplicate"] = df.duplicated(subset=["name", "location", "meal_type"], keep="first")

    print(f"Sample dataset: {len(df)} rows, {df['is_near_duplicate'].sum()} exact duplicates flagged")
    print("meal_type distribution:")
    print(df["meal_type"].value_counts().to_string())
    print()

    # Save parquet
    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)
    print(f"Saved {len(df)} rows → {CLEAN_PARQUET}")
    print()

    # Delete old ChromaDB and re-index
    import shutil
    db_path = CLEAN_PARQUET.parents[1] / "data" / "vectordb"
    if db_path.exists():
        shutil.rmtree(db_path)
        print("Wiped old ChromaDB")

    count = index_dataframe(df)
    print(f"\nDone — {count} documents in ChromaDB")


if __name__ == "__main__":
    main()
