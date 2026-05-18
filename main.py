import pandas as pd

from utils.data_cleaning import clean_data
from utils.matrix_builder import build_user_item_matrix

from models.popularity_model import PopularityRecommender
from models.ibcf_model import IBCFRecommender
from models.hybrid_model import HybridRecommender


from utils.eda import (
    top_menu_chart,
    category_chart,
    monthly_transaction_chart,
    heatmap_chart
)  


# Load data
df_raw = pd.read_csv(
    'data/coffee_shop_transactions.csv'
)

# Cleaning
df = clean_data(df_raw)

# Matrix
user_item_matrix = build_user_item_matrix(df)

# Menu info
menu_info = df[[
    'menu_id',
    'menu_name',
    'category'
]].drop_duplicates()

# Popularity Model
pop_model = PopularityRecommender()
pop_model.fit(df)

# IBCF Model
ibcf_model = IBCFRecommender()
ibcf_model.fit(user_item_matrix, menu_info)

# Hybrid Model
hybrid_model = HybridRecommender(
    pop_model,
    ibcf_model
)

# Testing
print("=" * 50)
print("POPULARITY RECOMMENDATION")
print("=" * 50)

print(pop_model.recommend())

print("=" * 50)
print("IBCF RECOMMENDATION")
print("=" * 50)

print(
    ibcf_model.recommend(
        customer_id='C001'
    )
)

print("=" * 50)
print("HYBRID RECOMMENDATION")
print("=" * 50)

print(
    hybrid_model.recommend(
        customer_id='C001'
    )
) 

# ====================================
# VISUALISASI DATA
# ====================================

top_menu_chart(df)

category_chart(df)

monthly_transaction_chart(df)

heatmap_chart(df)