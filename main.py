import pandas as pd

from utils.data_cleaning import clean_data
from utils.matrix_builder import build_user_item_matrix
from utils.eda import (
    top_menu_chart,
    category_chart,
    monthly_transaction_chart,
    heatmap_chart
)

from models.popularity_model import PopularityRecommender
from models.ibcf_model import IBCFRecommender
from models.hybrid_model import HybridRecommender


df_raw = pd.read_csv('data/coffee_shop_transactions.csv')

print(f"Shape data mentah : {df_raw.shape}")
print(f"Kolom             : {list(df_raw.columns)}")
print(df_raw.head(10))


df = clean_data(df_raw)


user_item_matrix = build_user_item_matrix(df)


menu_info = df[[
    'menu_id',
    'menu_name',
    'category'
]].drop_duplicates()


pop_model = PopularityRecommender()
pop_model.fit(df)


ibcf_model = IBCFRecommender()
ibcf_model.fit(user_item_matrix, menu_info)


hybrid_model = HybridRecommender(pop_model, ibcf_model)


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


top_menu_chart(df)
category_chart(df)
monthly_transaction_chart(df)
heatmap_chart(df)
