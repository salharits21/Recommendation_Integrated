from flask import Flask, jsonify
import pandas as pd


# IMPORT MODELS & UTILS
from models.popularity_model import PopularityRecommender
from models.ibcf_model import IBCFRecommender
from models.hybrid_model import HybridRecommender

from utils.matrix_builder import build_user_item_matrix
from utils.data_cleaning import clean_data


# INIT FLASK
app = Flask(__name__)


# LOAD DATASET
df_raw = pd.read_csv(
    "data/coffee_shop_transactions.csv"
)

print(f"Shape data mentah: {df_raw.shape}")
print(f"Kolom: {list(df_raw.columns)}")
print(df_raw.head(10))


# CLEAN DATA
df = clean_data(df_raw)


# BUILD USER ITEM MATRIX
user_item_matrix = build_user_item_matrix(df)


# MENU INFO
menu_info = df[[
    'menu_id',
    'menu_name',
    'category'
]].drop_duplicates()


# FIT POPULARITY MODEL
pop_model = PopularityRecommender()

pop_model.fit(df)


# FIT IBCF MODEL
ibcf_model = IBCFRecommender()

ibcf_model.fit(
    user_item_matrix,
    menu_info
)

# FIT HYBRID MODEL

hybrid_model = HybridRecommender(
    pop_model,
    ibcf_model
)


# HOME ROUTE

@app.route('/')
def home():
    return jsonify({
        "message": "API Recommendation System Running"
    })


# API TOP MENU

@app.route('/top-menu')
def top_menu():
    top_menu = (
        df.groupby('menu_name')['quantity']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    result = {
        "menu": top_menu.index.tolist(),
        "quantity": top_menu.values.tolist()
    }

    return jsonify(result)


# API CATEGORY
@app.route('/category')
def category():
    category_data = (
        df.groupby('category')['quantity']
        .sum()
    )

    result = {
        "category": category_data.index.tolist(),
        "quantity": category_data.values.tolist()
    }

    return jsonify(result)


# API MONTHLY TRANSACTION
@app.route('/monthly-transaction')
def monthly_transaction():
    temp_df = df.copy()

    temp_df['transaction_date'] = pd.to_datetime(
        temp_df['transaction_date']
    )

    temp_df['month'] = (
        temp_df['transaction_date']
        .dt
        .to_period('M')
        .astype(str)
    )

    monthly = (
        temp_df.groupby('month')['transaction_id']
        .count()
    )

    result = {
        "month": monthly.index.tolist(),
        "transaction": monthly.values.tolist()
    }

    return jsonify(result)

# API POPULARITY RECOMMENDATION
@app.route('/popularity')
def popularity_recommendation():
    recommendations = pop_model.recommend(
        top_n=5
    )

    result = recommendations.to_dict(
        orient='records'
    )

    return jsonify(result)


# API IBCF RECOMMENDATION
@app.route('/ibcf/<customer_id>')
def ibcf_recommendation(customer_id):
    recommendations = ibcf_model.recommend(
        customer_id=customer_id,
        top_n=3
    )

    if recommendations is None or recommendations.empty:
        return jsonify({
            "message": "Customer tidak ditemukan"
        })

    result = recommendations.to_dict(
        orient='records'
    )

    return jsonify(result)


# API HYBRID RECOMMENDATION
@app.route('/hybrid/<customer_id>')
def hybrid_recommendation(customer_id):
    recommendations = hybrid_model.recommend(
        customer_id=customer_id,
        top_n=5
    )

    if recommendations is None or recommendations.empty:
        return jsonify({
            "message": "Customer tidak ditemukan"
        })

    result = recommendations.to_dict(
        orient='records'
    )

    return jsonify(result)


# RUN SERVER
if __name__ == '__main__':
    app.run(debug=True)
