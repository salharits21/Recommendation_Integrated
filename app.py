from flask import Flask, jsonify
import pandas as pd
import requests
import sys
import os
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler # Tambahan baru

# IMPORT MODELS & UTILS
from models.popularity_model import PopularityRecommender
from models.ibcf_model import IBCFRecommender
from models.hybrid_model import HybridRecommender
from utils.matrix_builder import build_user_item_matrix
from utils.data_cleaning import clean_data

# INIT FLASK
app = Flask(__name__)
load_dotenv()

# Deklarasi variabel global agar bisa diakses oleh rute (endpoints)
df = None
pop_model = PopularityRecommender()
ibcf_model = IBCFRecommender()
hybrid_model = HybridRecommender(pop_model, ibcf_model)

def fetch_and_train_models():
    global df, pop_model, ibcf_model, hybrid_model
    
    try:
        print("Fetching transaction data from Laravel API...")
        response = requests.get(
            os.getenv("BACKEND_URL") + "/api/internal/transactions",
            headers={"X-API-KEY": os.getenv("X_API_KEY")}
        )
        
        if response.status_code == 200:
            data = response.json()
            df_raw = pd.DataFrame(data)
            print(f"Successfully loaded {len(df_raw)} transactions.")
            
            # CLEAN DATA
            df_new = clean_data(df_raw)
            
            # BUILD USER ITEM MATRIX
            user_item_matrix = build_user_item_matrix(df_new)
            
            # MENU INFO
            menu_info = df_new[['menu_id', 'menu_name', 'category']].drop_duplicates()
            
            # FIT MODELS (Update in memory)
            pop_model.fit(df_new)
            ibcf_model.fit(user_item_matrix, menu_info)
            hybrid_model = HybridRecommender(pop_model, ibcf_model)
            
            # Update variabel global df setelah semua proses berhasil
            df = df_new
            print("Models successfully retrained and updated in memory.")
            
        else:
            print(f"Failed to fetch data. Status Code: {response.status_code}")
            
    except Exception as e:
        print(f"Error connecting to Laravel API or training models: {e}")

# --- JALANKAN SEKALI SAAT APLIKASI STARTUP ---
fetch_and_train_models()

# --- SETUP SCHEDULER UNTUK RUN TIAP JAM 00:01 ---
scheduler = BackgroundScheduler()
scheduler.add_job(func=fetch_and_train_models, trigger="cron", hour=0, minute=1, timezone='Asia/Jakarta')
scheduler.start()


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
# API IBCF RECOMMENDATION
@app.route('/ibcf/<customer_id>')
def ibcf_recommendation(customer_id):
    # Pastikan customer memang terdaftar di dataset
    if df is not None and customer_id not in df['customer_id'].values:
        return jsonify({
            "status": "error",
            "message": f"Customer ID '{customer_id}' tidak ditemukan dalam database."
        }), 404

    recommendations = ibcf_model.recommend(
        customer_id=customer_id,
        top_n=5
    )

    if recommendations is None or recommendations.empty:
        return jsonify({
            "status": "success",
            "message": "Tidak ada rekomendasi item baru (Customer mungkin sudah membeli semua menu).",
            "data": []
        })

    result = recommendations.to_dict(orient='records')
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
    # scheduler mati dengan aman jika Flask dimatikan
    try:
        app.run(debug=True)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()