import matplotlib
matplotlib.use('Agg')  # non-interactive backend agar aman di server

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Pastikan folder output tersedia
os.makedirs('output_charts', exist_ok=True)


# ─── TOP MENU ────────────────────────────────────────────────
def top_menu_chart(df):
    top_menu = (
        df.groupby('menu_name')['quantity']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 6))
    plt.bar(
        top_menu.index,
        top_menu.values
    )
    plt.xlabel('Menu')
    plt.ylabel('Total Quantity')
    plt.title('Top 10 Menu Terlaris')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output_charts/top_menu_chart.png')
    plt.close()
    print("Saved: output_charts/top_menu_chart.png")


# ─── DISTRIBUSI KATEGORI ─────────────────────────────────────
def category_chart(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(
        data=df,
        x='category'
    )
    plt.title('Distribusi Kategori')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output_charts/category_chart.png')
    plt.close()
    print("Saved: output_charts/category_chart.png")


# ─── TRANSAKSI BULANAN ───────────────────────────────────────
def monthly_transaction_chart(df):
    temp_df = df.copy()
    temp_df['transaction_date'] = pd.to_datetime(
        temp_df['transaction_date']
    )

    temp_df['month'] = (
        temp_df['transaction_date']
        .dt.to_period('M')
    )

    monthly = (
        temp_df.groupby('month')['transaction_id']
        .count()
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        monthly.index.astype(str),
        monthly.values,
        marker='o'
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Tren Transaksi Per Bulan')
    plt.ylabel('Jumlah Transaksi')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output_charts/monthly_transaction_chart.png')
    plt.close()
    print("Saved: output_charts/monthly_transaction_chart.png")


# ─── HEATMAP KORELASI ────────────────────────────────────────
def heatmap_chart(df):
    numeric_df = df.select_dtypes(include='number')

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt='.2f',
        cmap='Blues'
    )
    plt.title('Heatmap Korelasi')
    plt.tight_layout()
    plt.savefig('output_charts/heatmap_chart.png')
    plt.close()
    print("Saved: output_charts/heatmap_chart.png")
