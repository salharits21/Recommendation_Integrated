import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================
# TOP MENU TERLARIS


def top_menu_chart(df):

    top_menu = (
        df.groupby('menu_name')['quantity']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10,6))

    plt.barh(
        top_menu.index,
        top_menu.values
    )

    plt.xlabel("Total Quantity")
    plt.ylabel("Menu")
    plt.title("Top 10 Menu Terlaris")

    plt.tight_layout()

    plt.savefig("top_menu_chart.png")

    plt.show()


# ==========================================
# DISTRIBUSI KATEGORI


def category_chart(df):

    plt.figure(figsize=(8,5))

    sns.countplot(
        data=df,
        x='category'
    )

    plt.title("Distribusi Kategori")

    plt.tight_layout()

    plt.savefig("category_chart.png")

    plt.show()


# ==========================================
# TREN TRANSAKSI


def monthly_transaction_chart(df):

    df['transaction_date'] = pd.to_datetime(
        df['transaction_date']
    )

    df['month'] = (
        df['transaction_date']
        .dt
        .to_period('M')
    )

    monthly = (
        df.groupby('month')['transaction_id']
        .count()
    )

    plt.figure(figsize=(12,5))

    plt.plot(
        monthly.index.astype(str),
        monthly.values,
        marker='o'
    )

    plt.xticks(rotation=45)

    plt.title("Tren Transaksi Per Bulan")

    plt.xlabel("Bulan")
    plt.ylabel("Jumlah Transaksi")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("monthly_transaction_chart.png")

    plt.show()


# ==========================================
# HEATMAP KORELASI


def heatmap_chart(df):

    numeric_df = df.select_dtypes(include='number')

    plt.figure(figsize=(8,6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap='Blues'
    )

    plt.title("Heatmap Korelasi")

    plt.tight_layout()

    plt.savefig("heatmap_chart.png")

    plt.show()