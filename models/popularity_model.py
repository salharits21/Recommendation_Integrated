import pandas as pd

class PopularityRecommender:

    def __init__(self):
        self.popularity_df = None

    def fit(self, df):

        menu_stats = df.groupby(
            ['menu_id', 'menu_name', 'category']
        ).agg(
            total_quantity=('quantity', 'sum'),
            total_transactions=('transaction_id', 'nunique'),
            unique_customers=('customer_id', 'nunique')
        ).reset_index()

        for col in [
            'total_quantity',
            'total_transactions',
            'unique_customers'
        ]:

            min_val = menu_stats[col].min()
            max_val = menu_stats[col].max()

            menu_stats[f'{col}_norm'] = (
                (menu_stats[col] - min_val) /
                (max_val - min_val + 1e-9)
            )

        menu_stats['popularity_score'] = (
            0.4 * menu_stats['total_quantity_norm'] +
            0.35 * menu_stats['total_transactions_norm'] +
            0.25 * menu_stats['unique_customers_norm']
        )

        self.popularity_df = menu_stats.sort_values(
            'popularity_score',
            ascending=False
        )

        return self

    def recommend(self, top_n=5):

        return self.popularity_df.head(top_n)[[
            'menu_id',
            'menu_name',
            'category',
            'popularity_score'
        ]]