import pandas as pd


class HybridRecommender:

    def __init__(self, pop_model, ibcf_model):
        self.pop_model = pop_model
        self.ibcf_model = ibcf_model

    def recommend(self, customer_id, top_n=5):
        # Ambil rekomendasi popularity (lebih banyak sebagai kandidat)
        pop_df = self.pop_model.recommend(20)

        # Ambil rekomendasi IBCF
        ibcf_df = self.ibcf_model.recommend(
            customer_id,
            top_n=20
        )

        # Pastikan kolom predicted_score ada di ibcf_df
        if ibcf_df is None or ibcf_df.empty:
            # Fallback ke popularity saja
            pop_df = pop_df.copy()
            pop_df['hybrid_score'] = pop_df['popularity_score']
            return pop_df.head(top_n)

        # Gabungkan kedua model
        merged = ibcf_df.merge(
            pop_df[['menu_id', 'popularity_score']],
            on='menu_id',
            how='left'
        )

        # Isi NaN popularity_score dengan 0
        merged['popularity_score'] = merged['popularity_score'].fillna(0)

        # Hitung hybrid score
        merged['hybrid_score'] = (
            0.7 * merged['predicted_score'] +
            0.3 * merged['popularity_score']
        )

        # Sorting
        merged = merged.sort_values(
            'hybrid_score',
            ascending=False
        )

        return merged.head(top_n)
