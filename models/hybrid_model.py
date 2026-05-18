class HybridRecommender:

    def __init__(self, pop_model, ibcf_model):

        self.pop_model = pop_model
        self.ibcf_model = ibcf_model

    def recommend(self, customer_id, top_n=5):

        # Mengambil popularity recommendation
        pop_df = self.pop_model.recommend(20)

        # Mengambil IBCF recommendation
        ibcf_df = self.ibcf_model.recommend(
            customer_id,
            top_n=20
        )

        # Menggabungkan kedua model
        merged = ibcf_df.merge(
            pop_df[['menu_id', 'popularity_score']],
            on='menu_id',
            how='left'
        )

        # Menghitung hybrid score
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