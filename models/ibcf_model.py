import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class IBCFRecommender:

    def __init__(self):

        self.item_similarity = None
        self.user_item_matrix = None
        self.menu_info = None

    def fit(self, user_item_matrix, menu_info_df):

        self.user_item_matrix = user_item_matrix

        item_matrix = user_item_matrix.T.values

        sim = cosine_similarity(item_matrix)

        self.item_similarity = pd.DataFrame(
            sim,
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns
        )

        self.menu_info = (
            menu_info_df
            .drop_duplicates('menu_id')
            .set_index('menu_id')
        )

        return self

    def predict_score(self, customer_id, menu_id):

        if customer_id not in self.user_item_matrix.index:
            return 0

        user_ratings = self.user_item_matrix.loc[customer_id]

        bought_items = user_ratings[user_ratings > 0]

        if len(bought_items) == 0:
            return 0

        sims = self.item_similarity.loc[
            menu_id,
            bought_items.index
        ]

        denom = sims.abs().sum()

        if denom == 0:
            return 0

        score = (sims * bought_items).sum() / denom

        return float(score)

    def recommend(self, customer_id, top_n=5):

        all_items = self.item_similarity.columns.tolist()

        scores = {}

        for item in all_items:
            scores[item] = self.predict_score(
                customer_id,
                item
            )

        scores = pd.Series(scores).sort_values(
            ascending=False
        ).head(top_n)

        result = pd.DataFrame({
            'menu_id': scores.index,
            'predicted_score': scores.values
        })

        result = result.merge(
            self.menu_info.reset_index(),
            on='menu_id',
            how='left'
        )

        return result