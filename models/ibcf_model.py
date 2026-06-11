import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class IBCFRecommender:

    def __init__(self):
        self.item_similarity  = None
        self.user_item_matrix = None
        self.menu_info        = None

   

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
            return 0.0

        user_ratings = self.user_item_matrix.loc[customer_id]

        
        bought_items = user_ratings[
            (user_ratings > 0) & (user_ratings.index != menu_id)
        ]

        if len(bought_items) == 0:
            return 0.0

        
        if menu_id not in self.item_similarity.index:
            return 0.0

        sims  = self.item_similarity.loc[menu_id, bought_items.index]
        denom = sims.abs().sum()

        if denom == 0:
            return 0.0

        score = (sims * bought_items[sims.index]).sum() / denom

        return float(score)


    def recommend(self, customer_id, top_n=5):
        all_items = self.item_similarity.columns.tolist()

        
        if customer_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[customer_id]
            already_bought = set(
                user_ratings[user_ratings > 0].index.tolist()
            )
        else:
            already_bought = set()

        
        scores = {}
        for item in all_items:
            if item in already_bought:
                continue  
            scores[item] = self.predict_score(customer_id, item)

        
        if not scores or max(scores.values()) == 0:
            return pd.DataFrame(
                columns=['menu_id', 'predicted_score', 'menu_name', 'category']
            )

        scores_series = (
            pd.Series(scores)
            .sort_values(ascending=False)
            .head(top_n)
        )

        result = pd.DataFrame({
            'menu_id'        : scores_series.index,
            'predicted_score': scores_series.values
        })

        result = result.merge(
            self.menu_info.reset_index(),
            on='menu_id',
            how='left'
        )

        return result