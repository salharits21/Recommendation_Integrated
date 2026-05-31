import pandas as pd


def build_user_item_matrix(df):
    matrix = (
        df.groupby(['customer_id', 'menu_id'])['quantity']
        .sum()
        .unstack(fill_value=0)
    )

    return matrix
