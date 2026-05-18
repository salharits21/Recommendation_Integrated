import pandas as pd
import numpy as np

def parse_date(s):
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y'):
        try:
            return pd.to_datetime(s, format=fmt)
        except:
            pass
    return pd.NaT


def clean_data(df):
    df = df.copy()

    # Hapus duplikat
    df = df.drop_duplicates()

    # Empty string -> NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Hapus customer kosong
    df = df.dropna(subset=['customer_id'])

    # Hapus menu kosong
    df = df.dropna(subset=['menu_id', 'menu_name'])

    # Quantity
    median_qty = df.groupby('menu_id')['quantity'].transform('median')
    df['quantity'] = df['quantity'].fillna(median_qty)
    df['quantity'] = df['quantity'].fillna(1)
    df['quantity'] = pd.to_numeric(
        df['quantity'],
        errors='coerce'
    ).fillna(1).astype(int)

    # Quantity > 0
    df = df[df['quantity'] > 0]

    # Parse tanggal
    df['transaction_date'] = df['transaction_date'].apply(parse_date)
    df = df.dropna(subset=['transaction_date'])

    # Numerik
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['total_price'] = pd.to_numeric(df['total_price'], errors='coerce')

    # Recalculate total price
    mask = (df['total_price'] != df['price'] * df['quantity'])
    df.loc[mask, 'total_price'] = (
        df.loc[mask, 'price'] *
        df.loc[mask, 'quantity']
    )

    return df