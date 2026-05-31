import pandas as pd
import numpy as np


def parse_date(s):
    for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']:
        try:
            return pd.to_datetime(s, format=fmt)
        except:
            pass
    return pd.NaT


def clean_data(df):
    df = df.copy()

    
    df = df.drop_duplicates()

    # Ganti empty string dengan NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Hapus baris customer_id kosong
    df = df.dropna(subset=['customer_id'])

    # Imputasi quantity kosong dengan median per menu
    median_qty = df.groupby('menu_id')['quantity'].transform('median')
    df['quantity'] = df['quantity'].fillna(median_qty)
    df['quantity'] = pd.to_numeric(
        df['quantity'], errors='coerce'
    ).fillna(1).astype(int)

    
    df['transaction_date'] = df['transaction_date'].apply(parse_date)
    df = df.dropna(subset=['transaction_date'])

    
    mask = (df['total_price'] != df['price'] * df['quantity'])
    df.loc[mask, 'total_price'] = (
        df.loc[mask, 'price'] * df.loc[mask, 'quantity']
    )

    df = df.reset_index(drop=True)

    print(f"Shape setelah cleaning: {df.shape}")
    print(df.isnull().sum())

    return df
