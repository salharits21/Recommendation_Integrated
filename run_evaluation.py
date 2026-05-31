import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.popularity_model import PopularityRecommender
from models.ibcf_model import IBCFRecommender
from models.hybrid_model import HybridRecommender
from utils.matrix_builder import build_user_item_matrix
from utils.data_cleaning import clean_data
from utils.evaluation import precision_recall_f1

# ══════════════════════════════════════════════════
# LOAD & CLEAN
# ══════════════════════════════════════════════════

print("=" * 55)
print("  EVALUASI SISTEM REKOMENDASI MENU KOPI")
print("=" * 55)

df_raw = pd.read_csv("data/coffee_shop_transactions.csv")
df     = clean_data(df_raw)

print(f"\n[INFO] Total transaksi : {len(df)}")
print(f" Total pelanggan : {df['customer_id'].nunique()}")
print(f" Total menu      : {df['menu_name'].nunique()}")

# ══════════════════════════════════════════════════
# SPLIT 80/20 PER PELANGGAN


df_sorted  = df.sort_values(['customer_id', 'transaction_date'])
train_rows = []
test_rows  = []

for cid, grp in df_sorted.groupby('customer_id'):
    n      = len(grp)
    cutoff = max(1, int(n * 0.8))
    train_rows.append(grp.iloc[:cutoff])
    test_rows.append(grp.iloc[cutoff:])

train_df = pd.concat(train_rows).reset_index(drop=True)
test_df  = pd.concat(test_rows).reset_index(drop=True)

# ══════════════════════════════════════════════════
# GROUND TRUTH
# Menu yang muncul di test 20% DAN BELUM PERNAH
# dibeli di train 80%  →  benar-benar "menu baru"
# bagi model (inilah yang harus diprediksi)
# ══════════════════════════════════════════════════

MIN_TXN   = 3
valid_cids = (
    train_df.groupby('customer_id')['transaction_id']
    .nunique()
    .loc[lambda x: x >= MIN_TXN]
    .index.tolist()
)

# Riwayat belanja per pelanggan di training
train_history = (
    train_df.groupby('customer_id')['menu_name']
    .apply(set)
    .to_dict()
)

test_data = {}
for cid, grp in test_df.groupby('customer_id'):
    if cid not in valid_cids:
        continue
    bought_in_train = train_history.get(cid, set())
    new_menus       = [m for m in grp['menu_name'] if m not in bought_in_train]
    if new_menus:
        test_data[cid] = list(set(new_menus))   # unik saja

avg_gt = np.mean([len(v) for v in test_data.values()]) if test_data else 0
print(f"\n Pelanggan dengan menu baru : {len(test_data)}")
print(f" Rata-rata ground truth     : {avg_gt:.2f} menu/pelanggan")
print(f" Total pelanggan valid      : {len(valid_cids)}")
print(f"[INFO] (Pelanggan tanpa menu baru tidak dimasukkan evaluasi )")


# FIT MODEL  (hanya pakai data latih)
print("\n[INFO] Melatih model ...")
menu_info    = df[['menu_id', 'menu_name', 'category']].drop_duplicates()
train_matrix = build_user_item_matrix(train_df)

pop_model    = PopularityRecommender()
pop_model.fit(train_df)

ibcf_model   = IBCFRecommender()
ibcf_model.fit(train_matrix, menu_info)

hybrid_model = HybridRecommender(pop_model, ibcf_model)
print("[INFO] Selesai.")

# ══════════════════════════════════════════════════
# HELPER


def extract_names(result):
    if result is None:
        return []
    if hasattr(result, 'empty') and result.empty:
        return []
    if 'menu_name' in result.columns:
        return result['menu_name'].dropna().tolist()
    return []


# EVALUASI
# Logika seragam untuk ketiga model:
#   1. Minta rekomendasi lebih banyak (top_n * 3)
#   2. Filter item yang sudah dibeli di train
#   3. Ambil top_n sisanya
#   4. Hitung precision / recall / f1 vs ground truth

def evaluate(recommend_fn, test_data, top_n=5):
    precisions, recalls, f1s = [], [], []
    n_empty = 0

    for cid, actual in test_data.items():
        bought = train_history.get(cid, set())

        try:
            # Minta lebih banyak agar setelah difilter masih cukup top_n
            raw             = recommend_fn(cid, top_n * 3)
            candidates      = extract_names(raw)
            # Buang yang sudah pernah dibeli di training
            filtered        = [m for m in candidates if m not in bought]
            predicted       = filtered[:top_n]
        except Exception:
            predicted = []

        if not predicted:
            n_empty += 1

        p, r, f = precision_recall_f1(actual, predicted)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        'precision' : np.mean(precisions),
        'recall'    : np.mean(recalls),
        'f1'        : np.mean(f1s),
        'p_std'     : np.std(precisions),
        'r_std'     : np.std(recalls),
        'f_std'     : np.std(f1s),
        'n'         : len(precisions),
        'n_empty'   : n_empty,
    }

def recommend_pop(cid, top_n):
    return pop_model.recommend(top_n=top_n)

def recommend_ibcf(cid, top_n):
    return ibcf_model.recommend(customer_id=cid, top_n=top_n)

def recommend_hybrid(cid, top_n):
    return hybrid_model.recommend(customer_id=cid, top_n=top_n)


# running
TOP_N = 5
print(f"\n[INFO] Evaluasi top-{TOP_N} rekomendasi ...\n")

r_pop    = evaluate(recommend_pop,    test_data, TOP_N)
r_ibcf   = evaluate(recommend_ibcf,   test_data, TOP_N)
r_hybrid = evaluate(recommend_hybrid, test_data, TOP_N)


# show
def show(label, r):
    print(f"  Model       : {label}")
    print(f"  Precision   : {r['precision']:.4f}  (±{r['p_std']:.4f})")
    print(f"  Recall      : {r['recall']:.4f}  (±{r['r_std']:.4f})")
    print(f"  F1-Score    : {r['f1']:.4f}  (±{r['f_std']:.4f})")
    print(f"  N eval      : {r['n']} pelanggan")
    if r['n_empty']:
        print(f"  ⚠ Kosong    : {r['n_empty']} pelanggan")

print("=" * 55)
print("  HASIL EVALUASI")
print("=" * 55)

print("\n[1] Popularity-Based")
print("-" * 45)
show("Popularity-Based", r_pop)

print("\n[2] IBCF")
print("-" * 45)
show("IBCF", r_ibcf)

print("\n[3] Hybrid (IBCF + Popularity)")
print("-" * 45)
show("Hybrid", r_hybrid)


# RINGKASAN PERBANDINGAN
models = [
    ("Popularity", r_pop['f1']),
    ("IBCF",       r_ibcf['f1']),
    ("Hybrid",     r_hybrid['f1']),
]

print("\n" + "=" * 55)
print("  PERBANDINGAN F1-SCORE")
print("=" * 55)
for name, f1 in models:
    bar = "█" * int(f1 * 40)
    print(f"  {name:<12}: {f1:.4f}  {bar}")

best = max(models, key=lambda x: x[1])
print(f"\n  >> Model terbaik : {best[0]}  (F1 = {best[1]:.4f})")

p_f1 = r_pop['f1']
i_f1 = r_ibcf['f1']
h_f1 = r_hybrid['f1']

if p_f1 > 0:
    print(f"  >> Hybrid vs Popularity : {((h_f1-p_f1)/p_f1)*100:+.2f}%")
if i_f1 > 0:
    print(f"  >> Hybrid vs IBCF       : {((h_f1-i_f1)/i_f1)*100:+.2f}%")

print("=" * 55)
print("\n  Catatan metodologi:")
print("  - Split data   : 80% train / 20% test per pelanggan")
print("  - Ground truth : menu di test yang BELUM dibeli di train")
print("  - Filter       : rekomendasi tidak menyertakan menu")
print("                   yang sudah pernah dibeli pelanggan")
print("  - Metrik       : rata-rata Precision, Recall, F1 atas")
print(f"                   {len(test_data)} pelanggan dengan menu baru")
print("=" * 55)