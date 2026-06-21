"""
Credit Card Customer Segmentation — end-to-end analysis.

Loads the raw cardholder dataset, cleans it, engineers behavioural features,
runs K-Means segmentation (with standardisation + PCA for diagnostics), profiles
the resulting segments, and exports:
  - data/processed/*.csv   cleaned data + per-customer cluster labels + profiles
  - figures/*.png          static charts for the report
  - dashboard/data.json     summary metrics consumed by the HTML dashboard/report

Run:  python src/analysis.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "credit_card_customers.csv"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
DASH = ROOT / "dashboard"
for d in (PROC, FIG, DASH):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
PALETTE = ["#E0533D", "#2E7C84", "#F0A35E", "#1C3A5E", "#8FB996"]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df = df.drop(columns=["CUST_ID"])
    # Two columns carry missing values; impute with the median (robust to skew).
    n_missing = int(df.isna().sum().sum())
    df = df.fillna(df.median(numeric_only=True))
    df.attrs["n_missing_imputed"] = n_missing
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform highly skewed monetary fields so distances are not dominated
    by a few whales, then standardise everything for K-Means."""
    monetary = [
        "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
        "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
    ]
    X = df.copy()
    for c in monetary:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X


def choose_k(X_scaled: np.ndarray, kmax: int = 8) -> dict:
    inertias, sils = {}, {}
    for k in range(2, kmax + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias[k] = float(km.inertia_)
        sils[k] = float(silhouette_score(X_scaled, labels))
    return {"inertia": inertias, "silhouette": sils}


def main():
    df = load_and_clean()
    X = engineer_features(df)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    sweep = choose_k(Xs)
    K = 3  # business-driven choice (3 actionable personas); see report

    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(Xs)
    sil_full = float(silhouette_score(Xs, labels))

    # PCA for 2-D visualisation + a diagnostic silhouette in reduced space
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Xs)
    var_explained = [float(v) for v in pca.explained_variance_ratio_]

    df_out = df.copy()
    df_out["cluster"] = labels
    df_out["pc1"] = coords[:, 0]
    df_out["pc2"] = coords[:, 1]

    # ---- profile clusters on ORIGINAL (un-logged) scale ----
    profile = df_out.groupby("cluster").agg(
        size=("BALANCE", "size"),
        balance=("BALANCE", "mean"),
        purchases=("PURCHASES", "mean"),
        cash_advance=("CASH_ADVANCE", "mean"),
        credit_limit=("CREDIT_LIMIT", "mean"),
        payments=("PAYMENTS", "mean"),
        prc_full_payment=("PRC_FULL_PAYMENT", "mean"),
        purchases_freq=("PURCHASES_FREQUENCY", "mean"),
        tenure=("TENURE", "mean"),
    ).round(2)
    profile["share"] = (profile["size"] / len(df_out) * 100).round(1)

    # name clusters by behaviour: high purchases -> transactors; high balance/low purchases -> revolvers
    names = {}
    p = profile.sort_values("purchases", ascending=False)
    names[p.index[0]] = "Power Transactors"
    # of the remaining two, the one with higher balance & cash advance = revolvers/shadow debtors
    rem = profile.drop(index=p.index[0])
    deb = rem.sort_values("balance", ascending=False)
    names[deb.index[0]] = "Revolving Debtors"
    names[deb.index[1]] = "Marginal Spenders"
    profile["segment"] = [names[i] for i in profile.index]
    df_out["segment"] = df_out["cluster"].map(names)

    # ---- tenure churn view: share of customers by tenure ----
    tenure_dist = (df_out["TENURE"].value_counts(normalize=True).sort_index() * 100).round(1)

    # ---- save processed data ----
    df_out.to_csv(PROC / "customers_with_segments.csv", index=False)
    profile.to_csv(PROC / "segment_profiles.csv")

    # ---- figures ----
    # 1. silhouette sweep
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ks = list(sweep["silhouette"].keys())
    ax.plot(ks, [sweep["silhouette"][k] for k in ks], "o-", color=PALETTE[0])
    ax.axvline(K, ls="--", color="#999")
    ax.set_xlabel("k (clusters)"); ax.set_ylabel("Silhouette score")
    ax.set_title("Choosing k — silhouette by cluster count")
    fig.tight_layout(); fig.savefig(FIG / "silhouette_sweep.png", dpi=130); plt.close(fig)

    # 2. PCA scatter
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for c in sorted(df_out["cluster"].unique()):
        m = df_out["cluster"] == c
        ax.scatter(df_out.loc[m, "pc1"], df_out.loc[m, "pc2"], s=6, alpha=.4,
                   color=PALETTE[c % len(PALETTE)], label=names[c])
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.0f}% var)")
    ax.set_title("Customer segments in PCA space")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "pca_segments.png", dpi=130); plt.close(fig)

    # 3. segment sizes
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ps = profile.sort_values("size", ascending=True)
    ax.barh(ps["segment"], ps["size"], color=PALETTE[:len(ps)])
    ax.set_title("Segment sizes"); ax.set_xlabel("customers")
    fig.tight_layout(); fig.savefig(FIG / "segment_sizes.png", dpi=130); plt.close(fig)

    # ---- dashboard JSON ----
    payload = {
        "meta": {
            "n_customers": int(len(df_out)),
            "n_features": int(X.shape[1]),
            "n_missing_imputed": int(df.attrs.get("n_missing_imputed", 0)),
            "k": K,
            "silhouette_full": round(sil_full, 3),
            "pca_var_explained": [round(v, 3) for v in var_explained],
            "pca_var_2d": round(sum(var_explained), 3),
        },
        "silhouette_sweep": {str(k): round(v, 3) for k, v in sweep["silhouette"].items()},
        "segments": [
            {
                "cluster": int(i),
                "name": profile.loc[i, "segment"],
                "size": int(profile.loc[i, "size"]),
                "share": float(profile.loc[i, "share"]),
                "balance": float(profile.loc[i, "balance"]),
                "purchases": float(profile.loc[i, "purchases"]),
                "cash_advance": float(profile.loc[i, "cash_advance"]),
                "credit_limit": float(profile.loc[i, "credit_limit"]),
                "payments": float(profile.loc[i, "payments"]),
                "prc_full_payment": float(profile.loc[i, "prc_full_payment"]),
                "purchases_freq": float(profile.loc[i, "purchases_freq"]),
                "tenure": float(profile.loc[i, "tenure"]),
            }
            for i in profile.index
        ],
        "tenure_distribution": {str(int(k)): float(v) for k, v in tenure_dist.items()},
        # downsampled scatter for the dashboard (keep payload small)
        "scatter": [
            {"x": round(float(r.pc1), 3), "y": round(float(r.pc2), 3), "c": int(r.cluster)}
            for r in df_out.sample(min(1200, len(df_out)), random_state=1).itertuples()
        ],
    }
    (DASH / "data.json").write_text(json.dumps(payload, indent=2))

    print(f"n={len(df_out)}  k={K}  silhouette={sil_full:.3f}  PCA2D var={sum(var_explained):.2f}")
    print(profile[["segment", "size", "share", "balance", "purchases", "cash_advance", "prc_full_payment", "tenure"]])


if __name__ == "__main__":
    main()
