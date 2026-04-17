"""
models.py — Temporal Pathway Edge Imputation
=============================================
Three baselines: REM (multinomial LR), GBDT (XGBoost), Hawkes (discrete marked).

Task 3 scoring is now via a dedicated Ridge time-regression head trained on
normalised event times, so pairwise accuracy and MAE measure genuine temporal
ordering ability — not classifier confidence.
"""

import random
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from reactome_graphs import NodeFeaturiser, ReactomeBioPAX, ReactomeViz

EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURISATION
# ─────────────────────────────────────────────────────────────────────────────


def load_graph(biopax_path: str = "../data/biopax3/R-HSA-168256.xml") -> tuple:
    parser = ReactomeBioPAX(uniprot_accession_num=True)
    G = parser.parse_biopax_into_networkx(
        biopax_path,
        reaction_partners=False,
        include_complexes=True,
    )
    ReactomeViz(G).print_stats()

    featuriser = NodeFeaturiser(
        G,
        xref_dict=parser.uniXrefs,
        protein_model_name="facebook/esm2_t6_8M_UR50D",
        protein_model_device="cuda",
        parser=parser,
    )
    featuriser.download_and_store()
    featuriser.featurise(add_go_embedding=False)

    feat_dim = next(
        G.nodes[n]["feature"].shape[0]
        for n in G.nodes
        if G.nodes[n].get("feature") is not None
    )
    print(f"Node feature dim: {feat_dim}  |  Edge feature dim: {2 * feat_dim}")
    return G, feat_dim, parser


# ─────────────────────────────────────────────────────────────────────────────
# SEMI-INDUCTIVE SPLIT
# ─────────────────────────────────────────────────────────────────────────────


def make_split(G, seed: int, train_frac=0.70, val_frac=0.15):
    rng = random.Random(seed)

    reaction_groups = defaultdict(list)
    for u, v, d in G.edges(data=True):
        key = (d.get("pathway"), d.get("time"))
        reaction_groups[key].append((u, v, d))

    keys = list(reaction_groups.keys())
    rng.shuffle(keys)
    n_train = int(train_frac * len(keys))
    n_val = int(val_frac * len(keys))

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set(keys[n_train + n_val :])

    train_edges = [e for k in train_keys for e in reaction_groups[k]]
    val_edges = [e for k in val_keys for e in reaction_groups[k]]
    test_edges = [e for k in test_keys for e in reaction_groups[k]]

    G_train = G.copy()
    G_train.remove_edges_from([(u, v) for u, v, _ in val_edges + test_edges])
    G_train.remove_nodes_from(list(nx.isolates(G_train)))

    print(
        f"[seed={seed}] Reactions  train: {len(train_keys)}"
        f"  val: {len(val_keys)}  test: {len(test_keys)}"
    )
    return train_edges, val_edges, test_edges, G_train


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def make_feature_fns(G, feat_dim: int):
    ZERO = np.zeros(feat_dim, dtype=np.float32)

    def node_feature(node) -> np.ndarray:
        if node in G.nodes and G.nodes[node].get("feature") is not None:
            feat = G.nodes[node]["feature"]
            if feat.shape[0] < feat_dim:
                feat = np.pad(feat, (0, feat_dim - feat.shape[0]))
            elif feat.shape[0] > feat_dim:
                feat = feat[:feat_dim]
            return feat.astype(np.float32)
        return ZERO.copy()

    def edge_feature(u, v) -> np.ndarray:
        return np.concatenate([node_feature(u), node_feature(v)])

    return node_feature, edge_feature


# ─────────────────────────────────────────────────────────────────────────────
# EDGE TYPE METADATA
# ─────────────────────────────────────────────────────────────────────────────


def make_type_meta(G):
    all_edge_types = sorted(
        {d.get("type", "unknown") for _, _, d in G.edges(data=True)}
    )
    etype_le = LabelEncoder().fit(all_edge_types)
    N_MARKS = len(all_edge_types)
    t_max = max(d.get("time", 1) for _, _, d in G.edges(data=True))
    print(f"Edge types ({N_MARKS}): {all_edge_types}")
    return all_edge_types, etype_le, N_MARKS, t_max


# ─────────────────────────────────────────────────────────────────────────────
# PCA FEATURE BIAS
# ─────────────────────────────────────────────────────────────────────────────


def fit_pca_bias(train_edges, edge_feature, N_MARKS: int, seed: int):
    print("\n[Feature proj] Fitting PCA on training edge features...")
    train_feat_matrix = np.array([edge_feature(u, v) for u, v, _ in train_edges])
    pca = PCA(n_components=N_MARKS, random_state=seed)
    pca.fit(train_feat_matrix)
    print(
        f"[Feature proj] PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}"
    )

    raw_bias_train = np.array(
        [
            pca.transform(edge_feature(u, v).reshape(1, -1)).squeeze(0)
            for u, v, _ in train_edges
        ],
        dtype=np.float32,
    )
    bias_scale = float(np.linalg.norm(raw_bias_train, axis=1).mean()) + EPS
    print(f"[Feature proj] Global bias scale (L2 mean): {bias_scale:.4f}")

    def feat_bias(u, v) -> np.ndarray:
        raw = (
            pca.transform(edge_feature(u, v).reshape(1, -1))
            .squeeze(0)
            .astype(np.float64)
        )
        return raw / bias_scale

    return feat_bias


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY COUNTS
# ─────────────────────────────────────────────────────────────────────────────


def build_history_counts(train_seq, etype_le, N_MARKS: int, t_max: int):
    if not train_seq:
        return lambda t: np.zeros(N_MARKS, dtype=np.float32)

    history_at: dict[int, np.ndarray] = {}
    counts = np.zeros(N_MARKS, dtype=np.float32)
    prev_t = -1

    for u, v, d in train_seq:
        t = int(d.get("time", 0))
        if t != prev_t:
            for bin_t in range(prev_t + 1, t + 1):
                history_at[bin_t] = counts.copy()
            prev_t = t
        k = etype_le.transform([d.get("type", "unknown")])[0]
        counts[k] += 1

    for bin_t in range(prev_t + 1, t_max + 2):
        history_at[bin_t] = counts.copy()

    def get_history(t: int) -> np.ndarray:
        return history_at.get(t, np.zeros(N_MARKS, dtype=np.float32))

    return get_history


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — EDGE EXISTENCE
# ─────────────────────────────────────────────────────────────────────────────


def fit_existence_classifier(train_edges, G, edge_feature, seed: int):
    rng = random.Random(seed)
    all_nodes = list(G.nodes())

    def sample_negative(u):
        while True:
            v = rng.choice(all_nodes)
            if not G.has_edge(u, v):
                return v

    print("\n[Stage 1] Building contrastive training set...")
    X_exist, y_exist = [], []
    for u, v, _ in train_edges:
        X_exist.append(edge_feature(u, v))
        y_exist.append(1)
        X_exist.append(edge_feature(u, sample_negative(u)))
        y_exist.append(0)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    clf.fit(np.array(X_exist, dtype=np.float32), np.array(y_exist))
    print("[Stage 1] Done.")
    return clf, sample_negative, all_nodes


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — TIME REGRESSION HEAD  (shared across all baselines)
# ─────────────────────────────────────────────────────────────────────────────


def fit_time_regressor(train_seq, edge_feature, get_history, t_max: int):
    """
    Ridge regression: [edge_feat ‖ history_counts] → t_norm ∈ [0, 1].

    This is the single source of `predict_order_score` for REM and GBDT,
    replacing the meaningless max-softmax-probability used previously.
    Hawkes uses its own intensity sum (also temporally grounded).
    """
    print("\n[Stage 3 — TimeReg] Fitting time regression head (Ridge)...")
    X, y = [], []
    for u, v, d in train_seq:
        t = int(d.get("time", 0))
        t_norm = float(t) / t_max
        feat = np.concatenate([edge_feature(u, v), get_history(t).astype(np.float32)])
        X.append(feat)
        y.append(t_norm)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    reg.fit(X, y)

    train_preds = reg.predict(X)
    train_mae = float(np.mean(np.abs(train_preds - y)))
    print(f"[Stage 3 — TimeReg] Train MAE: {train_mae:.4f}  (random baseline ≈ 0.25)")

    def predict_time(u, v, t: int) -> float:
        feat = np.concatenate([edge_feature(u, v), get_history(t).astype(np.float32)])
        raw = float(reg.predict(feat.reshape(1, -1))[0])
        return float(np.clip(raw, 0.0, 1.0))  # keep in [0, 1]

    return predict_time


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2a — RELATIONAL EVENT MODEL
# ─────────────────────────────────────────────────────────────────────────────


def fit_rem(
    train_seq,
    edge_feature,
    feat_bias,
    get_history,
    etype_le,
    all_edge_types,
    N_MARKS: int,
    t_max: int,
    n_neg: int = 20,
    seed: int = 42,
):
    """
    Approximate REM via case-control partial likelihood.

    For each observed event (u,v,t), sample n_neg alternative dyads from the
    risk set at time t and fit a binary conditional-choice model:
        y = 1 for observed dyad
        y = 0 for sampled non-event dyads

    This is much closer to a true REM than the previous multinomial classifier:
    it models which dyad occurs at time t given competing alternatives.

    Notes
    -----
    - This predicts whether a dyad is the realised event at time t.
    - It does NOT model exact waiting times via full interval likelihood.
    - Edge type prediction is produced by a second lightweight classifier on
      observed events only, since your evaluation still asks for type labels.
    """
    rng = random.Random(seed)

    # Global node list inferred from training events
    nodes = sorted({u for u, _, _ in train_seq} | {v for _, v, _ in train_seq})

    def rem_feature(u, v, t: int) -> np.ndarray:
        return np.concatenate(
            [
                edge_feature(u, v),
                feat_bias(u, v),
                get_history(t).astype(np.float32),
            ]
        ).astype(np.float32)

    def sample_risk_dyads(u_obs, v_obs, t: int, n_samples: int):
        """
        Sample alternative dyads from the risk set at time t.
        Current approximation: all ordered pairs over known nodes, excluding self-loops
        and excluding the observed dyad.
        """
        out = set()
        max_tries = max(100, 20 * n_samples)
        tries = 0
        while len(out) < n_samples and tries < max_tries:
            s = rng.choice(nodes)
            r = rng.choice(nodes)
            tries += 1
            if s == r:
                continue
            if s == u_obs and r == v_obs:
                continue
            out.add((s, r))
        return list(out)

    print("\n[Stage 2 — REM] Building case-control dataset...")

    # ------------------------------------------------------------------
    # A. Dyad-choice model: observed event vs sampled risk-set non-events
    # ------------------------------------------------------------------
    X_evt, y_evt, w_evt = [], [], []

    for u, v, d in train_seq:
        t = int(d.get("time", 0))

        # positive: realised dyad
        X_evt.append(rem_feature(u, v, t))
        y_evt.append(1)
        w_evt.append(1.0)

        # negatives: sampled alternative dyads
        neg_dyads = sample_risk_dyads(u, v, t, n_neg)
        for s, r in neg_dyads:
            X_evt.append(rem_feature(s, r, t))
            y_evt.append(0)
            w_evt.append(1.0 / max(n_neg, 1))

    X_evt = np.array(X_evt, dtype=np.float32)
    y_evt = np.array(y_evt, dtype=np.int64)
    w_evt = np.array(w_evt, dtype=np.float32)

    event_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    event_clf.fit(X_evt, y_evt, lr__sample_weight=w_evt)
    print("[Stage 2 — REM] Dyad-choice model done.")

    # ------------------------------------------------------------------
    # B. Mark/type model on realised events only
    # ------------------------------------------------------------------
    print("[Stage 2 — REM] Fitting mark/type model on realised events...")

    X_type = np.array(
        [rem_feature(u, v, int(d.get("time", 0))) for u, v, d in train_seq],
        dtype=np.float32,
    )
    y_type = np.array(
        [etype_le.transform([d.get("type", "unknown")])[0] for _, _, d in train_seq],
        dtype=np.int64,
    )

    type_counts = np.bincount(y_type, minlength=N_MARKS).astype(float)
    class_weights = dict(enumerate(1.0 / (type_counts + 1.0)))

    type_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weights,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    type_clf.fit(X_type, y_type)
    print("[Stage 2 — REM] Mark/type model done.")

    # ------------------------------------------------------------------
    # C. Shared time regressor (keep your current evaluation protocol)
    # ------------------------------------------------------------------
    predict_time = fit_time_regressor(train_seq, edge_feature, get_history, t_max)

    # ------------------------------------------------------------------
    # D. Prediction API
    # ------------------------------------------------------------------
    def predict_type(u, v, t: int) -> int:
        x = rem_feature(u, v, t).reshape(1, -1)
        return int(type_clf.predict(x)[0])

    def predict_proba(u, v, t: int) -> np.ndarray:
        x = rem_feature(u, v, t).reshape(1, -1)
        return type_clf.predict_proba(x)[0]

    def predict_event_score(u, v, t: int) -> float:
        """
        REM-style dyad score: probability that (u,v) is the realised dyad
        against sampled alternatives. This is the closest thing here to the
        REM intensity ranking signal.
        """
        x = rem_feature(u, v, t).reshape(1, -1)
        return float(event_clf.predict_proba(x)[0, 1])

    def predict_order_score(u, v, t: int) -> float:
        """
        Keep the temporally grounded score for your current Task 3 evaluation.
        If you want pure REM ranking instead, replace this with predict_event_score.
        """
        return predict_time(u, v, t)

    return {
        "predict_type": predict_type,
        "predict_proba": predict_proba,
        "predict_event_score": predict_event_score,
        "predict_order_score": predict_order_score,
        "rem_feature": rem_feature,
        "event_clf": event_clf,
        "type_clf": type_clf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2b — GBDT
# ─────────────────────────────────────────────────────────────────────────────


def fit_gbdt(
    train_seq,
    edge_feature,
    feat_bias,
    get_history,
    etype_le,
    all_edge_types,
    N_MARKS: int,
    seed: int,
    t_max: int,
):
    def rem_feature(u, v, t: int) -> np.ndarray:
        return np.concatenate([edge_feature(u, v), feat_bias(u, v), get_history(t)])

    print("\n[Stage 2 — GBDT] Building training set...")
    X_type = np.array(
        [rem_feature(u, v, int(d.get("time", 0))) for u, v, d in train_seq]
    )
    y_type_global = np.array(
        [etype_le.transform([d.get("type", "unknown")])[0] for _, _, d in train_seq]
    )

    # Re-encode to contiguous range (fixes missing-class crash)
    local_le = LabelEncoder()
    y_type = local_le.fit_transform(y_type_global)
    present_classes = local_le.classes_
    n_local_marks = len(present_classes)

    type_counts = np.bincount(y_type, minlength=n_local_marks).astype(float)
    sample_weights = 1.0 / (type_counts[y_type] + 1.0)
    sample_weights /= sample_weights.mean()

    print(f"  Training samples: {len(X_type)}  |  Feature dim: {X_type.shape[1]}")
    print(f"  Classes present in train: {list(present_classes)} (of {N_MARKS} total)")

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=n_local_marks,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        reg_alpha=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_type, y_type, sample_weight=sample_weights)
    print("[Stage 2 — GBDT] Done.")

    # Shared time regressor
    predict_time = fit_time_regressor(train_seq, edge_feature, get_history, t_max)

    def predict_type(u, v, t: int) -> int:
        local_pred = int(clf.predict(rem_feature(u, v, t).reshape(1, -1))[0])
        return int(present_classes[local_pred])

    def predict_proba(u, v, t: int) -> np.ndarray:
        local_proba = clf.predict_proba(rem_feature(u, v, t).reshape(1, -1))[0]
        full_proba = np.zeros(N_MARKS, dtype=np.float64)
        full_proba[present_classes] = local_proba
        return full_proba

    def predict_order_score(u, v, t: int) -> float:
        return predict_time(u, v, t)

    return {
        "predict_type": predict_type,
        "predict_proba": predict_proba,
        "predict_order_score": predict_order_score,
        "rem_feature": rem_feature,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2c — DISCRETE MARKED HAWKES
# ─────────────────────────────────────────────────────────────────────────────


class DiscreteMarkedHawkes:
    """
    λ_k(t|u,v) = softplus( μ_k  +  b_k(u,v)  +  Σ_{k'} α_{k,k'} · R_{k'}(t) )
    R_{k'}(t)  = Σ_{t'<t} exp(-β(t-t')) · N_{k'}(t')
    """

    def __init__(self, n_marks: int, beta: float = 0.3, l2_alpha: float = 1e-4):
        self.K = n_marks
        self.beta = beta
        self.l2_alpha = l2_alpha
        self.mu_ = np.zeros(n_marks, dtype=np.float64)
        self.alpha_ = np.zeros((n_marks, n_marks), dtype=np.float64)
        self._bias_fn = None
        self._train_times = None
        self._train_N_mat = None

    @staticmethod
    def _softplus(x):
        return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -500, 30))))

    @staticmethod
    def _sigmoid(x):
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
            np.exp(np.clip(x, -500, 0)) / (1.0 + np.exp(np.clip(x, -500, 0))),
        )

    def _unpack(self, params):
        return params[: self.K], params[self.K :].reshape(self.K, self.K)

    def _build_R(self, times, N_mat):
        T = len(times)
        R = np.zeros((T, self.K), dtype=np.float64)
        for i in range(1, T):
            dt = float(times[i] - times[i - 1])
            R[i] = np.exp(-self.beta * dt) * (R[i - 1] + N_mat[i - 1])
        return R

    def fit(self, train_seq, etype_le, bias_fn=None):
        K = self.K
        T = len(train_seq)
        self._bias_fn = bias_fn

        times = np.array([float(d.get("time", 0)) for _, _, d in train_seq])
        marks = np.array(
            [etype_le.transform([d.get("type", "unknown")])[0] for _, _, d in train_seq]
        )
        N_mat = np.eye(K, dtype=np.float64)[marks]

        self._train_times = times
        self._train_N_mat = N_mat

        R = self._build_R(times, N_mat)

        B = (
            np.array([bias_fn(u, v) for u, v, _ in train_seq], dtype=np.float64)
            if bias_fn is not None
            else np.zeros((T, K), dtype=np.float64)
        )

        type_counts = N_mat.sum(axis=0) + 1.0
        mu_init = np.log(type_counts / type_counts.sum())

        def neg_log_lik(params):
            mu, alpha = self._unpack(params)
            lin = mu[None, :] + B + R @ alpha.T
            lam = self._softplus(lin)
            ll_pos = np.log(lam[np.arange(T), marks] + EPS).sum()
            ll_neg = lam.sum()
            penalty = self.l2_alpha * (alpha**2).sum()
            return (-ll_pos + ll_neg + penalty) / T

        def grad(params):
            mu, alpha = self._unpack(params)
            lin = mu[None, :] + B + R @ alpha.T
            lam = self._softplus(lin)
            sig = self._sigmoid(lin)
            dl = -sig * N_mat / (lam + EPS) + sig
            g_mu = dl.sum(axis=0)
            g_alpha = dl.T @ R + 2 * self.l2_alpha * alpha
            return np.concatenate([g_mu, g_alpha.ravel()]) / T

        x0 = np.concatenate([mu_init, np.zeros(K * K)])
        res = minimize(
            neg_log_lik,
            x0,
            jac=grad,
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-7},
        )
        print(
            f"\n[Stage 2 — Hawkes] L-BFGS-B — Converged: {res.success}  "
            f"NLL: {res.fun:.6f}  Iter: {res.nit}"
        )

        if not res.success:
            res2 = minimize(
                neg_log_lik,
                res.x,
                method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-6},
            )
            if res2.fun < res.fun:
                res = res2
                print(f"  Nelder-Mead improved NLL to {res.fun:.6f}")

        self.mu_, self.alpha_ = self._unpack(res.x)
        print("[Stage 2 — Hawkes] Done.")

    def _history_at(self, t: int) -> np.ndarray:
        times = self._train_times
        N_mat = self._train_N_mat
        mask = times < t
        if not mask.any():
            return np.zeros(self.K, dtype=np.float64)
        past_t = times[mask]
        past_N = N_mat[mask]
        weights = np.exp(-self.beta * (t - past_t)).reshape(-1, 1)
        return (weights * past_N).sum(axis=0)

    def intensity(self, u, v, t: int) -> np.ndarray:
        R_t = self._history_at(t)
        b = self._bias_fn(u, v) if self._bias_fn else np.zeros(self.K)
        lin = self.mu_ + b + self.alpha_ @ R_t
        return self._softplus(lin)

    def predict_type(self, u, v, t: int) -> int:
        return int(np.argmax(self.intensity(u, v, t)))

    def predict_order_score(self, u, v, t: int) -> float:
        # Total intensity: higher = more cascade activity = earlier event
        return float(self.intensity(u, v, t).sum())


def fit_hawkes(
    train_seq,
    edge_feature,
    feat_bias,
    etype_le,
    N_MARKS: int,
    beta: float = 0.3,
    l2_alpha: float = 1e-4,
):
    hawkes = DiscreteMarkedHawkes(n_marks=N_MARKS, beta=beta, l2_alpha=l2_alpha)
    hawkes.fit(train_seq, etype_le, bias_fn=feat_bias)

    def predict_type(u, v, t: int) -> int:
        return hawkes.predict_type(u, v, t)

    def predict_order_score(u, v, t: int) -> float:
        return hawkes.predict_order_score(u, v, t)

    return {
        "predict_type": predict_type,
        "predict_order_score": predict_order_score,
        "hawkes": hawkes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────


def rank_normalise(scores) -> np.ndarray:
    arr = np.array(scores, dtype=np.float64)
    order = np.argsort(np.argsort(-arr))
    n = len(arr)
    return 1.0 - order / max(n - 1, 1)


def pairwise_precedence_accuracy(scores, labels) -> float:
    scores = np.array(scores)
    labels = np.array(labels)
    diff_true = labels[:, None] - labels[None, :]
    diff_pred = scores[:, None] - scores[None, :]
    ordered = diff_true != 0
    concordant = (np.sign(diff_true) == np.sign(diff_pred)) & ordered
    total = ordered.sum() // 2
    correct = np.triu(concordant, k=1).sum()
    return float(correct / total) if total > 0 else 0.0


def evaluate(
    edges,
    exist_clf,
    edge_feature,
    predict_type_fn,
    predict_order_score_fn,
    etype_le,
    all_edge_types,
    N_MARKS: int,
    t_max: int,
    sample_negative_fn,
    name: str,
    predict_event_score_fn=None,
    use_rank_normalise: bool = False,
) -> dict:
    """
    Returns: auc, f1, pair_acc, mae
    Kendall τ removed.
    Task 3 scores are now genuine temporal predictions (t_norm hat).
    """
    neg_edges = [(u, sample_negative_fn(u), d) for u, v, d in edges]

    # Task 1 — Edge existence
    X_pos = np.array([edge_feature(u, v) for u, v, _ in edges])
    X_neg = np.array([edge_feature(u, v_n) for u, v_n, _ in neg_edges])
    y_ev = np.array([1] * len(edges) + [0] * len(neg_edges))

    if predict_event_score_fn is not None:
        # REM: use dyad-choice model for existence scoring
        pos_scores = [
            predict_event_score_fn(u, v, int(d.get("time", 0))) for u, v, d in edges
        ]
        neg_scores = [
            predict_event_score_fn(u, v_n, int(d.get("time", 0)))
            for u, v_n, d in neg_edges
        ]
        auc = roc_auc_score(y_ev, np.array(pos_scores + neg_scores))
    else:
        auc = roc_auc_score(
            y_ev, exist_clf.predict_proba(np.vstack([X_pos, X_neg]))[:, 1]
        )

    # Tasks 2 & 3
    type_preds, type_labels = [], []
    order_scores, time_labels = [], []

    for u, v, d in edges:
        t = int(d.get("time", 0))
        t_norm = float(t) / t_max

        type_preds.append(predict_type_fn(u, v, t))
        type_labels.append(etype_le.transform([d.get("type", "unknown")])[0])
        order_scores.append(predict_order_score_fn(u, v, t))
        time_labels.append(t_norm)

    f1 = f1_score(type_labels, type_preds, average="macro", zero_division=0)

    if use_rank_normalise:
        # Hawkes: intensity sum is not in [0,1] — rank-normalise both sides
        order_arr = rank_normalise(order_scores)
        time_arr = rank_normalise(time_labels)
    else:
        # REM / GBDT: predict_time already outputs t_norm_hat ∈ [0,1]
        order_arr = np.array(order_scores)
        time_arr = np.array(time_labels)

    mae = float(np.mean(np.abs(order_arr - time_arr)))
    pair_acc = pairwise_precedence_accuracy(order_arr, time_arr)

    from sklearn.metrics import classification_report

    report = classification_report(
        type_labels,
        type_preds,
        labels=list(range(N_MARKS)),
        target_names=all_edge_types,
        zero_division=0,
    )

    print(f"\n{'─' * 54}")
    print(f"  {name}")
    print(f"{'─' * 54}")
    print(f"  Task 1  Edge existence   AUC-ROC  : {auc:.4f}")
    print(f"  Task 2  Edge type        Macro-F1 : {f1:.4f}")
    print(f"  Task 3  Pairwise order   Accuracy : {pair_acc:.4f}")
    print(f"  Task 3  Order score      MAE      : {mae:.4f}")
    print(f"\n  Per-class breakdown (Task 2):\n{report}")

    return {"auc": auc, "f1": f1, "pair_acc": pair_acc, "mae": mae}
