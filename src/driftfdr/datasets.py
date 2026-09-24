"""Model-error streams from real data sets, in their original time order.

A "fleet" of K models is built on one data set: every model is a multinomial
logistic regression on its own random subset of features, trained once on an
initial segment. Its monitored signals are the per-instance log-loss
(``values``) and 0/1 error (``errors``). Unlike the benchmark of Cerqueira et
al., rows are never shuffled, so autocorrelation, seasonality and slow
wandering of the data stay in the streams.

* ``insects_scenario``: INSECTS (Souza et al., 2020) with its documented
  change points, shared by all models: real, clustered drifts. In the river
  copy each concept ends with a long run of a single class (class 5, rare in
  training), which every model misclassifies. ``ground_truth="extended"``
  adds the starts of such runs, found from the labels alone, as changes.
* ``elec2_scenario``: Electricity (Harries, 1999) with no labelled drifts.
  For a chosen fraction of models the labels they are scored against are
  flipped with some probability after a known onset, which is a real drift in
  p(y | X) injected into real temporal structure.

Ground truth is regime based: a test is null iff no documented or injected
change lies between the start of its reference and the end of its window.
Undocumented natural changes (INSECTS has visible ones, Electricity has
seasons) therefore count against the detectors; this is the null-definition
problem of real data made explicit.

Data are downloaded by ``river.datasets`` on first use.
"""

from __future__ import annotations

import numpy as np

from .streams import NO_CHANGE, Scenario, ScenarioConfig

# Souza et al. (2020), abrupt_balanced variant
INSECTS_CHANGES = {"abrupt_balanced": [14352, 19500, 33240, 38682, 39510]}


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _load_river(dataset, drop=()):
    X, y = [], []
    for x, label in dataset:
        X.append([_to_float(v) for k, v in x.items() if k not in drop])
        y.append(label)
    return np.asarray(X), np.asarray(y)


def fit_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 1e-3, n_iter: int = 300, lr: float = 0.5):
    """Multinomial logistic regression by full-batch gradient descent on standardised features."""
    n, d = X.shape
    W = np.zeros((d + 1, n_classes))
    Xb = np.hstack([X, np.ones((n, 1))])
    Y = np.eye(n_classes)[y]
    for _ in range(n_iter):
        z = Xb @ W
        z -= z.max(axis=1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(axis=1, keepdims=True)
        W -= lr * (Xb.T @ (P - Y) / n + l2 * W)
    return W


def _predict_proba(W, X):
    z = np.hstack([X, np.ones((X.shape[0], 1))]) @ W
    z -= z.max(axis=1, keepdims=True)
    P = np.exp(z)
    return P / P.sum(axis=1, keepdims=True)


def model_fleet(X, y, n_models, n_features, train_size, rng, label_noise=None):
    """Losses and errors of ``n_models`` models on random feature subsets, shape ``(n_models, n)``."""
    classes, y_idx = np.unique(y, return_inverse=True)
    mu, sd = X[:train_size].mean(axis=0), X[:train_size].std(axis=0) + 1e-9
    Z = (X - mu) / sd
    losses = np.empty((n_models, len(y)))
    errors = np.empty((n_models, len(y)), dtype=np.int8)
    for k in range(n_models):
        cols = rng.choice(X.shape[1], size=min(n_features, X.shape[1]), replace=False)
        W = fit_softmax(Z[:train_size, cols], y_idx[:train_size], len(classes))
        P = _predict_proba(W, Z[:, cols])
        target = y_idx if label_noise is None else label_noise[k]
        losses[k] = -np.log(np.clip(P[np.arange(len(y)), target], 1e-6, 1.0))
        errors[k] = (P.argmax(axis=1) != target).astype(np.int8)
    return losses, errors


def _scenario(values, errors, changes, config, offset=0):
    """Scenario starting at ``offset`` (the end of the training segment), changes shifted accordingly."""
    values, errors = values[:, offset:], errors[:, offset:]
    changes = [[c - offset if c < NO_CHANGE else c for c in row] for row in np.atleast_2d(changes)]
    n = values.shape[0]
    changes = np.asarray(changes, dtype=np.int64).reshape(n, -1)
    first = changes[:, 0]
    later = np.stack([changes[:, 1:], changes[:, 1:]], axis=-1) if changes.shape[1] > 1 else None
    return Scenario(
        config,
        values,
        errors,
        change_start=first.copy(),
        change_end=first.copy(),
        drift_kind=np.where(first < NO_CHANGE, "abrupt", "none").astype(object),
        event=np.where(first < NO_CHANGE, 0, -1),
        later_changes=later,
    )


def label_runs(y: np.ndarray, min_length: int = 100) -> list[tuple[int, int]]:
    """``(start, end)`` of runs of one repeated label at least ``min_length`` long."""
    runs, start = [], 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[start]:
            if i - start >= min_length:
                runs.append((start, i))
            start = i
    return runs


def insects_scenario(
    n_models=50, n_features=8, train_size=3000, variant="abrupt_balanced", ground_truth="extended", seed=0
) -> Scenario:
    from river import datasets

    X, y = _load_river(datasets.Insects(variant=variant))
    rng = np.random.default_rng(seed)
    losses, errors = model_fleet(X, y, n_models, n_features, train_size, rng)
    points = set(INSECTS_CHANGES[variant])
    if ground_truth == "extended":
        points |= {start for start, _ in label_runs(y)}
    elif ground_truth != "documented":
        raise ValueError("ground_truth must be 'documented' or 'extended'")
    changes = np.tile(sorted(points), (n_models, 1))
    config = ScenarioConfig(n_streams=n_models, n_steps=len(y) - train_size, phi=np.nan, rho=np.nan, drift_fraction=1.0)
    return _scenario(losses, errors, changes, config, offset=train_size)


def elec2_scenario(
    n_models=50, n_features=4, train_size=5000, drift_fraction=0.2, flip=0.3, drift_events=0, seed=0
) -> Scenario:
    """Electricity with label-flip drift injected into ``drift_fraction`` of the models.

    With ``drift_events > 0`` the drifting models are split into that many groups
    sharing an onset (clustered drift); otherwise every drifting model gets its own.
    """
    from river import datasets

    # "date" is a running time index; a model must not see it
    X, y = _load_river(datasets.Elec2(), drop=("date",))
    X = X[:, ~np.isnan(X).any(axis=0)]
    y = y.astype(int)
    n = len(y)
    rng = np.random.default_rng(seed)
    n_drift = int(round(drift_fraction * n_models))
    drifting = rng.choice(n_models, size=n_drift, replace=False)
    lo, hi = int(0.3 * n), int(0.8 * n)
    if drift_events:
        event_onsets = rng.integers(lo, hi, size=drift_events)
        onsets = event_onsets[np.arange(n_drift) % drift_events]
    else:
        onsets = rng.integers(lo, hi, size=n_drift)
    target = np.tile(y, (n_models, 1))
    change = np.full(n_models, NO_CHANGE, dtype=np.int64)
    for k, tau in zip(drifting, onsets):
        flip_mask = rng.random(n - tau) < flip
        target[k, tau:] = np.where(flip_mask, 1 - y[tau:], y[tau:])
        change[k] = tau
    losses, errors = model_fleet(X, y, n_models, n_features, train_size, rng, label_noise=target)
    config = ScenarioConfig(n_streams=n_models, n_steps=n - train_size, phi=np.nan, rho=np.nan, drift_fraction=drift_fraction)
    return _scenario(losses, errors, change[:, None], config, offset=train_size)


def forward_error(errors: np.ndarray, span: int = 1000) -> np.ndarray:
    """Error rate over the next ``span`` steps from each step (shorter at the end): what
    the model will actually cost if it is not retrained now, known in hindsight."""
    e = np.asarray(errors, dtype=float)
    c = np.concatenate([np.zeros((e.shape[0], 1)), np.cumsum(e, axis=1)], axis=1)
    idx = np.arange(e.shape[1])
    hi = np.minimum(idx + span, e.shape[1])
    return (c[:, hi] - c[:, idx]) / (hi - idx)


def error_rate_view(scenario: Scenario, tolerance: float, span: int = 1000) -> Scenario:
    """Monitor the 0/1 error stream; judge alarms by *material and persistent* degradation.

    A test is null iff the error rate over the ``span`` steps following the tested
    window's start exceeds the observed error rate over the reference by at most
    ``tolerance``. Transient spikes that pass by themselves are therefore null:
    retraining for them would be wasted.
    """
    from dataclasses import replace

    errors = scenario.errors.astype(float)
    sc = replace(scenario, values=errors)
    return sc.with_material_null(tolerance, truth=forward_error(scenario.errors, span), truth_ref=errors)


def _fleet_scenario(X, y, n_models, n_features, train_size, seed, label):
    rng = np.random.default_rng(seed)
    losses, errors = model_fleet(X, y, n_models, n_features, train_size, rng)
    config = ScenarioConfig(n_streams=n_models, n_steps=len(y) - train_size, phi=np.nan, rho=np.nan, drift_fraction=np.nan)
    sc = _scenario(losses, errors, np.full((n_models, 1), NO_CHANGE), config, offset=train_size)
    sc.drift_kind = np.full(n_models, label, dtype=object)
    return sc


def covertype_scenario(n_models=50, n_features=10, train_size=5000, n_rows=100_000, seed=0) -> Scenario:
    """Forest Covertype (Blackard, 1998) in its original order, no drift labels.

    Needs scikit-learn (``fetch_covtype`` downloads the data on first use).
    """
    from sklearn.datasets import fetch_covtype

    X, y = fetch_covtype(return_X_y=True)
    return _fleet_scenario(X[:n_rows].astype(float), y[:n_rows], n_models, n_features, train_size, seed, "covertype")


def airlines_scenario(n_models=50, n_features=6, train_size=5000, n_rows=100_000, seed=0) -> Scenario:
    """Airlines delay data (OpenML 1169) in time order, no drift labels.

    Numeric columns plus a one-hot airline code; the high-cardinality airport codes
    are dropped. Needs scikit-learn (``fetch_openml`` downloads the data on first use).
    """
    from sklearn.datasets import fetch_openml

    frame = fetch_openml(data_id=1169, as_frame=True, parser="auto").frame.iloc[:n_rows]
    numeric = frame[["DayOfWeek", "Time", "Length"]].astype(float).to_numpy()
    airline = frame["Airline"].astype(str).to_numpy()
    codes = np.unique(airline)
    onehot = (airline[:, None] == codes[None, :]).astype(float)
    X = np.hstack([numeric, onehot])
    y = frame["Delay"].astype(int).to_numpy()
    return _fleet_scenario(X, y, n_models, n_features, train_size, seed, "airlines")


def airlines_hourly_scenario(n_models=50, n_features=6, seed=0) -> Scenario:
    """All 31 days of Airlines, 0/1 delay errors averaged per (day, hour) bucket.

    Models are trained on the first day. A step is one hour, so a window of 24
    steps covers a full daily cycle. Days are counted from changes of DayOfWeek.
    """
    from sklearn.datasets import fetch_openml

    from .preprocess import bucket_means

    frame = fetch_openml(data_id=1169, as_frame=True, parser="auto").frame
    dow = frame["DayOfWeek"].astype(int).to_numpy()
    day = np.concatenate([[0], np.cumsum(dow[1:] != dow[:-1])])
    hour = (frame["Time"].astype(float).to_numpy() // 60).astype(int).clip(0, 23)
    numeric = frame[["DayOfWeek", "Time", "Length"]].astype(float).to_numpy()
    airline = frame["Airline"].astype(str).to_numpy()
    onehot = (airline[:, None] == np.unique(airline)[None, :]).astype(float)
    X = np.hstack([numeric, onehot])
    y = frame["Delay"].astype(int).to_numpy()
    train_size = int(np.sum(day == 0))
    rng = np.random.default_rng(seed)
    _, errors = model_fleet(X, y, n_models, n_features, train_size, rng)
    bucket = (day * 24 + hour)[train_size:]
    bucket = bucket - bucket.min()
    rates = bucket_means(errors[:, train_size:], bucket)
    config = ScenarioConfig(n_streams=n_models, n_steps=rates.shape[1], phi=np.nan, rho=np.nan, drift_fraction=np.nan)
    sc = _scenario(rates, rates, np.full((n_models, 1), NO_CHANGE), config)
    sc.drift_kind = np.full(n_models, "airlines-hourly", dtype=object)
    return sc
