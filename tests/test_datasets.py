import numpy as np

from driftfdr.datasets import _scenario, fit_softmax, label_runs, model_fleet
from driftfdr.streams import NO_CHANGE, ScenarioConfig


def test_label_runs():
    y = np.array([1] * 5 + [2] * 150 + [1, 2] * 10 + [3] * 120)
    assert label_runs(y, 100) == [(5, 155), (175, 295)]


def test_scenario_with_several_changes_and_offset():
    values = np.zeros((2, 1000))
    sc = _scenario(values, values.astype(np.int8), [[300, 700], [300, 700]], ScenarioConfig(n_streams=2), offset=100)
    cs, _ = sc.changes()
    assert cs.tolist() == [[200, 600], [200, 600]]
    assert sc.is_null([0], 0, 200)[0] and not sc.is_null([0], 0, 201)[0]
    assert sc.is_null([0], 200, 600)[0] and not sc.is_null([0], 200, 601)[0]
    assert sc.is_null([0], 600, 900)[0]


def test_model_fleet_learns_separable_classes():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=2000)
    X = rng.normal(size=(2000, 6)) + 3 * np.eye(3)[y] @ rng.normal(size=(3, 6))
    losses, errors = model_fleet(X, y, n_models=3, n_features=6, train_size=1000, rng=rng)
    assert losses.shape == (3, 2000)
    assert errors[:, 1000:].mean() < 0.2
    assert fit_softmax(X[:100], y[:100], 3).shape == (7, 3)


def test_no_change_streams_are_always_null():
    values = np.zeros((1, 500))
    sc = _scenario(values, values.astype(np.int8), [[NO_CHANGE]], ScenarioConfig(n_streams=1))
    assert sc.is_null([0], 0, 500)[0] and sc.drifting.size == 0


def test_forward_error_and_material_null():
    from driftfdr.datasets import error_rate_view, forward_error

    e = np.zeros((1, 3000), dtype=np.int8)
    e[0, 1000:1030] = 1  # short spike: 3% of the next 1000 steps
    e[0, 2500:] = 1  # persistent degradation
    f = forward_error(e, span=1000)
    assert f[0, 0] == 0 and f[0, 2600] == 1
    sc = error_rate_view(_scenario(e.astype(float), e, [[NO_CHANGE]], ScenarioConfig(n_streams=1)), 0.05, span=1000)
    # the spike window is null (it passes by itself), the start of the lasting rise is not
    assert sc.is_null([0], 0, 1100, ref_len=300, window=100)[0]
    assert not sc.is_null([0], 0, 2600, ref_len=300, window=100)[0]


def test_bucket_means_and_tolerance():
    from driftfdr import bucket_means, tolerance_from_cost

    v = np.array([[1.0, 3.0, 5.0, 7.0]])
    assert bucket_means(v, np.array([0, 0, 2, 2]), 3).tolist() == [[2.0, 2.0, 6.0]]  # empty bucket 1 carried
    assert tolerance_from_cost(10.0, 200) == 0.05
