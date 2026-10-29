from data import FEATURES, TARGET, make_data


def test_columns_and_size():
    df = make_data(500)
    assert list(df.columns) == FEATURES + [TARGET]
    assert len(df) == 500


def test_same_seed_same_data():
    assert make_data(200, seed=1).equals(make_data(200, seed=1))


def test_positive_rate_is_realistic():
    rate = make_data()[TARGET].mean()
    assert 0.112 < rate < 0.448


def test_no_missing_values():
    assert not make_data().isna().any().any()
