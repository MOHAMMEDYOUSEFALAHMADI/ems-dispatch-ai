from datetime import date

import numpy as np

from ems_dispatch_ai import synth
from ems_dispatch_ai.forecast import DemandForecaster, HistoricalAverage, evaluate
from ems_dispatch_ai.geo import Grid


def test_gradient_boosting_beats_historical_average_on_ramadan():
    g = Grid(6, 6)
    # train covers Ramadan 2024; test is inside Ramadan 2025 (different Gregorian dates)
    train = synth.hourly_counts(date(2024, 2, 1), 120, g, seed=1, daily_total=400)
    test = synth.hourly_counts(date(2025, 3, 5), 14, g, seed=2, daily_total=400)
    gb = DemandForecaster(g, max_iter=150).fit(train)
    ha = HistoricalAverage().fit(train)
    m_gb = evaluate(test["count"].to_numpy(), gb.predict(test), test["rate_true"].to_numpy())
    m_ha = evaluate(test["count"].to_numpy(), ha.predict(test), test["rate_true"].to_numpy())
    assert m_gb["rate_mae"] < m_ha["rate_mae"]
    assert m_gb["poisson_deviance"] < m_ha["poisson_deviance"]
    assert abs(m_gb["total_bias_pct"]) < 10


def test_predict_block_shape_and_positive():
    g = Grid(4, 4)
    train = synth.hourly_counts(date(2025, 1, 1), 30, g, seed=1)
    f = DemandForecaster(g, max_iter=30).fit(train)
    blk = f.predict_block(date(2025, 2, 3), [8, 9, 10, 11])
    assert blk.shape == (16,) and (blk > 0).all()
