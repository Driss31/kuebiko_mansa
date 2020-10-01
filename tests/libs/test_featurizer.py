"""Test for featurizer lib."""
from datetime import datetime

import pandas as pd
from pandas.testing import assert_frame_equal

from kuebiko_mansa.libs.featurizer import (
    get_balance_unique_account,
    remove_outliers_yearly_aggregation,
    unique_account_week_resampling,
)


def test_remove_outliers_yearly_aggregation():
    """Should remove only the last row as an outlier."""
    df = pd.DataFrame(
        {
            "account_id": [1] * 1000,
            "date": [datetime(2019, 12, 10)] * 1000,
            "amount": [1] * 999 + [1000],
        }
    )
    df_answer = remove_outliers_yearly_aggregation(df=df, std_multiplier=3)

    assert len(df_answer) == 999
    assert set(df.index) - set(df_answer.index) == {999}


def test_unique_account_week_resampling():
    """Should return a 5-rows DataFrame with the right week dates."""
    df = pd.DataFrame(
        {
            "account_id": [1] * 2,
            "date": [str(datetime(2020, i, 1)) for i in range(1, 3)],
            "amount": [1000] * 2,
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df_answer = unique_account_week_resampling(df=df, account_id=1)
    df_answer["date"] = df_answer["date"].astype(str)

    assert_frame_equal(
        df_answer,
        pd.DataFrame(
            {
                "date": [
                    "2020-01-06",
                    "2020-01-13",
                    "2020-01-20",
                    "2020-01-27",
                    "2020-02-03",
                ],
                "amount": [1000, 0, 0, 0, 1000],
                "account_id": [1, 1, 1, 1, 1],
            }
        ),
        check_dtype=False,
    )


def test_get_balance_unique_account():
    """Should return the right balance."""
    df_answer = get_balance_unique_account(
        df_transactions_unique_account=pd.DataFrame(
            {
                "date": [
                    "2020-01-06",
                    "2020-01-13",
                    "2020-01-20",
                    "2020-01-27",
                    "2020-02-03",
                ],
                "amount": [1000, 0, 0, 0, 1000],
                "account_id": [1, 1, 1, 1, 1],
            }
        ),
        balance=3000,
    )
    assert_frame_equal(
        df_answer,
        pd.DataFrame(
            {
                "account_id": [1, 1, 1, 1, 1],
                "date": [
                    "2020-02-03",
                    "2020-01-27",
                    "2020-01-20",
                    "2020-01-13",
                    "2020-01-06",
                ],
                "amount": [1000, 0, 0, 0, 1000],
                "balance": [2000, 2000, 2000, 2000, 1000],
            }
        ),
    )
