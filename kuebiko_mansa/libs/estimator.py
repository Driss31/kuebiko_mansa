"""Predict the next week's outgoing given the past 3 months (12 weeks)."""
from datetime import datetime
from typing import List

import joblib
import pandas as pd
from pydantic import BaseModel

from kuebiko_mansa.libs.featurizer import compute_predict_features
from kuebiko_mansa.utils.constants import (
    XGB_REGRESSOR_PATH,
    XGB_REGRESSOR_PREDICT_COLUMNS,
)


class Account(BaseModel):
    update_date: datetime
    balance: float
    id: int


class Transaction(BaseModel):
    account_id: int
    amount: float
    date: datetime


class ResponsePredict(BaseModel):
    account_id: int
    predicted_amount: float


class EstimatorError(Exception):
    """Base exception when a prediction can't be made."""


class Estimator:
    """Predict the next week's outgoing given the past 3 months (12 weeks)."""

    def __init__(self, xgb_regressor_path: str = XGB_REGRESSOR_PATH):
        """Initialize the Estimator class."""
        self.xgb_regressor = joblib.load(xgb_regressor_path)

    def predict(
        self, transactions: List[Transaction], account: Account
    ) -> ResponsePredict:
        """Return a ResponsePredict object."""
        df_transactions = pd.DataFrame([t.dict() for t in transactions]).sort_values(
            by="date", ascending=True
        )
        df_features = compute_predict_features(
            df_transactions=df_transactions,
            account_id=account.id,
            balance=account.balance,
        )

        if df_features.empty:
            raise EstimatorError("Too many weeks without data")

        try:
            df_features = df_features[XGB_REGRESSOR_PREDICT_COLUMNS]
        except KeyError:
            raise EstimatorError("Missing some features to make a prediction")

        return ResponsePredict(
            account_id=account.id,
            predicted_amount=min(self.xgb_regressor.predict(df_features), 0),
        )
