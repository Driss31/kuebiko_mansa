"""Create an instance of FastAPI."""
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, validator

from kuebiko_mansa.libs.estimator import Account, Estimator, Transaction
from kuebiko_mansa.utils.constants import XGB_REGRESSOR_PATH


class RequestPredict(BaseModel):
    account: Account
    transactions: List[Transaction]

    @validator("transactions")
    def validate_transaction_history(cls, v):
        # validate that the transaction list passed has at least 6 months history
        if len(v) < 1:
            raise ValueError("Must have at least one Transaction")

        oldest_t = v[0].date
        newest_t = v[0].date
        for t in v[1:]:
            if t.date < oldest_t:
                oldest_t = t.date
            if t.date > newest_t:
                newest_t = t.date

        assert (newest_t - oldest_t).days > 183, "Not Enough Transaction History"

        return v


app = FastAPI()


@app.post("/predict")
async def root(predict_body: RequestPredict):
    transactions = predict_body.transactions
    account = predict_body.account

    estimator = Estimator(xgb_regressor_path=XGB_REGRESSOR_PATH)
    prediction = estimator.predict(transactions=transactions, account=account)

    return {
        "account_id": prediction.account_id,
        "predicted_amount": prediction.predicted_amount,
    }
