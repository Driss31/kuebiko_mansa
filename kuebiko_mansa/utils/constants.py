"""File that contains all constants that are used throughout the project."""
from typing import List

XGB_REGRESSOR_PATH: str = "xgb_regressor.sav"

DATES_RESAMPLING_RULE: str = "W-MON"
STD_MULTIPLIER: int = 3
COUNT_PREVIOUS_WEEKS: int = 12

XGB_REGRESSOR_PREDICT_COLUMNS: List[str] = [
    "balance",
    "previous_outgoings_1",
    "previous_outgoings_2",
    "previous_outgoings_3",
    "previous_outgoings_4",
    "previous_outgoings_5",
    "previous_outgoings_6",
    "previous_outgoings_7",
    "previous_outgoings_8",
    "previous_outgoings_9",
    "previous_outgoings_10",
    "previous_outgoings_11",
    "previous_outgoings_12",
    "previous_incomes_1",
    "previous_incomes_2",
    "previous_incomes_3",
    "previous_incomes_4",
    "previous_incomes_5",
    "previous_incomes_6",
    "previous_incomes_7",
    "previous_incomes_8",
    "previous_incomes_9",
    "previous_incomes_10",
    "previous_incomes_11",
    "previous_incomes_12",
    "previous_balance_1",
    "previous_balance_2",
    "previous_balance_3",
    "previous_balance_4",
    "previous_balance_5",
    "previous_balance_6",
    "previous_balance_7",
    "previous_balance_8",
    "previous_balance_9",
    "previous_balance_10",
    "previous_balance_11",
]
