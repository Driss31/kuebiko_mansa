"""A set of lib functions that help compute the features as needed by the model."""
import pandas as pd

from kuebiko_mansa.utils.constants import (
    COUNT_PREVIOUS_WEEKS,
    DATES_RESAMPLING_RULE,
    STD_MULTIPLIER,
)


def remove_outliers_yearly_aggregation(
    df: pd.DataFrame, std_multiplier: int
) -> pd.DataFrame:
    """Remove very high transactions by comparing them to the yearly `mean + std_multiplier * std`."""
    df["year"] = df["date"].dt.year

    yearly_transactions_df = (
        df.groupby(["account_id", "year"])["amount"].agg(["mean", "std"]).reset_index()
    )
    df = pd.merge(
        df, yearly_transactions_df, on=["account_id", "year"], how="left", sort=False,
    )
    df["outlier"] = df["amount"] > (df["mean"] + std_multiplier * df["std"])

    return df[~df["outlier"]]


def unique_account_week_resampling(df: pd.DataFrame, account_id: int) -> pd.DataFrame:
    """Resample dates into weeks and sum transactions amounts for a unique account."""
    temp_df = df.sort_values(by="date", ascending=True)[["date", "amount"]]

    temp_df = temp_df.set_index("date", drop=True)

    temp_df = temp_df.resample(DATES_RESAMPLING_RULE).sum().reset_index()
    temp_df["account_id"] = account_id

    return temp_df.reset_index(drop=True)


def get_balance_unique_account(
    df_transactions_unique_account: pd.DataFrame, balance: float
) -> pd.DataFrame:
    """Compute weekly balance using transactions for a unique account."""
    grouped_transactions = (
        df_transactions_unique_account.groupby(["account_id", "date"])["amount"]
        .sum()
        .reset_index()
        .sort_values(by=["date"], ascending=False)
    )
    grouped_transactions["balance"] = round(
        balance - grouped_transactions["amount"].cumsum(), 2
    )

    return grouped_transactions.reset_index(drop=True)


def concatenate_incomes_outgoings_balance(
    df_outgoings: pd.DataFrame, df_incomes: pd.DataFrame, df_balance: pd.DataFrame
) -> pd.DataFrame:
    """
    Return a DataFrame containing for each week the sum of outgoings,
    incomes and the balance at its beginning.
    """
    df_week_transactions = pd.merge(
        df_outgoings,
        df_incomes,
        on=["date", "account_id"],
        how="left",
        sort=False,
        suffixes=["_outgoings", "_incomes"],
    )

    df_week_transactions = pd.merge(
        df_week_transactions,
        df_balance[["date", "balance", "account_id"]],
        on=["account_id", "date"],
        how="left",
        sort=False,
    )
    df_week_transactions.fillna(0, inplace=True)

    return df_week_transactions


# def check_enough_data_in_last_weeks(df: pd.DataFrame, max_empty_weeks: int) -> bool:
#     """Check if there is less than `max_empty_weeks` weeks without outgoings in 1-row DataFrame."""
#     if df[
#         (df[[col for col in df.columns if "outgoing" in col]] == 0).sum(axis=1)[0]
#         >= max_empty_weeks
#     ]:
#         return False
#     return True


def generate_predict_features(df: pd.DataFrame, count_previous_weeks) -> pd.DataFrame:
    """Return a 1-row DataFrame containing last weeks incomes, outgoing and balance."""
    for i in range(1, count_previous_weeks + 1):
        df[f"previous_outgoings_{i}"] = df["amount_outgoings"].shift(i, fill_value=0)
        df[f"previous_incomes_{i}"] = df["amount_incomes"].shift(i, fill_value=0)
    for i in range(
        1, count_previous_weeks
    ):  # Only to 11, since the actual balance is the one from the beginning of the week
        df[f"previous_balance_{i}"] = df["amount_incomes"].shift(i, fill_value=0)

    return df.sort_values(by="date", ascending=False).iloc[0:1]


def compute_predict_features(
    df_transactions: pd.DataFrame,
    account_id: int,
    balance: float,
    std_multiplier: int = STD_MULTIPLIER,
    count_previous_weeks: int = COUNT_PREVIOUS_WEEKS,
) -> pd.DataFrame:
    """Apply the pre-processing steps to DataFrames for prediction."""
    df_incomes = df_transactions[df_transactions["amount"] > 0]
    df_outgoings = df_transactions[df_transactions["amount"] < 0]

    # Remove outliers
    df_incomes = remove_outliers_yearly_aggregation(
        df=df_incomes, std_multiplier=std_multiplier
    )

    df_outgoings["amount"] = abs(df_outgoings["amount"])
    df_outgoings = remove_outliers_yearly_aggregation(
        df=df_outgoings, std_multiplier=std_multiplier
    )

    # Week resampling
    df_outgoings = unique_account_week_resampling(
        df=df_outgoings, account_id=account_id
    )
    df_incomes = unique_account_week_resampling(df=df_incomes, account_id=account_id)

    # Compute history balance
    df_outgoings["amount"] = df_outgoings["amount"] * (-1)
    df_week_transactions = pd.concat([df_incomes, df_outgoings], axis=0)

    df_balance = get_balance_unique_account(
        df_transactions_unique_account=df_week_transactions, balance=balance
    )

    # Merge DataFrames
    df_week_transactions = concatenate_incomes_outgoings_balance(
        df_outgoings=df_outgoings, df_incomes=df_incomes, df_balance=df_balance
    )

    predict_features = generate_predict_features(
        df=df_week_transactions, count_previous_weeks=count_previous_weeks
    )

    return predict_features
