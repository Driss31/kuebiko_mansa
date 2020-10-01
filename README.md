# kuebiko_mansa

I decided to work on `option B: Predict the next week's outgoing given the past 3 months (12 weeks)` (reason explained bellow - notebooks 2).

You can find all what I did for the analysis part on the notebooks folder. Bellow I have listed some of the contents of the different notebooks: 

#### 0_history_filter_and_first_plots:
- Wrote a function fo filter data having less than 180 day of data `keep_accounts_by_history`.
- Check if all transactions happen before the balance `update_date`.
- Noticed two exact accounts (same balance and same transactions). Removed one of them.
- Plotted in a two dimensional plot, the distribution of number of days of history and the number of transactions per account. It shows that we could create 2 clusters and make a model for each one: for example active and less active users.
- I looked at the transactions with an amount of 0. Removing them didn't impact the filter of 180 days of history. 

#### 1_duplicate_transactions
- Some transactions (account_id, amount, date) appear many times. One can think of it as duplicates to remove. But since these (account_id, amount) can have different dates, they might be som regular expenses/incomes... Having the time of the transactions and some additional metadata about accounts would help understand if these are really duplicates or not.
- The data contains sometimes expenses and incomes that happen the same day and have the exact absolute value: -5000.84 / +5000.84 or +1842.50/-1842.50.
This might be transactions to clean:
1. Let say we expect a high expense, so we make a transfer of money to our main account to back up the expense. However, the chances that we do it the same day are very low because (usually we make a transfer before it appears in our bank account).
2. These expenses aren't seasonal, hence they are not regular expenses that can be planned in advance and for which the date is known (repayment of a loan for example).

I guess that the objective behind predicting the income of the next month or the expenses of the next week, is to know if an account balance is positive enough to pay back a credit. Hence, having these kind of transactions that happen the same day and in a non-seasonal way might be confusing and lead to poor predictions.

A better understanding of Mansa's business can help me understand the meaning of these transactions. They may be due to something else.

- In the other hand there are transactions that do not happen in the same day, but show the exact pattern as mentionned before: very specific positive values thatt equal to negative values: +109.99/-109.99.
These transactions need more time to be analyzed in order to take a decision about whether to keep or remove them.
onclusion: I will keep them.

I printed some examples.


#### 2_datetimes_months_incomes / 2_datetimes_weeks_expenses
These notebooks treat datetimes. 
- I aggregated incomes by month (expenses by week) and looked at the availability of data. I plotted a graph showing the weeks without data compared to those with.
This showed that some accounts, even though they have more than 180 days between the latest timestamp and the oldest one, they lack of data. I decided to create a function that removed these accounts for the training set.
- I also plotted aggregated value over time to see if one problem can be easier to predict than the other.

These notebooks helped me decided which problem to take:

I have decided to move forward on Problem B: Predict the next week's outgoing given the past 3 months (12 weeks) - for the following reasons:
- Aggregating data by week allows to see more seasonality compared to a month aggregation. In one hand, we are allowed to use 12 weeks == 3 months. Hence we might see some seasonality over the first week of each month, or the last. In the other hand we are allowed to use only 6 months, so we can't anticipate some obvious incomes like end of the year bonuses of yearly salary increase...
- We have more expenses data (16406 compared to 2439 for incomes)
- Having a week without any expense in a group of 12 weeks may be less impactful than having a month without income in a group of 6 months.


#### 3_outliers
When I plotted expenses over time in previous notebook, I saw many transaction that were very high compared to the rest. And these transactions were not expected (no seasonnality nor trend).

I have decided to remove them from the data. For each transaction, I looked at the mean and std of amounts in the same month and removed the transaction if it was higher than 3*std + mean. By doing that I removed many transactions that weren't that high. 
Then I aggregated by year and it was better. I only removed 2.5% of the data, and by looking at the distribution of removed data, the median was higher than the one when aggregating by month.

#### 4_enough_week_data
In this notebook, I made window samples. Each sample contained 12 consecutive weeks of expenses. Whenever  a sample had more than 6 empty values (6 weeks without expenses) I deleted it.

#### 5_ARIMA_models
In this notebook I made a gridsearch over the parameters of an ARIMA model. 

I first splitted data into train / test while keeping the right chronological order.

Concerning the evaluation I looked at both the MAE and MAPE. The MAE would be the main metric to look at if MANSA was only looking to reduced the global error of the predictions. However if MANSA is looking to reduce the error for each account, looking at the MAPE is more adequate. The issue with computing the MAPE is when the target is equal to 0. I looked at the percentage of zero-value target and it was very low so I just removed them for the evaluation.
 
#### 5_ML_models
Here I benchmarked a simple linear regression with a tree-basel model (an xgboost regressor).

Here i had to change my data set for training and testing:
- I computed the history balance for account. 
- I added the incomes as features as well.
- I splitted data into train, validation and test. I chose between the two models by looking at the results on the validation set. Then I made a prediction on the test set to get a final metric for our model to expect later.
- I computed both the quantiles of MAE and MAPE on the test set and compared them to the quantiles of the target values: 40% of data have an error twice higher than the real value.
- The results are obviously not great. A lot of work needs to be done to improve the model (preprocessing and choice of model).

### Scope
- This model will give a prediction only if there is at least 1 expense in the last 12 weeks. 


### Install necessary packages
```pip install -r requirements.txt```

### Run API
```uvicorn kuebiko_mansa.api.main:app --reload --workers 1 --host 0.0.0.0 --port 8000```
This should start the local server and you should be able to see the automatically generated API docs at http://0.0.0.0:8000/docs.
I have decided to raise an error when input data is not as scopped instead of returning a default value.

### Run test for API
```python -m tests.api.test_main```

### Run unit tests
```pytest```
I have only written few unit tests. I didn't write them all because I have already spent a lot of time on the analysis part (notebooks). But I wanted to show an example of how I would write tests.
