from datetime import datetime
import json

import requests


# You can use this function to test your api
# Make sure the uvicorn server is running locally on `http://127.0.0.1:8000/`
# or change the hardcoded localhost below
def test_predict():
    """
    Test the predict route with test data
    """
    test_account = {"id": 1, "balance": 10000, "update_date": str(datetime(2020, 1, 1))}
    test_transactions = [
        {"account_id": 1, "date": str(datetime(2019, i, 1)), "amount": -1000}
        for i in range(1, 10)
    ]

    test_data = {"account": test_account, "transactions": test_transactions}

    print("Calling API with test data:")
    print(json.dumps(test_data))

    response = requests.post("http://0.0.0.0:8001/predict", data=json.dumps(test_data))

    print("Response: ", response)
    print(response.json())

    assert response.status_code == 200


if __name__ == "__main__":
    test_predict()
