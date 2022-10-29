import numpy as np

import bentoml
from bentoml.io import JSON
from pydantic import BaseModel
import warnings

class CarPurchaseApp(BaseModel):
    age: int
    annual_salary: float
    credit_card_debt: float
    net_worth: float

model_ref = bentoml.sklearn.get("car_purchase_model:latest")
model_runner = model_ref.to_runner()

svc = bentoml.Service("car_purchase_prediction", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=CarPurchaseApp), output=JSON())
async def predict(application_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = np.array(list(application_data.dict().values())).reshape(1, -1)
        prediction = await model_runner.predict.async_run(arr)
        print(f"Car purchase value is {prediction[0]}")
        result = prediction[0]
        return result