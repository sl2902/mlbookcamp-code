import numpy as np

import bentoml
from bentoml.io import JSON
from pydantic import BaseModel
import warnings

class CreditCardChurnApp(BaseModel):
    customer_age: int
    dependent_count: int
    gender: str
    education_level: str
    marital_status: str 
    income_category: str 
    card_category: str 
    credit_limit: int
    months_on_book: int 
    total_relationship_count: int
    total_revolving_bal: float

model_ref = bentoml.sklearn.get("credit_card_churn_model:latest")
dv = model_ref.custom_objects["dictVectorizer"]
model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_card_churn_prediction", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=CreditCardChurnApp), output=JSON())
async def predict(application_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # arr = np.array(list(application_data.dict().values())).reshape(1, -1)
        credit_card_info = application_data.dict()
        arr = dv.transform(credit_card_info)
        prediction = await model_runner.predict_proba.async_run(arr)
        print(f"Probablity of churn is {prediction[:, 1]}")
        result = prediction[:, 1]
        return {"status": "Attrited Customer" if result > 0.5 else "Existing Customer"}