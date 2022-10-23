import numpy as np

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
#dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=JSON())
async def classify(application_data):
    # vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(application_data)
    print(prediction)
    result = prediction[0]
    return result
