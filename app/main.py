from fastapi import FastAPI
from .schema import ApiResponse
from pydantic import ValidationError
app = FastAPI()


@app.get("/")
def health():
    return ApiResponse(True, data="Server Ready...")



@app.post("/predict")
def predict():
    data = {"heart_disease_chances":89.33}
    try :
        return ApiResponse(True, data=data,message="Successfully Predicted ")
    except ValidationError:
        return ApiResponse(False,data=None,message="Pydantic Validation Error")
    # will add model loading or prediction errors also



if __name__ =="__main__":
    import uvicorn
    uvicorn.run(app=app,host="0.0.0.0",port='8000')
