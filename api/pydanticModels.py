from pydantic import BaseModel

class featureColumns(BaseModel):
    gender:str
    SeniorCitizen:int
    Partner:str
    Dependents:str
    tenure:int
    PhoneService:str
    MultipleLines:str
    InternetService:str
    OnlineSecurity:str
    OnlineBackup:str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    MonthlyCharges:float
    TotalCharges:float
    

class predictionData(BaseModel):
    Churn: bool
    probability: int

class modelMetadata(BaseModel):
    model: str
    metric : str

