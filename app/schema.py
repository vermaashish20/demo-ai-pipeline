from typing import Generic, TypeVar, Optional, List
from pydantic import BaseModel

T = TypeVar("T")


# 1. UNIFIED RESPONSE SCHEMA
class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    message: str


# 2. ML MODEL INPUT SCHEMA
class MLModelInput(BaseModel):
    age: int
    sex: str
    resting_bp_systolic: int
    resting_bp_diastolic: int
    hdl: int
    ldl: int
    triglycerides: int
    hba1c: float
    bmi: float
    resting_heart_rate: int
    max_heart_rate_achieved: int
    chest_pain_type: str
    exercise_induced_angina: bool
    st_depression: float
    family_history: bool
    smoker_status: str
    alcohol_units_per_week: float
    exercise_minutes_per_week: int
    sleep_hours: float
    stress_score: float
    wearable_owner: bool
    daily_steps: int
    diet_quality_score: float

    class Config:
        json_schema_extra = {
            "example": {
                "age": 44,
                "sex": "Male",
                "resting_bp_systolic": 117,
                "resting_bp_diastolic": 74,
                "hdl": 57,
                "ldl": 106,
                "triglycerides": 119,
                "hba1c": 5.2,
                "bmi": 26.5,
                "resting_heart_rate": 74,
                "max_heart_rate_achieved": 165,
                "chest_pain_type": "Atypical Angina",
                "exercise_induced_angina": False,
                "st_depression": 0.3,
                "family_history": False,
                "smoker_status": "Never",
                "alcohol_units_per_week": 2.9,
                "exercise_minutes_per_week": 86,
                "sleep_hours": 5.4,
                "stress_score": 19.8,
                "wearable_owner": True,
                "daily_steps": 7731,
                "diet_quality_score": 62.9
            }
        }


# 3. ML MODEL OUTPUT SCHEMA
class MLModelOutput(BaseModel):
    has_heart_disease: int
    heart_disease_probability_percent: float
    probability: float
    risk_level: str
    model_used: str

