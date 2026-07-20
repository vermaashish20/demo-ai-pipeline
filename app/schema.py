from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, Field

T = TypeVar("T")

# 1. UNIFIED RESPONSE SCHEMA
class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    message: str



# ML model Input validation here 
class MLModelInput(BaseModel):
    pass 


class MLModelOutput(BaseModel):
    pass
