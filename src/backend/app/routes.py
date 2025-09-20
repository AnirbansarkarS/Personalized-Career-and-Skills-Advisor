from fastapi import APIRouter


from .models import userInput

router = APIRouter(prefix="/api")

@router.get("/")
def read_root():
   return {"message" : "FastApi is running"}

@router.post("/career")
def get_prediction(data: userInput):
    
    result = "ok"
    
    return result
