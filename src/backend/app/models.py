from pydantic import BaseModel
from typing import Optional, List

class userInput(BaseModel):
    skils: List[str]
    interest: str
    aspirations: str 
