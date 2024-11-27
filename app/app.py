import cv2
import numpy as np
from tableocr import TableOCR
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Prediction(BaseModel):
    energi: float
    protein: float
    lemak: float
    karbohidrat: float
    serat: float
    natrium: float

class NutritionFactExtractor:
    
    EXTENSIONS = ['png', 'jpg', 'jpeg', 'heic']
    
    def __init__(self):
        self.table_engine = PPStructure(layout=False, show_log=True)
        self.nutrients = {
            'energi': ['energi total', 'total energi', 'energy total', 'total energy', 'calories', 'energi', 'energy'],
            'protein': ['protein'],
            'lemak': ['lemak total', 'total lemak', 'total fat'],
            'karbohidrat': ['karbohidrat', 'karbohidrat total', 'total carbohydrate'],
            'serat': ['serat', 'serat pangan', 'dietary fiber'],
            'natrium': ['natrium', 'garam', 'salt', 'sodium']
        }