from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR
import numpy as np
import cv2
import re

app = FastAPI()

class Prediction(BaseModel):
    nutriscore: float
    grade: str
    calories: float
    fat: float
    sugar: float
    fiber: float
    protein: float
    natrium: float
    vegetable: float

class NutritionFactExtractor:
    
    ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'heic']
    
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='id', det_model_dir='./Models/en_PP-OCRv3_det_infer', rec_model_dir='./Models/latin_PP-OCRv3_rec_infer', cls_model_dir='./Models/ch_ppocr_mobile_v2.0_cls_infer', use_space_char=True)
        self.nutrition_keyword_units = {
            'serving_size': (['takaran saji', 'takaransaji', 'ukuran sajian', 'ukuransajian'], ['g']),
            'calories': (['energi total', 'total energi', 'energy total', 'total energy', 'calories', 'energi', 'energy'], ['kkal', 'kcal', 'kj']),
            'protein': (['protein'], ['g']),
            'fat': (['lemak jenuh', 'lemakjenuh', 'saturated fat', 'saturatedfat'], ['g']),
            'sugar': (['gula', 'sugar'], ['g']),
            'total_carbohydrate': (['karbohidrat total', 'karbohidrattotal', 'total karbohidrat', 'totalkarbohidrat', 'total carbohydrate'], ['g']),
            'fiber': (['serat', 'serat pangan', 'dietary fiber'], ['g']),
            'natrium': (['(natrium)', 'garam', 'salt', 'sodium'], ['mg', 'g'])
        }
        self.negative_quantization = {
            'calories': [335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350],
            'sugar': [3.4, 6.8, 10, 14, 17, 20, 24, 27, 31, 34, 37, 41, 44, 48, 51],
            'fat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'natrium': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
        }

        self.positive_quatization = {
            'fiber': [3.0, 4.1, 5.2, 6.3, 7.4],
            'protein': [2.4, 4.8, 7.2, 9.6, 12, 14, 17]
        }

        self.nutrition_value = {
            'vegetable': 0.0
        }
        
        self.nutri_grade_class = {
            'A': 0,
            'B': 2,
            'C': 10,
            'D': 18,
            'E': 19,
        }
        
    
    def check_file(self, filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
               
    def read_img(self, img_data):
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        result = self.ocr.ocr(img, cls=True)
        for box in result[0]:
            box[1] = (box[1][0].lower(), box[1][1])
        return result[0]
    
    def find_nutrition_value_units(self, text, *keywords):
        for keyword in keywords[0]:
            for j, line in enumerate(text):
                line[1] = (line[1][0].lower(), line[1][1])
                
                if not keyword in line[1][0]: continue
                
                value = None
                founded_unit = ''
                
                for unit in keywords[1]:
                    value = re.search(r'\D*'+ re.escape(keyword) +r'\s?(\d+,?\.?\d*)[\s]?'+re.escape(unit), line[1][0])
                    if not value:
                        value = re.search(r'(\d+,?\.?\d*)[\s]?'+re.escape(unit), text[j+1][1][0])
                        if not value:
                            value = re.search(r'(\d+,?\.?\d*)[\s]?'+re.escape(unit), text[j-1][1][0])
                            if not value: continue
                    
                    founded_unit = unit
                    break
                if not value: continue
                
                return [float(value.group(1).replace(',', '.')), founded_unit]
        return [0.0, '']
    
    def get_100_g_scale(self, nutrition_value_units):
        return 100/nutrition_value_units['serving_size'][0]
    
    def get_nutrition_value(self, nutrition_value_units, scaled_nutrition):
        extracted_nutrition_value = {}
        for nutrition, value in nutrition_value_units.items():
            if nutrition == 'calories' and (value[1]=='kkal' or value[1]=='kcal'):
                extracted_nutrition_value[nutrition] = round(value[0]*4.184, 3)*scaled_nutrition
            elif value[1]=='mg':
                extracted_nutrition_value[nutrition] = (value[0]/1000)*scaled_nutrition
            elif value[1]=='':
                if nutrition == 'fiber' and 'total_carbohydrate' in nutrition_value_units and 'sugar' in nutrition_value_units:
                    extracted_nutrition_value[nutrition] = (nutrition_value_units['total_carbohydrate'][0] - nutrition_value_units['sugar'][0])*scaled_nutrition
                else: extracted_nutrition_value[nutrition] = value[0]*scaled_nutrition
            else: extracted_nutrition_value[nutrition] = value[0]*scaled_nutrition
        return extracted_nutrition_value
    
    def count_nutri_score(self, extracted_nutrition_value):
        nutri_score = 0
        nutrition_value = self.nutrition_value

        for nutrition, value in extracted_nutrition_value.items():
            if nutrition in self.negative_quantization:
                nutrition_value.update({nutrition:value})
                for i, limit in enumerate(self.negative_quantization[nutrition]):
                    if value <= limit:
                        nutri_score += i
                        break
                    elif i == len(self.negative_quantization[nutrition])-1: nutri_score += i+1
            if nutrition in self.positive_quatization:
                nutrition_value.update({nutrition:value})
                for i, limit in enumerate(self.positive_quatization[nutrition]):
                    if value <= limit:
                        nutri_score -= i
                        break
                    elif i == len(self.positive_quatization[nutrition])-1: nutri_score -= i+1
        
        return nutrition_value, nutri_score
        
    def get_grade(self, nutri_score):
        nutri_grade = ''

        for grade, limit in self.nutri_grade_class.items():
            if nutri_score <= limit:
                nutri_grade = grade
                break
            elif grade == 'E': nutri_score = grade
            
        return nutri_grade
        
nutrition_fact_extractor = NutritionFactExtractor()

@app.get("/")
def home():
    return {"health_check": "OK"}
@app.post("/predict", response_model=Prediction)
async def prediction(file: UploadFile = File(...)):
    if nutrition_fact_extractor.check_file(file.filename):
        contents = await file.read()
        img_text = nutrition_fact_extractor.read_img(contents)

        nutrient_value_units = {}
        for nutrient, variations in nutrition_fact_extractor.nutrition_keyword_units.items():
            value = nutrition_fact_extractor.find_nutrition_value_units(img_text, *variations)
            nutrient_value_units[nutrient] = value
        scaled_nutrition = nutrition_fact_extractor.get_100_g_scale(nutrient_value_units)
        extracted_nutrition_value = nutrition_fact_extractor.get_nutrition_value(nutrient_value_units, scaled_nutrition)
        nutrition_value, nutriscore = nutrition_fact_extractor.count_nutri_score(extracted_nutrition_value)
        grade = nutrition_fact_extractor.get_grade(nutriscore)
        nutrition_value.update({'nutriscore': nutriscore})
        nutrition_value.update({'grade': grade})
        print(nutrition_value)
        return Prediction(**nutrition_value)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
