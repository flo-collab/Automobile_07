import numpy as np
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from sklearn.linear_model import LinearRegression
from typing import Optional


#print('The scikit-learn version is {}.'.format(sklearn.__version__))
# Import modele for IA
path_model_2_var_poly = '../car_reglin_2_var_poly.sav'
model_2_var_poly = pickle.load(open(path_model_2_var_poly, 'rb'))

path_model_2_var = '../car_reglin_2_var.sav'
model_2_var = pickle.load(open(path_model_2_var, 'rb'))

my_X_test = np.matrix([2548,130])
model_2_var.predict(my_X_test)
print("my_result_price is : " ,model_2_var.predict(my_X_test))



app = FastAPI()
'''
class Fuel(BaseModel):
    gaz : 
    diesel :
'''
'''
class Energy(str,Enum):
    gaz : "gaz"
    diesel : "diesel"

@app.get("/energy/{energy}")
async def get_energy(energy : Energy):
    if energy == Energy.gaz:
        return {"energy" : energy , 'message': "C'est la meilleur bravo!"}
    if energy == Energy.diesel:
        return {"energy" : energy , 'message': "C'est la pire alala!"}

    #return {"energy" : energy , 'message': "De tou"}
'''
#price_predict

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/print_data/{curbweight}&{enginesize}")
async def print_data(curbweight : int , enginesize : int):
    stuff : float
    stuff = curbweight + 0.5
    return {"curbweight": curbweight, "enginesize":enginesize,"stuff":stuff }

'''
class Item(BaseModel):
    toto : str
    lalala : int

@app.post("/post_yannick/{item}")
async def create_item_y(item : Item):
    return item
'''





# price_predict 
@app.get("/predict_2_var/{curbweight}&{enginesize}")
async def predict_2_var(curbweight : int , enginesize : int):

    price_predict : float
    price_predict = float(model_2_var.predict(np.matrix([curbweight, enginesize])))

    return {"curbweight": curbweight, "enginesize":enginesize, "The price predicted is : ":price_predict}


#@app.POST('/')
#async def car_data():


#    pass





class Energy(str, Enum):
    gas = "gas"
    diesel = "diesel"

@app.get("/models/{energy}")
async def get_energy(energy: Energy):
    if energy == Energy.gas:
        return {"energy": energy, "message": "Ho ça roule bien "}

    if energy == Energy.diesel:
        return {"energy": energy, "message": "Alalala ça polue!!"}

    return {"energy": energy, "message": "Have some residuals"}
































if (__name__=="__main__"):
    uvicorn.run(app, host='http://127.0.0.1',port = 8000)

