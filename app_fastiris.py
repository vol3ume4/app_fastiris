# Try 5.1,3.5,1.4,0.2	Setosa
# Try 6,	2.9,	4.5,	1.5	Versicolor
# Try 6.2,3.4,5.4,2.3 for Virginica


#import os
#os.chdir('D:\\ssridhar\\research\\AnalyticsBigdataML\\ML Course\\GreatLearning\\PES ML-2\\Session 7\\fast_logreg')
#os.getcwd()
import numpy as np
# 1. Library imports
import uvicorn
from fastapi import FastAPI
from IrisSpecies import IrisSpecies
import pickle

# Create the app object
app = FastAPI()
pickle_in = open('logreg.pkl',"rb")
logreg=pickle.load(pickle_in)

# data dictionary for label
variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# Route with a single parameter, returns the parameter within a message
# Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}

# Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_iris_species(data:IrisSpecies):
    data = data.dict()
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width'] 
    petal_length = data['petal_length']
    petal_width = data['petal_width']
    features = np.array([sepal_length,sepal_width,petal_length,petal_width])
    query = features.reshape(1,-1)
    prediction = variety_mappings[logreg.predict(query)[0]]
    return {'prediction': prediction}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload