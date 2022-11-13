from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path_2 = 'models/DecisionTree_5.pkl'
model = load(model_path_2)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# get x and y somehow    
#     - query parameter
#     - get call / methods
#     - post call / methods ** 

@app.route("/sum", methods=['POST'])
def sum():
    x = request.json['x']
    y = request.json['y']
    z = x + y 
    return {'sum':z}



@app.route("/predict", methods=['POST'])
def predict_digit():
    image1 = request.json['image1']
    image2 = request.json['image2']
    predicted1 = model.predict([image1])
    predicted2 = model.predict([image2])
    resp = ""
    if int(predicted1[0]) == int(predicted2[0]):
        resp = "Both the given images belongs to the same digit"
    else:
        resp = "Both the given iimages does not belong to the same digit"
    return resp
