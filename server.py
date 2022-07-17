from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import utilities
app = Flask(__name__)
CORS(app)


@app.route('/')
def Index():
    return render_template('index.html')


@app.route('/MSZoning', methods=['GET'])
def MSZoning():
    response = jsonify({
        'MSZoning': utilities.MSZoning()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    print(response)
    return response


@app.route('/Neighborhood')
def Neighborhood():
    response = jsonify({
        'Neighborhood': utilities.Neighborhood()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/BldgType')
def BldgType():
    response = jsonify({
        'BldgType': utilities.BldgType()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/HouseStyle')
def HouseStyle():
    response = jsonify({
        'HouseStyle': utilities.HouseStyle()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/RoofStyle')
def RoofStyle():
    response = jsonify({
        'RoofStyle': utilities.RoofStyle()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/ExterCond')
def ExterCond():
    response = jsonify({
        'ExterCond': utilities.ExterCond()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/Heating')
def Heating():
    response = jsonify({
        'Heating': utilities.Heating()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/Electrical')
def Electrical():
    response = jsonify({
        'Electrical': utilities.Electrical()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/PoolQC')
def PoolQC():
    response = jsonify({
        'PoolQC': utilities.PoolQC()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/Utilities')
def Utilities():
    response = jsonify({
        'Utilities': utilities.Utilities()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/Alley')
def Alley():
    response = jsonify({
        'Alley': utilities.Alley()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/Street')
def Street():
    response = jsonify({
        'Street': utilities.Street()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/Porch')
def Porch():
    response = jsonify({
        'Porch': utilities.Porch()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/Garage')
def Garage():
    response = jsonify({
        'Garage': utilities.Garage()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/CentralAir')
def CentralAir():
    response = jsonify({
        'CentralAir': utilities.CentralAir()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    _Garage = request.form['Garage']
    _Porch = request.form['Porch']
    _PoolQC = request.form['PoolQC']
    _Electrical = request.form['Electrical']
    _CentralAir = request.form['CentralAir']
    _Heating = request.form['Heating']
    _ExterCond = request.form['ExterCond']
    _RoofStyle = request.form['RoofStyle']
    _HouseStyle = request.form['HouseStyle']
    _BldgType = request.form['BldgType']
    _Neighborhood = request.form['Neighborhood']
    _Utilities = request.form['Utilities']
    _Alley = request.form['Alley']
    _Street = request.form['Street']
    _MSZoning = request.form['MSZoning']
    _LotArea = float(request.form['LotArea'])
    _TotalBsmtSF = float(request.form['TotalBsmtSF'])
    _GrLivArea = float(request.form['GrLivArea'])
    _FullBath = float(request.form['FullBath'])
    _BedroomAbvGr = float(request.form['BedroomAbvGr'])
    _KitchenAbvGr = float(request.form['KitchenAbvGr'])

    response = jsonify({
        'price': utilities.predict_price(_Garage, _Porch, _PoolQC, _Electrical, _CentralAir, _Heating, _ExterCond, _RoofStyle, _HouseStyle, _BldgType, _Neighborhood, _Utilities, _Alley, _Street, _MSZoning, _LotArea, _TotalBsmtSF, _GrLivArea, _FullBath, _BedroomAbvGr, _KitchenAbvGr)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == '__main__':
    utilities.load_saved_artifacts()
    app.run()
