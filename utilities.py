import json
import joblib
import numpy as np

__MSZoning = None
__columns = None
__model = None
__Neighborhood = None
__BldgType = None
__HouseStyle = None
__RoofStyle = None
__ExterCond = None
__Heating = None
__Electrical = None
__PoolQC = None
__Garage = None
__Porch = None
__CentralAir = None
__Utilities = None
__Alley = None
__Street = None


def Garage():
    return __Garage


def Porch():
    return __Porch


def CentralAir():
    return __CentralAir


def Utilities():
    return __Utilities


def Alley():
    return __Alley


def Street():
    return __Street


def MSZoning():
    return __MSZoning


def Neighborhood():
    return __Neighborhood


def BldgType():
    return __BldgType


def HouseStyle():
    return __HouseStyle


def RoofStyle():
    return __RoofStyle


def ExterCond():
    return __ExterCond


def Heating():
    return __Heating


def Electrical():
    return __Electrical


def PoolQC():
    return __PoolQC


def load_saved_artifacts():
    global __columns
    global __model
    global __MSZoning
    global __Neighborhood
    global __BldgType
    global __HouseStyle
    global __RoofStyle
    global __ExterCond
    global __Heating
    global __Electrical
    global __PoolQC
    global __Garage
    global __Porch
    global __CentralAir
    global __Utilities
    global __Alley
    global __Street

    with open('./artifacts/columns.json', 'r') as f:
        __columns = json.load(f)['columns']
        __MSZoning = __columns[78:83]
        __Neighborhood = __columns[45:70]
        __BldgType = __columns[40:45]
        __HouseStyle = __columns[32:40]
        __RoofStyle = __columns[26:32]
        __ExterCond = __columns[21:26]
        __Heating = __columns[15:21]
        __Electrical = __columns[8:13]
        __PoolQC = __columns[4:8]
        __Garage = __columns[0:2]
        __Porch = __columns[2:4]
        __CentralAir = __columns[13:15]
        __Utilities = __columns[70:73]
        __Alley = __columns[73:76]
        __Street = __columns[76:78]

    with open('./artifacts/house_price_prediction_1.jl', 'rb') as f:
        __model = joblib.load(f)


def predict_price(_Garage, _Porch, _PoolQC, _Electrical, _CentralAir, _Heating, _ExterCond, _RoofStyle, _HouseStyle, _BldgType, _Neighborhood, _Utilities, _Alley, _Street, _MSZoning, _LotArea, _TotalBsmtSF, _GrLivArea, _FullBath, _BedroomAbvGr, _KitchenAbvGr):
    global __columns
    X = np.zeros(len(__columns))
    Garage_index = __columns.index(_Garage)
    if Garage_index > 0:
        X[Garage_index] = 1
    Porch_index = __columns.index(_Porch)
    if Porch_index > 0:
        X[Porch_index] = 1
    PoolQC_index = __columns.index(_PoolQC)
    if PoolQC_index > 0:
        X[PoolQC_index] = 1
    Electrical_index = __columns.index(_Electrical)
    if Electrical_index >= 0:
        X[Electrical_index] = 1
    CentralAir_index = __columns.index(_CentralAir)
    if CentralAir_index >= 0:
        X[CentralAir_index] = 1
    Heating_index = __columns.index(_Heating)
    if Heating_index >= 0:
        X[Heating_index] = 1
    ExterCond_index = __columns.index(_ExterCond)
    if ExterCond_index >= 0:
        X[ExterCond_index] = 1
    RoofStyle_index = __columns.index(_RoofStyle)
    if RoofStyle_index >= 0:
        X[RoofStyle_index] = 1
    HouseStyle_index = __columns.index(_HouseStyle)
    if HouseStyle_index >= 0:
        X[HouseStyle_index] = 1
    BldgType_index = __columns.index(_BldgType)
    if BldgType_index >= 0:
        X[BldgType_index] = 1
    Neighborhood_index = __columns.index(_Neighborhood)
    if Neighborhood_index >= 0:
        X[Neighborhood_index] = 1
    Utilities_index = __columns.index(_Utilities)
    if Utilities_index >= 0:
        X[Utilities_index] = 1
    Alley_index = __columns.index(_Alley)
    if Alley_index >= 0:
        X[Alley_index] = 1
    Street_index = __columns.index(_Street)
    if Street_index >= 0:
        X[Street_index] = 1
    MSZoning_index = __columns.index(_MSZoning)
    if MSZoning_index >= 0:
        X[MSZoning_index] = 1
    X[83] = _LotArea
    X[84] = _TotalBsmtSF
    X[85] = _GrLivArea
    X[86] = _FullBath
    X[87] = _BedroomAbvGr
    X[88] = _KitchenAbvGr
    print(round(__model.predict([X])[0], 2))
    return round(__model.predict([X])[0], 2)


if __name__ == '__main__':
    load_saved_artifacts()
    print(MSZoning())
    print(Neighborhood())
    print(BldgType())
    print(HouseStyle())
    print(RoofStyle())
    print(Heating())
    print(ExterCond())
    print(Electrical())
    print(PoolQC())
    print(Garage())
    print(Porch())
    print(Utilities())
    print(Alley())
    print(Street())
    print(CentralAir())
    print(predict_price('Garage_YES', 'Porch_YES', 'PoolQC_Ex', 'Electrical_FuseA', 'CentralAir_N', 'Heating_Floor', 'ExterCond_Ex', 'RoofStyle_Flat',
                        'HouseStyle_1.5Fin', 'BldgType_1Fam', 'Neighborhood_Blmngtn', 'Utilities_AllPub', 'Alley_Grvl', 'Street_Grvl', 'MSZoning_FV', 16388, 1740, 2560, 2, 4, 1))
