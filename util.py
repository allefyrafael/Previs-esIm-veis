import pickle
import json
import numpy as np
from sklearn.linear_model import LinearRegression  # Certifique-se de que o scikit-learn está instalado

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    if __model is None:
        with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
            try:
                __model = pickle.load(f)
            except ModuleNotFoundError:
                from sklearn.linear_model import _base as base
                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'sklearn.linear_model.base':
                            module = 'sklearn.linear_model._base'
                        return super().find_class(module, name)
                __model = CustomUnpickler(f).load()

    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))  # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
