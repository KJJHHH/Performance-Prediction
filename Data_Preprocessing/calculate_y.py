import pandas as pd
# encoding
def calculate_ymean(data):
    y_mean = (data['life_insurance']+data['property_insurance'])/2
    data['y_mean'] = y_mean
    return data