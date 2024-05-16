from geopy.geocoders import Nominatim
from etna.datasets import TSDataset
from etna.models import CatBoostMultiSegmentModel
from etna.transforms import LagTransform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import pickle

@st.cache_data
def load_model_forcasting():
    model =  CatBoostMultiSegmentModel()
    fit_model = model.load('../models/model_forcasting.sav')
    return fit_model
    
@st.cache_data
def load_model_kmeans():
    filename = '../models/model_kmeans.sav'
    model = pickle.load(open(filename, 'rb')) 
    return model

@st.cache_data
def load_data_pd():
    data = pd.read_pickle('../models/data.sav')
    return data

def predict_clastering(lat, lon):
    moscow_centre = (55.751003 * np.pi / 180.0, 37.617964 * np.pi / 180.0)  # перевод из градусов в радины
    R = 6371.0 # км
    x = (lat * np.pi / 180.0 - moscow_centre[0]) * R
    y = (lon * np.pi / 180.0 - moscow_centre[1]) * R * np.cos(moscow_centre[0]) 
    X = np.array([[x,y]])
    model_kmeans = load_model_kmeans()  
    number_claster = model_kmeans.predict([X[0]])[0]
    return number_claster

def predict_forcasting(number_claster : str):
    data = load_data_pd()
    ts = TSDataset(data, freq="MS")
    if horizon == 1:
        _test_end = "2020-12-01"
    if horizon == 2:
        _test_end = "2021-01-01"
    if horizon == 3:
        _test_end = "2021-02-01"
    if horizon == 4:
        _test_end = "2021-03-01"
    if horizon == 5:
        _test_end = "2021-04-01"
    if horizon == 6:
        _test_end = "2021-04-01"
    train_ts, test_ts = ts.train_test_split(
    train_start="2018-09-01",
    train_end="2020-11-01",
    test_start="2020-12-01",
    test_end=_test_end,
    )
    model =  CatBoostMultiSegmentModel()
    model = load_model_forcasting()
    HORIZON = horizon
    lags = LagTransform(in_column="target", lags=[6])
    transforms = [ lags]
    train_ts.fit_transform(transforms)
    future_ts = train_ts.make_future(future_steps=HORIZON, transforms=transforms)
    forecast_ts = model.forecast(future_ts)
    forecast_ts.inverse_transform(transforms)
    return forecast_ts


def predict_model():
    address = house + ', ' + street + ', Москва'
    geolocator = Nominatim(user_agent="base")
    location = geolocator.geocode(address)
    print(location)
    lat , lon = location.latitude, location.longitude
    print(lat , lon)
    # predict claster
    number_claster = predict_clastering(lat, lon)
    # find segment
    k = number_claster
    t = 11
    r = number_room
    print(type_house)
    if type_house == "Новостройка":
        t = 11
    elif type_house == "Вторичка":
         t = 1

    segment = f'k={k}_t={t}_r={r}'
    # forcasting
    forecast_ts = predict_forcasting(number_claster)
    forecast_df = forecast_ts.to_pandas()
    serias = forecast_df[segment]['target']

    return serias
  
st.title("Предиктивная система для анализа цен на невдижемость", anchor=None, help=None)
st.write("\n")
st.header("Введите параметры жилья")

st.subheader('Введите адрес жилья')
street = st.text_input("Введите улицу")
house = st.text_input("Введиет дом")

st.subheader('Введите другие параметры')
type_house = st.selectbox(
    "Тип жилья",
    ("Новостройка", "Вторичка"))
number_room = st.selectbox(
    "Количество комнат",
    (1, 2, 3, 4))
horizon = st.select_slider(
    "Продолжительность предсказания",
    options=[1, 2, 3, 4, 5, 6])


if st.button('Predict!!!'):
    serias = predict_model()
    fig, ax = plt.subplots()
    ax.plot(serias)
    st.pyplot(fig)



    