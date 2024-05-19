from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from etna.datasets import TSDataset
from etna.models import CatBoostMultiSegmentModel
from etna.transforms import LagTransform
from streamlit_option_menu import option_menu

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import pickle
from matplotlib.ticker import ScalarFormatter

@st.cache_data
def load_model_forcasting():
    model =  CatBoostMultiSegmentModel()
    fit_model = model.load('../models/model_forcasting.sav')
    return fit_model

@st.cache_data
def load_model_forcasting_days():
    model =  CatBoostMultiSegmentModel()
    fit_model = model.load('../models/model_forcasting_days.sav')
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

@st.cache_data
def load_data_days_pd():
    data = pd.read_pickle('../models/data_days.sav')
    return data

def predict_clastering(lat, lon):
    moscow_centre = (55.751003 * np.pi / 180.0, 37.617964 * np.pi / 180.0)  # перевод из градусов в радианы
    R = 6371.0 # км
    x = (lat * np.pi / 180.0 - moscow_centre[0]) * R
    y = (lon * np.pi / 180.0 - moscow_centre[1]) * R * np.cos(moscow_centre[0]) 
    X = np.array([[x, y]])
    model_kmeans = load_model_kmeans()  
    number_claster = model_kmeans.predict([X[0]])[0]
    return number_claster

def predict_forcasting(number_claster: str):
    data = load_data_pd()
    ts = TSDataset(data, freq="MS")
    _test_end = f"2020-{11 + horizon}-01"
    
    train_ts, test_ts = ts.train_test_split(
        train_start="2018-09-01",
        train_end="2020-11-01",
        test_start="2020-12-01",
        test_end="2021-05-01",
    )
    model = load_model_forcasting()
    HORIZON = horizon
    lags = LagTransform(in_column="target", lags=[6])
    transforms = [lags]
    train_ts.fit_transform(transforms)
    future_ts = train_ts.make_future(future_steps=HORIZON, transforms=transforms)
    forecast_ts = model.forecast(future_ts)
    forecast_ts.inverse_transform(transforms)
    return forecast_ts, train_ts, test_ts

def predict_forcasting_days(number_claster: str):
    data = load_data_days_pd()
    ts = TSDataset(data, freq="D")
    _test_end = f"2020-{11 + horizon}-01"
    
    train_ts, test_ts = ts.train_test_split(
        train_start="2018-09-25",
        train_end="2020-11-01",
        test_start="2020-11-01",
        test_end=_test_end,
    )
    model = load_model_forcasting_days()
    HORIZON = horizon * 30
    lags = LagTransform(in_column="target", lags=[30 * 6])
    transforms = [lags]
    train_ts.fit_transform(transforms)
    future_ts = train_ts.make_future(future_steps=HORIZON, transforms=transforms)
    forecast_ts = model.forecast(future_ts)
    forecast_ts.inverse_transform(transforms)
    return forecast_ts, train_ts, test_ts

def predict_model(lat, lon, days):
    # predict claster
    number_claster = predict_clastering(lat, lon)
    # find segment
    k = number_claster
    t = 11
    r = number_room
    if type_house == "Новостройка":
        t = 11
    elif type_house == "Вторичка":
        t = 1

    segment = f'k={k}_t={t}_r={r}'
    # forcasting
    if days == False:
        forecast_ts, train_ts, test_ts = predict_forcasting(number_claster)
    if days == True:
        forecast_ts, train_ts, test_ts = predict_forcasting_days(number_claster)
    forecast_df = forecast_ts.to_pandas()
    train_df = train_ts.to_pandas()
    test_df = test_ts.to_pandas()
    forecast = forecast_df[segment]['target']
    train = train_df[segment]['target']
    test = test_df[segment]['target']
    return forecast, train, test

def get_coordinates_from_address(street, house):
    address = house + ', ' + street + ', Москва'
    geolocator = Nominatim(user_agent="base")
    location = geolocator.geocode(address)
    return location.latitude, location.longitude

def get_address_from_coordinates(lat, lon):
    geolocator = Nominatim(user_agent="base")
    try:
        location = geolocator.reverse((lat, lon), timeout=10)
        address = location.address
        return address
    except GeocoderTimedOut:
        return None

with st.sidebar:
    selected = option_menu(
        menu_title="Меню",
        options=['Краткое руководство оператора',
                 'Предсказание по месяцам', 
                 'Предсказание по дням'],
        default_index=0,
    )

if selected == "Предсказание по месяцам":
    st.title("Предиктивная система для анализа цен на недвижимость", anchor=None, help=None)
    st.write("\n")
    st.header("Введите параметры жилья")

    st.subheader('Выберите местоположение жилья на карте или введите адрес')
    m = folium.Map(location=[55.751244, 37.618423], zoom_start=12)
    map_data = st_folium(m, width=700, height=500)
    
    lat, lon = None, None
    street, house = "", ""  # Инициализируем переменные пустыми значениями
    if map_data['last_clicked']:
        lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
        st.write(f"Выбрана точка: Широта {lat}, Долгота {lon}")
        
        # Добавление маркера на карту
        folium.Marker([lat, lon], tooltip='Кликнутая точка').add_to(m)

        # Обратный геокодинг для автозаполнения
        address = get_address_from_coordinates(lat, lon)
        if address:
            st.write(f"Адрес: {address}")
            address_parts = address.split(',')
            if len(address_parts) > 1:
                house = address_parts[0].strip()
                street = address_parts[1].strip()

    st.subheader('Введите адрес жилья')
    street = st.text_input("Введите улицу", value=street)
    house = st.text_input("Введите дом", value=house)
    
    if street and house:
        lat, lon = get_coordinates_from_address(street, house)

    if lat is not None and lon is not None:
        st.write(f"Используем координаты: Широта {lat}, Долгота {lon}")

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

    on_train = st.checkbox("Отобразить исторические данные цен")
    on_test = st.checkbox("Показать реальные цена на предсказании")

    if st.button('Predict!!!'):
        if lat is not None and lon is not None:
            forecast, train, test = predict_model(lat, lon, days=False)
            fig, ax = plt.subplots()
            ax.plot(forecast, 'r', label='Прогноз')
            if on_train:
                ax.plot(train, 'b', label='Исторические данные')
            if on_test:
                ax.plot(test, 'g', label='Реальные данные')
            ax.grid()
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)
        else:
            st.write("Пожалуйста, введите адрес или выберите точку на карте.")

if selected == "Предсказание по дням":
    st.title("Предиктивная система для анализа цен на недвижимость", anchor=None, help=None)
    st.write("\n")
    st.header("Введите параметры жилья")

    st.subheader('Выберите местоположение жилья на карте или введите адрес')
    m = folium.Map(location=[55.751244, 37.618423], zoom_start=12)
    map_data = st_folium(m, width=700, height=500)
    
    lat, lon = None, None
    street, house = "", ""  # Инициализируем переменные пустыми значениями
    if map_data['last_clicked']:
        lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
        st.write(f"Выбрана точка: Широта {lat}, Долгота {lon}")
        
        # Добавление маркера на карту
        folium.Marker([lat, lon], tooltip='Кликнутая точка').add_to(m)

        # Обратный геокодинг для автозаполнения
        address = get_address_from_coordinates(lat, lon)
        if address:
            st.write(f"Адрес: {address}")
            address_parts = address.split(',')
            if len(address_parts) > 1:
                house = address_parts[0].strip()
                street = address_parts[1].strip()

    st.subheader('Введите адрес жилья')
    street = st.text_input("Введите улицу", value=street)
    house = st.text_input("Введите дом", value=house)
    
    if street and house:
        lat, lon = get_coordinates_from_address(street, house)

    if lat is not None and lon is not None:
        st.write(f"Используем координаты: Широта {lat}, Долгота {lon}")

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
    
    on_train = st.checkbox("Отобразить исторические данные цен")
    on_test = st.checkbox("Показать реальные цена на предсказании")

    if st.button('Predict!!!'):
        if lat is not None and lon is not None:
            forecast, train, test = predict_model(lat, lon, days=True)
            fig, ax = plt.subplots()
            ax.plot(forecast, 'r', label='Прогноз')
            if on_train:
                ax.plot(train, 'b', label='Исторические данные')
            if on_test:
                ax.plot(test, 'g', label='Реальные данные')
            ax.grid()
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)
        else:
            st.write("Пожалуйста, введите адрес или выберите точку на карте.")

if selected == "Краткое руководство оператора":
    st.subheader("Встречайте предиктивную систему, которая поможет Вам посмотреть прогноз цен на желаемую квартиру!", anchor=None, help=None)
    st.divider()
    st.subheader("Прогноз цен можно посмотреть по разным промежуткам времени: как по месяцам, так и по дням. Выбрать интересующий интервал предсказания можно в разделе 'Меню'")
    st.markdown("Для того, чтобы расчитать прогноз цен, необходимо ввести ряд параметров:")
    st.markdown("<span style='font-style: italic;'>Шаг 1:</span> Выбрать точку на карте или ввести адрес", unsafe_allow_html=True)
    st.markdown("<span style='font-style: italic;'>Шаг 2:</span> Ввести улицу", unsafe_allow_html=True)
    st.markdown("<span style='font-style: italic;'>Шаг 3:</span> Ввести номер дома", unsafe_allow_html=True)
    st.markdown("<span style='font-style: italic;'>Шаг 4:</span> Выбрать тип жилья (новостройка/ вторичка)", unsafe_allow_html=True)
    st.markdown("<span style='font-style: italic;'>Шаг 5:</span> Выбрать количество комнат (от 1 до 4 комнат)", unsafe_allow_html=True)
    st.markdown("<span style='font-style: italic;'>Шаг 6:</span> На ползунке выбрать дальность предсказания (от 1 до 6 месяцев)", unsafe_allow_html=True)
    st.markdown("<span style='font-style: italic;'>Шаг 7:</span> Прожать необходимые кнопки активации. Доступны 2:", unsafe_allow_html=True)
    st.markdown("<div style='margin-left: 40px;'>1. Отобразить исторические данные цен</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-left: 40px;'>2. Показать реальные цена на предсказании</div>", unsafe_allow_html=True)
