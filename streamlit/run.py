from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from etna.datasets import TSDataset
from etna.transforms import LagTransform
from streamlit_option_menu import option_menu

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from matplotlib.ticker import ScalarFormatter

from cache_data import load_model_forcasting
from cache_data import load_model_forcasting_days
from cache_data import load_model_kmeans
from cache_data import load_data_pd
from cache_data import load_data_days_pd
from cache_data import load_model_forcasting_days_many

hirezon2timestamp = {
    1: "2020-12-01",
    2: "2021-01-01",
    3: "2021-02-01",
    4: "2021-03-01",
    5: "2021-04-01",
    6: "2021-05-01",
}  # преобразует число горизонт предсказания в тип timestamp

hirezon2timestamp_days = {
    1: "2020-08-30",
    2: "2020-09-29",
    3: "2020-10-29",
    4: "2020-11-28",
    5: "2020-12-28",
    6: "2021-01-27",
}
# преобразует число горизонт предсказания в тип timestamp

type_str2int = {
    "Новостройка": 11,
    "Вторичка": 1,
}  # преобразует тип дома заданный ввиде строки в число


def predict_clastering(lat, lon):
    """Функция, отрабатывающая predict модели кластеризации
    args:
        lat : широта (градусы)
        lon : долгота (градусы)
    return:
        number_claster: номер кластера (от 0 до 19)
    """
    moscow_centre = (
        55.751003 * np.pi / 180.0,
        37.617964 * np.pi / 180.0,
    )  # перевод из градусов в радианы
    R = 6371.0  # км
    x = (lat * np.pi / 180.0 - moscow_centre[0]) * R
    y = (lon * np.pi / 180.0 - moscow_centre[1]) * R * np.cos(moscow_centre[0])
    X = np.array([[x, y]])
    model_kmeans = load_model_kmeans()
    number_claster = model_kmeans.predict([X[0]])[0]
    return number_claster


def predict_forcasting(number_claster: int):
    """Функция, отрабатывающая predict модели предсказания временого ряда по месяцам
    args:
        number_claster : номер кластера
    return:
        forecast_ts : DataFrame предсказаний по месяцам
        train_ts : DataFrame обучающей выборки (исторических данных)
        test_ts : DataFrame тестовой выборки (реальные цена на предсказании)
    """
    data = load_data_pd()
    ts = TSDataset(data, freq="MS")

    _test_end = hirezon2timestamp[horizon]

    train_ts, test_ts = ts.train_test_split(
        train_start="2018-09-01",
        train_end="2020-11-01",
        test_start="2020-12-01",
        test_end=_test_end,
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


def predict_forcasting_days(number_claster: int):
    """Функция, отрабатывающая predict модели предсказания временого ряда по дням
    args:
        number_claster : номер кластера
    return:
        forecast_ts : DataFrame предсказаний по дням
        train_ts : DataFrame обучающей выборки (исторических данных)
        test_ts : DataFrame тестовой выборки (реальные цена на предсказании)
    """
    data = load_data_days_pd()
    ts = TSDataset(data, freq="D")
    train_ts, test_ts = ts.train_test_split(
        train_start="2018-10-25",
        train_end="2020-07-31",
        test_start="2020-08-01",
        test_end=hirezon2timestamp_days[horizon],
    )

    # model = load_model_forcasting_days()
    model = load_model_forcasting_days_many(horizon)
    HORIZON = horizon * 30
    # MIN_LAG = 180
    MAX_LAG = HORIZON + 400
    lags = LagTransform(in_column="target", lags=list(range(HORIZON, MAX_LAG)))
    # lags = LagTransform(in_column="target", lags=[6*30])
    transforms = [lags]
    train_ts.fit_transform(transforms)
    future_ts = train_ts.make_future(future_steps=HORIZON, transforms=transforms)
    forecast_ts = model.forecast(future_ts)
    forecast_ts.inverse_transform(transforms)
    return forecast_ts, train_ts, test_ts


def predict_model(lat: float, lon: float, days: bool):
    """Функция, отрабатывающая predict всех моделей
    args:
        lat : широта (градусы)
        lon : долгота (градусы)
        days : выбор стратегии прогнозирования (True - по дням / False - по месяцам)
    return:
        forecast_ts : DataFrame предсказаний по дням
        train_ts : DataFrame обучающей выборки (исторических данных)
        test_ts : DataFrame тестовой выборки (реальные цена на предсказании)
    """
    # predict claster
    number_claster = predict_clastering(lat, lon)
    # find segment
    k = number_claster
    r = number_room
    t = type_str2int[type_house]
    segment = f"k={k}_t={t}_r={r}"
    # forcasting
    if days == False:
        forecast_ts, train_ts, test_ts = predict_forcasting(number_claster)
    else:
        forecast_ts, train_ts, test_ts = predict_forcasting_days(number_claster)
    forecast_df = forecast_ts.to_pandas()
    train_df = train_ts.to_pandas()
    test_df = test_ts.to_pandas()
    forecast = forecast_df[segment]["target"]
    train = train_df[segment]["target"]
    test = test_df[segment]["target"]
    return forecast, train, test


def get_coordinates_from_address(street: str, house: str):
    """Функция, отрабатывающая геокоднг из адреса в координаты (OpenStreatMap API)
    args:
        streat : название улицы
        house : номер дома
    return:
        location.latitude : значение широты (градусы)
        location.longitude : значение доготы (градусы)
    """
    address = house + ", " + street + ", Москва"
    geolocator = Nominatim(user_agent="base")
    location = geolocator.geocode(address)
    return location.latitude, location.longitude


def get_address_from_coordinates(lat: float, lon: float):
    """Функция, отрабатывающая обратный геокоднг из координат в адресс (OpenStreatMap API)
    args:
        lat : значение широты (градусы)
        lon : значение доготы (градусы)
    return:
        address : адрес
    """
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
        options=[
            "Краткое руководство оператора",
            "Предсказание по месяцам",
            "Предсказание по дням",
        ],
        default_index=0,
    )

if selected == "Предсказание по месяцам":
    st.title(
        "Предиктивная система для анализа цен на недвижимость", anchor=None, help=None
    )
    st.write("\n")
    st.header("Введите параметры жилья")

    st.subheader("Выберите местоположение жилья на карте или введите адрес")
    m = folium.Map(location=[55.751244, 37.618423], zoom_start=12)
    map_data = st_folium(m, width=700, height=500)

    lat, lon = None, None
    street, house = "", ""  # Инициализируем переменные пустыми значениями
    if map_data["last_clicked"]:
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        st.write(f"Выбрана точка: Широта {lat}, Долгота {lon}")

        # Добавление маркера на карту
        folium.Marker([lat, lon], tooltip="Кликнутая точка").add_to(m)

        # Обратный геокодинг для автозаполнения
        address = get_address_from_coordinates(lat, lon)
        if address:
            st.write(f"Адрес: {address}")
            address_parts = address.split(",")
            if len(address_parts) > 1:
                house = address_parts[0].strip()
                street = address_parts[1].strip()

    st.subheader("Введите адрес жилья")
    street = st.text_input("Введите улицу", value=street)
    house = st.text_input("Введите дом", value=house)

    if street and house:
        lat, lon = get_coordinates_from_address(street, house)

    if lat is not None and lon is not None:
        st.write(f"Используем координаты: Широта {lat}, Долгота {lon}")

    st.subheader("Введите другие параметры")
    type_house = st.selectbox("Тип жилья", ("Новостройка", "Вторичка"))
    number_room = st.selectbox("Количество комнат", (1, 2, 3, 4))
    horizon = st.select_slider(
        "Продолжительность предсказания", options=[1, 2, 3, 4, 5, 6]
    )

    on_train = st.checkbox("Отобразить исторические данные цен")
    on_test = st.checkbox("Показать реальные цена на предсказании")

    if st.button("Предсказать"):
        if lat is not None and lon is not None:
            forecast, train, test = predict_model(lat, lon, days=False)
            fig, ax = plt.subplots()
            ax.plot(forecast, "r", label="Прогноз")
            if on_train:
                ax.plot(train, "b", label="Исторические данные")
            if on_test:
                ax.plot(test, "g", label="Реальные данные")
            ax.grid()
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)
        else:
            st.write("Пожалуйста, введите адрес или выберите точку на карте.")

if selected == "Предсказание по дням":
    st.title(
        "Предиктивная система для анализа цен на недвижимость", anchor=None, help=None
    )
    st.write("\n")
    st.header("Введите параметры жилья")

    st.subheader("Выберите местоположение жилья на карте или введите адрес")
    m = folium.Map(location=[55.751244, 37.618423], zoom_start=12)
    map_data = st_folium(m, width=700, height=500)

    lat, lon = None, None
    street, house = "", ""  # Инициализируем переменные пустыми значениями
    if map_data["last_clicked"]:
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        st.write(f"Выбрана точка: Широта {lat}, Долгота {lon}")

        # Добавление маркера на карту
        folium.Marker([lat, lon], tooltip="Кликнутая точка").add_to(m)

        # Обратный геокодинг для автозаполнения
        address = get_address_from_coordinates(lat, lon)
        if address:
            st.write(f"Адрес: {address}")
            address_parts = address.split(",")
            if len(address_parts) > 1:
                house = address_parts[0].strip()
                street = address_parts[1].strip()

    st.subheader("Введите адрес жилья")
    street = st.text_input("Введите улицу", value=street)
    house = st.text_input("Введите дом", value=house)

    if street and house:
        lat, lon = get_coordinates_from_address(street, house)

    if lat is not None and lon is not None:
        st.write(f"Используем координаты: Широта {lat}, Долгота {lon}")

    st.subheader("Введите другие параметры")
    type_house = st.selectbox("Тип жилья", ("Новостройка", "Вторичка"))
    number_room = st.selectbox("Количество комнат", (1, 2, 3, 4))
    horizon = st.select_slider(
        "Продолжительность предсказания", options=[1, 2, 3, 4, 5, 6]
    )

    on_train = st.checkbox("Отобразить исторические данные цен")
    on_test = st.checkbox("Показать реальные цена на предсказании")

    if st.button("Предсказать"):
        if lat is not None and lon is not None:
            forecast, train, test = predict_model(lat, lon, days=True)
            fig, ax = plt.subplots()
            ax.plot(forecast, "r", label="Прогноз")
            if on_train:
                ax.plot(train, "b", label="Исторические данные")
            if on_test:
                ax.plot(test, "g", label="Реальные данные")
            ax.grid()
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)
        else:
            st.write("Пожалуйста, введите адрес или выберите точку на карте.")

if selected == "Краткое руководство оператора":
    st.subheader(
        "Встречайте предиктивную систему, которая поможет Вам посмотреть прогноз цен на желаемую квартиру!",
        anchor=None,
        help=None,
    )
    st.divider()
    st.subheader(
        "Прогноз цен можно посмотреть по разным промежуткам времени: как по месяцам, так и по дням. Выбрать интересующий интервал предсказания можно в разделе 'Меню'"
    )
    st.markdown(
        "Для того, чтобы расчитать прогноз цен, необходимо ввести ряд параметров:"
    )
    st.markdown(
        "<span style='font-style: italic;'>Шаг 1:</span> Выбрать точку на карте или ввести адрес",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='font-style: italic;'>Шаг 2:</span> Ввести улицу",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='font-style: italic;'>Шаг 3:</span> Ввести номер дома",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='font-style: italic;'>Шаг 4:</span> Выбрать тип жилья (новостройка/ вторичка)",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='font-style: italic;'>Шаг 5:</span> Выбрать количество комнат (от 1 до 4 комнат)",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='font-style: italic;'>Шаг 6:</span> На ползунке выбрать дальность предсказания (от 1 до 6 месяцев)",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='font-style: italic;'>Шаг 7:</span> Прожать необходимые кнопки активации. Доступны 2:",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='margin-left: 40px;'>1. Отобразить исторические данные цен</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='margin-left: 40px;'>2. Показать реальные цена на предсказании</div>",
        unsafe_allow_html=True,
    )
