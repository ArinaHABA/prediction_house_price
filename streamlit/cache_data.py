import streamlit as st
import pandas as pd
import pickle
from etna.models import CatBoostMultiSegmentModel

@st.cache_data
def load_model_forcasting():
    model = CatBoostMultiSegmentModel()
    fit_model = model.load("../models/model_forcasting.sav")
    return fit_model


@st.cache_data
def load_model_forcasting_days():
    model = CatBoostMultiSegmentModel()
    fit_model = model.load("../models/model_forcasting_days_clear.sav")
    return fit_model

@st.cache_data
def load_model_forcasting_days_many(horizon : int):
    model = CatBoostMultiSegmentModel()
    if horizon == 1:
        return model.load("../models/model_forcasting_days_clear_1.sav")
    if horizon == 2:
        return model.load("../models/model_forcasting_days_clear_2.sav")
    if horizon == 3:
        return model.load("../models/model_forcasting_days_clear_3.sav")
    if horizon == 4:
        return model.load("../models/model_forcasting_days_clear_4.sav")
    if horizon == 5:
        return model.load("../models/model_forcasting_days_clear_5.sav")
    if horizon == 6:
        return model.load("../models/model_forcasting_days_clear_6.sav")
    return None


@st.cache_data
def load_model_kmeans():
    filename = "../models/model_kmeans.sav"
    model = pickle.load(open(filename, "rb"))
    return model


@st.cache_data
def load_data_pd():
    data = pd.read_pickle("../models/data.sav")
    return data


@st.cache_data
def load_data_days_pd():
    data = pd.read_pickle("../models/data_days_clear_0.5.sav")
    return data