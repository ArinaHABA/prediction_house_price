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
    fit_model = model.load("../models/model_forcasting_days.sav")
    return fit_model


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
    data = pd.read_pickle("../models/data_days.sav")
    return data