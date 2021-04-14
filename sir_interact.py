from __future__ import division # always-float division
import numpy as np
import pandas as pd
import glob
import pprint
import os
import requests
from datetime import date

# Easy interactive plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Interactive plots in notebook
from IPython.display import HTML, Image, display
from ipywidgets.widgets import interact, IntSlider, FloatSlider, Layout, ToggleButton, ToggleButtons, fixed, Checkbox
import ipywidgets as widgets
from ipywidgets import interact

# Maths
from scipy.integrate import odeint
import scipy.signal.windows as window
from sklearn.preprocessing import normalize
import scipy.stats as stats

# Long computations
from tqdm import tqdm_notebook as tqdm
import pickle

# Fitter
from lmfit import Model, Parameters, Parameter, report_fit, minimize

#Import from utils
from utils import *

#Set variables
# Relative path to EPI data folder
DATA_PATH = './data/clean/EPI'
# Studied country for OWID data
COUNTRY = 'France'
owid_file = update_owid(DATA_PATH)


#Function to get covid data
def country_covid(country, owid_file):
    """
    Extracts 'country' time series from OWID database saved in 'owid_file'. Can add hospital data if 'country'==France and 'hosp_file' is specified.
    Time series starts when infected are positive
    """
    # Get ISO-3 code for country to later use as index for OWID database
    try:
        code = pycountry.countries.search_fuzzy(country)[0].alpha_3
    except LookupError:
        print(f'{country} not found in country dictionary.')
        return

    covid_data, country_attrs = extract_owid(owid_file, country_code=code)

    covid_data = covid_data.sort_values(by='date')
    covid_data = covid_data.reset_index(drop=True)
    # Oldest EPI values are all 0 (I, R, D)
    covid_data.loc[0, covid_data.columns != 'date'] = covid_data.loc[0, covid_data.columns != 'date'].apply(lambda x: 0)
    # Forward-fill NaN: old value is maintained until not-NaN value
    covid_data.ffill(axis=0, inplace=True)
    # Rename columns
    covid_data.columns = ['date', 'I', 'D', 's']
    # Compute S
    #covid_data['S'] = country_attrs['population'] - covid_data['I'] - covid_data['D']
    covid_data['S'] = country_attrs['population'] - covid_data['I']
    covid_data = covid_data[covid_data['I'] > 0]
    covid_data.reset_index(drop=True, inplace=True)
    covid_data['N_effective'] = country_attrs['population'] - covid_data['D']
    covid_data.bfill(axis=0, inplace=True)

    return covid_data, country_attrs

#Get data
EPI_data, country_attrs = country_covid(COUNTRY, owid_file)
df = EPI_data.drop(['N_effective'], axis=1).reindex(columns=['date', 'S', 'I', 'R','D', 's'])
df.ffill(axis=0, inplace=True)
df.bfill(axis=0, inplace=True)

#df = df[df['date'] <= '2020-05-10']
initN = country_attrs['population']/200
df.R.fillna(0, inplace=True)
#Get array of data
france_data = df.loc[0:107, ['I']].values.ravel()


"""Main Function"""
def simulation_SIR(beta, gamma, odeint_=True):

    def SIR_deriv(start_values, t, beta, gamma):

        S, I, R = start_values
        N = S + I + R
        #print(N == S + I + R)

        #Compartment derivatives
        dSdt = -beta*I*S/N
        dIdt = beta*I*S/N - gamma*I
        dRdt = gamma*I
        return [dSdt, dIdt, dRdt]

    #Initialization
    nb_days = 108
    initI, initR, initN = [3.0, 0.0, 68147687.0/350]
    # initial Susceptible
    initS = initN - (initI + initR)

    S_final = [initS]
    I_final = [initI]
    R_final = [initR]

    cases_cumul = [initI]

    inputs = (initS, initI, initR)

    #Timeline
    tspan = np.arange(1, nb_days+1, 1)

    "TEST WITH SCIPY FUNCTION"
    if odeint_:
        res = odeint(SIR_deriv, [initS, initI, initR], tspan, args=(beta, gamma))

        S_final, I_final, R_final = res[:,0], res[:,1], res[:,2]

        #Compute S' (i.e S_t - S_t-1)
        deriv_S = list(np.diff(S_final,1))

        for i in range(nb_days-1):
            cases_cumul.append(cases_cumul[i]+np.abs(deriv_S[i]))

    #By hand
    elif not odeint_:
        for i in range(nb_days-1):
            deriv_S, deriv_I, deriv_R = SIR_deriv(inputs, tspan, beta, gamma)

            new_S = S_final[i]+deriv_S
            new_I = I_final[i]+deriv_I
            new_R = R_final[i]+deriv_R

            S_final.append(new_S)
            I_final.append(new_I)
            R_final.append(new_R)

            #Update inputs and cumul cases
            inputs = (new_S, new_I, new_R)
            cases_cumul.append(cases_cumul[i]+np.abs(deriv_S))

    _ = np.linspace(1, nb_days, nb_days)
    bbox = dict(boxstyle="round", fc="0.8")
    plt.figure(figsize=(15,6))
    plt.title('SIR model', size=25)
    plt.plot(_, S_final, label='S', color='b')
    plt.plot(_, I_final, label='I', color='red')
    plt.plot(_, R_final, label='R', color ='green')
    plt.plot(_, cases_cumul, label='Cumulated infections', color ='orange',  lw=3.5)
    plt.plot(_, france_data, label='FRA observed infections', color='darkorange', ls='--', alpha=0.5, lw=3.5)
   # plt.annotate('Beta: {0} \nGamma: {1}'.format(beta, np.round(gamma,2)),   (0, 40000000), bbox=bbox, size=13)
    plt.legend(fontsize=15, bbox_to_anchor=(0.8, -0.2), ncol=3, fancybox=True, shadow=True)
    plt.grid()
    plt.xlabel('Days after first infected', size=18), plt.ylabel('Count', size=18), plt.tick_params(labelsize=18)
    plt.show()


#Run visualization
interact(simulation_SIR, beta=widgets.FloatSlider(min=0.0001, max=3.0, step=0.01, value=0.5),
         gamma=widgets.FloatSlider(min=0.0001, max=1.5, step=0.01, value=0.1),odeint_=True)
