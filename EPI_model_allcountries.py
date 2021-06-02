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
import plotly.figure_factory as ff

# Interactive plots in notebook
from IPython.display import HTML, Image, display
from ipywidgets.widgets import interact, IntSlider, FloatSlider, Layout, ToggleButton, ToggleButtons, fixed, Checkbox

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
from utils import country_covid, get_data_w_mobility


class EPIfit_allcountries():
    """
    Class for fitting compartmental epidemiologic model (either SIR, SEIR, SEIRD)
    """
    def __init__(self, model='SIR', beta_constant_step=False, beta_sigmoid_step=False, beta_linear_step=False, mu_sigmoid_step=False, stringency_fit=False,
    PARAMS_FILE=None, OWID_FILE=None, HOSP_FILE=None, cutoff_date='2020-07-01', ratio_susceptible=1/300):

        if stringency_fit:
            assert beta_sigmoid_step, "Stringency must be used when fitting beta parameter as a sigmoid function. Please put 'beta_sigmoid_step=True'."

        assert OWID_FILE != None, "Data file not specified."
        assert PARAMS_FILE != None, "Parameters file not specified."

        self.model=model
        self.beta_constant_step=beta_constant_step
        self.beta_sigmoid_step=beta_sigmoid_step
        self.beta_linear_step=beta_linear_step
        self.mu_sigmoid_step = mu_sigmoid_step
        self.stringency_fit= stringency_fit
        self.PARAMS_FILE = PARAMS_FILE
        self.owid_file = OWID_FILE
        self.hosp_file = HOSP_FILE
        self.cutoff_date = cutoff_date
        self.ratio_susceptible = ratio_susceptible
        self.countries = ['Australia', 'Austria', 'Belgium', 'Bulgaria', 'Canada', 'China', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'India',
        'Israel', 'Italy', 'Japan', 'Luxembourg', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Russia', 'South Africa', 'Korea, Republic of', 'Spain', 'Sweden',
        'Taiwan', 'United Kingdom', 'United States']

        self.iso_code2 = ['AU', 'AT', 'BE', 'BG', 'CA', 'CH', 'DK', 'FI', 'FR', 'DE', 'GR', 'IN', 'IL', 'IT', 'JP', 'LU', 'NL', 'NO', 'PL', 'PT', 'RU', 'ZA',
                     'KR', 'ES', 'SE', 'TW', 'GB', 'US']

        if stringency_fit:
            #List of fitted beta curve on stringency
            self.result_stringency = []
            self.a_beta_init = []
            self.stringency_fitted = []
        self.final_result = []
        self.data = []
        self.recovered=[]
        self.data_s = []
        self.data_mob=[]
        self.cases_fitted=[]
        self.derivative = []
        self.mae = []
        self.days=[]
        self.mse=[]
        self.test_mse=[]
        self.std_error=[]
        self.train_fitted=[]
        self.fitted_deriv=[]
        self.train_result=[]
        self.train_params = []
        self.test_predicted=[]
        self.test_deriv=[]
        self.test_on_optim = False
        if self.beta_constant_step:
            method = 'constant'
        elif self.beta_sigmoid_step:
            method = 'sigmoid'
        elif self.beta_linear_step:
            method = 'linear'

        self.file_method = 'results_{}'.format(method)
        self.file_model = model

        #Create directories for future results
        if not os.path.exists('./{}'.format(self.file_method)):
            os.makedirs('./{}'.format(self.file_method))

        if not os.path.exists('./{0}/{1}'.format(self.file_method, self.file_model)):
            os.makedirs('./{0}/{1}'.format(self.file_method, self.file_model))

        #Chi2 table for 7 liberty degrees, alpha=5%
        #self.Chi2 = [(0.00,5.02),(0.05,7.38),(0.22,9.35),(0.48,11.14),(0.83,12.83), (1.24,14.45),(1.69,16.01)]


    """Logistic functions for sigmoid fits"""
    @staticmethod
    def logistic(x, L, k, x0):
        """
        Logistic function on 'x' with 'L' maximum value, 'k' steepness and 'x0' midpoint
        """
        return L/(1+np.exp(-k*(x-x0)))

    def logistic_params(self, x, params):
        """
        Logistic function from a list of parameters 'params'=[L, k, x0]
        """
        L, k, x0 = params
        return self.logistic(x, L, k, x0)

    def logistic_step(self, t, params):
        """
        Not-0 baseline logistic function with 'params'=[C, L, a, b] where
            C initial value
            C+L final value
            a step steepness (a>0)
            b step midpoint
        """
        C, L, a, b = params
        return C + self.logistic(t, L, a, b)

    @staticmethod
    def linear_step(s, mob, params):
        """ Trying a simple linear step to approximate beta parameter with stringency index as input
        """
        weight1, weight2, bias = params
        return weight1*s + weight2*mob + bias

    @staticmethod
    def surface_beta(data_s, data_mob, weights):
        """To plot surface plot in 3d and assess sensibility to parameters"""
        #mat = []
        #for x in np.linspace(-1, 1, len(france_s)):
          #  z = weight2*x +weight1*np.linspace(0, 1, len(france_s)) + bias
          #  mat.append(z)
        weight1, weight2, bias = weights
        X, Y = np.meshgrid(np.linspace(0,1,len(data_s)), np.linspace(-1,1,len(data_mob)))
        Z = weight1*X + weight2*Y + bias

        return Z


    """PLOTS"""
    def show_rates(self, result, df, tspan, STATIC_PLOTS=True):
        """
        Show fitted rates from 'result' and visualize stringency from 'df' over 'tspan'
        """
        pnames = list((result.params.valuesdict()))

        if self.beta_sigmoid_step and not self.mu_sigmoid_step:
            b_params = [result.params[p].value for p in pnames[:4]]
            s_params = [self.result_stringency.params[p].value for p in list((self.result_stringency.params.valuesdict()))[:4]]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            beta_curve = self.logistic_step(tspan, b_params)
            fig.add_trace(go.Scatter(x=tspan, y=beta_curve, mode='lines+markers', name='Transmission rate',line = dict(dash='dot', color='orange')))
            #Annotate inflexion points
            fig.add_annotation(x=int(b_params[-1]), y=(np.max(beta_curve)-np.min(beta_curve))/2+np.min(beta_curve), text="Beta IP \n= {}".format(round(b_params[-1])), showarrow=True, arrowhead=1,
            xref="x", yref="y", ax=20, ay=-30, font=dict(family="Courier New, monospace",size=15),bordercolor="#c7c7c7",borderwidth=2,borderpad=4, bgcolor="#ff7f0e", opacity=0.8)

            fig.add_trace(go.Scatter(x=tspan, y=df.s, mode='lines+markers', name='Stringency', line = dict(dash='dot', color='green')), secondary_y=True)
            #Annotate inflexion points
            fig.add_annotation(x=int(s_params[-1]), y=(np.max(df.s)-np.min(df.s))/2+np.min(df.s), text="Stringency IP \n= {}".format(int(s_params[-1])), showarrow=True, arrowhead=1,
            xref="x",yref="y2", font=dict(family="Courier New, monospace",size=15), bordercolor="#c7c7c7",borderwidth=2,borderpad=4, bgcolor="lightgreen", opacity=0.8)

            fig.update_layout(title='Fitted Beta rate & stringency data',
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Rate value',
                                   yaxis2_title='OxCGRT index',
                                   title_x=0.5,
                                   title_font_size=18,
                                   font_size=18,
                                  width=1000, height=600
                                 )

        elif self.beta_sigmoid_step and self.mu_sigmoid_step:
            b_params = [result.params[p].value for p in pnames[:4]]
            m_params = [result.params[p].value for p in pnames[4:8]]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=tspan, y=self.logistic_step(tspan, b_params), mode='lines+markers', name='Transmission rate',line = dict(dash='dot', color='orange')))
            fig.add_trace(go.Scatter(x=tspan, y=self.logistic_step(tspan, m_params), mode='lines+markers', name='Morbidity rate',line = dict(dash='dot', color='purple')))

            fig.add_trace(go.Scatter(x=tspan, y=df.s, mode='lines+markers', name='Stringency', line = dict(dash='dot', color='green')), secondary_y=True)

            fig.update_layout(title='Fitted compartment rates & stringency data',
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Rate value',
                                   yaxis2_title='OxCGRT index',
                                   title_x=0.5,
                                   title_font_size=18,
                                   font_size=18,
                                  width=1000, height=600
                                 )

        elif self.mu_sigmoid_step and not self.beta_sigmoid_step:
            m_params = [result.params[p].value for p in pnames[:4]]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=tspan, y=self.logistic_step(tspan, m_params), mode='lines+markers', name='Morbidity rate',line = dict(dash='dot', color='orange')))
            fig.add_trace(go.Scatter(x=tspan, y=df.s, mode='lines+markers', name='Stringency', line = dict(dash='dot', color='green')), secondary_y=True)

            fig.update_layout(title='Fitted compartment Mu rate & stringency data',
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Rate value',
                                   yaxis2_title='OxCGRT index',
                                   title_x=0.5,
                                   title_font_size=18,
                                   font_size=18,
                                  width=1000, height=600
                                 )

        elif not self.beta_sigmoid_step and not self.mu_sigmoid_step:
            pass



        if STATIC_PLOTS:
            img_bytes = fig.to_image(format="png")
            display(Image(img_bytes))
        else:
            fig.show()

    """Turn stringency into continuous value"""
    def get_strin_continuous(self, t):
        return self.stringency_country[int(np.floor(t))] + (t - np.floor(t))*(self.stringency_country[int(np.ceil(t))]-self.stringency_country[int(np.floor(t))])

    def get_mob_continuous(self, t):
        return self.mob_country[int(np.floor(t))] + (t - np.floor(t))*(self.mob_country[int(np.ceil(t))]-self.mob_country[int(np.floor(t))])

    """SIR, SEIR or SEIRD model"""
    def SIRD_derivs(self, z, t, rate_params, evo_params=None, stringency=None):
        """
        Derivatives for ODE solver
        """
        # Separate compartments
        if self.model=='SIR':
            S, I, R = z
            N = S +I + R

            #Fit beta as sigmoid or not
            if self.beta_constant_step:
                beta, gamma = rate_params

            elif self.beta_linear_step:
                gamma = rate_params
                beta_params = evo_params
                #print('t sird derivs', t)
                #if t > len(self.stringency_country):
                    #t=len(self.stringency_country)-1
                strin_continuous =  self.get_strin_continuous(t-1.0) #stringency data starts at index 0 so we remove 1
                mob_continuous =  self.get_mob_continuous(t-1.0)
                #print('strin_continuous', strin_continuous)
                beta = self.linear_step(strin_continuous, mob_continuous, beta_params)
                #print('beta linear step', beta)

            elif self.beta_sigmoid_step:
                gamma = rate_params
                # Define evolution rates
                beta_params = evo_params
                beta= self.logistic_step(t, beta_params)
                #Prevent from having negative beta values:
                #beta = max(0.0005, beta)

            #Compartment derivatives
            dSdt = -beta*S*I/N
            dIdt = beta*S*I/N - gamma*I
            dRdt = gamma*I

            #print('[dSdt, dIdt, dRdt]', [dSdt, dIdt, dRdt])

            return [dSdt, dIdt, dRdt]

        elif self.model=='SEIR':
            S, E, I, R = z
            N = S + E + I + R
            #Fit beta as sigmoid or not
            if self.beta_constant_step:
                beta, sigma, gamma = rate_params

            elif self.beta_linear_step:
                sigma, gamma = rate_params
                beta_params = evo_params
                strin_continuous =  self.get_strin_continuous(t-1.0) #stringency data starts at index 0 so we remove 1
                mob_continuous =  self.get_mob_continuous(t-1.0)
                #print('strin_continuous', strin_continuous)
                beta = self.linear_step(strin_continuous, mob_continuous, beta_params)

            elif self.beta_sigmoid_step:
                sigma, gamma = rate_params
                # Define evolution rates
                beta_params = evo_params
                beta= self.logistic_step(t, beta_params)
                #beta = max(0.0005, beta)

            # Compartment derivatives
            dSdt = -beta*S*I/N
            dEdt = beta*S*I/N - sigma*E
            dIdt = sigma*E - gamma*I
            dRdt = gamma*I
            return [dSdt, dEdt, dIdt, dRdt]

        elif self.model=='SEIRD':
            S, E, I, R, D = z
            N = S + E + I + R + D
            #Fit beta and/or mu as sigmoids or not
            if not self.beta_sigmoid_step and not self.mu_sigmoid_step:
                beta, sigma, gamma, mu = rate_params
                #No evolution rates
            elif self.beta_sigmoid_step and not self.mu_sigmoid_step:
                sigma, gamma, mu = rate_params
                # Define evolution rates
                beta_params = evo_params
                beta= self.logistic_step(t, beta_params)
            elif self.beta_sigmoid_step and self.mu_sigmoid_step:
                sigma, gamma = rate_params
                # Define evolution rates
                beta_params, mu_params = evo_params
                beta= self.logistic_step(t, beta_params)
                mu = self.logistic_step(t, mu_params)
            elif not self.beta_sigmoid_step and self.mu_sigmoid_step:
                beta, sigma, gamma = rate_params
                # Define evolution rates
                mu_params = evo_params
                mu = self.logistic_step(t, mu_params)

            # Compartment derivatives
            dSdt = -beta*S*I/(N-D)
            dEdt = beta*S*I/(N-D) - sigma*E
            dIdt = sigma*E - gamma*I - mu*I
            dRdt = gamma*I
            dDdt = mu*I
            return [dSdt, dEdt, dIdt, dRdt, dDdt]

    def ode_solver(self, t, initial_conditions, params, i):
        """
        ODE solver per sector.
        """
        if self.model=='SIR':
            initI, initR, initN = initial_conditions
            # initial Susceptible
            initS = initN - (initI + initR)
            # Make param lists
            if self.beta_sigmoid_step:
                gamma = params['gamma'].value
                # beta and mu params
                C, L, a, b = params[f'C_{i}'].value, params[f'L_{i}'].value, params[f'a_{i}'].value, params[f'b_{i}'].value
                # Static params and param lists
                rate_params = gamma
                beta_params = [C, L, a, b]
                evo_params = beta_params
                res = odeint(self.SIRD_derivs, [initS,initI, initR], t, args=(rate_params, evo_params))
                return res

            elif self.beta_linear_step:
                gamma = params['gamma'].value
                w1, w2, b = params[f'w1_{i}'].value, params[f'w2_{i}'].value, params[f'b_{i}'].value
                # Static params and param lists
                rate_params = gamma
                beta_params = [w1, w2, b]
                evo_params = beta_params
                #print('rate_params', rate_params)
                #print('evo params', evo_params)
                #print('SIR',[initS,initI, initR])
                #print('t', t)

                res = odeint(self.SIRD_derivs, [initS,initI, initR], t, args=(rate_params, evo_params))
                return res

            elif self.beta_constant_step:
                beta, gamma = params['beta'].value, params['gamma'].value
                # Static params and param lists
                rate_params = [beta, gamma]
                res = odeint(self.SIRD_derivs, [initS, initI, initR], t, args=(rate_params, [])) #args has to be in a tuple

                return res


        elif self.model == 'SEIR':
            initE, initI, initR, initN = initial_conditions
            # initial Susceptible
            initS = initN - (initE + initI + initR)
            if self.beta_sigmoid_step:
                # Make param lists
                sigma, gamma = params['sigma'].value, params['gamma'].value
                # beta params
                C, L, a, b = params[f'C_{i}'].value, params[f'L_{i}'].value, params[f'a_{i}'].value, params[f'b_{i}'].value

                # Static params and param lists
                rate_params = [sigma, gamma]
                beta_params = [C, L, a, b]
                evo_params = beta_params
                # Solve ODE
                res =odeint(self.SIRD_derivs, [initS, initE, initI, initR], t, args=(rate_params, evo_params))
                return res

            elif self.beta_linear_step:
                # Make param lists
                sigma, gamma = params['sigma'].value, params['gamma'].value
                # beta params
                w1, w2, b = params[f'w1_{i}'].value, params[f'w2_{i}'].value, params[f'b_{i}'].value

                # Static params and param lists
                rate_params = [sigma, gamma]
                beta_params = [w1, w2, b]
                evo_params = beta_params
                # Solve ODE
                res =odeint(self.SIRD_derivs, [initS, initE, initI, initR], t, args=(rate_params, evo_params))
                return res

            elif self.beta_constant_step:
                # Make param lists
                beta, sigma, gamma = params['beta'].value, params['sigma'].value, params['gamma'].value
                # Static params and param lists
                rate_params = [beta, sigma, gamma]
                # Solve ODE
                res =odeint(self.SIRD_derivs, [initS, initE, initI, initR], t, args=(rate_params, []))
                return res


        elif self.model =='SEIRD':
            initE, initI, initR, initN, initD = initial_conditions
            # initial Susceptible
            initS = initN - (initE + initI + initR + initD)

            if self.beta_sigmoid_step and not self.mu_sigmoid_step:
                # Make param lists
                sigma, gamma, mu= params['sigma'].value, params['gamma'].value, params['mu'].value
                # beta params
                C, L, a, b = params[f'C_{i}'].value, params[f'L_{i}'].value, params[f'a_{i}'].value, params[f'b_{i}'].value
                # Static params and param lists
                rate_params = [sigma, gamma, mu]
                beta_params = [C, L, a, b]
                evo_params = beta_params
                # Solve ODE
                res =odeint(self.SIRD_derivs, [initS, initE, initI, initR, initD], t, args=(rate_params, evo_params))
                return res

            elif self.mu_sigmoid_step and not self.beta_sigmoid_step:
                # Make param lists
                beta, sigma, gamma = params['beta'].value, params['sigma'].value, params['gamma'].value
                # mu params
                muC, muL, mua, mub = params[f'muC_{i}'].value, params[f'muL_{i}'].value,params[f'mua_{i}'].value, params[f'mub_{i}'].value
                # Static params and param lists
                rate_params = [beta, sigma, gamma]
                mu_params = [muC, muL, mua, mub]
                evo_params = mu_params
                # Solve ODE
                res =odeint(self.SIRD_derivs, [initS, initE, initI, initR, initD], t, args=(rate_params, evo_params))
                return res


            elif not self.beta_sigmoid_step and not self.mu_sigmoid_step:
                # Make param lists
                beta, sigma, gamma, mu = params['beta'].value, params['sigma'].value, params['gamma'].value, params['mu'].value
                # Static params and param lists
                rate_params = [beta, sigma, gamma, mu]
                # Solve ODE
                res =odeint(self.SIRD_derivs, [initS, initE, initI, initR, initD], t, args=(rate_params, []))
                return res

            elif self.beta_sigmoid_step and self.mu_sigmoid_step:
                # Make param lists
                sigma, gamma = params['sigma'].value, params['gamma'].value
                # beta and mu params
                C, L, a, b = params[f'C_{i}'].value, params[f'L_{i}'].value, params[f'a_{i}'].value, params[f'b_{i}'].value
                muC, muL, mua, mub = params[f'muC_{i}'].value, params[f'muL_{i}'].value,params[f'mua_{i}'].value, params[f'mub_{i}'].value
                # Static params and param lists
                rate_params = [sigma, gamma]
                beta_params = [C, L, a, b]
                mu_params = [muC, muL, mua, mub]
                evo_params = [beta_params, mu_params]
                # Solve ODE
                res =odeint(self.SIRD_derivs, [initS, initE, initI, initR, initD], t, args=(rate_params, evo_params))
                return res


    """Solver functions"""
    def init_sectors(self, sector_props, disease_vary=True, initN=0, initI=1, initR=0, initE=0, stringency_fit=False):
        """
        Makes initial conditions for each sector: one infected and parameter initial values for optimization.
            'disease_params' are initial values for sigma, gamma mu, nu
            'sector_props' is a list of the proportion of the population in each sector
            'disease_vary' freezes disease_params for optimization
        """
        assert abs(np.sum(sector_props) - 100) < 0.1, "Sector population proportions aren't normalized"
        params = Parameters()
        init_params = pd.read_csv(self.PARAMS_FILE, sep=";",index_col='name', header=0, skipinitialspace=True)
    #     print('Fitted parameter list:')
    #     print(init_params)

        if stringency_fit:
            for i in range(len(sector_props)):
                params.add(f'Cs{i}', value=init_params.at['Cs', 'init_value'], min=init_params.at['Cs', 'min'], max=init_params.at['Cs', 'max'], vary=True)
                params.add(f'Ls{i}', value=init_params.at['Ls', 'init_value'], min=init_params.at['Ls', 'min'], max=init_params.at['Ls', 'max'], vary=True)
                params.add(f'as{i}', value=init_params.at['as', 'init_value'], min=init_params.at['as', 'min'], max=init_params.at['as', 'max'], vary=True)
                params.add(f'bs{i}', value=init_params.at['bs', 'init_value'], min=init_params.at['bs', 'min'], max=init_params.at['bs', 'max'], vary=True)

            #print('Params for stringency fit:', params)
            return params

        elif self.test_on_optim:
            #Init conditions
            initial_conditions = []

            for i in range(len(sector_props)):
                initN_i = initN * sector_props[i]/100
                initI_i = initI
                initR_i = initR

                initial_conditions_i = [initI_i, initR_i, initN_i]
                initial_conditions.append(initial_conditions_i)

                if self.beta_sigmoid_step:
                    params.add(f'C_{i}', value=init_params.at['C', 'init_value'], min=init_params.at['C', 'min'], max=init_params.at['C', 'max'], vary=False)
                    params.add(f'L_{i}', value=init_params.at['L', 'init_value'], min=init_params.at['L', 'min'], max=init_params.at['L', 'max'], vary=True)
                    params.add(f'a_{i}', value=init_params.at['a', 'init_value'], min=init_params.at['a', 'min'], max=init_params.at['a', 'max'], vary=False)
                    params.add(f'b_{i}', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                elif self.beta_linear_step:
                    params.add(f'w1_{i}', value=init_params.at['w1', 'init_value'], min=init_params.at['w1', 'min'], max=init_params.at['w1', 'max'], vary=True)
                    params.add(f'w2_{i}', value=init_params.at['w2', 'init_value'], min=init_params.at['w2', 'min'], max=init_params.at['w2', 'max'], vary=True)
                    params.add(f'b_{i}', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=False)

            #Add gamma
            params.add('gamma', value=init_params.at['gamma', 'init_value'], min=init_params.at['gamma', 'min'], max=init_params.at['gamma', 'max'], vary=False)

            return params, initial_conditions


        elif not stringency_fit:
            #Init conditions
            initial_conditions = []

            if self.model=='SIR':
                for i in range(len(sector_props)):
                    initN_i = initN * sector_props[i]/100
                    initI_i = initI
                    initR_i = initR

                    initial_conditions_i = [initI_i, initR_i, initN_i]
                    initial_conditions.append(initial_conditions_i)

                    if self.beta_sigmoid_step and not self.stringency_fit:
                        params.add(f'C_{i}', value=init_params.at['C', 'init_value'], min=init_params.at['C', 'min'], max=init_params.at['C', 'max'], vary=True)
                        params.add(f'L_{i}', value=init_params.at['L', 'init_value'], min=init_params.at['L', 'min'], max=init_params.at['L', 'max'], vary=True)
                        params.add(f'a_{i}', value=init_params.at['a', 'init_value'], min=init_params.at['a', 'min'], max=init_params.at['a', 'max'], vary=True)
                        params.add(f'b_{i}', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                    elif self.beta_sigmoid_step and self.stringency_fit:
                        params.add(f'C_{i}', value=init_params.at['C', 'init_value'], min=init_params.at['C', 'min'], max=init_params.at['C', 'max'], vary=True)
                        params.add(f'L_{i}', value=init_params.at['L', 'init_value'], min=init_params.at['L', 'min'], max=init_params.at['L', 'max'], vary=True)
                        params.add(f'a_{i}', value=self.a_beta_init[-1], min=self.a_beta_init[-1], max=self.a_beta_init[-1]+0.1, vary=False)
                        params.add(f'b_{i}', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                    elif self.beta_linear_step:
                        params.add(f'w1_{i}', value=init_params.at['w1', 'init_value'], min=init_params.at['w1', 'min'], max=init_params.at['w1', 'max'], vary=True)
                        params.add(f'w2_{i}', value=init_params.at['w2', 'init_value'], min=init_params.at['w2', 'min'], max=init_params.at['w2', 'max'], vary=True)
                        params.add(f'b_{i}', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                if self.beta_constant_step:
                    params.add('beta', value=init_params.at['beta', 'init_value'], min=init_params.at['beta', 'min'], max=init_params.at['beta', 'max'], vary=True)

                #Add gamma
                params.add('gamma', value=init_params.at['gamma', 'init_value'], min=init_params.at['gamma', 'min'], max=init_params.at['gamma', 'max'], vary=disease_vary)


            elif self.model=='SEIR':
                for i in range(len(sector_props)):
                    initN_i = initN * sector_props[i]/100
                    initE_i = initE
                    initI_i = initI
                    initR_i = initR

                    initial_conditions_i = [initE_i, initI_i, initR_i, initN_i]
                    initial_conditions.append(initial_conditions_i)

                    if self.beta_sigmoid_step:
                        params.add(f'C_{i}', value=init_params.at['C', 'init_value'], min=init_params.at['C', 'min'], max=init_params.at['C', 'max'], vary=True)
                        params.add(f'L_{i}', value=init_params.at['L', 'init_value'], min=init_params.at['L', 'min'], max=init_params.at['L', 'max'], vary=True)
                        if self.stringency_fit:
                            params.add(f'a_{i}', value=self.a_beta_init[-1], min=self.a_beta_init[-1], max=self.a_beta_init[-1]+0.1, vary=False)
                        elif not self.stringency_fit:
                            params.add(f'a_{i}', value=init_params.at['a', 'init_value'], min=init_params.at['a', 'min'], max=init_params.at['a', 'max'], vary=True)
                        params.add(f'b_{i}', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                    elif self.beta_linear_step:
                        params.add(f'w1_{i}', value=init_params.at['w1', 'init_value'], min=init_params.at['w1', 'min'], max=init_params.at['w1', 'max'], vary=True)
                        params.add(f'w2_{i}', value=init_params.at['w2', 'init_value'], min=init_params.at['w2', 'min'], max=init_params.at['w2', 'max'], vary=True)
                        params.add(f'b_{i}', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                if self.beta_constant_step:
                    params.add('beta', value=init_params.at['beta', 'init_value'], min=init_params.at['beta', 'min'], max=init_params.at['beta', 'max'], vary=True)

                #Add sigma and gamma
                params.add('sigma', value=init_params.at['sigma', 'init_value'], min=init_params.at['sigma', 'min'], max=init_params.at['sigma', 'max'], vary=disease_vary)
                params.add('gamma', value=init_params.at['gamma', 'init_value'], min=init_params.at['gamma', 'min'], max=init_params.at['gamma', 'max'], vary=disease_vary)


            elif self.model =='SEIRD':
                for i in range(len(sector_props)):
                    initN_i = initN * sector_props[i]/100
                    initE_i = 0
                    initI_i = 1
                    initR_i = 0
                    initD_i = 0

                    initial_conditions_i = [initE_i, initI_i, initR_i, initN_i, initD_i]
                    initial_conditions.append(initial_conditions_i)

                    if self.beta_sigmoid_step and not self.mu_sigmoid_step:
                        params.add(f'C_{i}', value=init_params.at['C', 'init_value'], min=init_params.at['C', 'min'], max=init_params.at['C', 'max'], vary=True)
                        params.add(f'L_{i}', value=init_params.at['L', 'init_value'], min=init_params.at['L', 'min'], max=init_params.at['L', 'max'], vary=True)
                        params.add(f'a_{i}', value=init_params.at['a', 'init_value'], min=init_params.at['a', 'min'], max=init_params.at['a', 'max'], vary=True)
                        params.add(f'b_{i}', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                        params.add('mu', value=init_params.at['mu', 'init_value'], min=init_params.at['mu', 'min'], max=init_params.at['mu', 'max'], vary=disease_vary)

                    elif self.mu_sigmoid_step and not self.beta_sigmoid_step:
                        params.add(f'muC_{i}', value=init_params.at['muC', 'init_value'], min=init_params.at['muC', 'min'], max=init_params.at['muC', 'max'], vary=True)
                        params.add(f'muL_{i}', value=init_params.at['muL', 'init_value'], min=init_params.at['muL', 'min'], max=init_params.at['muL', 'max'], vary=True)
                        params.add(f'mua_{i}', value=init_params.at['mua', 'init_value'], min=init_params.at['mua', 'min'], max=init_params.at['mua', 'max'], vary=True)
                        params.add(f'mub_{i}', value=init_params.at['mub', 'init_value'], min=init_params.at['mub', 'min'], max=init_params.at['mub', 'max'], vary=True)

                        params.add('beta', value=init_params.at['beta', 'init_value'], min=init_params.at['beta', 'min'], max=init_params.at['beta', 'max'], vary=disease_vary)

                    elif self.beta_sigmoid_step and self.mu_sigmoid_step:
                        params.add(f'C_{i}', value=init_params.at['C', 'init_value'], min=init_params.at['C', 'min'], max=init_params.at['C', 'max'], vary=True)
                        params.add(f'L_{i}', value=init_params.at['L', 'init_value'], min=init_params.at['L', 'min'], max=init_params.at['L', 'max'], vary=True)
                        params.add(f'a_{i}', value=init_params.at['a', 'init_value'], min=init_params.at['a', 'min'], max=init_params.at['a', 'max'], vary=True)
                        params.add(f'b_{i}', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)
                        params.add(f'muC_{i}', value=init_params.at['muC', 'init_value'], min=init_params.at['muC', 'min'], max=init_params.at['muC', 'max'], vary=True)
                        params.add(f'muL_{i}', value=init_params.at['muL', 'init_value'], min=init_params.at['muL', 'min'], max=init_params.at['muL', 'max'], vary=True)
                        params.add(f'mua_{i}', value=init_params.at['mua', 'init_value'], min=init_params.at['mua', 'min'], max=init_params.at['mua', 'max'], vary=True)
                        params.add(f'mub_{i}', value=init_params.at['mub', 'init_value'], min=init_params.at['mub', 'min'], max=init_params.at['mub', 'max'], vary=True)

                    elif not self.beta_sigmoid_step and not self.mu_sigmoid_step:
                        params.add('beta', value=init_params.at['beta', 'init_value'], min=init_params.at['beta', 'min'], max=init_params.at['beta', 'max'], vary=disease_vary)
                        params.add('mu', value=init_params.at['mu', 'init_value'], min=init_params.at['mu', 'min'], max=init_params.at['mu', 'max'], vary=disease_vary)

                #Add sigma and gamma
                params.add('sigma', value=init_params.at['sigma', 'init_value'], min=init_params.at['sigma', 'min'], max=init_params.at['sigma', 'max'], vary=disease_vary)
                params.add('gamma', value=init_params.at['gamma', 'init_value'], min=init_params.at['gamma', 'min'], max=init_params.at['gamma', 'max'], vary=disease_vary)

            return params, initial_conditions

    def callback_func(self, params, iter_, resid, init_conditions, tspan, data, eps, bool):
        if iter_ != -1:
            for name in self.dict_training_params.keys():
                self.dict_training_params[name].append(params[name].value)


    def error_sectors(self, params, initial_conditions, tspan, data, eps, stringency_fit=False):
        n_sectors = len(initial_conditions)
        sol = np.zeros_like(data)

        if stringency_fit:
            for i in range(n_sectors):

                #Get params values
                Cs, Ls, a_s, b_s = params[f'Cs{i}'].value, params[f'Ls{i}'].value, params[f'as{i}'].value, params[f'bs{i}'].value

                stringency_params = [Cs, Ls, a_s, b_s]
                sol_i = self.logistic_step(tspan, stringency_params).reshape(data.shape)
                sol += sol_i

                return ((sol - data)/eps).ravel()

        elif not stringency_fit:
            if self.model=='SIR':
                idx = 1
            elif self.model=='SEIR':
                idx= 2
            elif self.model=='SEIRD':
                idx=2

            for i in range(n_sectors):
                sol_i = self.ode_solver(tspan, initial_conditions[i], params, i)
                #print('sol_i', sol_i)

                #Init I cases
                initI = initial_conditions[i][idx-1]

                #Compute cumulated I cases to minimize
                sol_ = self.get_cumulated_cases(sol_i, self.last_I_cumul, sol)
                #sol_ = self.get_cumulated_cases(sol_i, initI, sol)

                sol_=sol_[self.split_day:self.split_day+len(data)]

            return ((sol_ - data)/eps).ravel()

    def get_cumulated_cases(self, sol, initI, final_sol, test_phase=False):
        #Compute cumulated I cases to minimize
        cumul_i = [initI]

        if self.model=='SIR': #Compute dSdt
            #print("sol", sol)
            diff_i = np.diff(sol[:,0], 1)
            #print('diff_i', diff_i)
            #self.last_I = round(sol[-1,1], 0)
            #self.last_R = round(sol[-1,2], 0)
        if self.model=='SEIR':
            idx=2
            #self.last_E = round(sol[-1,idx-1],0)
            #self.last_I = round(sol[-1,idx], 0)
            #self.last_R = round(sol[-1,idx+1], 0)
            diff_S = np.diff(sol[:,0], 1)
            diff_E = np.diff(sol[:,idx-1], 1)

            diff_i = diff_E - np.abs(diff_S)

        if test_phase:
            diff_i = diff_i[self.split_day+self.train_window:self.split_day+self.train_window+7]
            #Add absolute value of dSdt to previous cumulated values
            for i, diff_ in enumerate(diff_i):
                cumul_i.append(np.abs(diff_)+cumul_i[i]) #round number to next integer (float type)

            final_sol_= np.asarray(cumul_i[:])
            return final_sol_

        elif not test_phase:
            #Add absolute value of dSdt to previous cumulated values
            for i, diff_ in enumerate(diff_i):
                cumul_i.append(np.abs(diff_)+cumul_i[i])

            final_sol_ = np.asarray(cumul_i[:])
            #print('final_sol', final_sol)
            return final_sol_


    @staticmethod
    def compute_mse(errors):
        return np.mean(errors**2)

    @staticmethod
    def compute_mae(errors):
        return np.mean(np.abs(errors))

    def train_test_split(self, data, split_day):
        train_date_start = data.loc[split_day, 'date']
        train_date = data.loc[split_day+self.train_window, 'date']
        #train_date = data.loc[split_day+38, 'date']
        test_date = data.loc[split_day+self.train_window+7, 'date']
        #test_date = data.loc[split_day+45, 'date']

        train_df = data[data['date'] < train_date]
        train_df = train_df[train_date_start <= train_df['date']]

        test_df = data[data['date'] >= train_date]
        test_df = test_df[test_date > test_df['date']]

        return train_df, test_df

    @staticmethod
    def remove_artefacts(dataframe):
        #Clean artefacts
        dataframe['dI'] = dataframe['I'].diff()
        #Init
        dataframe.loc[0, 'dI'] = 0
        indexes = list(np.where(dataframe['dI']<0)[0])

        while len(indexes) > 0:
            for i in indexes:
                dataframe.loc[i, 'I'] = dataframe.loc[i-1, 'I']

            #Update
            dataframe['dI'] = dataframe['I'].diff()
            indexes = list(np.where(dataframe['dI']<0)[0])

        return dataframe

    def rolling_window_fit(self, disease_vary=True, sector_props=[100.0]):
        #Loop over countries
        for k, country in enumerate(self.countries):
            try:
                EPI_data, country_attrs = country_covid(country, self.owid_file, self.hosp_file, model=self.model)
            except ValueError:
                print(f'incomplete data on {country}')
                return [], []

            df = EPI_data.drop(['N_effective'], axis=1).reindex(columns=['date', 'S', 'I', 'R', 'D', 's'])
            df.R.fillna(0, inplace=True)
            df.ffill(axis=0, inplace=True)
            df.bfill(axis=0, inplace=True)
            eps=1.0
            thresh =self.cutoff_date
            #Get first wave timeframe
            df = df[df['date'] <= thresh]

            #Remove artefacts
            df = self.remove_artefacts(df)
            days = len(df)
            data = df.loc[0:121, ['I']].values
            data_s = df.loc[0:121, ['s']].values
            data_s = data_s.reshape(len(data_s))
            data_s_norm = (data_s - data_s.min())/(data_s.max() - data_s.min())
            derivative = df.loc[1:121, 'dI'].values

            #Compute recovered cases
            #df['R'] = df.I - df.dI
            df['R'] = df['I'].shift(10) #lag of 10 days
            df.R.fillna(0, inplace=True)
            data_recovered = df.loc[0:121, ['R']].values


            #Append country full data and derivative
            self.data.append(data)
            self.recovered.append(data_recovered)
            self.data_s.append(data_s_norm)
            self.derivative.append(derivative)

            print(f'Fitting {country} data.')
            #List of data to store for plots
            mse = []
            STD=[]
            test_mse=[]
            list_train_fitted=[]
            list_fitted_deriv=[]
            list_result=[]
            list_params=[]
            list_test_predicted=[]
            list_test_deriv=[]

            #TRAIN_TEST SPLIT
            for i, split_day in enumerate([0, 38, 76]):

                self.split_day = split_day

                #SPLIT
                train_df, test_df = self.train_test_split(df, self.split_day)

                #Initialization
                if self.model=='SIR':
                    initI, initR = 1, 0

                elif self.model=='SEIR':
                    initE, initI, initR = 0, 1, 0

                if self.beta_constant_step:
                    max_cases = self.data[k][-1].item()
                    initN = max_cases

                elif self.beta_sigmoid_step or self.beta_linear_step:
                    initN = country_attrs['population']

                #Keep data
                if self.model=='SIR' or self.model=='SEIR':
                    train_data = train_df['I'].values
                    test_data = test_df['I'].values

                tspan_train = np.arange(1, 38+1+self.split_day, 1)
                self.stringency_country = data_s_norm
                #tspan_train = np.arange(1, 30+1, 1)

                #Fitting parameters
                if self.model=='SIR':
                    params, initial_conditions = self.init_sectors(sector_props, disease_vary=disease_vary, initN=initN, initI=initI, initR=initR, stringency_fit=False)

                elif self.model=='SEIR':
                    params, initial_conditions = self.init_sectors(sector_props, disease_vary=disease_vary, initN=initN, initE=initE, initI=initI, initR=initR, stringency_fit=False)

                #print('init train conditions (I,R,N)', initial_conditions)
                #If first training period, initialize with initI, else initialize with last value of cumulative cases
                self.last_I_cumul = initI


                #Fit model on train data
                #print('TRAIN {}'.format(i))
                if self.beta_linear_step:
                    #stringency_train = self.data_s[k][0:self.split_day+38]
                    result = minimize(self.error_sectors, params, args=(initial_conditions, tspan_train, train_data, eps, False), method='leastsq', full_output = 1)
                    print(report_fit(result))

                elif self.beta_constant_step or self.beta_sigmoid_step:
                    #Minimize only the data from the train timespan
                    result = minimize(self.error_sectors, params, args=(initial_conditions, tspan_train, train_data, eps, False), method='leastsq', full_output = 1)
                    #print(report_fit(result))
                    #result.params.pretty_print()

                #Init TESTING
                if self.model=='SIR':
                    self.last_I, self.last_R, self.last_N =initial_conditions[0]

                elif self.model=='SEIR':
                    self.last_E, self.last_I, self.last_R, self.last_N =initial_conditions[0]

                self.last_I_cumul = self.data[k][self.split_day+38-1].item()

                #Store train data
                train_final = train_data + result.residual.reshape(train_data.shape)
                #print('residual', result.residual)
                list_train_fitted.append(train_final)
                list_result.append(result)
                list_params.append(result.params)

                #Update Infections
                fitted_deriv = np.diff(train_final.ravel(), 1).reshape(train_data.shape[0]-1, 1)
                list_fitted_deriv.append(fitted_deriv)

                #Update MSE for country
                MSE = self.compute_mse(result.residual.reshape(train_data.shape))
                mse.append(MSE)

                #Compute error std for confidence interval
                std = np.std(result.residual)
                STD.append(std)

                #TESTING
                sol = np.zeros_like(test_data)
                tspan_test = np.arange(1, self.split_day+45+1, 1)
                #tspan_test = np.arange(1, 15+1, 1)
                #self.stringency_country = data_s_norm
                #stringency_test = self.data_s[k][0:self.split_day+45]
                params_fitted = result.params

                if self.model=='SIR':
                    initial_conditions_test = [self.last_I, self.last_R, self.last_N]
                elif self.model=='SEIR':
                    initial_conditions_test = [self.last_E, self.last_I, self.last_R, self.last_N]
                #print('TEST {}'.format(i))

                if self.beta_linear_step:
                    predicted = self.ode_solver(tspan_test, initial_conditions_test, params_fitted, 0)

                elif self.beta_constant_step or self.beta_sigmoid_step:
                    #print('init test conditions', initial_conditions_test)
                    predicted = self.ode_solver(tspan_test, initial_conditions_test, params_fitted, 0)

                #Compute cumulated I cases to minimize
                sol = self.get_cumulated_cases(predicted, self.last_I_cumul, sol, test_phase=True)
                test_error = ((sol - test_data)/eps).ravel()
                test_MSE = self.compute_mse(test_error)
                test_mse.append(test_MSE)

                #Keep test data
                list_test_predicted.append(sol)
                #Update Infections
                if self.model =='SIR':
                    test_deriv = predicted[self.split_day+38:self.split_day+45,1]
                elif self.model == 'SEIR':
                    test_deriv = predicted[self.split_day+38:self.split_day+45,2]
                list_test_deriv.append(test_deriv)

            #Store for each country
            self.mse.append(mse)
            self.test_mse.append(test_mse)
            self.std_error.append(STD)
            self.train_fitted.append(list_train_fitted)
            self.fitted_deriv.append(list_fitted_deriv)
            self.train_result.append(list_result)
            self.train_params.append(list_params)
            self.test_predicted.append(list_test_predicted)
            self.test_deriv.append(list_test_deriv)

        #Get dataframe of results
        results_df = self.results_to_dataframe()

        return results_df

    def results_to_dataframe(self):
        df_temp = pd.DataFrame(data=self.mse, index=self.countries, columns=["mse_train1", "mse_train2", "mse_train3"])

        if self.model=='SIR':
            if self.beta_constant_step:
                #Get beta
                beta  = np.asarray([[param['beta'] for param in sublist] for sublist in self.train_params])
                gamma  = np.asarray([[param['gamma'] for param in sublist] for sublist in self.train_params])

                for i in range(3):
                    df_temp["mse_test{}".format(i+1)]=np.asarray(self.test_mse)[:,i]
                    df_temp["beta_train{}".format(i+1)]=beta[:,i]
                    df_temp["gamma_train{}".format(i+1)]=gamma[:,i]

                #Compute statistics
                beta_ = np.vstack((df_temp['beta_train1'].values, df_temp['beta_train2'].values, df_temp['beta_train3'].values))
                gamma_ = np.vstack((df_temp['gamma_train1'].values, df_temp['gamma_train2'].values, df_temp['gamma_train3'].values))

                df_temp["mean_beta"] = np.mean(beta_, axis=0)
                df_temp["std_beta"] = np.std(beta_, axis=0)
                df_temp["mean_gamma"] = np.mean(gamma_, axis=0)
                df_temp["std_gamma"] = np.std(gamma_, axis=0)

            elif self.beta_linear_step:
                #Get beta params
                w  = np.asarray([[param[f'w_0'] for param in sublist] for sublist in self.train_params])
                b  = np.asarray([[param[f'b_0'] for param in sublist] for sublist in self.train_params])
                gamma  = np.asarray([[param['gamma'] for param in sublist] for sublist in self.train_params])

                for i in range(3):
                    df_temp["mse_test{}".format(i+1)]=np.asarray(self.test_mse)[:,i]
                    df_temp["w_train{}".format(i+1)]=w[:,i]
                    df_temp["b_train{}".format(i+1)]=b[:,i]
                    df_temp["gamma_train{}".format(i+1)]=gamma[:,i]

                #Compute statistics
                w_ = np.vstack((df_temp['w_train1'].values, df_temp['w_train2'].values, df_temp['w_train3'].values))
                b_ = np.vstack((df_temp['b_train1'].values, df_temp['b_train2'].values, df_temp['b_train3'].values))
                gamma_ = np.vstack((df_temp['gamma_train1'].values, df_temp['gamma_train2'].values, df_temp['gamma_train3'].values))

                df_temp["mean_w"] = np.mean(w_, axis=0)
                df_temp["std_w"] = np.std(w_, axis=0)
                df_temp["mean_b"] = np.mean(b_, axis=0)
                df_temp["std_b"] = np.std(b_, axis=0)
                df_temp["mean_gamma"] = np.mean(gamma_, axis=0)
                df_temp["std_gamma"] = np.std(gamma_, axis=0)

            elif self.beta_sigmoid_step:
                #Get beta params
                C  = np.asarray([[param[f'C_0'] for param in sublist] for sublist in self.train_params])
                L  = np.asarray([[param[f'L_0'] for param in sublist] for sublist in self.train_params])
                a  = np.asarray([[param[f'a_0'] for param in sublist] for sublist in self.train_params])
                b  = np.asarray([[param[f'b_0'] for param in sublist] for sublist in self.train_params])
                gamma  = np.asarray([[param['gamma'] for param in sublist] for sublist in self.train_params])

                for i in range(3):
                    df_temp["mse_test{}".format(i+1)]=np.asarray(self.test_mse)[:,i]
                    df_temp["C_train{}".format(i+1)]=C[:,i]
                    df_temp["L_train{}".format(i+1)]=L[:,i]
                    df_temp["a_train{}".format(i+1)]=a[:,i]
                    df_temp["b_train{}".format(i+1)]=b[:,i]
                    df_temp["gamma_train{}".format(i+1)]=gamma[:,i]

                #Compute statistics
                C_ = np.vstack((df_temp['C_train1'].values, df_temp['C_train2'].values, df_temp['C_train3'].values))
                L_ = np.vstack((df_temp['L_train1'].values, df_temp['L_train2'].values, df_temp['L_train3'].values))
                a_ = np.vstack((df_temp['a_train1'].values, df_temp['a_train2'].values, df_temp['a_train3'].values))
                b_ = np.vstack((df_temp['b_train1'].values, df_temp['b_train2'].values, df_temp['b_train3'].values))
                gamma_ = np.vstack((df_temp['gamma_train1'].values, df_temp['gamma_train2'].values, df_temp['gamma_train3'].values))

                df_temp["mean_C"] = np.mean(C_, axis=0)
                df_temp["std_C"] = np.std(C_, axis=0)
                df_temp["mean_L"] = np.mean(L_, axis=0)
                df_temp["std_L"] = np.std(L_, axis=0)
                df_temp["mean_a"] = np.mean(a_, axis=0)
                df_temp["std_a"] = np.std(a_, axis=0)
                df_temp["mean_b"] = np.mean(b_, axis=0)
                df_temp["std_b"] = np.std(b_, axis=0)
                df_temp["mean_gamma"] = np.mean(gamma_, axis=0)
                df_temp["std_gamma"] = np.std(gamma_, axis=0)

            #Compute MSE stats
            mse = np.vstack((df_temp['mse_train1'].values, df_temp['mse_train2'].values, df_temp['mse_train3'].values))
            mse_test = np.vstack((df_temp['mse_test1'].values, df_temp['mse_test2'].values, df_temp['mse_test3'].values))
            df_temp["mean_mse_train"] = np.mean(mse, axis=0)
            df_temp["std_mse_train"] = np.std(mse, axis=0)
            df_temp["mean_mse_test"] = np.mean(mse_test, axis=0)
            df_temp["std_mse_test"] = np.std(mse_test, axis=0)


        elif self.model=='SEIR':
            if self.beta_constant_step:
                #Get beta
                sigma  = np.asarray([[param['sigma'] for param in sublist] for sublist in self.train_params])
                beta  = np.asarray([[param['beta'] for param in sublist] for sublist in self.train_params])
                gamma  = np.asarray([[param['gamma'] for param in sublist] for sublist in self.train_params])

                for i in range(3):
                    df_temp["mse_test{}".format(i+1)]=np.asarray(self.test_mse)[:,i]
                    df_temp["sigma_train{}".format(i+1)]=sigma[:,i]
                    df_temp["beta_train{}".format(i+1)]=beta[:,i]
                    df_temp["gamma_train{}".format(i+1)]=gamma[:,i]

                #Compute statistics
                sigma_ = np.vstack((df_temp['sigma_train1'].values, df_temp['sigma_train2'].values, df_temp['sigma_train3'].values))
                beta_ = np.vstack((df_temp['beta_train1'].values, df_temp['beta_train2'].values, df_temp['beta_train3'].values))
                gamma_ = np.vstack((df_temp['gamma_train1'].values, df_temp['gamma_train2'].values, df_temp['gamma_train3'].values))

                df_temp["mean_sigma"] = np.mean(sigma_, axis=0)
                df_temp["std_sigma"] = np.std(sigma_, axis=0)
                df_temp["mean_beta"] = np.mean(beta_, axis=0)
                df_temp["std_beta"] = np.std(beta_, axis=0)
                df_temp["mean_gamma"] = np.mean(gamma_, axis=0)
                df_temp["std_gamma"] = np.std(gamma_, axis=0)

            elif self.beta_linear_step:
                #Get beta params
                w  = np.asarray([[param[f'w_0'] for param in sublist] for sublist in self.train_params])
                b  = np.asarray([[param[f'b_0'] for param in sublist] for sublist in self.train_params])
                sigma  = np.asarray([[param['sigma'] for param in sublist] for sublist in self.train_params])
                gamma  = np.asarray([[param['gamma'] for param in sublist] for sublist in self.train_params])

                for i in range(3):
                    df_temp["mse_test{}".format(i+1)]=np.asarray(self.test_mse)[:,i]
                    df_temp["w_train{}".format(i+1)]=w[:,i]
                    df_temp["b_train{}".format(i+1)]=b[:,i]
                    df_temp["sigma_train{}".format(i+1)]=sigma[:,i]
                    df_temp["gamma_train{}".format(i+1)]=gamma[:,i]

                #Compute statistics
                w_ = np.vstack((df_temp['w_train1'].values, df_temp['w_train2'].values, df_temp['w_train3'].values))
                b_ = np.vstack((df_temp['b_train1'].values, df_temp['b_train2'].values, df_temp['b_train3'].values))
                sigma_ = np.vstack((df_temp['sigma_train1'].values, df_temp['sigma_train2'].values, df_temp['sigma_train3'].values))
                gamma_ = np.vstack((df_temp['gamma_train1'].values, df_temp['gamma_train2'].values, df_temp['gamma_train3'].values))

                df_temp["mean_w"] = np.mean(w_, axis=0)
                df_temp["std_w"] = np.std(w_, axis=0)
                df_temp["mean_b"] = np.mean(b_, axis=0)
                df_temp["std_b"] = np.std(b_, axis=0)
                df_temp["mean_sigma"] = np.mean(sigma_, axis=0)
                df_temp["std_sigma"] = np.std(sigma_, axis=0)
                df_temp["mean_gamma"] = np.mean(gamma_, axis=0)
                df_temp["std_gamma"] = np.std(gamma_, axis=0)

            elif self.beta_sigmoid_step:
                #Get beta params
                C  = np.asarray([[param[f'C_0'] for param in sublist] for sublist in self.train_params])
                L  = np.asarray([[param[f'L_0'] for param in sublist] for sublist in self.train_params])
                a  = np.asarray([[param[f'a_0'] for param in sublist] for sublist in self.train_params])
                b  = np.asarray([[param[f'b_0'] for param in sublist] for sublist in self.train_params])
                sigma  = np.asarray([[param['sigma'] for param in sublist] for sublist in self.train_params])
                gamma  = np.asarray([[param['gamma'] for param in sublist] for sublist in self.train_params])

                for i in range(3):
                    df_temp["mse_test{}".format(i+1)]=np.asarray(self.test_mse)[:,i]
                    df_temp["C_train{}".format(i+1)]=C[:,i]
                    df_temp["L_train{}".format(i+1)]=L[:,i]
                    df_temp["a_train{}".format(i+1)]=a[:,i]
                    df_temp["b_train{}".format(i+1)]=b[:,i]
                    df_temp["sigma_train{}".format(i+1)]=sigma[:,i]
                    df_temp["gamma_train{}".format(i+1)]=gamma[:,i]

                #Compute statistics
                C_ = np.vstack((df_temp['C_train1'].values, df_temp['C_train2'].values, df_temp['C_train3'].values))
                L_ = np.vstack((df_temp['L_train1'].values, df_temp['L_train2'].values, df_temp['L_train3'].values))
                a_ = np.vstack((df_temp['a_train1'].values, df_temp['a_train2'].values, df_temp['a_train3'].values))
                b_ = np.vstack((df_temp['b_train1'].values, df_temp['b_train2'].values, df_temp['b_train3'].values))
                sigma_ = np.vstack((df_temp['sigma_train1'].values, df_temp['sigma_train2'].values, df_temp['sigma_train3'].values))
                gamma_ = np.vstack((df_temp['gamma_train1'].values, df_temp['gamma_train2'].values, df_temp['gamma_train3'].values))

                df_temp["mean_C"] = np.mean(C_, axis=0)
                df_temp["std_C"] = np.std(C_, axis=0)
                df_temp["mean_L"] = np.mean(L_, axis=0)
                df_temp["std_L"] = np.std(L_, axis=0)
                df_temp["mean_a"] = np.mean(a_, axis=0)
                df_temp["std_a"] = np.std(a_, axis=0)
                df_temp["mean_b"] = np.mean(b_, axis=0)
                df_temp["std_b"] = np.std(b_, axis=0)
                df_temp["mean_sigma"] = np.mean(sigma_, axis=0)
                df_temp["std_sigma"] = np.std(sigma_, axis=0)
                df_temp["mean_gamma"] = np.mean(gamma_, axis=0)
                df_temp["std_gamma"] = np.std(gamma_, axis=0)

            #Compute MSE stats
            mse = np.vstack((df_temp['mse_train1'].values, df_temp['mse_train2'].values, df_temp['mse_train3'].values))
            mse_test = np.vstack((df_temp['mse_test1'].values, df_temp['mse_test2'].values, df_temp['mse_test3'].values))
            df_temp["mean_mse_train"] = np.mean(mse, axis=0)
            df_temp["std_mse_train"] = np.std(mse, axis=0)
            df_temp["mean_mse_test"] = np.mean(mse_test, axis=0)
            df_temp["std_mse_test"] = np.std(mse_test, axis=0)

        return df_temp


    def plot_rolling_window(self, cases=False, derivative = False, beta_param=False, beta_param_linear=False,
                              mse_train=False, mse_test=False, STATIC_PLOTS=True):

        assert len(self.train_fitted) > 1, "The model has not been previously fitted. Please call `rolling_window_fit` function before."

        complete_list_countries = [ x for x in self.countries for _ in (0,1,2)]

        #Plots with all countries
        if cases:

            #Compute confidence interval bounds
            bounds_95 = np.asarray([float(1.96*self.std_error[0][0]*np.sqrt(h)) for h in range(1,8)])
            bounds_90 = np.asarray([float(1.64*self.std_error[0][0]*np.sqrt(h)) for h in range(1,8)])

            tspan_train = np.arange(1, 38+1, 1)
            tspan_test = np.arange(39, 39+7, 1)
            tspan_full = np.arange(1, 45+1, 1)
            train_idxs = [1, 38, 76]*28

            #Plots with all countries
            fig = make_subplots(rows=28, cols=3, subplot_titles=(complete_list_countries))
            fig.add_trace(go.Scatter(x=tspan_train, y=self.train_fitted[0][0], mode='markers', name='Train fitted cases',
                                     line = dict(dash='dot', color='red')), row=1, col=1)

            fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[0][0]+bounds_95, mode='lines', name='95% Confidence Interval',
                                     line = dict(color='gold')),row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[0][0]-bounds_95, mode='lines',
                                     line = dict(color='gold'),  fill='tonexty', fillcolor = 'khaki' , showlegend=False),row=1, col=1)

            fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[0][0]+bounds_90, mode='lines', name='90% Confidence Interval',
                                     line = dict(color='burlywood')),row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[0][0]-bounds_90, mode='lines',
                                     line = dict(color='burlywood'),  fill='tonexty', fillcolor = 'burlywood' , showlegend=False),row=1, col=1)

            fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[0][0], mode='lines+markers', name='Test predicted cases',
                                     line = dict(dash='dot', color='sienna')),row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_full, y=self.data[0][0:45].ravel(), mode='lines', name='Real cases',
                                                              line = dict(dash='dot', color='black')), row=1, col=1)


            rows_ = [val for val in range(1,29) for _ in (0, 1, 2)]
            cols_ = [1,2,3]*28

            for i, j, train_idx in zip(rows_[1:], cols_[1:], train_idxs[1:]):
                tspan_train = np.arange(train_idx, train_idx+38, 1)
                tspan_test = np.arange(train_idx+38, train_idx+45, 1)
                tspan_full = np.arange(train_idx, train_idx+45, 1)

                #Compute confidence interval bounds
                bounds_95 = np.asarray([float(1.96*self.std_error[i-1][j-1]*np.sqrt(h)) for h in range(1,8)])
                bounds_90 = np.asarray([float(1.64*self.std_error[i-1][j-1]*np.sqrt(h)) for h in range(1,8)])


                fig.add_trace(go.Scatter(x=tspan_train, y=self.train_fitted[i-1][j-1], mode='markers', name='Train fitted cases',
                                         line = dict(dash='dot', color='red'), showlegend=False),row=i, col=j)

                #IC
                fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[i-1][j-1]+bounds_95, mode='lines',
                                         line = dict(color='gold'), showlegend=False),row=i, col=j)
                fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[i-1][j-1]-bounds_95, mode='lines', name='95% Confidence Interval',
                                         line = dict(color='gold'), showlegend=False, fill='tonexty', fillcolor = 'khaki' ),row=i, col=j)


                fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[i-1][j-1]+bounds_90, mode='lines',
                                         line = dict(color='burlywood'), showlegend=False),row=i, col=j)
                fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[i-1][j-1]-bounds_90, mode='lines', name='90% Confidence Interval',
                                         line = dict(color='burlywood'), showlegend=False, fill='tonexty', fillcolor = 'burlywood' ),row=i, col=j)

                fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[i-1][j-1], mode='lines+markers', name='Test predicted cases',
                                         line = dict(dash='dot', color='sienna'), showlegend=False), row=i, col=j)
                fig.add_trace(go.Scatter(x=tspan_full, y=self.data[i-1][train_idx-1:train_idx-1+45].ravel(), mode='lines', name='Real cases',
                                     line = dict(dash='dot', color='black'), showlegend=False), row=i, col=j)


                fig.update_xaxes(title_text="Days since first infected", row=i, col=j, showgrid=True)

                fig.update_yaxes(title_text="Count", row=i, col=1)


            fig.update_layout(title='{} per country: Observed vs Fitted Cumulated Cases'.format(self.model),
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Counts',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,
                                  width=950, height=5600, legend=dict(font_size=14))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

        elif derivative:
            tspan_train = np.arange(2, 38+1, 1)
            tspan_test = np.arange(39, 39+7, 1)
            tspan_full = np.arange(2, 45+1, 1)
            train_idxs = [2, 38, 76]*28

            #Plots with all countries
            fig = make_subplots(rows=28, cols=3, subplot_titles=(complete_list_countries))
            fig.add_trace(go.Scatter(x=tspan_train, y=self.fitted_deriv[0][0].ravel(), mode='markers', name='Train fitted daily cases',
                                     line = dict(dash='dot', color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_test, y=self.test_deriv[0][0].ravel(), mode='lines+markers', name='Test predicted daily cases',
                                     line = dict(dash='dot', color='orange')),row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_full, y=self.derivative[0][0:45].ravel(), mode='lines', name='Real daily cases',
                                     line = dict(dash='dot', color='black')), row=1, col=1)


            rows_ = [val for val in range(1,29) for _ in (0, 1, 2)]
            cols_ = [1,2,3]*28

            for i, j, train_idx in zip(rows_[1:], cols_[1:], train_idxs[1:]):
                tspan_train = np.arange(train_idx, train_idx+37, 1)
                tspan_test = np.arange(train_idx+37, train_idx+44, 1)
                tspan_full = np.arange(train_idx, train_idx+44, 1)

                fig.add_trace(go.Scatter(x=tspan_train, y=self.fitted_deriv[i-1][j-1].ravel(), mode='markers', name='Train fitted daily cases',
                                         line = dict(dash='dot', color='red'), showlegend=False),row=i, col=j)

                fig.add_trace(go.Scatter(x=tspan_test, y=self.test_deriv[i-1][j-1].ravel(), mode='lines+markers', name='Test predicted daily cases',
                                         line = dict(dash='dot', color='orange'), showlegend=False), row=i, col=j)
                fig.add_trace(go.Scatter(x=tspan_full, y=self.derivative[i-1][train_idx-1:train_idx+44].ravel(), mode='lines', name='Real daily cases',
                                     line = dict(dash='dot', color='black'), showlegend=False), row=i, col=j)


                fig.update_xaxes(title_text="Days since first infected", row=i, col=j, showgrid=True)

                fig.update_yaxes(title_text="Daily Count", row=i, col=1)


            fig.update_layout(title='{} per country: Observed vs Fitted and Predicted Daily Cases'.format(self.model),
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Daily Counts',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,
                                  width=950, height=5600, legend=dict(font_size=14))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

        elif beta_param:
            pnames = list((self.train_params[0][0].valuesdict()))
            tspan_train = np.arange(2, 60+1, 1)
            tspan_test = np.arange(61, 61+7, 1)
            tspan_full = np.arange(2, 67+1, 1)
            train_idxs = [2, 60, 120]*28

            b_params = [self.train_params[0][0][p].value for p in pnames[:4]]
            beta_curve_train = self.logistic_step(tspan_train, b_params)
            beta_curve_test = self.logistic_step(tspan_test, b_params)


            #Plots with all countries
            fig = make_subplots(rows=28, cols=3, subplot_titles=(complete_list_countries))
            fig.add_trace(go.Scatter(x=tspan_train, y=beta_curve_train, mode='lines+markers', name='Train beta curve',
                                     line = dict(dash='dot', color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_test, y=beta_curve_test, mode='lines+markers', name='Test beta curve',
                                     line = dict(dash='dot', color='orange')),row=1, col=1)


            rows_ = [val for val in range(1,29) for _ in (0, 1, 2)]
            cols_ = [1,2,3]*28

            for i, j, train_idx in zip(rows_[1:], cols_[1:], train_idxs[1:]):
                tspan_train = np.arange(train_idx, train_idx+59, 1)
                tspan_test = np.arange(train_idx+59, train_idx+66, 1)
                tspan_full = np.arange(train_idx, train_idx+66, 1)

                b_params = [self.train_params[i-1][j-1][p].value for p in pnames[:4]]
                beta_curve_train = self.logistic_step(tspan_train, b_params)
                beta_curve_test = self.logistic_step(tspan_test, b_params)

                fig.add_trace(go.Scatter(x=tspan_train, y=beta_curve_train, mode='lines+markers', name='Train beta curve',
                                         line = dict(dash='dot', color='red'), showlegend=False), row=i, col=j)
                fig.add_trace(go.Scatter(x=tspan_test, y=beta_curve_test, mode='lines+markers', name='Test beta curve',
                                         line = dict(dash='dot', color='orange'), showlegend=False),row=i, col=j)


                fig.update_xaxes(title_text="Days since first infected", row=i, col=j, showgrid=True)

                fig.update_yaxes(title_text="Beta value", row=i, col=1)


            fig.update_layout(title='{} per country: Beta fitted'.format(self.model),
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Beta value',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,
                                  width=950, height=5600, legend=dict(font_size=14))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

        elif beta_param_linear:
            pnames = list((self.train_params[0][0].valuesdict()))
            tspan_train = np.arange(1, 38+1, 1)
            tspan_test = np.arange(39, 45+1, 1)
            tspan_full = np.arange(1, 45+1, 1)
            train_idxs = [1, 38, 76]*28

            b_params = [self.train_params[0][0][p].value for p in pnames[:2]]
            stringency_train = self.data_s[0][0:38]
            stringency_test = self.data_s[0][38:45]
            beta_curve_train = self.linear_step(np.asarray(stringency_train), b_params)
            beta_curve_test = self.linear_step(np.asarray(stringency_test), b_params)


            #Plots with all countries
            fig = make_subplots(rows=28, cols=3, subplot_titles=(complete_list_countries), horizontal_spacing=0.1,
                    specs=[x*3 for x in [[{"secondary_y": True}]]*28])
            fig.add_trace(go.Scatter(x=tspan_train, y=beta_curve_train, mode='lines+markers', name='Train beta curve',
                                     line = dict(dash='dot', color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_test, y=beta_curve_test, mode='lines+markers', name='Test beta curve',
                                     line = dict(dash='dot', color='orange')),row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_full, y=self.data_s[0][0:45], mode='lines', name='Stringency',
                                     line = dict(dash='dot', color='black')),row=1, col=1, secondary_y=True)

            fig.update_yaxes(row=1, col=1, secondary_y=True, tickfont=dict(color="blue"))


            rows_ = [val for val in range(1,29) for _ in (0, 1, 2)]
            cols_ = [1,2,3]*28

            for i, j, train_idx in zip(rows_[1:], cols_[1:], train_idxs[1:]):
                tspan_train = np.arange(train_idx, train_idx+38, 1)
                tspan_test = np.arange(train_idx+38, train_idx+45, 1)
                tspan_full = np.arange(train_idx, train_idx+45, 1)

                b_params = [self.train_params[i-1][j-1][p].value for p in pnames[:2]]
                stringency_train = self.data_s[i-1][train_idx-1:train_idx+38-1]
                stringency_test = self.data_s[i-1][train_idx+38-1:train_idx+45-1]
                beta_curve_train = self.linear_step(stringency_train, b_params)
                beta_curve_test = self.linear_step(stringency_test, b_params)

                fig.add_trace(go.Scatter(x=tspan_train, y=beta_curve_train, mode='lines+markers', name='Train beta curve',
                                         line = dict(dash='dot', color='red'), showlegend=False), row=i, col=j)
                fig.add_trace(go.Scatter(x=tspan_test, y=beta_curve_test, mode='lines+markers', name='Test beta curve',
                                         line = dict(dash='dot', color='orange'), showlegend=False),row=i, col=j)
                fig.add_trace(go.Scatter(x=tspan_full, y=self.data_s[i-1][train_idx-1:train_idx+45-1], mode='lines', name='Stringency',
                                         line = dict(dash='dot', color='black'),  showlegend=False),row=i, col=j,  secondary_y=True)


                #Update axis colors and texts
                fig.update_xaxes(title_text="Days since first infected", row=i, col=j, showgrid=True)
                fig.update_yaxes(title_text="Beta value", row=i, col=1, secondary_y=False, titlefont=dict(color="black"),
                                tickfont=dict(color="black"))
                fig.update_yaxes(title_text="OxCGRT index", row=i, col=3, secondary_y=True, titlefont=dict(color="blue"),
                                tickfont=dict(color="blue"))
                fig.update_yaxes(row=i, col=j, secondary_y=False, tickfont=dict(color="black"))
                fig.update_yaxes(row=i, col=j, secondary_y=True, tickfont=dict(color="blue"))


            fig.update_layout(title='{} per country: Beta fitted'.format(self.model),
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Beta value',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,
                                  width=950, height=5600, legend=dict(font_size=14))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

        elif mse_train:
            #Get standardizing denominator
            range_ = np.asarray([np.max(i)- np.min(i)+0.0001 for i in self.data])
            #Standardize
            numerator1 = np.asarray(np.asarray(self.mse)[:,0], dtype=float)
            numerator2 = np.asarray(np.asarray(self.mse)[:,1], dtype=float)
            numerator3 = np.asarray(np.asarray(self.mse)[:,2], dtype=float)
            mse_n1 = np.divide(numerator1, range_)
            mse_n2 = np.divide(numerator2, range_)
            mse_n3 = np.divide(numerator3, range_)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=self.countries,
                y=mse_n1,
                name='Train 1',
                marker_color='#AF0038'
            ))
            fig.add_trace(go.Bar(
                x=self.countries,
                y=mse_n2,
                name='Train 2',
                marker_color='indianred'
            ))
            fig.add_trace(go.Bar(
                x=self.countries,
                y=mse_n3,
                name='Train 3',
                marker_color='lightsalmon'
            ))

            # Here we modify the tickangle of the xaxis, resulting in rotated labels.
            fig.update_layout(barmode='group', xaxis_tickangle=-45,
                                title='{} per country: Training MSE'.format(self.model),
                                   yaxis_title='MSE',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,legend=dict(font_size=14))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

        elif mse_test:
            #Get standardizing denominator
            range_ = np.asarray([np.max(i)- np.min(i)+0.0001 for i in self.data])
            #Standardize
            numerator1 = np.asarray(np.asarray(self.test_mse)[:,0], dtype=float)
            numerator2 = np.asarray(np.asarray(self.test_mse)[:,1], dtype=float)
            numerator3 = np.asarray(np.asarray(self.test_mse)[:,2], dtype=float)
            mse_n1 = np.divide(numerator1, range_)
            mse_n2 = np.divide(numerator2, range_)
            mse_n3 = np.divide(numerator3, range_)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=self.countries,
                y= mse_n1,
                name='Test 1',
                marker_color='#AF0038'
            ))
            fig.add_trace(go.Bar(
                x=self.countries,
                y= mse_n2,
                name='Test 2',
                marker_color='indianred'
            ))
            fig.add_trace(go.Bar(
                x=self.countries,
                y=mse_n3,
                name='Test 3',
                marker_color='lightsalmon'
            ))

            # Here we modify the tickangle of the xaxis, resulting in rotated labels.
            fig.update_layout(barmode='group', xaxis_tickangle=-45,
                title='{} per country: Test MSE'.format(self.model),
               yaxis_title='MSE',
               title_x=0.5,
               title_font_size=14,
               font_size=14,legend=dict(font_size=14))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()


    def fit(self, disease_vary=True, sector_props=[100.0], train_window=38):

        self.train_window = train_window
        full_window = self.train_window + 7

        #Loop over countries
        for k, country in enumerate(self.countries):
            try:
                EPI_data, country_attrs = country_covid(country, self.owid_file, self.hosp_file, model=self.model)
            except ValueError:
                print(f'incomplete data on {country}')
                return [], []

            df = EPI_data.drop(['N_effective'], axis=1).reindex(columns=['date', 'S', 'I', 'R', 'D', 's'])
            df.R.fillna(0, inplace=True)
            df.ffill(axis=0, inplace=True)
            df.bfill(axis=0, inplace=True)

            #if country in ['USA', 'Sweden', 'United Kingdom', 'South Africa', 'Poland', 'Portugal', 'Russia', 'India', 'Denmark', 'Canada', 'Bulgaria']:
                #thresh = '2020-06-30'
            #else:
            #thresh =self.cutoff_date

            #Stringency
            data_s = df.loc[0:, ['s']].values
            data_s = data_s.reshape(len(data_s))
            data_s_norm = (data_s - data_s.min())/(data_s.max() - data_s.min())
            #Smooth stringency
            df['s_norm'] = data_s_norm
            df['stringency_smooth']= df['s_norm'].rolling(20, center=False).mean().reset_index(0, drop=True).fillna(0.001)
            s_smooth = df.loc[0:, ['stringency_smooth']].values
            self.data_s.append(s_smooth.ravel())
            self.stringency_country = s_smooth

            #Add mobility data
            if self.beta_linear_step:
                mobility_df = get_data_w_mobility(self.iso_code2[k])
                #Merge datasets
                df = df.merge(mobility_df[['mobility_index','date']], left_on=['date'], right_on=['date'], how='left')
                df['mobility_index'].bfill(inplace=True) #fill nan
                df['mobility_index_smooth']= df['mobility_index'].rolling(15, center=False).mean().reset_index(0, drop=True).fillna(0.5)
                self.data_mob.append(df.loc[0:, ['mobility_index_smooth']].values.ravel())
                self.mob_country = df.loc[0:, ['mobility_index_smooth']].values.ravel()

            #Crop df
            df = df[0:full_window+1]
            #df = df[df['date'] <= thresh]
            #df = df[df['date'] < str(date.today())]
            #Remove artefacts
            df = self.remove_artefacts(df)
            days = len(df)
            data = df.loc[0:full_window, ['I']].values
            derivative = df.loc[1:full_window, 'dI'].values

            #Compute recovered cases
            #df['R'] = df.I - df.dI
            df['R'] = df['I'].shift(10) #lag of 10 days
            df.R.fillna(0, inplace=True)
            #data_recovered = df.loc[0:121, ['R']].values

            #Append country full data and derivative
            self.data.append(data.ravel())
            #self.recovered.append(data_recovered)
            self.derivative.append(derivative.ravel())

            print(f'Fitting {country} data.')
            #List of data to store for plots
            #mse = []
            #STD=[]
            test_mse=[]
            list_train_fitted=[]
            #list_fitted_deriv=[]
            list_result=[]
            list_params=[]
            list_test_predicted=[]
            #list_test_deriv=[]

            split_day = 0
            self.split_day = split_day
            #SPLIT
            train_df, test_df = self.train_test_split(df, self.split_day)
            #print(train_df)

            #Initialization
            if self.model=='SIR':
                initI, initR = 1, 0

            elif self.model=='SEIR':
                initE, initI, initR = 0, 1, 0

            if self.beta_constant_step:
                max_cases = self.data[k][-1].item()
                initN = max_cases

            elif self.beta_sigmoid_step or self.beta_linear_step:
                initN = country_attrs['population']

            #Keep data
            if self.model=='SIR' or self.model=='SEIR':
                train_data = train_df['I'].values
                test_data = test_df['I'].values


            eps=1.0
            #tspan = np.arange(1, self.days[k]+1, 1)
            tspan_train = np.arange(1, self.train_window+1+self.split_day, 1)
            #tspan_full = np.arange(1, self.days[k]+1, 1)

            self.last_I_cumul = initI

            # Fitting stringency curve
            if self.stringency_fit:
                params_stringency = self.init_sectors(sector_props, disease_vary=disease_vary, initN=initN, stringency_fit=self.stringency_fit)
                # fit model
                result_s = minimize(self.error_sectors, params_stringency, args=(sector_props, tspan, self.data_s[k], eps, self.stringency_fit), method='leastsq', full_output = 1)
                #result_s.params.pretty_print()

                #Keep stringency results
                self.result_stringency.append(result_s)
                #Keep inflexion point fitted
                self.a_beta_init.append(result_s.params['as0'].value)
                #Keep fitted stringency curve
                final_stringency = self.data_s[k] + result_s.residual.reshape(self.data_s[k].shape)
                self.stringency_fitted.append(final_stringency)


            #Fitting other parameters
            if self.model=='SIR':
                params, initial_conditions = self.init_sectors(sector_props, disease_vary=disease_vary, initN=initN, initI=initI, initR=initR, stringency_fit=False)
            elif self.model=='SEIR':
                params, initial_conditions = self.init_sectors(sector_props, disease_vary=disease_vary, initN=initN, initE=initE ,initI=initI, initR=initR, stringency_fit=False)


            #print(f'Fitting {country} data with {self.days[k]} days.')
            #Fit model and find predicted values
            if self.beta_linear_step:
                result = minimize(self.error_sectors, params, args=(initial_conditions, tspan_train, train_data, eps, False), method='leastsq', full_output = 1)

            elif self.beta_constant_step or self.beta_sigmoid_step:
                #Minimize only the data from the train timespan
                #print('train data shape ', train_data.shape)
                #print('tspan train ', tspan_train.shape)
                result = minimize(self.error_sectors, params, args=(initial_conditions, tspan_train, train_data, eps, False), method='leastsq', full_output = 1)
            #print(report_fit(result))
            #result.params.pretty_print()

            #Store locally
            save_to_path ='./{0}/{1}/temp_res/best_result_{2}_{3}days.npy'.format(self.file_method, self.file_model, country, train_window)
            np.save(save_to_path, result)

            #Store for each country
            train_final = train_data + result.residual.reshape(train_data.shape)
            self.train_fitted.append(train_final.ravel())
            self.train_result.append(result)
            self.train_params.append(result.params)

            #Update MSE for country
            MSE = self.compute_mse(result.residual.reshape(train_data.shape))
            self.mse.append(MSE)
            #Compute error std for confidence interval
            STD = np.std(result.residual)
            self.std_error.append(STD)
            #Update the 1st derivative of cases
            self.derivative.append(df.loc[1:self.train_window, 'dI'].values)
            self.fitted_deriv.append(np.diff(train_final.ravel(), 1).reshape(train_data.shape[0]-1, 1))


            ##########################Init TESTING
            if self.model=='SIR':
                self.last_I, self.last_R, self.last_N =initial_conditions[0]

            elif self.model=='SEIR':
                self.last_E, self.last_I, self.last_R, self.last_N =initial_conditions[0]

            self.last_I_cumul = self.data[k][self.split_day+self.train_window-1].item()

            #TESTING
            sol = np.zeros_like(test_data)
            tspan_test = np.arange(1, self.split_day+full_window+1, 1)
            params_fitted = result.params

            if self.model=='SIR':
                initial_conditions_test = [self.last_I, self.last_R, self.last_N]
            elif self.model=='SEIR':
                initial_conditions_test = [self.last_E, self.last_I, self.last_R, self.last_N]


            if self.beta_linear_step:
                predicted = self.ode_solver(tspan_test, initial_conditions_test, params_fitted, 0)

            elif self.beta_constant_step or self.beta_sigmoid_step:
                predicted = self.ode_solver(tspan_test, initial_conditions_test, params_fitted, 0)

            #Compute cumulated I cases
            sol = self.get_cumulated_cases(predicted, self.last_I_cumul, sol, test_phase=True)
            test_error = ((sol - test_data)/eps).ravel()
            test_MSE = self.compute_mse(test_error)
            self.test_mse.append(test_MSE)
            self.test_predicted.append(sol)
            self.std_error.append(STD)

            #Update Infections
            if self.model =='SIR':
                test_deriv = predicted[self.split_day+self.train_window:self.split_day+full_window, 1]
            elif self.model == 'SEIR':
                test_deriv = predicted[self.split_day+self.train_window:self.split_day+full_window, 2]
            self.test_deriv.append(test_deriv)

        return self.mse


    def plot(self, stringency = False, cases=False, beta_param=False, beta_param_linear=False, beta_linear_surface=False, derivative = False, MSE = False, MSE_norm=False, MAE_norm=False, rate_params = False, STATIC_PLOTS=True):

        assert len(self.train_fitted) > 1, "The model has not been previously fitted. Please call 'fit' function before."

        #tspan_full = [np.arange(1, days+1, 1) for days in self.days]
        full_window = self.split_day+self.train_window+7
        train_window = self.split_day+self.train_window
        tspan_full = np.arange(1, full_window+1, 1)
        tspan_train = np.arange(1, train_window+1, 1)
        tspan_test = np.arange(train_window, full_window+1, 1)

        #Plots with all countries
        if stringency:
            assert len(self.stringency_fitted) > 1, "Stringency was not used for model fitting and thus cannot be plotted."

            #Plot stringency fits for each country
            fig = make_subplots(rows=10, cols=3, subplot_titles=(self.countries))
            fig.add_trace(go.Scatter(x=tspan_full[0], y=self.data_s[0], mode='markers', name='Observed Stringency',line = dict(dash='dot', color='purple')),
                    row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_full[0], y=self.stringency_fitted[0], mode='lines+markers', name='Fitted Stringency',line = dict(dash='dot', color='green')),
                      row=1, col=1)

            rows_ = [val for val in range(1,11) for _ in (0, 1, 2)]
            cols_ = [1,2,3]*10

            for idx, i, j in zip(range(1, len(self.data_s)), rows_[1:-1], cols_[1:-1]):
                fig.add_trace(go.Scatter(x=tspan_full[idx], y=self.data_s[i], mode='markers', name='Observed Stringency',line = dict(dash='dot', color='purple'), showlegend=False),
                        row=i, col=j)

                fig.add_trace(go.Scatter(x=tspan_full[idx], y=self.stringency_fitted[i], mode='lines+markers', name='Fitted Stringency',line = dict(dash='dot', color='green'), showlegend=False),
                          row=i, col=j)
                fig.update_xaxes(title_text="Days since first infected", row=i, col=j, showgrid=True)
                fig.update_yaxes(title_text="Level", row=i, col=1)


            fig.update_layout(title='Observed vs Fitted stringency for each country',
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Level',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,
                                  width=950, height=2600,  legend=dict(font_size=14))
                                 # legend=dict(font_size=14, yanchor="top",y=0.99, xanchor="left", x=0.01))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

        elif cases:

            fig = make_subplots(rows=10, cols=3, subplot_titles=(self.countries))
            fig.add_trace(go.Scatter(x=tspan_full, y=self.data[0], mode='markers', name='Observed Infections',line = dict(dash='dot', color='black')),
                    row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_train, y=self.train_fitted[0], mode='lines+markers', name='Fitted Infections',line = dict(dash='dot', color='red')),
                      row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[0], mode='lines+markers', name='Predicted Infections',line = dict(dash='dot', color='orange')),
                                row=1, col=1)

            rows_ = [val for val in range(1,11) for _ in (0, 1, 2)]
            cols_ = [1,2,3]*10

            for idx, i, j in zip(range(1, len(self.data)), rows_[1:-1], cols_[1:-1]):
                fig.add_trace(go.Scatter(x=tspan_full, y=self.data[idx], mode='markers', name='Observed Infections',line = dict(dash='dot', color='black'), showlegend=False),
                        row=i, col=j)

                fig.add_trace(go.Scatter(x=tspan_train, y=self.train_fitted[idx], mode='lines+markers', name='Fitted Infections',line = dict(dash='dot', color='red'), showlegend=False),
                          row=i, col=j)

                fig.add_trace(go.Scatter(x=tspan_test, y=self.test_predicted[idx], mode='lines+markers', name='Predicted Infections',line = dict(dash='dot', color='orange'), showlegend=False),
                                              row=i, col=j)

                fig.update_xaxes(title_text="Days since first infected", row=i, col=j, showgrid=True)
                fig.update_yaxes(title_text="Count", row=i, col=1)


            fig.update_layout(title='{} per country: Observed vs Fitted'.format(self.model),
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Counts',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,
                                  width=950, height=2600, legend=dict(font_size=14))
                                  #legend=dict(font_size=14, yanchor="top",y=0.99, xanchor="left", x=0.01))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

        elif derivative:
            #Plot the 1st derivative of cases
            fig = make_subplots(rows=10, cols=3, subplot_titles=(self.countries))
            fig.add_trace(go.Scatter(x=tspan_full[0][1:], y=self.derivative[0], mode='markers', name='Observed Daily Infections',line = dict(dash='dot', color='red')),
                    row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_full[0][1:], y=self.fitted_deriv[0][:, 0], mode='lines+markers', name='Fitted Daily Infections',line = dict(dash='dot', color='orange')),
                      row=1, col=1)

            rows_ = [val for val in range(1,11) for _ in (0, 1, 2)]
            cols_ = [1,2,3]*10

            for idx, i, j in zip(range(1, len(self.data_s)), rows_[1:-1], cols_[1:-1]):
                fig.add_trace(go.Scatter(x=tspan_full[idx][1:], y=self.derivative[idx], mode='markers', name='Observed Daily Infections',line = dict(dash='dot', color='red'), showlegend=False),
                        row=i, col=j)

                fig.add_trace(go.Scatter(x=tspan_full[idx][1:], y=self.fitted_deriv[idx][:, 0], mode='lines+markers', name='Fitted Daily Infections',line = dict(dash='dot', color='orange'), showlegend=False),
                          row=i, col=j)
                fig.update_xaxes(title_text="Days since first infected", row=i, col=j, showgrid=True)
                fig.update_yaxes(title_text=" Daily count", row=i, col=1)


            fig.update_layout(title='{} per country: Observed vs Fitted'.format(self.model),
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Daily count',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,
                                  width=950, height=2600, legend=dict(font_size=14))
                                  #legend=dict(font_size=14, yanchor="top",y=0.99, xanchor="left", x=0.01))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

        elif beta_param:
            pnames = list((self.train_params[0].valuesdict()))

            b_params = [self.train_params[0][p].value for p in pnames[:4]]
            beta_curve = self.logistic_step(tspan_full, b_params)
            beta_curve_train = self.logistic_step(tspan_train, b_params)
            beta_curve_test = self.logistic_step(tspan_test, b_params)

            fig = make_subplots(rows=10, cols=3, subplot_titles=(self.countries), horizontal_spacing=0.1,
                    specs=[x*3 for x in [[{"secondary_y": True}]]*10])

            #First subplot
            fig.add_trace(go.Scatter(x=tspan_train, y=beta_curve_train, mode='markers', name='Transmission rate train',line = dict(dash='dot', color='red')),
                    row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_test, y=beta_curve_test, mode='markers', name='Transmission rate test',line = dict(dash='dot', color='orange')),
                    row=1, col=1)
            #Add inflexion point
            fig.add_annotation(x=round(b_params[-1]), y=(np.max(beta_curve)-np.min(beta_curve))/2+np.min(beta_curve),
                text="{}".format(round(b_params[-1])), showarrow=True, arrowhead=2, arrowcolor= "black", arrowwidth=2,
                xref="x1", yref="y1", ax=20, ay=-25, font=dict(family="Courier New, monospace",size=10, color='black'),
                bordercolor="#c7c7c7",borderwidth=1,borderpad=1, bgcolor="orange", opacity=0.75)

            fig.add_trace(go.Scatter(x=tspan_full, y=self.data_s[0], mode='lines', name='Stringency',line = dict(dash='dot', color='purple')),
                    row=1, col=1, secondary_y=True)
            #Add stringency inflexion point
            #fig.add_annotation(x=round(s_params[-1]), y=(np.max(self.data_s[0])-np.min(self.data_s[0]))/2+np.min(self.data_s[0]),
                #text="{}".format(round(s_params[-1])), showarrow=True, arrowhead=2, arrowcolor= "black", arrowwidth=2,
                #xref="x1", yref="y2", ax=20, ay=-25, font=dict(family="Courier New, monospace",size=10, color='black'),
                #bordercolor="#c7c7c7",borderwidth=1,borderpad=1, bgcolor="green", opacity=0.85)
            fig.update_yaxes(row=1, col=1, secondary_y=True, tickfont=dict(color="blue"))


            rows_ = [val for val in range(1,11) for _ in (0, 1, 2)]
            cols_ = [1,2,3]*10

            for idx, idx_y, i, j in zip(range(2, len(self.data_s)+2), range(3,58,2), rows_[1:-2], cols_[1:-2]):
                b_params = [self.train_params[idx-1][p].value for p in pnames[:4]]
                beta_curve = self.logistic_step(tspan_full, b_params)
                beta_curve_train = self.logistic_step(tspan_train, b_params)
                beta_curve_test = self.logistic_step(tspan_test, b_params)


                #Subplots
                fig.add_trace(go.Scatter(x=tspan_train, y=beta_curve_train, mode='markers', name='Transmission rate train',line = dict(dash='dot', color='red'), showlegend=False),
                        row=i, col=j)
                fig.add_trace(go.Scatter(x=tspan_test, y=beta_curve_test, mode='markers', name='Transmission rate test',line = dict(dash='dot', color='orange'), showlegend=False),
                        row=i, col=j)
                #Annotate inflexion points
                fig.add_annotation(x=round(b_params[-1]), y=(np.max(beta_curve)-np.min(beta_curve))/2+np.min(beta_curve),
                text="{}".format(round(b_params[-1])), showarrow=True, arrowhead=2, arrowcolor= "black", arrowwidth=2,
                xref="x{}".format(idx), yref="y{}".format(idx_y), ax=20, ay=-25, font=dict(family="Courier New, monospace",size=10, color='black'),
                bordercolor="#c7c7c7",borderwidth=1,borderpad=1, bgcolor="orange", opacity=0.75)


                fig.add_trace(go.Scatter(x=tspan_full, y=self.data_s[idx-1], mode='lines', name='Stringency',
                                         line = dict(dash='dot', color='purple'), showlegend=False),row=i, col=j, secondary_y=True)

                #fig.add_annotation(x=round(s_params[-1]), y=(np.max(self.data_s[idx-1])-np.min(self.data_s[idx-1]))/2+np.min(self.data_s[idx-1]),
                #text="{}".format(round(s_params[-1])), showarrow=True, arrowhead=2, arrowcolor= "black", arrowwidth=2,
                #xref="x{}".format(idx), yref="y{}".format(idx_y+1), ax=20, ay=-25, font=dict(family="Courier New, monospace",size=10, color='black'),
                #bordercolor="#c7c7c7",borderwidth=1,borderpad=1, bgcolor="green", opacity=0.85)

                #Update axis colors and texts
                fig.update_xaxes(title_text="Days since first infected", row=i, col=j, showgrid=True)
                fig.update_yaxes(title_text="Level", row=i, col=1, secondary_y=False, titlefont=dict(color="black"),
                                tickfont=dict(color="black"))
                fig.update_yaxes(title_text="OxCGRT index", row=i, col=3, secondary_y=True, titlefont=dict(color="blue"),
                                tickfont=dict(color="blue"))
                fig.update_yaxes(row=i, col=j, secondary_y=False, tickfont=dict(color="black"))
                fig.update_yaxes(row=i, col=j, secondary_y=True, tickfont=dict(color="blue"))


            fig.update_layout(title='Observed vs Fitted stringency for each country',
                                   xaxis_title='Days since first infected',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,
                                  width=950, height=2600, legend=dict(font_size=14))
                              #legend=dict(font_size=14, yanchor="top",y=0.99, xanchor="left", x=0.01))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            fig.show()


        elif beta_param_linear:
            pnames = list((self.train_params[0].valuesdict()))

            b_params = [self.train_params[0][p].value for p in pnames[:3]]
            beta_curve = self.linear_step(self.data_s[0], self.data_mob[0], b_params)
            beta_curve_train = self.linear_step(self.data_s[0][self.split_day: train_window], self.data_mob[0][self.split_day: train_window], b_params)
            beta_curve_test = self.linear_step(self.data_s[0][train_window: full_window], self.data_mob[0][train_window:full_window], b_params)

            fig = make_subplots(rows=10, cols=3, subplot_titles=(self.countries), horizontal_spacing=0.1,
                    specs=[x*3 for x in [[{"secondary_y": True}]]*10])

            #First subplot
            fig.add_trace(go.Scatter(x=tspan_train, y=beta_curve_train, mode='markers', name='Transmission rate train',line = dict(dash='dot', color='red')),
                    row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_test, y=beta_curve_test, mode='markers', name='Transmission rate test',line = dict(dash='dot', color='orange')),
                    row=1, col=1)
            fig.add_trace(go.Scatter(x=tspan_full, y=self.data_s[0], mode='lines', name='Stringency',line = dict(dash='dot', color='purple')),
                    row=1, col=1, secondary_y=True)

            fig.update_yaxes(row=1, col=1, secondary_y=True, tickfont=dict(color="blue"))


            rows_ = [val for val in range(1,11) for _ in (0, 1, 2)]
            cols_ = [1,2,3]*10

            for idx, idx_y, i, j in zip(range(2, len(self.data_s)+2), range(3,58,2), rows_[1:-2], cols_[1:-2]):
                b_params = [self.train_params[idx-1][p].value for p in pnames[:3]]
                beta_curve = self.linear_step(self.data_s[idx-1], self.data_mob[idx-1], b_params)
                beta_curve_train = self.linear_step(self.data_s[idx-1][self.split_day: train_window], self.data_mob[idx-1][self.split_day: train_window], b_params)
                beta_curve_test = self.linear_step(self.data_s[idx-1][train_window: full_window], self.data_mob[idx-1][train_window:full_window], b_params)


                #Subplots
                fig.add_trace(go.Scatter(x=tspan_train, y=beta_curve_train, mode='markers', name='Transmission rate train',line = dict(dash='dot', color='red'), showlegend=False),
                        row=i, col=j)
                fig.add_trace(go.Scatter(x=tspan_test, y=beta_curve_test, mode='markers', name='Transmission rate test',line = dict(dash='dot', color='orange'), showlegend=False),
                        row=i, col=j)

                fig.add_trace(go.Scatter(x=tspan_full, y=self.data_s[idx-1], mode='lines', name='Stringency',
                                         line = dict(dash='dot', color='purple'), showlegend=False),row=i, col=j, secondary_y=True)

                #Update axis colors and texts
                fig.update_xaxes(title_text="Days since first infected", row=i, col=j, showgrid=True)
                fig.update_yaxes(title_text="Level", row=i, col=1, secondary_y=False, titlefont=dict(color="black"),
                                tickfont=dict(color="black"))
                fig.update_yaxes(title_text="OxCGRT index", row=i, col=3, secondary_y=True, titlefont=dict(color="blue"),
                                tickfont=dict(color="blue"))
                fig.update_yaxes(row=i, col=j, secondary_y=False, tickfont=dict(color="black"))
                fig.update_yaxes(row=i, col=j, secondary_y=True, tickfont=dict(color="blue"))


            fig.update_layout(title='Beta linear curve vs observed stringency',
                                   xaxis_title='Days since first infected',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,
                                  width=950, height=2600, legend=dict(font_size=14))
                              #legend=dict(font_size=14, yanchor="top",y=0.99, xanchor="left", x=0.01))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            fig.show()


        elif beta_linear_surface:
            pnames = list((self.train_params[0].valuesdict()))
            b_params = [self.train_params[0][p].value for p in pnames[:3]]
            beta_curve = self.linear_step(self.data_s[0], self.data_mob[0], b_params)
            b_surface = self.surface_beta(self.data_s[0],self.data_mob[0], b_params)
            x, y = np.linspace(0, 1, len(beta_curve)), np.linspace(-1, 1, len(beta_curve))

            fig = make_subplots(rows=10, cols=3, subplot_titles=(self.countries), specs = [x*3 for x in [[{"type": 'surface'}]]*10])
            fig.add_trace(go.Surface(x=x, y=y, z=b_surface, colorscale='RdBu', showscale=True), row=1, col=1)
            fig.add_scatter3d(x=self.data_s[0], y=self.data_mob[0], z = beta_curve, mode='markers',marker=dict(size=2, colorscale='Greys'))

            rows_ = [val for val in range(1,11) for _ in (0, 1, 2)]
            cols_ = [1,2,3]*10

            for idx, i, j in zip(range(1, len(self.data)), rows_[1:-1], cols_[1:-1]):

                b_params = [self.train_params[idx][p].value for p in pnames[:3]]
                beta_curve = self.linear_step(self.data_s[idx], self.data_mob[idx], b_params)
                b_surface = self.surface_beta(self.data_s[idx],self.data_mob[idx], b_params)
                x, y = np.linspace(0, 1, len(beta_curve)), np.linspace(-1, 1, len(beta_curve))

                fig.add_trace(go.Surface(x=x, y=y, z=b_surface, colorscale='RdBu', showscale=True),row=i, col=j)
                fig.add_scatter3d(x=self.data_s[idx], y=self.data_mob[idx], z = beta_curve, mode='markers',marker=dict(size=2, colorscale='Greys'),row=i, col=j)

                #fig.update_xaxes(title_text="Days since first infected", row=i, col=j, showgrid=True)
                fig.update_yaxes(title_text="Beta value", row=i, col=1)


            fig.update_layout(title='Beta fitted (surface plot)'.format(self.model),
                                   xaxis_title='Stringency',
                                   yaxis_title='Mobility',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,
                                  width=950, height=2600, legend=dict(font_size=14))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()


        elif MSE:
            #Bar plot of each country mse sorted
            list_mse = self.mse
            list_mse.sort(key=lambda x: x[1])
            y_mse = np.asarray(np.asarray(list_mse)[:,1], dtype=float)

            fig = go.Figure(data=[go.Bar(x=np.asarray(list_mse)[:,0], y=y_mse)])
            # Customize aspect
            fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                              marker_line_width=1.5, opacity=0.6)
            fig.update_layout(title_text='MSE per country for {} model'.format(self.model),
            xaxis_title='Country', yaxis_title='MSE', title_font_size=23, font_size=14,)

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

        elif MSE_norm:
            #Bar plot of each country mse sorted
            list_mse = self.mse

            #Get standardizing denominator
            range_ = np.asarray([np.max(i)- np.min(i)+0.0001 for i in self.data])
            #Standardize
            numerator = np.asarray(np.asarray(list_mse)[:,1], dtype=float)
            mse_n = np.divide(numerator, range_)

            list_mse_norm = [(i, j) for i,j in zip(np.asarray(list_mse)[:,0], mse_n)]
            list_mse_norm.sort(key=lambda x: x[1])

            norm_mse = np.asarray(np.asarray(list_mse_norm)[:,1], dtype=float)

            fig = go.Figure(data=[go.Bar(x=np.asarray(list_mse)[:,0], y=norm_mse)])
            # Customize aspect
            fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                              marker_line_width=1.5, opacity=0.6)
            fig.update_layout(title_text='MSE standardized per country for {} model'.format(self.model),
            xaxis_title='Country', yaxis_title='MSE', title_font_size=23, font_size=14,)

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

        elif MAE_norm:
            #Bar plot of each country mse sorted
            list_mae = self.mae
            #Get standardizing denominator
            range_ = np.asarray([np.max(i)- np.min(i)+0.0001 for i in self.data])
            #Standardize
            numerator = np.asarray(np.asarray(list_mae)[:,1], dtype=float)
            mae_n = np.divide(numerator, range_)

            list_mae_norm = [(i, j) for i,j in zip(np.asarray(list_mae)[:,0], mae_n)]
            list_mae_norm.sort(key=lambda x: x[1])
            norm_mae = np.asarray(np.asarray(list_mae_norm)[:,1], dtype=float)

            fig = go.Figure(data=[go.Bar(x=np.asarray(list_mae_norm)[:,0], y=norm_mae)])
            # Customize aspect
            fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                              marker_line_width=1.5, opacity=0.6)
            fig.update_layout(title_text='MAE standardized per country for {} model'.format(self.model),
            xaxis_title='Country', yaxis_title='MAE', title_font_size=23, font_size=14,)

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()


        elif rate_params:
            pnames = list((self.final_result[0].params.valuesdict()))

            b_params = [self.final_result[0].params[p].value for p in pnames[:4]]
            s_params = [self.result_stringency[0].params[p].value for p in list((self.result_stringency[0].params.valuesdict()))[:4]]
            beta_curve = self.logistic_step(tspan_full[0], b_params)

            fig = make_subplots(rows=10, cols=3, subplot_titles=(self.countries), horizontal_spacing=0.1,
                    specs=[x*3 for x in [[{"secondary_y": True}]]*10])

            #First subplot
            fig.add_trace(go.Scatter(x=tspan_full[0], y=beta_curve, mode='markers', name='Transmission rate',line = dict(dash='dot', color='orange')),
                    row=1, col=1)
            #Add inflexion point
            fig.add_annotation(x=round(b_params[-1]), y=(np.max(beta_curve)-np.min(beta_curve))/2+np.min(beta_curve),
                text="{}".format(round(b_params[-1])), showarrow=True, arrowhead=2, arrowcolor= "black", arrowwidth=2,
                xref="x1", yref="y1", ax=20, ay=-25, font=dict(family="Courier New, monospace",size=10, color='black'),
                bordercolor="#c7c7c7",borderwidth=1,borderpad=1, bgcolor="orange", opacity=0.75)


            fig.add_trace(go.Scatter(x=tspan_full[0], y=self.data_s[0], mode='lines+markers', name='Stringency',line = dict(dash='dot', color='green')),
                    row=1, col=1, secondary_y=True)
            #Add stringency inflexion point
            fig.add_annotation(x=round(s_params[-1]), y=(np.max(self.data_s[0])-np.min(self.data_s[0]))/2+np.min(self.data_s[0]),
                text="{}".format(round(s_params[-1])), showarrow=True, arrowhead=2, arrowcolor= "black", arrowwidth=2,
                xref="x1", yref="y2", ax=20, ay=-25, font=dict(family="Courier New, monospace",size=10, color='black'),
                bordercolor="#c7c7c7",borderwidth=1,borderpad=1, bgcolor="green", opacity=0.85)

            fig.update_yaxes(row=1, col=1, secondary_y=True, tickfont=dict(color="blue"))


            rows_ = [val for val in range(1,11) for _ in (0, 1, 2)]
            cols_ = [1,2,3]*10

            for idx, idx_y, i, j in zip(range(2, len(self.data_s)+2), range(3,58,2), rows_[1:-2], cols_[1:-2]):
                b_params = [self.final_result[idx-1].params[p].value for p in pnames[:4]]
                s_params = [self.result_stringency[idx-1].params[p].value for p in list((self.result_stringency[idx-1].params.valuesdict()))[:4]]
                beta_curve = self.logistic_step(tspan_full[idx-1], b_params)

                #Subplots
                fig.add_trace(go.Scatter(x=tspan_full[idx-1], y=beta_curve, mode='markers', name='Transmission rate',line = dict(dash='dot', color='orange'), showlegend=False),
                        row=i, col=j)
                #Annotate inflexion points
                fig.add_annotation(x=round(b_params[-1]), y=(np.max(beta_curve)-np.min(beta_curve))/2+np.min(beta_curve),
                text="{}".format(round(b_params[-1])), showarrow=True, arrowhead=2, arrowcolor= "black", arrowwidth=2,
                xref="x{}".format(idx), yref="y{}".format(idx_y), ax=20, ay=-25, font=dict(family="Courier New, monospace",size=10, color='black'),
                bordercolor="#c7c7c7",borderwidth=1,borderpad=1, bgcolor="orange", opacity=0.75)


                fig.add_trace(go.Scatter(x=tspan_full[idx-1], y=self.data_s[idx-1], mode='lines+markers', name='Stringency',
                                         line = dict(dash='dot', color='green'), showlegend=False),
                    row=i, col=j, secondary_y=True)

                fig.add_annotation(x=round(s_params[-1]), y=(np.max(self.data_s[idx-1])-np.min(self.data_s[idx-1]))/2+np.min(self.data_s[idx-1]),
                text="{}".format(round(s_params[-1])), showarrow=True, arrowhead=2, arrowcolor= "black", arrowwidth=2,
                xref="x{}".format(idx), yref="y{}".format(idx_y+1), ax=20, ay=-25, font=dict(family="Courier New, monospace",size=10, color='black'),
                bordercolor="#c7c7c7",borderwidth=1,borderpad=1, bgcolor="green", opacity=0.85)

                #Update axis colors and texts
                fig.update_xaxes(title_text="Days since first infected", row=i, col=j, showgrid=True)
                fig.update_yaxes(title_text="Level", row=i, col=1, secondary_y=False, titlefont=dict(color="black"),
                                tickfont=dict(color="black"))
                fig.update_yaxes(title_text="OxCGRT index", row=i, col=3, secondary_y=True, titlefont=dict(color="blue"),
                                tickfont=dict(color="blue"))

                fig.update_yaxes(row=i, col=j, secondary_y=False, tickfont=dict(color="black"))
                fig.update_yaxes(row=i, col=j, secondary_y=True, tickfont=dict(color="blue"))


            fig.update_layout(title='Observed vs Fitted stringency for each country',
                                   xaxis_title='Days since first infected',
                                   title_x=0.5,
                                   title_font_size=14,
                                   font_size=14,
                                  width=950, height=2600, legend=dict(font_size=14))
                              #legend=dict(font_size=14, yanchor="top",y=0.99, xanchor="left", x=0.01))

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            fig.show()

    def sum_errors(self, country, value_x, value_y, initial_conditions, ground_truth, tspan, sol):
        #parameters
        eps=1.0
        params = Parameters()
        #init_params = pd.read_csv(self.PARAMS_FILE, sep=";",index_col='name', header=0, skipinitialspace=True)
        #Init with best fitted params to fix (a, C, gamma)
        init_params = np.load('./{0}/{1}/temp_res/best_result_{2}_100days.npy'.format(self.file_method, self.file_model, country), allow_pickle=True).item()

        for i in range(1):

            if self.beta_sigmoid_step:
                params.add(f'C_{i}', value=init_params.params['C_0'].value, vary=False)
                params.add(f'L_{i}', value=value_x, min=value_x-0.0001, max=value_x+0.001, vary=True)
                params.add(f'a_{i}', value=init_params.params['a_0'].value, vary=False)
                params.add(f'b_{i}', value= value_y, min = value_y-0.001, max = value_y+0.001, vary=True)

            elif self.beta_linear_step:
                params.add(f'w1_{i}', value=value_x, min=value_x-0.0001, max=value_x+0.001, vary=True)
                params.add(f'w2_{i}', value= value_y, min = value_y-0.001, max = value_y+0.001, vary=True)
                params.add(f'b_{i}', value=init_params.params['b_0'].value, vary=False)

        #Add gamma
        params.add('gamma', value=init_params.params['gamma'].value, vary=False)

        predicted = self.ode_solver(tspan, initial_conditions, params, 0)

        #Compute cumulated I cases
        sol = self.get_cumulated_cases(predicted, 1.0, sol, test_phase=False)
        test_error = ((sol - ground_truth)/eps).ravel()
        z = np.sum(test_error**2)/len(test_error)

        return np.sqrt(z)

    def compute_obj_function(self, country, train_window):
        try:
            EPI_data, country_attrs = country_covid(country, self.owid_file, self.hosp_file, model=self.model)
        except ValueError:
            print(f'incomplete data on {country}')
            return [], []

        df = EPI_data.drop(['N_effective'], axis=1).reindex(columns=['date', 'S', 'I', 'R', 'D', 's'])
        df.R.fillna(0, inplace=True)
        df.ffill(axis=0, inplace=True)
        df.bfill(axis=0, inplace=True)

        if self.beta_linear_step:
            #Stringency
            data_s = df.loc[0:, ['s']].values
            data_s = data_s.reshape(len(data_s))
            data_s_norm = (data_s - data_s.min())/(data_s.max() - data_s.min())
            #Smooth stringency
            df['s_norm'] = data_s_norm
            df['stringency_smooth']= df['s_norm'].rolling(20, center=False).mean().reset_index(0, drop=True).fillna(0.001)
            s_smooth = df.loc[0:, ['stringency_smooth']].values
            self.data_ = s_smooth.ravel()
            self.stringency_country = s_smooth

            #Add mobility data
            k = np.where(np.asarray(self.countries) == country)[0].item()
            mobility_df = get_data_w_mobility(self.iso_code2[k])
            #Merge datasets
            df = df.merge(mobility_df[['mobility_index','date']], left_on=['date'], right_on=['date'], how='left')
            df['mobility_index'].bfill(inplace=True) #fill nan
            df['mobility_index_smooth']= df['mobility_index'].rolling(15, center=False).mean().reset_index(0, drop=True).fillna(0.5) # 0.5 means no difference in mobility from previous year
            self.mob_country = df.loc[0:, ['mobility_index_smooth']].values.ravel()

        #Crop df
        df = df[0:train_window+1]
        #Remove artefacts
        df = self.remove_artefacts(df)
        data = df.loc[0:train_window-1, ['I']].values

        #Append country full data and derivative
        true_data = data.ravel()

        #Compute
        sol = np.zeros_like(true_data)
        tspan = np.arange(1, train_window+1, 1)
        initial_conditions = [1, 0, country_attrs['population']]

        #Compute obj function values for couple of parameters

        if self.beta_sigmoid_step:
            Lmin, Lmax, Lstep = -0.95, 0.0, 0.001
            bmin, bmax, bstep = 1.0, 100.0, 0.5

            #Get initial value
            #init_L, init_b = -0.3, 60.0
            #init_z = self.sum_errors(country, init_L, init_b, initial_conditions, true_data, tspan, sol)

        elif self.beta_linear_step:
            xmin, xmax, xstep = -1.0, -0.00001, 0.001
            ymin, ymax, ystep = 0.00001, 1.0, 0.001

        X, Y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
        z_final = np.zeros((X.shape))

        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                X_temp, Y_temp = X[i][j], Y[i][j]
                z = self.sum_errors(country, X_temp, Y_temp, initial_conditions, true_data, tspan, sol)
                z_final[i][j] = z

        #Get fitted values
        #Import best fitted params for given train window and country

        #save obj function as a tuple
        np.save('./{0}/{1}/obj_functions/{2}_{3}days.npy'.format(self.file_method, self.file_model, country, train_window), {'obj_func': (X, Y, z_final)})

    def plot_obj_function(self, country, train_window):
        #Load computed objective function if pre computed
        try:
            print(f'Importing file with precomputed objective function for {country} ...')
            obj_dict = np.load('./res_sigmoid/sir/obj_functions/{0}_{1}days.npy'.format(country, train_window), allow_pickle=True).item()
        except FileNotFoundError:
            print(f'File not found. Objective function for {country} is being computed...')
            self.compute_obj_function(country, train_window)
            obj_dict = np.load('./res_sigmoid/sir/obj_functions/{0}_{1}days.npy'.format(country, train_window), allow_pickle=True).item()

        L, b, z_final = obj_dict['obj_func'] #obj function values
        init_L, init_b, init_z = obj_dict['init'] #init values
        L_fit, b_fit, z_fit = obj_dict['fit'] #fitted values

        #Get minima
        minima = np.argwhere(z_final == np.amin(z_final))
        minima_value = [np.amin(z_final).item()]*len(minima)
        x, y = minima[:,0].item(), minima[:,1].item()
        L_min, b_min = L[x][y], b[x][y]

        fig = make_subplots(rows=1, cols=1, subplot_titles=([country]), specs = [[{"type": 'surface'}]])
        fig.add_trace(go.Surface(x=L, y=b, z=z_final, colorscale="bluered", showscale=True, name='Objective function', opacity=0.6), row=1, col=1)
        fig.add_scatter3d(x=[L_min], y=[b_min], z = minima_value, mode='markers+text', marker=dict(size=4, color='black'),
                            text='MINIMUM', textfont=dict(size=12, color='black'), showlegend=False)
        fig.add_scatter3d(x=[init_L], y=[init_b], z = [init_z], mode='markers+text', marker=dict(size=4, color='green'),
                            text = 'INIT',textfont=dict(size=12, color='black'), showlegend=False)
        fig.add_scatter3d(x=[L_fit], y=[b_fit], z = [z_fit], mode='markers+text', marker=dict(size=4, color='yellow'),
                            text = 'FIT',textfont=dict(size=12, color='black'), showlegend=False)

        fig.update_yaxes(title_text="Beta value", row=1, col=1)
        fig.update_layout(title='Objective function (surface plot) for {0} with {1} model for {2} days'.format(country, self.model, train_window),
                               xaxis_title='L',
                               yaxis_title='b',
                               title_x=0.5,
                               title_font_size=14,
                               font_size=14,
                              width=950, height=800, legend=dict(font_size=14))
        fig.show()


        # Zoom around minimum
        L_bound = L[max(0, x-100): min(x+100, L.shape[0]), max(0, y-100): min(y+100, L.shape[1])]
        b_bound = b[max(0, x-100): min(x+100, b.shape[0]), max(0, y-100): min(y+100, b.shape[1])]
        z_bound = z_final[max(0, x-100): min(x+100, z_final.shape[0]),max(0, y-100): min(y+100, z_final.shape[1])]

        fig = make_subplots(rows=1, cols=1, subplot_titles=([country]), specs = [[{"type": 'surface'}]])
        fig.add_trace(go.Surface(x=L_bound, y=b_bound, z=z_bound, colorscale="bluered", showscale=True, name='Objective function', opacity=0.6), row=1, col=1)
        fig.add_scatter3d(x=[L_min], y=[b_min], z = minima_value, mode='markers+text', marker=dict(size=4, color='black'),
                            text='MINIMUM', textfont=dict(size=12, color='black'), showlegend=False)

        fig.update_layout(title='[ZOOM] Objective function (surface plot) for {0} with {1} model for {2} days'.format(country, self.model, train_window),
                               xaxis_title='L',
                               yaxis_title='b',
                               title_x=0.5,
                               title_font_size=14,
                               font_size=14,
                              width=950, height=800, legend=dict(font_size=14))
        fig.show()

    """Fitting only 1 country either at random, with initialization at solution or near solution for further investigation of minimization algorithm."""
    def fit_one_country(self, country, train_window, mode='init_random'):
        try:
            results = np.load('./{0}/{1}/result_{2}_{3}days_{4}.npy'.format(self.file_method, self.file_model, country, train_window, mode), allow_pickle=True).item()
            print('{0} model already fitted for {1} country on {2} days with {3} mode.'.format(self.model, country, train_window, mode))

        except FileNotFoundError:
            print(f'Fitted parameters not found. Fitting {self.model} for {country} on {train_window} days with {mode}...')
            try:
                EPI_data, country_attrs = country_covid(country, self.owid_file, self.hosp_file, model=self.model)
            except ValueError:
                print(f'incomplete data on {country}')
                return [], []

            df = EPI_data.drop(['N_effective'], axis=1).reindex(columns=['date', 'S', 'I', 'R', 'D', 's'])
            df.R.fillna(0, inplace=True)
            df.ffill(axis=0, inplace=True)
            df.bfill(axis=0, inplace=True)

            if self.beta_linear_step:
                #Stringency
                data_s = df.loc[0:, ['s']].values
                data_s = data_s.reshape(len(data_s))
                data_s_norm = (data_s - data_s.min())/(data_s.max() - data_s.min())
                #Smooth stringency
                df['s_norm'] = data_s_norm
                df['stringency_smooth']= df['s_norm'].rolling(20, center=False).mean().reset_index(0, drop=True).fillna(0.001)
                s_smooth = df.loc[0:, ['stringency_smooth']].values
                self.data_ = s_smooth.ravel()
                self.stringency_country = s_smooth

                #Add mobility data
                k = np.where(np.asarray(self.countries) == country)[0].item()
                mobility_df = get_data_w_mobility(self.iso_code2[k])
                #Merge datasets
                df = df.merge(mobility_df[['mobility_index','date']], left_on=['date'], right_on=['date'], how='left')
                df['mobility_index'].bfill(inplace=True) #fill nan
                df['mobility_index_smooth']= df['mobility_index'].rolling(15, center=False).mean().reset_index(0, drop=True).fillna(0.5) # 0.5 means no difference in mobility from previous year
                self.mob_country = df.loc[0:, ['mobility_index_smooth']].values.ravel()

            #Crop df
            self.train_window = train_window
            full_window = train_window + 7
            df = df[0:full_window+1]
            #Remove artefacts
            df = self.remove_artefacts(df)
            split_day = 0
            self.split_day = split_day
            #SPLIT
            train_df, test_df = self.train_test_split(df, self.split_day)

            #Keep data
            train_data = train_df['I'].values
            test_data = test_df['I'].values
            self.data = df.loc[0:full_window,'I'].values

            eps=1.0
            tspan_train = np.arange(1, self.train_window+1+self.split_day, 1)

            sol = np.zeros_like(train_data)
            tspan_train = np.arange(1, train_window+1, 1)
            initI, initR, initN = 1.0, 0.0, country_attrs['population']
            initial_conditions = [initI, initR, initN]

            if mode == 'init_random':
                print('Fitting with initialization at random...')
                #get best fitted params on 100 days
                best_params = np.load('./{0}/{1}/temp_res/best_result_{2}_100days.npy'.format(self.file_method, self.file_model, country), allow_pickle=True).item()
                path_params_file = self.PARAMS_FILE
                params = pd.read_csv(path_params_file, sep=";",index_col='name', header=0, skipinitialspace=True)
                if self.beta_sigmoid_step:
                    init_X, init_Y = params.at['L', 'init_value'], params.at['b', 'init_value']
                    init_z = self.sum_errors(country, init_X, init_Y, initial_conditions, train_data, tspan_train, sol)
                    #Fix best params
                    params.at['C', 'init_value'] = best_params.params['C_0'].value
                    params.at['a', 'init_value'] = best_params.params['a_0'].value
                elif self.beta_linear_step:
                    init_X, init_Y = params.at['w1', 'init_value'], params.at['w2', 'init_value']
                    init_z = self.sum_errors(country, init_X, init_Y, initial_conditions, train_data, tspan_train, sol)
                    #Fix best params
                    params.at['b', 'init_value'] = best_params.params['b_0'].value
                #fix gamma
                params.at['gamma', 'init_value'] = best_params.params['gamma'].value
                params.to_csv('./params_new/params_countries/params_{0}_{1}_{2}_best.csv'.format(self.file_model, country, self.file_method), sep=";", header=1)
                self.PARAMS_FILE = './params_new/params_countries/params_{0}_{1}_{2}_best.csv'.format(self.file_model, country, self.file_method)

            else:
                #Get minimum from obj function
                #Load computed objective function if pre computed
                try:
                    print(f'Importing file with precomputed objective function for {country} ...')
                    obj_dict = np.load('./{0}/{1}/obj_functions/{2}_{3}days.npy'.format(self.file_method, self.file_model, country, train_window), allow_pickle=True).item()
                except FileNotFoundError:
                    print(f'File not found. Objective function for {country} is being computed...')
                    self.compute_obj_function(country, train_window)
                    obj_dict = np.load('./{0}/{1}/obj_functions/{2}_{3}days.npy'.format(self.file_method, self.file_model, country, train_window), allow_pickle=True).item()

                X, Y, z_final = obj_dict['obj_func'] #obj function values
                #Get minima
                minima = np.argwhere(z_final == np.amin(z_final))
                minima_value = [np.amin(z_final).item()]*len(minima)
                x, y = minima[:,0].item(), minima[:,1].item()
                X_min, Y_min = X[x][y], Y[x][y]

                if mode == 'init_at_sol':
                    print('Fitting with initialization at solution...')
                    #Create new csv file with min params for b and L
                    init_X, init_Y = X_min, Y_min
                    init_z = minima_value
                    #Get best params
                    init_params = pd.read_csv('./params_new/params_countries/params_{0}_{1}_{2}_best.csv'.format(self.file_model, country, self.file_method), sep=";",index_col='name', header=0, skipinitialspace=True)
                    if self.beta_sigmoid_step:
                        init_params.at['L', 'init_value'] = X_min
                        init_params.at['b', 'init_value'] = Y_min
                    elif self.beta_linear_step:
                        init_params.at['w1', 'init_value'] = X_min
                        init_params.at['w2', 'init_value'] = Y_min

                    init_params.to_csv('./params_new/params_countries/params_{0}_{1}_sol_{2}.csv'.format(self.file_model, country, self.file_method), sep=";", header=1)
                    self.PARAMS_FILE = './params_new/params_countries/params_{0}_{1}_sol_{2}.csv'.format(self.file_model, country, self.file_method)

                elif mode == 'init_near_sol':
                    print('Fitting with initialization near solution...')
                    #Near solution
                    init_params = pd.read_csv('./params_new/params_countries/params_{0}_{1}_{2}_best.csv'.format(self.file_model, country, self.file_method), sep=";",index_col='name', header=0, skipinitialspace=True)
                    if self.beta_sigmoid_step:
                        init_X, init_Y = max(X_min+0.1, init_params.at['L', 'max']), min(Y_min+10, init_params.at['b', 'min'])

                    elif self.beta_linear_step:
                        init_X, init_Y = min(X_min+0.1, init_params.at['w1', 'max']), min(Y_min+0.1, init_params.at['w2', 'min'])

                    init_z =  self.sum_errors(country, init_X, init_Y, initial_conditions, train_data, tspan_train, sol)
                    #Get best params
                    init_params = pd.read_csv('./params_new/params_countries/params_{0}_{1}_{2}_best.csv'.format(self.file_model, country, self.file_method), sep=";",index_col='name', header=0, skipinitialspace=True)
                    if self.beta_sigmoid_step:
                        init_params.at['L', 'init_value'] = init_X
                        init_params.at['b', 'init_value'] = init_Y
                    elif self.beta_linear_step:
                        init_params.at['w1', 'init_value'] = init_X
                        init_params.at['w2', 'init_value'] = init_Y

                    init_params.to_csv('./params_new/params_countries/params_{0}_{1}_nearsol_{2}.csv'.format(self.file_model, country, self.file_method), sep=";", header=1)
                    self.PARAMS_FILE = './params_new/params_countries/params_{0}_{1}_nearsol_{2}.csv'.format(self.file_model, country, self.file_method)

            #Next
            self.last_I_cumul = initI
            self.test_on_optim = True
            params, initial_conditions = self.init_sectors([100.0], disease_vary=True, initN=initN, initI=initI, initR=initR, stringency_fit=False)
            #Minimization
            if self.beta_sigmoid_step:
                self.dict_training_params = {i : [] for i in ['L_0', 'b_0']}
            elif self.beta_linear_step:
                self.dict_training_params = {i : [] for i in ['w1_0', 'w2_0']}

            result = minimize(self.error_sectors, params, args=(initial_conditions, tspan_train, train_data, eps, False), iter_cb = self.callback_func, method='leastsq', full_output = 1)
            train_final = train_data + result.residual.reshape(train_data.shape)
            #print(report_fit(result))
            if self.beta_sigmoid_step:
                X_fit = result.params['L_0'].value
                Y_fit = result.params['b_0'].value
            elif self.beta_linear_step:
                X_fit = result.params['w1_0'].value
                Y_fit = result.params['w2_0'].value
            fit_z = self.sum_errors(country, X_fit, Y_fit, initial_conditions[0], train_data, tspan_train, sol)

            #Compute obj function for every parameter pairs visited in the gradient descent
            if self.beta_sigmoid_step:
                training_z = [self.sum_errors(country, train_X, train_Y, initial_conditions[0], train_data, tspan_train, sol) for train_X, train_Y in zip(self.dict_training_params['L_0'], self.dict_training_params['b_0'])]
            elif self.beta_linear_step:
                training_z = [self.sum_errors(country, train_X, train_Y, initial_conditions[0], train_data, tspan_train, sol) for train_X, train_Y in zip(self.dict_training_params['w1_0'], self.dict_training_params['w2_0'])]

            ####TESTING
            self.last_I, self.last_R, self.last_N =initial_conditions[0]
            self.last_I_cumul = self.data[self.train_window-1].item()
            sol = np.zeros_like(test_data)
            tspan_test = np.arange(1, self.split_day+full_window+1, 1)
            params_fitted = result.params
            if self.model=='SIR':
                initial_conditions_test = [self.last_I, self.last_R, self.last_N]
            elif self.model=='SEIR':
                initial_conditions_test = [self.last_E, self.last_I, self.last_R, self.last_N]

            predicted = self.ode_solver(tspan_test, initial_conditions_test, params_fitted, 0)
            #Compute cumulated I cases
            test_final = self.get_cumulated_cases(predicted, self.last_I_cumul, sol, test_phase=True)
            #test_error = ((sol_test - test_data)/eps).ravel()
            save_to_path = './{0}/{1}/result_{2}_{3}days_{4}.npy'.format(self.file_method, self.file_model, country, train_window, mode)
            np.save(save_to_path,  {'init': (init_X, init_Y, init_z), 'fit': (X_fit, Y_fit, fit_z), 'result':result, 'train_fit': train_final.ravel(), 'test': test_final.ravel(), 'true_data':self.data.ravel(), 'training_obj_func': (training_z, self.dict_training_params)})

    """Function to plot 4 subplots to inverstigate the minimization process. 2 contour plots, 1 convergence plot and 1 plot of the final fit."""
    def subplots_obj_func(self, country, train_window, mode='init_random'):
        #Import fit or compute it
        try:
            results = np.load('./{0}/{1}/result_{2}_{3}days_{4}.npy'.format(self.file_method, self.file_model, country, train_window, mode), allow_pickle=True).item()
        except FileNotFoundError:
            self.fit_one_country(country, train_window, mode)
            results = np.load('./{0}/{1}/result_{2}_{3}days_{4}.npy'.format(self.file_method, self.file_model, country, train_window, mode), allow_pickle=True).item()

        #Load computed objective function if pre computed
        try:
            print(f'Importing file with precomputed objective function for {country} ...')
            obj_dict = np.load('./{0}/{1}/obj_functions/{2}_{3}days.npy'.format(self.file_method, self.file_model, country, train_window), allow_pickle=True).item()
        except FileNotFoundError:
            print(f'File not found. Objective function for {country} is being computed...')
            self.compute_obj_function(country, train_window)
            obj_dict = np.load('./{0}/{1}/obj_functions/{2}_{3}days.npy'.format(self.file_method, self.file_model, country, train_window), allow_pickle=True).item()

        X, Y, z_final = obj_dict['obj_func'] #obj function values
        #Get minima
        minima = np.argwhere(z_final == np.amin(z_final))
        minima_value = [np.amin(z_final).item()]*len(minima)
        x, y = minima[:,0].item(), minima[:,1].item()
        X_min, Y_min = X[x][y], Y[x][y]

        #Get fitted and init values for each fit
        X_init, Y_init, z_init = results['init']
        X_fit, Y_fit, z_fit = results['fit']

        #Get fitted cases and predictions
        true_data = results['true_data']
        train_fit, test = results['train_fit'], results['test']
        tspan_train = np.arange(1, train_window+1, 1)
        tspan_test = np.arange(train_window+1, train_window+7+1, 1)
        tspan_full = np.arange(1, train_window+7+1, 1)

        #Get intermediary points
        training_z, training_params = results['training_obj_func']
        if self.beta_sigmoid_step:
            X0 = np.asarray([training_params['L_0']])
            Y0 = np.asarray([training_params['b_0']])
        elif self.beta_linear_step:
            X0 = np.asarray([training_params['w1_0']])
            Y0 = np.asarray([training_params['w2_0']])
        path = np.concatenate((X0, Y0), axis=0)

        # Zoom around minimum
        #L_bound = L[max(0, x-10): min(x+10, L.shape[0]), max(0, y-10): min(y+10, L.shape[1])]
        #b_bound = b[max(0, x-10): min(x+10, b.shape[0]), max(0, y-10): min(y+10, b.shape[1])]
        X_bound = X[0,:][int(max(0, y-80)) : int(min(y+80, len(X[0,:])-1))]
        Y_bound = Y[:,0][int(max(0, x-15)) : int(min(x+15, len(Y[:,0])-1))]
        z_bound = z_final[int(max(0, x-15)): int(min(x+15, z_final.shape[0])), int(max(0, y-80)): int(min(y+80, z_final.shape[1]))]

        #List of points inside the bounds
        path_bound = []
        for i in range(len(path[0,])):
            x_, y_ = path[:,i]
            if X_bound.min() <= x_ <= X_bound.max() and Y_bound.min() <= y_ <= Y_bound.max():
                path_bound.append([x_, y_])

        #cbarlocs = [.85, .25]
        #Subplots
        ####SURFACE PLOT
        fig = make_subplots(rows=2, cols=2, subplot_titles=(['Minimization [log scale]', '[ZOOM] Minimum', 'Convergence and minimum (red)', 'Fit result']))
        #fig.add_trace(go.Surface(x=L, y=b, z=z_final, colorscale="bluered", showscale=False, name='Objective function', opacity=0.6), row=1, col=1)
        #fig.add_scatter3d(x=[L_min], y=[b_min], z = minima_value, mode='markers+text', marker=dict(size=4, color='black'),
        #                    text='MINIMUM', textfont=dict(size=12, color='black'), showlegend=False, row=1, col=1)
        #fig.add_scatter3d(x=[L_init], y=[b_init], z = [z_init], mode='markers+text', marker=dict(size=4, color='green'),
        #                    text = 'INIT',textfont=dict(size=12, color='black'), showlegend=False, row=1, col=1)
        #fig.add_scatter3d(x=[L_fit], y=[b_fit], z = [z_fit], mode='markers+text', marker=dict(size=4, color='yellow'),
                #            text = 'FIT',textfont=dict(size=12, color='black'), showlegend=False, row=1, col=1)

        #add intermediary points
        #for train_z, train_L, train_b in zip(training_z, training_params['L_0'], training_params['b_0']):
            #fig.add_scatter3d(x=[train_L], y=[train_b], z = [train_z], mode='markers', marker=dict(size=2, color='blue'),
                                #textfont=dict(size=12, color='black'), showlegend=False, row=1, col=1)

        #2nd subplot:zoom
        #fig.add_trace(go.Surface(x=L_bound, y=b_bound, z=z_bound, colorscale="bluered", showscale=True, name='Objective function', opacity=0.6), row=1, col=2)
        #fig.add_scatter3d(x=[L_min], y=[b_min], z = minima_value, mode='markers+text', marker=dict(size=4, color='black'),
                            #text='MINIMUM', textfont=dict(size=12, color='black'), showlegend=False, row=1, col=2)

        #fig.update_yaxes(title_text="RMSE", row=1, col=1)
        #fig.update_yaxes(title_text="RMSE", row=1, col=2)

        contour_colorscale = [[0.0, "rgb(49,54,149)"],
[0.1111111111111111, "rgb(69,117,180)"],
[0.2222222222222222, "rgb(116,173,209)"],
[0.3333333333333333, "rgb(171,217,233)"],
[0.4444444444444444, "rgb(224,243,248)"],
[0.5555555555555556, "rgb(254,224,144)"],
[0.6666666666666666, "rgb(253,174,97)"],
[0.7777777777777778, "rgb(244,109,67)"],
[0.8888888888888888, "rgb(215,48,39)"],
[1.0, "rgb(165,0,38)"]]

        #1st contour subplot
        fig.add_trace(go.Contour(z=np.log(z_final), x=X[0,:], y=Y[0:,0], colorscale=contour_colorscale, showscale=True, showlegend=False, opacity=1,
        colorbar=dict(nticks=10, ticks='outside', ticklen=5, tickwidth=1,showticklabels=True, tickangle=0, tickfont_size=12, len=0.5, title = 'log-RMSE')), row=1, col=1)
        #Add gradient descent
        fig.add_trace(go.Scatter(x=path[0,:-1], y=path[1,:-1], name='Path', mode='lines+markers',marker_symbol =46,
        marker_size=10, marker_color='black', line = dict(dash='dot', color='black'),showlegend=True), row=1, col=1)
        #add minimum, init and fit
        fig.add_scatter(x=[X_min], y=[Y_min], mode='markers',name = 'Minimum', marker=dict(size=8, color='red'), marker_symbol='star',  marker_size=15,
                             showlegend=True, row=1, col=1)
        fig.add_scatter(x=[X_init], y=[Y_init], mode='markers', name = 'Init', marker=dict(size=8, color='green'), marker_symbol='star', marker_size=15,
                            showlegend=True, row=1, col=1)
        fig.add_scatter(x=[X_fit], y=[Y_fit], mode='markers', name = 'Fit', marker=dict(size=8, color='yellow'),  marker_symbol='star', marker_size=15,
                             showlegend=True, row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=1, showgrid=True)
        fig.update_yaxes(title_text="y", row=1, col=1)

        #2nd subplot : zoom on contour plot
        fig.add_trace(go.Contour(z=np.log(z_bound), x=X_bound, y=Y_bound, colorscale=contour_colorscale, showscale=False, showlegend=False, opacity=1,
        colorbar=dict(nticks=10, ticks='outside', ticklen=5, tickwidth=1,showticklabels=True, tickangle=0, tickfont_size=12, len=0.5)), row=1, col=2)
        #Add gradient descent

        if len(path_bound) >= 1:
            fig.add_trace(go.Scatter(x=np.asarray(path_bound).T[0,:-1], y=np.asarray(path_bound).T[1,:-1], name='Path', mode='lines+markers',marker_symbol =46, marker_size=15, marker_color='black',
                                    line = dict(dash='dot', color='black'), showlegend=False), row=1, col=2)
        #add minimum, init and fit only if inside bounds
        fig.add_scatter(x=[X_min], y=[Y_min], mode='markers', marker=dict(size=8, color='red'),marker_symbol='star', marker_size=15,
                            showlegend=False, row=1, col=2)
        if X_bound.min() <= X_init <= X_bound.max() and Y_bound.min() <= Y_init <= Y_bound.max():
            fig.add_scatter(x=[X_init], y=[Y_init], mode='markers', marker=dict(size=8, color='green'),marker_symbol='star',  marker_size=15,
                                 showlegend=False, row=1, col=2)
        if X_bound.min() <=X_fit <= X_bound.max() and Y_bound.min() <= Y_fit <= Y_bound.max():
            fig.add_scatter(x=[X_fit], y=[Y_fit], mode='markers', marker=dict(size=8, color='yellow'),marker_symbol='star', marker_size=15,
                                 showlegend=False, row=1, col=2)
        fig.update_xaxes(title_text="x", row=1, col=2, showgrid=True)

        #3rd Subplot: convergence over iterations
        iterations = np.arange(1, len(training_z), 1)
        fig.add_trace(go.Scatter(x=iterations, y=np.log(training_z), mode='lines', name='Training RMSE',line = dict(color='black'), showlegend=False),
                row=2, col=1)
        fig.add_trace(go.Scatter(x=iterations, y=[np.log(np.amin(z_final).item())]*len(iterations), mode='lines', name='Minimum',line = dict(dash='dot', color='red'), showlegend=False),
                row=2, col=1)
        fig.update_xaxes(title_text="Iterations", row=2, col=1, showgrid=True)
        fig.update_yaxes(title_text="Log-RMSE", row=2, col=1)

        #4th plot
        fig.add_trace(go.Scatter(x=tspan_full, y=true_data, mode='markers', name='Observed Infections',line = dict(dash='dot', color='black'), showlegend=False),
                row=2, col=2)
        fig.add_trace(go.Scatter(x=tspan_train, y=train_fit, mode='lines+markers', name='Fitted Infections',line = dict(dash='dot', color='red'), showlegend=False),
                  row=2, col=2)
        fig.add_trace(go.Scatter(x=tspan_test, y=test, mode='lines+markers', name='Predicted Infections',line = dict(dash='dot', color='orange'), showlegend=False),
                            row=2, col=2)
        fig.update_xaxes(title_text="Days since first infected", row=2, col=2, showgrid=True)
        fig.update_yaxes(title_text="Count", row=2, col=2)


        fig.update_layout(title='{0} model for {1} on {2} days'.format(self.model,country, train_window),
                               title_x=0.5,
                               title_font_size=14,
                               font_size=14,
                              width=950, height=700, legend=dict(font_size=14))
        fig.show()
