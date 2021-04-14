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
from utils import country_covid


class EPIfit():
    """
    Class for fitting compartmental epidemiologic model (either SIR, SEIR, SEIRD)
    """
    def __init__(self, model='SIR', beta_sigmoid_step=False, mu_sigmoid_step=False, stringency_fit=False,
    PARAMS_FILE=None, OWID_FILE=None, HOSP_FILE=None, cutoff_date='2020-07-01', ratio_susceptible=1/300):
        self.model=model
        self.beta_sigmoid_step=beta_sigmoid_step
        self.mu_sigmoid_step = mu_sigmoid_step
        self.stringency_fit= stringency_fit
        self.PARAMS_FILE = PARAMS_FILE
        self.owid_file = OWID_FILE
        self.hosp_file = HOSP_FILE
        self.cutoff_date = cutoff_date
        self.ratio_susceptible = ratio_susceptible


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




    """PLOTS"""
    def show_rates(self, result, df, tspan, STATIC_PLOTS=True):
        """
        Show fitted rates from 'result' and visualize stringency from 'df' over 'tspan'
        """
        pnames = list((result.params.valuesdict()))

        if self.beta_sigmoid_step and not self.mu_sigmoid_step:
            b_params = [result.params[p].value for p in pnames[:4]]
            s_params = [self.result_s.params[p].value for p in list((self.result_s.params.valuesdict()))[:4]]

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
        #    fig = make_subplots(specs=[[{"secondary_y": True}]])

        #    res = [result.params[p].value for p in pnames]

            #for p in pnames:
        #    fig.add_trace(go.Bar(x=pnames, y=res, name='parameter', width=0.5))

        #    fig.update_layout(title='Fitted compartment rates',
                            #       xaxis_title='Parameters',
                            #       yaxis_title='Rate value',
                            #       title_x=0.5,
                            #       title_font_size=18,
                            #       font_size=18,
                            #      width=1000, height=600
                            #     )


        if STATIC_PLOTS:
            img_bytes = fig.to_image(format="png")
            display(Image(img_bytes))
        else:
            fig.show()

    """SIR, SEIR or SEIRD model"""
    def SIRD_derivs(self, z, t, rate_params, evo_params=None):
        """
        Derivatives for ODE solver
        """
        # Separate compartments
        if self.model=='SIR':
            S, I, R = z
            N = S +I + R

            #Fit beta as sigmoid or not
            if not self.beta_sigmoid_step:
                beta, gamma = rate_params

            elif self.beta_sigmoid_step:
                gamma = rate_params
                # Define evolution rates
                beta_params = evo_params
                beta= self.logistic_step(t, beta_params)

            #Compartment derivatives
            dSdt = -beta*S*I/N
            dIdt = beta*S*I/N - gamma*I
            dRdt = gamma*I

            return [dSdt, dIdt, dRdt]

        elif self.model=='SEIR':
            S, E, I, R = z
            N = S + E + I + R
            #Fit beta as sigmoid or not
            if not self.beta_sigmoid_step:
                beta, sigma, gamma = rate_params
            elif self.beta_sigmoid_step:
                sigma, gamma = rate_params
                # Define evolution rates
                beta_params = evo_params
                beta= self.logistic_step(t, beta_params)

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

            elif not self.beta_sigmoid_step:
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

            elif not self.beta_sigmoid_step:
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
    def init_sectors(self, sector_props, disease_vary=True, initN=0, stringency_fit=False):
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

            print('Params for stringency fit:', params)
            return params


        elif not stringency_fit:
            #Init conditions
            initial_conditions = []

            if self.model=='SIR':
                for i in range(len(sector_props)):
                    initN_i = initN * sector_props[i]/100
                    initI_i = 1
                    initR_i = 0

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
                        params.add(f'a_{i}', value=self.a_beta_init, min=self.a_beta_init, max=self.a_beta_init+0.1, vary=False)
                        params.add(f'b_{i}', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                if not self.beta_sigmoid_step:
                    params.add('beta', value=init_params.at['beta', 'init_value'], min=init_params.at['beta', 'min'], max=init_params.at['beta', 'max'], vary=True)

                #Add gamma
                params.add('gamma', value=init_params.at['gamma', 'init_value'], min=init_params.at['gamma', 'min'], max=init_params.at['gamma', 'max'], vary=disease_vary)


            elif self.model=='SEIR':
                for i in range(len(sector_props)):
                    initN_i = initN * sector_props[i]/100
                    initE_i = 0
                    initI_i = 1
                    initR_i = 0

                    initial_conditions_i = [initE_i, initI_i, initR_i, initN_i]
                    initial_conditions.append(initial_conditions_i)

                    if self.beta_sigmoid_step:
                        params.add(f'C_{i}', value=init_params.at['C', 'init_value'], min=init_params.at['C', 'min'], max=init_params.at['C', 'max'], vary=True)
                        params.add(f'L_{i}', value=init_params.at['L', 'init_value'], min=init_params.at['L', 'min'], max=init_params.at['L', 'max'], vary=True)
                        if self.stringency_fit:
                            params.add(f'a_{i}', value=self.a_beta_init, min=self.a_beta_init, max=self.a_beta_init+0.1, vary=False)
                        elif not self.stringency_fit:
                            params.add(f'a_{i}', value=init_params.at['a', 'init_value'], min=init_params.at['a', 'min'], max=init_params.at['a', 'max'], vary=True)
                        params.add(f'b_{i}', value=init_params.at['b', 'init_value'], min=init_params.at['b', 'min'], max=init_params.at['b', 'max'], vary=True)

                if not self.beta_sigmoid_step:
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

                #Init I cases
                initI = initial_conditions[i][idx-1]

                #Compute cumulated I cases to minimize
                cumul_i = [initI]

                if self.model=='SIR': #Compute dSdt
                    diff_i = np.diff(sol_i[:,0], 1)
                if self.model=='SEIR':
                    diff_S = np.diff(sol_i[:,0], 1)
                    diff_E = np.diff(sol_i[:,idx-1], 1)

                    diff_i = diff_E - np.abs(diff_S)


                #Add absolute value of dSdt to previous cumulated values
                for i, diff_ in enumerate(diff_i):
                    cumul_i.append(np.abs(diff_)+cumul_i[i])

                sol[:,0] += np.asarray(cumul_i[:])
                #sol[:,0] += sol_i[:, idx]

                if self.model=='SEIRD':
                    # D
                    sol[:,1] += sol_i[:, 4]
            return ((sol - data)/eps).ravel()


    def fit(self, country='France', out_days=0, plot=True, disease_vary=True, sector_props=[100.0], STATIC_PLOTS=False, remove_artefacts=False):
        try:
            EPI_data, country_attrs = country_covid(country, self.owid_file, self.hosp_file, model=self.model)
        except ValueError:
            print(f'incomplete data on {country}')
            return [], []
        # Add daily deaths data if available
        #if DEATHS_FILE != '' and country == 'France':
        #    deaths_data = pd.read_csv(DATA_PATH+'/'+DEATHS_FILE, sep=",",header=0)
        #    EPI_data = EPI_data.merge(deaths_data, on='date', how='outer')
            # Compute collateral deaths
        #    EPI_data['F'] = EPI_data['D_total'] - EPI_data['D']
        #    EPI_data.drop(['D_total'], axis=1, inplace=True)
        #    EPI_data['dD'] = EPI_data['D'].diff()
        #    EPI_data['dF'] = EPI_data['F'].diff()
        #    EPI_data.ffill(axis=0, inplace=True)
        #    EPI_data.bfill(axis=0, inplace=True)
        #    df = EPI_data.drop(['H', 'Reanimation', 'N_effective'], axis=1).reindex(columns=['date', 'S', 'I', 'R', 'D', 'F', 's'])

        df = EPI_data.drop(['N_effective'], axis=1).reindex(columns=['date', 'S', 'I', 'R', 'D', 's'])
        #df = EPI_data.drop(['H', 'Reanimation', 'N_effective'], axis=1).reindex(columns=['date', 'S', 'I', 'R', 'D', 's'])
        df.R.fillna(0, inplace=True)
        df.ffill(axis=0, inplace=True)
        df.bfill(axis=0, inplace=True)
        #FIRST WAVE and corresponds to Martin's fit
        df = df[df['date'] <= self.cutoff_date]
        #df = df[df['date'] < str(date.today())]
        initN = country_attrs['population']*self.ratio_susceptible

        #Clean artefacts
        if remove_artefacts:
            df['dI'] = df['I'].diff()
            #Init
            df.loc[0, 'dI'] = 0
            indexes = list(np.where(df['dI']<0)[0])

            while len(indexes) > 0:
                for i in indexes:
                    df.loc[i, 'I'] = df.loc[i-1, 'I']

                #Update
                df['dI'] = df['I'].diff()
                indexes = list(np.where(df['dI']<0)[0])

        #stringency
        days = len(df) - out_days
        data_s = df.loc[0:(days-1), 's'].values
        data_full = df.loc[0:(len(df)-1), ['I', 'D']].values
        eps=1.0
        # initial_conditions = [initE, initI, initR, initN, initD]
        tspan = np.arange(1, days+1, 1)
        tspan_full = np.arange(1, days+out_days+1, 1)

        # Fitting stringency curve
        if self.stringency_fit:
            params_stringency = self.init_sectors(sector_props, disease_vary=disease_vary, initN=initN, stringency_fit=self.stringency_fit)
            # fit model
            result_s = minimize(self.error_sectors, params_stringency, args=(sector_props, tspan, data_s, eps, self.stringency_fit), method='leastsq', full_output = 1)
            result_s.params.pretty_print()

            self.result_s = result_s
            self.a_beta_init = result_s.params['as0'].value

            final_stringency = data_s + result_s.residual.reshape(data_s.shape)

            #Plot
            fig = go.Figure()
            #Plot Observed cumulated cases
            fig.add_trace(go.Scatter(x=tspan_full, y=data_s, mode='markers', name='Observed Stringency',line = dict(dash='dot', color='purple')))
            #Plot fitted cases
            fig.add_trace(go.Scatter(x=tspan_full, y=final_stringency, mode='lines+markers', name='Fitted Stringency',line = dict(dash='dot', color='green')))
            fig.update_layout(title='Observed vs Fitted stringency',
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Level',
                                   title_x=0.5,
                                   title_font_size=18,
                                   font_size=18,
                                  width=1000, height=600
                                 )

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()


        #Fitting other parameters
        params, initial_conditions = self.init_sectors(sector_props, disease_vary=disease_vary, initN=initN, stringency_fit=False)

        if self.model=='SIR' or self.model=='SEIR':
            data = df.loc[0:(days-1), ['I']].values
        elif self.model =='SEIRD':
            data = df.loc[0:(days-1), ['I', 'D']].values


        if plot:
            print(f'Fitting {country} data with {out_days} last days out')

        # fit model and find predicted values
        result = minimize(self.error_sectors, params, args=(initial_conditions, tspan, data, eps, False), method='leastsq', full_output = 1)
        if plot:
            print(report_fit(result))
            result.params.pretty_print()

            print(f'{days} fitted days out of {days+out_days} available data points')

        final = data + result.residual.reshape(data.shape)
        tspan_full = np.arange(1, days+out_days+1, 1)

        if plot:
            fig = go.Figure()
            #Plot Observed cumulated cases
            fig.add_trace(go.Scatter(x=tspan_full, y=data[:, 0], mode='markers', name='Observed Infections',line = dict(dash='dot', color='red')))
            #Plot fitted cases
            fig.add_trace(go.Scatter(x=tspan_full, y=final[:, 0], mode='lines+markers', name='Fitted Infections',line = dict(dash='dot', color='orange')))

            if self.model =='SEIRD': #Plot cumulated deaths and fitted deaths
                fig.add_trace(go.Scatter(x=tspan_full, y=data[:, 1], mode='markers', name='Observed COVID Deaths', line = dict(dash='dot', color='blue')))
                fig.add_trace(go.Scatter(x=tspan_full, y=final[:, 1], mode='lines+markers', name='Fitted Deaths',line = dict(dash='dot', color='purple')))

            #Plot collateral deaths
            # fig.add_trace(go.Scatter(x=tspan_full, y=data[:, 2], mode='markers', name='Observed Collateral Deaths', line = dict(dash='dot')))
            # fig.add_trace(go.Scatter(x=tspan_full, y=final[:, 2], mode='lines+markers', name='Fitted Collateral Deaths'))

            if out_days > 0:
                fig.add_trace(go.Scatter(x=tspan_full, y=data_full[:, 0], mode='markers', name='Observed Infections (Full)', line = dict(dash='dot')))
                if self.model=='SEIRD':
                    fig.add_trace(go.Scatter(x=tspan_full, y=data_full[:, 1], mode='markers', name='Observed Deaths (Full)', line = dict(dash='dot')))
            #     fig.add_trace(go.Scatter(x=tspan_full, y=data_full[:, 2], mode='markers', name='Observed Collateral Deaths (Full)', line = dict(dash='dot')))

            fig.update_layout(title='{} per sector: Observed vs Fitted'.format(self.model),
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Counts',
                                   title_x=0.5,
                                   title_font_size=18,
                                   font_size=18,
                                  width=1000, height=600
                                 )

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

            #Plot the 1st derivative
            dI = df.loc[1:(days-1), 'dI'].values

            if self.model=='SEIRD':
                df['dD'] = df['D'].diff()
                dD = df.loc[1:(days-1), 'dD'].values

            fig = make_subplots(specs=[[{"secondary_y": False}]])
            #Infections
            derivs = np.diff(final.ravel(), 1).reshape(data.shape[0]-1, 1)
            #derivs = np.diff(final[:, 0])*(np.diff(final[:, 0]) >0)
            fig.add_trace(go.Scatter(x=tspan_full[1:], y=dI, mode='markers', name='Observed Daily Infections',line = dict(dash='dot', color='red')))
            fig.add_trace(go.Scatter(x=tspan_full[1:], y=derivs[:,0], mode='lines+markers', name='Fitted Daily Infections',line = dict(dash='dot', color='orange')))

            if self.model=='SEIRD':
                fig.add_trace(go.Scatter(x=tspan_full[1:], y=dD, mode='markers', name='Observed Daily Deaths', line = dict(dash='dot', color='blue')))
                fig.add_trace(go.Scatter(x=tspan_full[1:], y=derivs[:,1], mode='lines+markers', name='Fitted Daily Deaths',line = dict(dash='dot', color='purple')))

            fig.update_layout(title='Observed vs Fitted on new number of cases/deaths',
                                   xaxis_title='Days since first infected',
                                   yaxis_title='Daily Count',
                                   title_x=0.5,
                                   title_font_size=18,
                                   font_size=18,
                                  width=1000, height=600
                                 )

            if STATIC_PLOTS:
                img_bytes = fig.to_image(format="png")
                display(Image(img_bytes))
            else:
                fig.show()

            #Third plot
            if self.beta_sigmoid_step or self.mu_sigmoid_step:
                self.show_rates(result, df, tspan, STATIC_PLOTS=STATIC_PLOTS)

        return data_s, result, initN, final
