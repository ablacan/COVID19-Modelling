from __future__ import division # always-float division
import numpy as np
import pandas as pd
import glob
import os
import requests
from datetime import date

#Import from utils
from utils import *

#Import data generator
from generator_script import Generator

#Import epidemiological model, either SIR or SEIR
from EPI_model_country.py import EPI_Model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


class ErrorPredictor():

    def __init__(self, model='SIR', PARAMS_FILE='None'):
        self.model = model
        self.PARAMS_FILE = PARAMS_FILE
        self.file_model = model
        self.file_method = 'results_error_predictor'

        #Create directories for future results
        if not os.path.exists('./{}'.format(self.file_method)):
            os.makedirs('./{}'.format(self.file_method))

        if not os.path.exists('./{0}/{1}'.format(self.file_method, self.file_model)):
            os.makedirs('./{0}/{1}'.format(self.file_method, self.file_model))


    #Generate training data
    @staticmethod
    def get_data(samples):
        """
        Returns training set of stringency and mobility data to use in simulations of covid19 cases.
        """
        Generator = Generator()
        stringency = Generator.generate(samples, type='stringency')
        mobility = Generator_mobility.generate(samples, type = 'mobility')

        beta = Generator.generate(samples, type='beta')

        return stringency, mobility, beta


    #Generate simulations of cases according to beta
    def simulate(self, samples):
        """
        Returns simulated cases from EPI_Model
            samples : number of simulations, int
        """

        Model = EPI_Model(model=self.model, beta_step = 'linear')

        #Get data for simulation
        stringency, mobility, beta = self.get_data(samples)
        #Generate simulations with respective population
        simulations, population = Model.simulations(beta, samples)

        return simulations, population, stringency, mobility, beta


    #Train epi_model
    def fit(self, samples, train_window):
        """
        Fit model on simulations
            train_window : number of days to train the epi model, int
        """
        self.train_window = train_window

        Model = EPI_Model(model=self.model, beta_step = 'linear', PARAMS_FILE=self.PARAMS_FILE)

        #Get simulations
        covid_cases, population, stringency_data, mobility_data, beta = self.simulate(samples)

        results_dict = []

        for i in range(len(covid_cases)):

            #Create dataframe
            df = pd.DataFrame(data={'date' : pd.date_range("2020", freq="D", periods=len(covid_cases[i])), 'I': covid_cases[i]})
            country_attrs = {'population' : population[i].item()}

            #Prepare training with fixed gamma, sigma (disease_vary=False)
            train_data, test_data, params, initial_conditions = Model.prepare_training(df, country_attrs, disease_vary=False)

            #Training and forecast
            tspan_train = np.arange(1, 1+self.train_window, 1)
            train_data, test_data, result, train_final, MSE, MAE, sol, test_MSE, test_MAE = Model.fit_test(df, country_attrs, disease_vary=False)

            #Save dictionary of results for each simulation
            results_dict.append({'result': result, 'train_data': train_data,  'stringency_data' : stringency_data[i], 'mobility_data': mobility_data[i], 'beta': beta[i],  'train_fit': train_final,
            'test_data': test_data, 'test_predicted': sol})

        #Saving results
        save_to_path = './{0}/{1}/results_simulations_{3}samples_{4}days.npy'.format(self.file_method, self.file_model, samples, train_window)

        np.save(save_to_path,  results_dict)


    #Get training inputs for error predictor model
    def prepare_inputs(self, RES, prediction_horizon=1):
        """
        Returns tuple of target and training inputs for error predictor model for a given prediction horizon.
            RES : dictionnary of results previoulsy fitted, dict
            prediction_horizon : days ahead to predict error on, int (>0)
        """
        assert prediction_horizon !=0, 'Error: prediction_horizon must be between 15 and 1.'

        #Past inputs
        past_cases = RES['train_fit']
        past_derivative = np.concatenate(np.zeros(1).reshape(1,1), np.diff(past_cases, 1), axis=1)
        past_error = np.abs(RES['train_data'] - RES['train_fit'])
        past_error2 = past_error**2

        past_stringency = RES['stringency_data'][:250]
        past_mobility = RES['mobility_data'][:250]
        past_beta =  RES['beta'][:250]

        #Add nb_lookback_days ??

        #Target at different horizons
        targets = np.abs(RES['test_data'] - RES['test_predicted'])


        #Get target at horizon i
        target = np.abs(targets[prediction_horizon-1])

        if prediction_horizon ==1:
            inputs0 = np.stack((past_cases, past_derivative, past_error, past_error2, past_stringency, past_mobility, past_beta), axis=0)
            return (target, inputs0)

        elif prediction_horizon >1:
            #Stack inputs
            past_cases = np.stack(past_cases,  RES['test_data'][:prediction_horizon-2], axis=0)
            past_derivative = np.stack(past_cases, 1)
            past_error = np.stack(past_error, targets[:prediction_horizon-2])
            past_error2 = np.stack(past_error2, targets[:prediction_horizon-2]**2)

            past_stringency = RES['stringency_data'][:250+prediction_horizon-1]
            past_mobilit = RES['mobility_data'][:250+prediction_horizon-1]
            past_beta =  RES['beta'][:250+prediction_horizon-1]

            inputs = np.stack((past_cases, past_derivative, past_error, past_error2, past_stringency, past_mobility, past_beta), axis=0)
            return (target, inputs)


    #Train error predictor model on given time horizon
    def train(self, samples=10, train_window = 250, prediction_horizon=1):
        """
        Returns..
            prediction_horizon : horizon of days to predict error on, int (>0).
        """

        #Fit epi_model on a number of simulations=samples
        self.fit(samples, train_window = train_window)

        #Load results previously fitted
        from_path = './{0}/{1}/results_simulations_{3}samples_{4}days.npy'.format(self.file_method, self.file_model, samples, train_window)
        RES = np.load(from_path, allow_pickle=True).item()

        #Get target and inputs for error predictor training given a prediction_horizon and a dictionnary of results
        #Init
        target, x = self.prepare_inputs(RES[0], prediction_horizon)
        X = x
        Y = np.asarray([target]).reshape(1,)

        #Loop over all dicts of results
        for res_dict in RES[1:len(RES)]:
            target, x = self.prepare_training_inputs(res_dict, prediction_horizon)
            X = np.stack(X, x)
            Y = np.stack(Y, np.asarray(target).reshape(1,))

        #Split train_test
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

        #Train the error predictor model
        RF_Model = RandomForestRegressor(max_depth=5, random_state=0)
        RF_Model.fit(X_train, y_train)

        #Predict on test samples
        RF_Model.predict(X_test)

        # Save model
        joblib.dump(RF_Model,'./{0}/{1}/Error_Predictor_Model_{3}samples_{4}days.npy'.format(self.file_method, self.file_model, samples, train_window))

    def predict(self, inputs, model_path, prediction_horizon=1):
        """
        Returns error predictions on input samples given a horizon of prediction.
            inputs : dict of results from epidemiological model, list of dicts
            model_path : path from which to load the error predictor model, str
            prediction_horizon : error prediction horizon, int
        """

        #Prepare inputs
        target, x = self.prepare_inputs(inputs, prediction_horizon)

        # Load
        RF_Model = joblib.load(model_path)

        #Predict on samples
        predictions = RF_Model.predict(inputs)

        return predictions
