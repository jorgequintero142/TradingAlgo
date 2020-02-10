#########################################################################################
# We use a Gaussian Naive Bayes model to predict if a stock will have a high return 
# or low return next Monday (num_holding_days = 5),  using as input decision variables 
# the assets growthto yesterday from 2,3,,4,5,6,7,8,9 and 10 days before  
# We use the code form post [“How to Leverage the Pipeline to Conduct Machine Learning in the IDE”][2] 
# by Jim Obreen to preprocess teh data
#########################################################################################
 
##################################################
# Imports
##################################################

from __future__ import division
from collections import OrderedDict
import time

# Pipeline, Morningstar, and Quantopian Trading Functions
from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio
from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.optimize import TargetWeights
from quantopian.pipeline.factors import Returns
from quantopian.pipeline.data import EquityPricing
from quantopian.pipeline.data.psychsignal import (twitter_withretweets)
import quantopian.pipeline.data.factset.estimates as fe

# The basics
import pandas as pd
import numpy as np

# SKLearn :)
from sklearn.naive_bayes import GaussianNB

##################################################
# Globals
##################################################

num_holding_days = 5 # holding our stocks for five trading days.
days_for_fundamentals_analysis = 20
upper_percentile = 20
lower_percentile = 30

MAX_GROSS_EXPOSURE = 1.0
MAX_POSITION_CONCENTRATION = 0.05

##################################################
# Initialize
##################################################

def initialize(context):
    """ Called once at the start of the algorithm. """

    # Configure the setup
    set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))
    set_asset_restrictions(security_lists.restrict_leveraged_etfs)

    # Schedule our function
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_open(minutes=1))

    # Build the Pipeline
    attach_pipeline(make_pipeline(), 'my_pipeline')

##################################################
# Pipeline-Related Code
##################################################
            
class Predictor(CustomFactor):
    """ Defines our machine learning model. """
    
    # The factors that we want to pass to the compute function. We use an ordered dict for clear labeling of our inputs.
    factor_dict = OrderedDict([
              ('Asset_Growth_2d' , Returns(window_length=2)),
              ('Asset_Growth_3d' , Returns(window_length=3)),
              ('Asset_Growth_4d' , Returns(window_length=4)),
              ('Asset_Growth_5d' , Returns(window_length=5)),
              ('Asset_Growth_6d' , Returns(window_length=6)),
              ('Asset_Growth_7d' , Returns(window_length=7)),
              ('Asset_Growth_8d' , Returns(window_length=8)),
              ('Asset_Growth_9d' , Returns(window_length=9)),
              ('Asset_Growth_10d' , Returns(window_length=10)),
              ('Asset_Growth_15d' , Returns(window_length=15)),
              ('Asset_Growth_10d' , Returns(window_length=10)),
              ('Asset_Growth_20d' , Returns(window_length=20)),
              ('Return' , Returns(inputs=[USEquityPricing.open],window_length=5))
              ])

    columns = factor_dict.keys()
    inputs = factor_dict.values()

    # Run it.
    def compute(self, today, assets, out, *inputs):
        """ Through trial and error, I determined that each item in the input array comes in with rows as days and securities as columns. Most recent data is at the "-1" index. Oldest is at 0.

        !!Note!! In the below code, I'm making the somewhat peculiar choice  of "stacking" the data... you don't have to do that... it's just a design choice... in most cases you'll probably implement this without stacking the data.
        """

        ## Import Data and define y.
        inputs = OrderedDict([(self.columns[i] , pd.DataFrame(inputs[i]).fillna(0,axis=1).fillna(0,axis=1)) for i in range(len(inputs))]) # bring in data with some null handling.
        num_secs = len(inputs['Return'].columns)
        y = inputs['Return'].shift(-num_holding_days)
        y=y.dropna(axis=0,how='all')
        
        for index, row in y.iterrows():
            
             upper = np.nanpercentile(row, upper_percentile)            
             lower = np.nanpercentile(row, lower_percentile)
             auxrow = np.zeros_like(row)
             
             for i in range(0,len(row)):
                if row[i] <= lower: 
                    auxrow[i] = -1
                elif row[i] >= upper: 
                    auxrow[i] = 1 
        
             y.iloc[index] = auxrow
            
        y=y.stack(dropna=False)
        
        
        ## Get rid of our y value as an input into our machine learning algorithm.
        #del inputs['Return']

        ## Munge x and y
        x = pd.concat([df.stack(dropna=False) for df in inputs.values()], axis=1).fillna(0)
        
        ## Run Model
        model = GaussianNB() 
        model_x = x[:-num_secs*(num_holding_days)]
        model.fit(model_x, y)
        
        out[:] =  model.predict(x[-num_secs:])

def make_pipeline():

    universe = QTradableStocksUS()
    
    #Se importa la libreria para tener en cuenta las recomendaciones del Broker
    #Estos datos solo estan disponibles hasta Febrero del 2019.
    fe_rec = fe.ConsensusRecommendations
    
    #Obtenemos el ultimo precio de cierre del dia, de todas las acciones    
    yesterday_close = EquityPricing.close.latest
    #Obtenemos el ultimo volumen de operaciones diarias, para todas las acciones
    yesterday_volume = EquityPricing.volume.latest
    #Obtiene el conjunto de datos para los mensajes de twitter, incluyendo re-tweets
    #Se realiza la proporcion entre mensajes relacionados con el alza sobre
    #el total de mensajes
    tweets_prop = (twitter_withretweets.bull_scored_messages.latest/
            twitter_withretweets.total_scanned_messages.latest)
    
    #suma entre ultimo precio de cierre del dia y ultimo volumen de operaciones diarias
    last_prices = yesterday_close.zscore()+yesterday_volume.zscore()
    
    #recomendaciones de compra
    rec_buy  = fe_rec.buy.latest
    #recomenraciones de venta
    rec_sell = fe_rec.sell.latest
    #diferencia en las recomendaciones
    recomendacion = (rec_buy-rec_sell).zscore()
    
    #Se agregan las columnas 'Prices', 'Tweets' y 'DiferenciaRec'
    #De esta manera se tendran 4 factores en el pipeline, y se han agrupado segun 
    #caracteristicas generales de las variables seleccionadas
    pipe = Pipeline(columns={'Model': Predictor(window_length=days_for_fundamentals_analysis, mask=universe), 'Prices':last_prices, 'Tweets': tweets_prop, 'DiferenciaRec':recomendacion},screen = universe)

    return pipe

##################################################
# Execution Functions
##################################################

def rebalance(context,data):
    """ Execute orders according to our schedule_function() timing."""

    # Timeit!
    start_time = time.time()

    ## Run pipeline
    pipeline_output_df = pipeline_output('my_pipeline').dropna(how='any')
    
    todays_predictions = pipeline_output_df.Model

    # Demean pipeline scores
    target_weight_series = todays_predictions.sub(todays_predictions.mean())

    # Reweight scores to prepare for portfolio ordering.
    target_weight_series = target_weight_series/target_weight_series.abs().sum()
    
    order_optimal_portfolio(objective=TargetWeights(target_weight_series),constraints=[])

    # Print useful things. You could also track these with the "record" function.
    print ('Full Rebalance Computed Seconds: '+'{0:.2f}').format(time.time() - start_time)
    print ("Number of total securities trading: ")+ str(len(target_weight_series[target_weight_series > 0]))
    print ("Leverage: ") + str(context.account.leverage)