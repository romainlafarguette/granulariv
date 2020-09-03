# -*- coding: utf-8 -*-
"""
Granular IV for the Macro Stress-Test project
Romain Lafarguette, rlafarguette@imf.org
Time-stamp: "2020-08-31 21:42:01 Romain"
"""

###############################################################################
#%% Packages
###############################################################################
# Base modules
import os, sys                                           # System packages
import importlib                                         # Import tools
import pandas as pd                                      # Dataframes 
import numpy as np                                       # Numerical tools

# Functional loading
from linearmodels import OLS, PanelOLS                   # Linear models
from sklearn.decomposition import PCA                    # PCA
from collections import namedtuple                       # Containers

# Graphics
import matplotlib
matplotlib.use('TkAgg') # Must be called before importing plt
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
#%% Class: GIV (Granulary IV)
###############################################################################
class GIV(object):
    """ 
    Construct an instrument through Granular IV
    Based on Gabaix and Koijen (2019)

    Inputs
    ------
    endog: str
        Name of the endogeneous variable to instrument

    data: pandas dataframe
        Multi-index frame with individual, date (in this order) and weights

    wgt_col: str
        Name of the weight column

    ind_col: str
        Name of the identifier for the individuals (e.g. banks)

    date_str: str, default 'date'
        Name of the date column

    pca_num_factors: int, default 3
        Number of factors in the PCA fit

    threshold: int, default 0.05 (5%)
        Define the largest entities in terms of weights shares

    Output
    ------
    A GIV class object, wrapping .plot() class

    Usage:
    giv = GIV()

    """
    __description = "Granular IV under Python, based on Gabaix and Koijen 2019"
    __author = "Romain Lafarguette, IMF/MCM, rlafarguette@imf.org"

    # Initializer
    def __init__(self,
                 endog, # Name of the endogeneous variable
                 data, # Data pandas dataframe
                 wgt_col, # Name of the weights
                 ind_col, # Name of the identifier variable
                 date_col='date', # Default date name
                 pca_num_factors=3, # Default 3
                 threshold=0.05, # Default 5%
    ):

        # Unit tests (definition at the bottom of the file)
        GIV_unit_tests(endog, data, wgt_col, ind_col, date_col,
                       pca_num_factors, threshold)
        
        # Attributes (user defined)
        self.endog = endog
        self.ind_col = ind_col
        self.date_col = date_col
        self.df = data.dropna(subset=[self.endog]).copy() 
        self.wgt_col = wgt_col
        self.pca_num_factors = pca_num_factors
        self.threshold = threshold
                    
        # # Step 1: compute the residuals
        # self.df[f'{self.endog}_residuals'] = self.__reg_residuals().copy()

        # # Step 2: estimate the idiosyncratic shocks via PCA
        # self.d_idiosyncratic = self.__pca_idiosyncratic().copy()

        # # Step 3: aggregate the idiosyncratic shocks
        # self.instrument = self.agg_idiosyncratic(self.d_idiosyncratic)

    # Step 1/2/3: compute the residuals    
    # Class-methods (methods which returns a class defined below)    
    def fit(self, exog_l=None):
        # Note that the 'self' arg below is the GIV object itself
        return(GIVFit(self, exog_l=exog_l))
    
###############################################################################
#%% GIVFit
###############################################################################
class GIVFit(object): # Fit the model and retrieve the residuals
    """ 
    Estimate (total) residuals the panel regression with either: 
        - total absorbing effects (entity & time effects) 
        - or a specification with control variables

    Remember: total residuals = common factors + idiosyncratic shocks

    Inputs
    ------
    exog_l: list or None, default None
        List of exogeneous regressors to run the panel reg in first step   

    """

    # Initialization
    def __init__(self, GIV, exog_l=None):
               
        self.__dict__.update(GIV.__dict__) # Import all attributes from GIV
        
        # Add a new variable for the regressions        
        self.df['Intercept'] = 1

        # Name of the instrumented variable
        self.endog_instrument = f'{self.endog}_instrument'
        
        # Deal with exogenous variables, if any, and fit the model
        if exog_l: # With exogeneous variables
            self.exog_l = ['Intercept'] + exog_l
            self.mod = PanelOLS(self.df[self.endog], self.df[self.exog_l],
                                entity_effects=True)
            
        else: # without exogeneous variables
            self.exog_l = ['Intercept']
            self.mod = PanelOLS(self.df[self.endog], self.df[self.exog_l],
                                entity_effects=True, time_effects=True)
            
            
        self.panel_res = self.mod.fit(cov_type='clustered',
                                      cluster_entity=True)

        print(self.panel_res.summary)
        
        # Return the residuals, by entities
        self.df[f'{self.endog}_residuals'] = self.panel_res.resids
        
        # Prepare the data in wide format 
        self.dresids = self.df.pivot_table(index=[self.date_col], 
                                           columns=self.ind_col,
                                           values=f'{self.endog}_residuals')

        dresidsc = self.dresids.dropna(axis='columns') # Balanced panel
                
        # Fit a PCA with a given number of components
        # TODO: choice of num factors in PCA with the variance explained
        resids_pca = PCA(n_components=self.pca_num_factors) 
        resids_pca_factors = resids_pca.fit_transform(dresidsc)
        resids_pca_loadings = resids_pca.components_ # Varies by individuals

        cum_var_exp = np.cumsum(resids_pca.explained_variance_ratio_)

        self.cum_var = resids_pca.explained_variance_ratio_.cumsum()
        
        print(f'Cumulated explained variance with {self.pca_num_factors} ' 
              f'factors for {self.endog}: {round(100*self.cum_var[-1], 2)} %')
        
        resids_pca_reduc = resids_pca.inverse_transform(resids_pca_factors)

        # Verification of the PCA inverse transform, with mean 0 residuals
        resids_pca_reduc2 = (resids_pca_factors.dot(resids_pca_loadings)
                             + resids_pca.mean_)
        np.testing.assert_array_almost_equal(resids_pca_reduc,
                                             resids_pca_reduc2) 

        # Compute the "pure" idiosyncratic shocks
        # resids_pca_reduc is common shock, each ind with different loadings
        d_common = pd.DataFrame(resids_pca_reduc,
                                columns=dresidsc.columns,
                                index=dresidsc.index)
        self.resids_common = d_common
        self.resids_idiosyncratic = dresidsc - d_common # Simple difference

        #### Aggregate the idiosyncratic shocks
        # Relative weights (time varying, but take historical largest)
        dwgt = self.df.groupby(self.ind_col)[self.wgt_col].mean()
        dwgm = dwgt.sort_values(ascending=False) # Sort from largest
        self.avg_weights = dwgm # Save it for plotting
        
        # Only keep the weights above a certain threhold
        large_l = list(dwgm[dwgm>=self.threshold].index) 

        # In case the weights of some entities are not available
        avl_large_l = [x for x in large_l
                       if x in self.resids_idiosyncratic.columns]

        # Give an information message if some entities are missing
        if len(avl_large_l) < len(large_l):
            missing_l = [x for x in large_l if x not in avl_large_l]
            print(f'Entities not available {missing_l}')

        # Extract the largest idiosyncratic shocks, weighted average
        self.large_resids = self.resids_idiosyncratic[avl_large_l]
        instrument = self.large_resids.dot(dwgm[avl_large_l]) # Wgt average   
        self.instrument = pd.DataFrame(instrument,
                                       index=instrument.index,
                                       columns=[f'{self.endog}_instrument'])
                
        # Class attributes (attributes based on a class below)
        self.plot = GIVPlot(self)
        
                       
###############################################################################
#%% GIV Plot class
###############################################################################
class GIVPlot(object): # Fit the model and retrieve the residuals
    """ 
    Plot the results of the GIV estimation

    Inherit directly from the GIVFit class

    """

    # Initialization
    def __init__(self, GIVFit, exog_l=None):
        self.__dict__.update(GIVFit.__dict__) # Import all GIVFit attributes 


    # Variance explained
    def variance_explained(self,
                           title=None,
                           yticks_l=None):

        # Plot
        fig, ax = plt.subplots(1,1)
        factors = np.arange(len(self.cum_var))
        ax.bar([1 + x for x in factors], 100*self.cum_var)

        # Add a red horizontal bar
        top_floor = round(np.max(100*self.cum_var), 0)
        ax.axhline(y=top_floor, c='red', lw=2, ls='--')

        # Manage the ticks
        if yticks_l:
            ax.set_yticks(yticks_l + [top_floor])
            ax.get_yticklabels()[-1].set_color('red')
            
        else: # By default, the basic
            ax.set_yticks([0, 25, 50, 75, 100] + [top_floor])
            ax.get_yticklabels()[-1].set_color('red') 
        
        # Axis
        ax.set_xlabel('Factors', labelpad=20)
        ax.set_ylabel('Variance explained, in percent', labelpad=20)
        if title: ax.set_title(title, y=1.02)

        # Layout
        fig.set_size_inches(25, 15)
        fig.tight_layout()
        
        return(fig)
     

    # Weights plot (which entities are kept)
    def wgt_threshold(self,
                      title=None):

        # Sort the data
        wgt = 100*self.avg_weights.sort_values(ascending=True).copy()

        # Plot structure
        fig, ax = plt.subplots(1,1)

        # Main plot
        ax.barh(wgt.index, wgt, align='center') # Simple barplot
        ax.axvline(100*self.threshold, c='red', lw=2, ls='--',
                   label='Threshold for large entities')
               
        # Manage the color of the bars
        large_l = wgt[wgt >= 100*self.threshold]
        large_ind = [list(wgt).index(x) for x in large_l]
        for x in large_ind: ax.get_children()[x].set_color('r') 

        # Manage the ticks
        new_ticks_l = [100*self.threshold]
        old_ticks_l = [x for x in list(ax.get_xticks())]
        ax.set_xticks(list(ax.get_xticks()) + new_ticks_l)
        
        # Captions
        ax.set_xlabel('Weight, percent')
        if title: ax.set_title(title, y=1.02)

        # Legend
        ax.legend(loc='lower right', framealpha=0)
        
        # Layout
        fig.set_size_inches(25, 15)
        fig.tight_layout()

        fig.subplots_adjust(bottom=0.01, top=0.99)
        
        return(fig)

    
    # Instrument and residuals plot (last step of the process)
    def resids_instrument(self,
                          title=None,
                          legendloc='best', legendfont=None):
        # Data preparation
        idx = [x for x in self.instrument.index if x in self.large_resids.index]
        ins = self.instrument.loc[idx, :]
        lr = self.large_resids.loc[idx, :]

        # Plot
        fig, ax = plt.subplots(1, 1)
        ax.plot(lr.index, lr.iloc[:,0], color='gray', alpha=0.8, lw=1, ls='-',
                label='Idiosyncratic shocks')
        for var in lr.columns[1:]:
            ax.plot(lr.index, lr, color='gray', alpha=0.8, lw=1, ls='-')
        ax.plot(ins.index, ins, color='red', lw=4, ls='-',
                label='Instrument')

        # Captions and legend
        ax.set_ylabel('Shocks',
                      labelpad=20)
        if title: ax.set_title(title, y=1.02)
        if legendfont:
            ax.legend(numpoints=2, loc=legendloc, framealpha=0,
                      fontsize=legendfont)
        else:
            ax.legend(numpoints=2, loc=legendloc, framealpha=0,
                      fontsize=legendfont)


        # Layout
        fig.set_size_inches(25, 15)
        fig.tight_layout()

        return(fig)
        
        
###############################################################################
#%% Unit tests
###############################################################################
def GIV_unit_tests(endog, data, wgt_col, ind_col, date_col,
                   pca_num_factors, threshold):
    """ 
    Unit tests for the GIV class 
    """
    assert isinstance(endog, str), 'Endogeneous variable should be str'   
    assert isinstance(date_col, str), 'Date name should be str'
    assert isinstance(ind_col, str), 'Individual name should be str'   
    assert isinstance(pca_num_factors, int), 'PCA factors should be int'
    
    return(None)
    
