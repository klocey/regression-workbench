import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
#from dash.exceptions import PreventUpdate

import pandas as pd
import random
import plotly.graph_objects as go
import warnings
import sys
#import re
import numpy as np

import base64
import io
#import json
#import ast
#import time
#from functools import reduce
#import operator
import math
#import random_equations
#import os
#import scipy as sc
from scipy import stats

import statsmodels.api as sm
#import statsmodels.discrete.discrete_model as sdm
#from statsmodels.regression.linear_model import OLS
#from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import summary_table
#import statsmodels.formula.api as smf

from sklearn.preprocessing import PolynomialFeatures
#from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score #r2_score
#from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, LogisticRegression

#########################################################################################
################################# CONFIG APP ############################################
#########################################################################################

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
server = app.server

xvars = ['Nothing uploaded']
yvar = 'Nothing uploaded'

#########################################################################################
########################### CUSTOM FUNCTIONS ############################################
#########################################################################################

def obs_pred_rsquare(obs, pred):
    # Determines the prop of variability in a data set accounted for by a model
    # In other words, this determines the proportion of variation explained by
    # the 1:1 line in an observed-predicted plot.
    r2 = 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)
    if r2 < 0:
        r2 = 0
    return r2


def myround(n):
    if n == 0:
        return 0
    sgn = -1 if n < 0 else 1
    scale = int(-math.floor(math.log10(abs(n))))
    if scale <= 0:
        scale = 2
    factor = 10**scale
    return sgn*math.floor(abs(n)*factor)/factor


def dummify(df, cat_vars, dropone=True):

    dropped = []
    cat_var_ls = []
    
    interxn = list(set(cat_vars) & set(list(df)))
    
    for i in interxn:
        labs = list(set(df[i].tolist()))
        df[i] = df[i].replace(r"^ +| +$", r"", regex=True)
        
        one_hot = pd.get_dummies(df[i])
        one_hot = one_hot.add_prefix(i + ':')
        ls2 = list(one_hot)
        
        if dropone == True:
            nmax = 0
            lab = 0
            for ii in ls2:
                x = one_hot[ii].tolist()
                n = x.count(1)
                if n > nmax:
                    nmax = int(n)
                    lab = ii
            
            one_hot.drop(labels=[lab], axis = 1, inplace=True)
            dropped.append(lab)
            
        labs = list(one_hot)
        cat_var_ls.append(labs)
        df = df.join(one_hot)
        df.drop(labels=[i], axis = 1, inplace=True)
        
    return df, dropped, cat_var_ls
            

def dummify_logistic(df, cat_vars, y_prefix, dropone=True):

    dropped = []
    cat_var_ls = []
    
    interxn = list(set(cat_vars) & set(list(df)))
    
    for i in interxn:
        labs = list(set(df[i].tolist()))
        df[i] = df[i].replace(r"^ +| +$", r"", regex=True)
        
        one_hot = pd.get_dummies(df[i])
        one_hot = one_hot.add_prefix(i + ':')
        ls2 = list(one_hot)
        
        if dropone == True and i != y_prefix:
            nmax = 0
            lab = 0
            for ii in ls2:
                x = one_hot[ii].tolist()
                n = x.count(1)
                if n > nmax:
                    nmax = int(n)
                    lab = ii
            
            one_hot.drop(labels=[lab], axis = 1, inplace=True)
            dropped.append(lab)
            
        labs = list(one_hot)
        cat_var_ls.append(labs)
        df = df.join(one_hot)
        df.drop(labels=[i], axis = 1, inplace=True)
        
    return df, dropped, cat_var_ls

def run_MLR(df_train, xvars, yvar, cat_vars, rfe_val):

    y_train = df_train.pop(yvar)
    X_train = df_train
    
    X_train, dropped, cat_vars_ls = dummify(X_train, cat_vars)
    
    if X_train.shape[1] < 2:
        return [], [], [], [], [], [], []
    
    ########## Eliminating features with many 0's ###########
    x_vars = list(X_train)
    drop = []
    for var in x_vars:
        vals = X_train[var].tolist()
        frac_0 = vals.count(0)/len(vals)
        if frac_0 > 0.95:
            drop.append(var)
    
    X_train.drop(labels=drop, axis=1, inplace=True)
    
    ########## RFECV ############
    model = LinearRegression()
    rfecv = RFECV(model, step=1)
    rfecv = rfecv.fit(X_train, y_train)
    #support = rfecv.support_
    ranks = rfecv.ranking_
    xlabs = rfecv.feature_names_in_
    
    supported_features = []
    unsupported = []
    
    for i, lab in enumerate(xlabs):
        if ranks[i] == 1:
            supported_features.append(lab)
        else:
            unsupported.append(lab)
    
    for ls in cat_vars_ls:
        
        check = list(set(ls) & set(supported_features)) # elements of ls that are in supported_features
        if len(check) == 0:
            supported_features = list(set(supported_features) - set(ls))
            for l in ls:
                try:
                    X_train.drop(l, axis=1, inplace=True)
                    unsupported.append(l)
                except:
                    pass
                    
        elif len(check) > 0:
            supported_features.extend(ls)
            supported_features = list(set(supported_features))
    
    if len(supported_features) >= 2:
        if rfe_val == 'Yes':
            X_train = X_train.filter(items = supported_features, axis=1)
    
    X_train_lm = sm.add_constant(X_train, has_constant='add')
    results = sm.OLS(y_train, X_train_lm).fit()
    
    y_pred = results.predict(X_train_lm)
    pval_df = results.pvalues
    R2 = results.rsquared_adj
    if R2 < 0: R2 = 0
    
    results_summary = results.summary()
    
    results_as_html1 = results_summary.tables[1].as_html()
    df1_summary = pd.read_html(results_as_html1, header=0, index_col=0)[0]
    
    results_as_html2 = results_summary.tables[0].as_html()
    df2_summary = pd.read_html(results_as_html2, header=0, index_col=0)[0]
    
    vifs = [variance_inflation_factor(X_train.values, j) for j in range(X_train.shape[1])]
    
    methods = ['bonf']#, 'sidak', 'holm-sidak',
               #'holm', 'simes-hochberg', 'hommel',
               #'fdr_bh', 'fdr_by']

    outlier_df = results.outlier_test(method='bonf', alpha=0.05)
    
    colors = []
    for i in outlier_df['bonf(p)'].tolist():
        if i < 1.0:
            colors.append("#ff0000")
        else:
            colors.append("#3399ff")
    
    cols = ['Parameter', 'coef', 'std err', 't', 'P>|t|', '[0.025]', '[0.975]']
    df_table = pd.DataFrame(columns=cols)
    df_table['Parameter'] = df1_summary.index.tolist()
    df_table['coef'] = df1_summary['coef'].tolist()
    df_table['std err'] = df1_summary['std err'].tolist()
    df_table['t'] = df1_summary['t'].tolist()
    df_table['P>|t|'] = df1_summary['P>|t|'].tolist()
    df_table['[0.025]'] = df1_summary['[0.025'].tolist()
    df_table['[0.975]'] = df1_summary['0.975]'].tolist()
    
    xlabs = list(X_train)
    vifs2 = []
    for p in df_table['Parameter'].tolist():
        if p == 'const':
            vifs2.append(np.nan)
        else:
            i = xlabs.index(p)
            vif = vifs[i]
            vifs2.append(np.round(vif,3))
        
    df_table['VIF'] = vifs2
    df1_summary = df_table
    
    return y_train, y_pred, df1_summary, df2_summary, supported_features, unsupported, colors
    



def run_logistic_regression(df, xvars, yvar, cat_vars):
    
    coefs = []
    r2s = []
    #intercepts = []
    pvals = []
    #rvals = []
    aics = []
    llf_ls = []
    PredY = []
    Ys = []
            
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how='any', inplace=True)
    df = df.loc[:, df.nunique() != 1]
    
    y_o = df[yvar]
    x_o = df.drop(labels=[yvar], axis=1, inplace=False)
    
    ########## Eliminating features that are perfectly correlated to the y-variable ###########
    perfect_correlates = []
    for xvar in list(x_o):
        x = x_o[xvar].tolist()
        y = y_o.tolist()
        slope, intercept, r, p, se = stats.linregress(x, y)
        if r**2 == 1.0:
            print('remove perfect correlate:', xvar)
            perfect_correlates.append(xvar)
    x_o.drop(labels=perfect_correlates, axis=1, inplace=True)
    
    ########## Eliminating features that only have one value ###########
    singularities = []
    for xvar in list(x_o):
        x = len(list(set(x_o[xvar].tolist())))
        if x == 1:
            print('remove singularity:', xvar)
            singularities.append(xvar)
    x_o.drop(labels=singularities, axis=1, inplace=True)
    
    ########## Eliminating features with many 0's ###########
    x_vars = list(x_o)
    drop = []
    for var in x_vars:
        vals = x_o[var].tolist()
        frac_0 = vals.count(0)/len(vals)
        if frac_0 > 0.95:
            drop.append(var)
    
    x_o.drop(labels=drop, axis=1, inplace=True)
    
    ########## Eliminating features using vif ###########
    while x_o.shape[1] > 100:
        cols = list(x_o)
        vifs = [variance_inflation_factor(x_o.values, j) for j in range(x_o.shape[1])]
            
        max_vif = max(vifs)
        if max_vif > 10:
            i = vifs.index(max(vifs))
            col = cols[i]
            x_o.drop(labels=[col], axis=1, inplace=True)
        else:
            break
            
    ########## RFECV ############
    model = LogisticRegression()
    try:
        rfecv = RFECV(model, step=1, min_features_to_select=2)
        
        print('list(set(y_o)):', list(set(y_o)))
        rfecv = rfecv.fit(x_o, y_o)
        
        #support = rfecv.support_
        ranks = rfecv.ranking_
        xlabs = rfecv.feature_names_in_
        
        supported_features = []
        unsupported = []
        
        for i, lab in enumerate(xlabs):
            if ranks[i] == 1:
                supported_features.append(lab)
            else:
                unsupported.append(lab)
        
        #if len(supported_features) >= 2:
        #    if rfe_val == 'Yes':
                #X_train = X_train.filter(items = supported_features, axis=1)
    
        x_o = x_o.filter(items = supported_features, axis=1)
    except:
        pass
    
    model = 0
    x_o_lm = sm.add_constant(x_o, has_constant='add')
    
    try:
        model = sm.Logit(y_o, x_o_lm).fit(maxiter=30)
    except:
        return None, None, None, 1, None
    
    results_summary = model.summary()
            
    results_as_html1 = results_summary.tables[1].as_html()
    df1_summary = pd.read_html(results_as_html1, header=0, index_col=0)[0]
    
    results_as_html2 = results_summary.tables[0].as_html()
    df2_summary = pd.read_html(results_as_html2, header=0, index_col=0)[0]
    
    #results_as_html2 = results_summary.tables[2].as_html()
    #df2_summary = pd.read_html(results_as_html2, header=0, index_col=0)[0]
            
    vifs = [variance_inflation_factor(x_o.values, j) for j in range(x_o.shape[1])]
            
    cols = ['Parameter', 'coef', 'std err', 'z', 'P>|z|', '[0.025]', '[0.975]']
    df_table = pd.DataFrame(columns=cols)
    df_table['Parameter'] = df1_summary.index.tolist()
    df_table['coef'] = df1_summary['coef'].tolist()
    df_table['std err'] = df1_summary['std err'].tolist()
    df_table['z'] = df1_summary['z'].tolist()
    df_table['P>|z|'] = df1_summary['P>|z|'].tolist()
    df_table['[0.025]'] = df1_summary['[0.025'].tolist()
    df_table['[0.975]'] = df1_summary['0.975]'].tolist()
            
    xlabs = list(x_o)
    vifs2 = []
    for p in df_table['Parameter'].tolist():
        if p == 'const':
            vifs2.append(np.nan)
        else:
            i = xlabs.index(p)
            vif = vifs[i]
            vifs2.append(np.round(vif,3))
                
    df_table['VIF'] = vifs2
    df1_summary = df_table
    df1_summary.sort_values(by='P>|z|', inplace=True, ascending=True)
    #ypred = model.fittedvalues
    ypred = model.predict(x_o_lm)
    ypred = ypred.tolist()
    
    #df['predicted probability'] = ypred
    
    ####### ROC CURVE #######################################
    fpr, tpr, thresholds = roc_curve(y_o, ypred, pos_label=1)
    auroc = auc(fpr, tpr)
            
    ####### PRECISION-RECALL CURVE ##############################################
    ppv, recall, thresholds_ppv = precision_recall_curve(y_o, ypred, pos_label=1)
    auprc = average_precision_score(y_o, ypred, pos_label=1)
            
    dist = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    dist = dist.tolist()
    thresholds = thresholds.tolist()
    di = dist.index(min(dist))
    t = thresholds[di]
    ypred2 = []
    for i in ypred:
        if i < t:
            ypred2.append(0)
        else:
            ypred2.append(1)
    ypred = list(ypred2)
    
    df['Predicted'] = ypred
    
    coefs.append(model.params[0])
                    
    pr2 = model.prsquared
    if pr2 < 0:
        pr2 = 0
                    
    aic = model.aic
    #bic = model.bic
    tp = model.pvalues[0]
    llf = model.llf
                    
    #m_coefs = model.params[0].tolist()
    #m_coefs.reverse()
                            
    r2s.append(np.round(pr2, 3))
    pvals.append(np.round(tp, 3))
    aics.append(np.round(aic, 3))
    llf_ls.append(np.round(llf, 5))
    Ys.append(y_o)
    PredY.append(ypred)
            
    y_o = y_o.tolist()
    prc_null = y_o.count(1)/len(y_o)
            
    cols = ['r-square']
    df_models = pd.DataFrame(columns=cols)
    df_models['r-square'] = r2s
    df_models['p-value'] = pvals
    df_models['AIC'] = aics
    df_models['log-likelihood'] = llf_ls
    df_models['FPR'] = [fpr]
    df_models['TPR'] = [tpr]
    df_models['PPV'] = [ppv]
    df_models['Recall'] = [recall]
    df_models['auprc'] = [auprc]
    df_models['auroc'] = [auroc]
    df_models['pred_y'] = PredY
    df_models['prc_null'] = [prc_null]
    #df_models['intercept'] = intercepts
    df_models['coefficients'] = coefs
            
    #df_models = df_models.replace('_', ' ', regex=True)
    #for col in list(df_models):
    #    col2 = col.replace("_", " ")
    #    df_models.rename(columns={col: col2})
            
    df_models.reset_index(drop=True, inplace=True)
    
    #col = df.pop('probability of ')
    #df.insert(0, col.name, col)
    
    col = df.pop('Predicted')
    df.insert(0, col.name, col)
    
    col = df.pop(yvar)
    df.insert(0, col.name, col)
    
    return df_models, df1_summary, df2_summary, 0, df
    

#########################################################################################
#################### DASH APP CONTROL CARDS  ############################################
#########################################################################################


def description_card1():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card1",
        children=[
            html.H5("Regression workbench",
                    style={
            'textAlign': 'left',
            }),
            html.P("This app automates regression-based tools, allowing you to explore relationships between individual variables, combinations of variables, and to find the most powerful relationships between variables.",
                    style={
            'textAlign': 'left',
            }),
            html.P("This beta version limits uploaded data to 10,000 randomly selected rows and does not provide downloadable results.",
                    style={
            'textAlign': 'left',
            }),
        ],
    )


def description_card_final():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card-final",
        children=[
            html.H5("Developer",
                    style={
            'textAlign': 'left',
            }),
            html.P("Kenneth J. Locey, PhD. Senior clinical data scientist. Center for Quality, Safety and Value Analytics. Rush University Medical Center.",
                    style={
            'textAlign': 'left',
            }),
        ],
    )



def control_card_upload():
    
    return html.Div(
        id="control-card-upload1",
        children=[
            html.H5("Begin by uploading your data", style={'display': 'inline-block',
                                                    'width': '90%'},),
            html.I(className="fas fa-question-circle fa-lg", id="target1",
                style={'display': 'inline-block', 'width': '10%', 'color':'#99ccff'},
                ),
            dbc.Tooltip("Use sound data practices: No commas in column headers. No values with mixed data types (like 10cm or 10%). If your Excel file does not load, then it probably contains special formatting (frozen panes, etc.). If this happens, save your file as a csv and upload that.", target="target1",
                style = {'font-size': 12},
                ),
            html.P("Your file (csv or Excel) must only have rows, columns, and column headers. Excel files must have one sheet and no special formatting."),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a File', style={'color':'#2c8cff', "text-decoration": "underline"},),
                ]),
                style={
                    'lineHeight': '68px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '20px',
                },
                multiple=False
            ),
            html.P("Do not upload sensitive information. Data are deleted when the app is closed, refreshed, or when another file is uploaded."),
        ],
    )

    

def control_card1():

    return html.Div(
        id="control-card1",
        children=[
                html.H5("Find the strongest relationships between numeric variables",
                        style={'display': 'inline-block', 'width': '44%'},),
                html.I(className="fas fa-question-circle fa-lg", id="target_select_vars",
                            style={'display': 'inline-block', 'width': '3%', 'color':'#99ccff'},),
                dbc.Tooltip("These analyses are based on ordinary least squares regression. They exclude categorical variables but include dichotomous numerical variables (those only having 2 values) for exploratory purposes. OLS regressions based on dichotomous variables should be interpreted with caution.", target="target_select_vars", style = {'font-size': 12},),
                html.Hr(),
                
                html.B("Choose your x-variables",
                    style={'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-right': '10px',
                       }),
                dcc.Dropdown(
                        id='xvar',
                        options=[{"label": i, "value": i} for i in []],
                        multi=True, value=None,
                        style={'width': '100%',
                               #'display': 'inline-block',
                             },
                        ),
                        
                html.Br(),
                html.B("Choose your y-variables",
                    style={'display': 'inline-block',
                           'vertical-align': 'top',
                           'margin-right': '10px',
                    }),
                dcc.Dropdown(
                        id='yvar',
                        options=[{"label": i, "value": i} for i in []],
                        multi=True, value=None,
                        style={'width': '100%',
                               #'display': 'inline-block',
                             },
                        ),
                html.Hr(),
                html.Br(),
                html.Button('Run regressions', id='btn1', n_clicks=0,
                    style={#'width': '100%',
                            'display': 'inline-block',
                            'margin-right': '3%',
                    },
                    ),
                #html.Button('Download results', id='btn1b', n_clicks=0,
                #    style={#'width': '100%',
                #            'display': 'inline-block',
                            #'margin-right': '10px',
                #    },
                #    ),
                
                html.P("", id='rt0'),
                ],
                style={'margin-bottom': '0px',
                       'margin': '10px',
                       'width': '98.5%',
                    },
            )


def control_card2():

    return html.Div(
        id="control-card2",
        children=[
                html.H5("Conduct a single regression for deeper insights",
                        style={'display': 'inline-block', 'width': '35.4%'},),
                html.I(className="fas fa-question-circle fa-lg", id="target_select_vars2",
                            style={'display': 'inline-block', 'width': '3%', 'color':'#99ccff'},),
                dbc.Tooltip("These analyses are based on ordinary least squares regression. They exclude categorical variables but include dichotomous numerical variables (those only having 2 values) for exploratory purposes. OLS regressions based on dichotomous variables should be interpreted with caution.", target="target_select_vars2", style = {'font-size': 12},),
                html.Hr(),
                
                html.Div(
                id="control-card2a",
                children=[
                    html.B("Choose an x-variable",
                        style={'display': 'inline-block',
                                'vertical-align': 'top',
                                'margin-right': '10px',
                           }),
                    dcc.Dropdown(
                            id='xvar2',
                            options=[{"label": i, "value": i} for i in []],
                            multi=False,
                            placeholder='Select a variable',
                            style={'width': '100%',
                                   'display': 'inline-block',
                                 },
                            ),
                        ],
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'margin-right': '20px',
                               'width': '20%',
                        }),
                
                html.Div(
                id="control-card2b",
                children=[
                    html.B("Choose a data transform for x",
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'margin-right': '10px',
                        }),
                    dcc.Dropdown(
                            id='x_transform',
                            options=[{"label": i, "value": i} for i in ['None', 'log10', 'square root']],
                            multi=False, value='None',
                            style={'width': '100%',
                                   'display': 'inline-block',
                                 },
                            ),
                    ],
                    style={'display': 'inline-block',
                           'vertical-align': 'top',
                           'margin-right': '20px',
                           'width': '20%',
                    }),
                    
                html.Div(
                id="control-card2c",
                children=[
                    html.B("Choose a y-variable",
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'margin-right': '10px',
                        }),
                    dcc.Dropdown(
                            id='yvar2',
                            options=[{"label": i, "value": i} for i in []],
                            multi=False,
                            placeholder='Select a variable',
                            style={'width': '100%',
                                   'display': 'inline-block',
                                 },
                            ),
                    ],
                    style={'display': 'inline-block',
                           'vertical-align': 'top',
                           'margin-right': '20px',
                           'width': '20%',
                    }),
                
                html.Div(
                id="control-card2d",
                children=[
                    html.B("Choose a data transform for y",
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'margin-right': '10px',
                        }),
                    dcc.Dropdown(
                            id='y_transform',
                            options=[{"label": i, "value": i} for i in ['None', 'log10', 'square root']],
                            multi=False, value='None',
                            style={'width': '100%',
                                   'display': 'inline-block',
                                 },
                            ),
                    ],
                    style={'display': 'inline-block',
                           'vertical-align': 'top',
                           'margin-right': '20px',
                           'width': '20%',
                    }),
                    
                html.Div(
                id="control-card2e",
                children=[
                    html.B("Choose a model",
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'margin-right': '10px',
                        }),
                    dcc.Dropdown(
                            id='model2',
                            options=[{"label": i, "value": i} for i in ['linear', 'quadratic', 'cubic']],
                            multi=False, value='linear',
                            style={'width': '100%',
                                   'display': 'inline-block',
                                 },
                            ),
                    ],
                    style={'display': 'inline-block',
                           'vertical-align': 'top',
                           'margin-right': '20px',
                           'width': '10%',
                    }),
                html.Hr(),
                html.Br(),
                html.Button('Run regression', id='btn2', n_clicks=0,
                    style={#'width': '100%',
                            'display': 'inline-block',
                            'margin-right': '3%',
                    },
                    ),
                #html.Button('Download results', id='btn2b', n_clicks=0,
                #    style={#'width': '100%',
                #            'display': 'inline-block',
                #            #'margin-right': '10px',
                #    },
                #    ),
                html.P("", id='rt3')
                ],
                style={'margin-bottom': '0px',
                       'margin': '10px',
                       'width': '98.5%',
                    },
            )



def control_card3():

    return html.Div(
        id="control-card3",
        children=[
                html.H5("Conduct multiple linear regression",
                        style={'display': 'inline-block', 'width': '26.3%'},),
                html.I(className="fas fa-question-circle fa-lg", id="target_select_vars3",
                            style={'display': 'inline-block', 'width': '3%', 'color':'#99ccff'},),
                dbc.Tooltip("This analysis is based on ordinary least squares regression and reveals predicted values, outliers, and the significance of individual variables.", target="target_select_vars3", style = {'font-size': 12},),
                html.P("When trying to explain or predict a numerical variable using several other variables."),
                html.Hr(),
                
                html.B("Choose 2 or more x-variables.",
                    style={'vertical-align': 'top',
                           'margin-right': '10px',
                       }),
                dcc.Dropdown(
                        id='xvar3',
                        options=[{"label": i, "value": i} for i in []],
                        multi=True, value=None,
                        style={'width': '100%',
                             },
                        ),
                html.Br(),
                
                html.Div(id='choosey',
                    children = [
                        html.B("Choose your y-variable",
                            style={'vertical-align': 'top',
                                   'margin-right': '10px',
                                   'display': 'inline-block', 'width': '42.%'}),
                        html.I(className="fas fa-question-circle fa-lg", id="target_y1",
                            style={'display': 'inline-block', 'width': '5%', 'color':'#bfbfbf'},),
                        dbc.Tooltip("Does not include categorical variables.",
                            target="target_y1", style = {'font-size': 12},),
                        dcc.Dropdown(
                                id='yvar3',
                                options=[{"label": i, "value": i} for i in []],
                                multi=False, value=None,
                                style={'width': '100%',
                                       'display': 'inline-block',
                                     },
                                ),
                            ],
                            style={'width': '30%',
                              'display': 'inline-block',
                              'margin-right': '5%',
                            },
                        ),
                html.Div(id='choose_ref',
                    children = [
                        html.B("Remove unimportant variables",
                            style={'display': 'inline-block', 'width': '58.%'},),
                        html.I(className="fas fa-question-circle fa-lg", id="target_rfe",
                                    style={'display': 'inline-block', 'width': '5%', 'color':'#bfbfbf'},),
                        dbc.Tooltip("Remove variables that have little-to-no effect on model performance. Removal is done using recursive feature elimination with 5-fold cross-validation. Unimportant features will not be removed if the resulting number of variables is less 2.", target="target_rfe", style = {'font-size': 12},),
                        dcc.Dropdown(
                            id='rfecv',
                            options=[{"label": i, "value": i} for i in ['Yes', 'No']],
                            multi=False, value='Yes',
                            style={'width': '50%', 'display': 'inline-block',
                             },
                            ),
                        ],
                        style={'width': '30%',
                        'display': 'inline-block',
                        },
                    ),
                        
                html.Hr(),
                html.Br(),
                html.Button('Run regression', id='btn3', n_clicks=0,
                    style={#'width': '100%',
                            'display': 'inline-block',
                            'margin-right': '3%',
                    },
                    ),
                #html.Button('Download results', id='btn3b', n_clicks=0,
                #    style={#'width': '100%',
                #            'display': 'inline-block',
                #            #'margin-right': '10px',
                #    },
                #    ),
                html.P("", id = 'rt1'),
                ],
                style={'margin-bottom': '0px',
                       'margin': '10px',
                       'width': '98.5%',
                    },
            )
    

def control_card4():

    return html.Div(
        id="control-card4",
        children=[
                html.H5("Conduct multiple logistic regression",
                        style={'display': 'inline-block', 'width': '27.5%'},),
                html.I(className="fas fa-question-circle fa-lg", id="target_SLR_vars",
                            style={'display': 'inline-block', 'width': '3%', 'color':'#99ccff'},),
                dbc.Tooltip("In statistics, multiple logistic regression is used to find explanatory relationships and to understand the significance of variables. In machine learning, it is used to obtain predictions. This app does both.", target="target_SLR_vars", style = {'font-size': 12},),
                html.Br(),
                html.P("When trying to explain, predict, or classify a binary variable (1/0, yes/no) using several other variables",
                       style={'display': 'inline-block', 'width': '52%'},),
                html.I(className="fas fa-question-circle fa-lg", id="target_SLR_vars2",
                            style={'display': 'inline-block', 'width': '3%', 'color':'#bfbfbf'},),
                dbc.Tooltip("To improve efficiency, x-variables that are 95% zeros will be removed from analysis. Highly multicollinear x-variables are also removed during analysis, as are x-variables that are perfect correlates of the y-variable and any x-variable that only has one value. The app attempts to use recursive feature elimination to remove statistically unimportant variables.", target="target_SLR_vars2", style = {'font-size': 12},),
                
                html.Hr(),
                
                html.B("Choose two or more x-variables",
                    style={'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-right': '10px',
                            'width': '17.5%',
                       }),
                html.I(className="fas fa-question-circle fa-lg", id="target_select_x",
                            style={'display': 'inline-block', 'width': '5%', 'color':'#bfbfbf'},),
                dbc.Tooltip("Any that contain your y-variable will be removed from analysis. Example: If one of your x-variables is 'sex' and your y-variable is 'sex:male', then 'sex' will be removed from your x-variables during regression.", target="target_select_x", style = {'font-size': 12},),
                dcc.Dropdown(
                        id='xvar_logistic',
                        options=[{"label": i, "value": i} for i in []],
                        multi=True, value=None,
                        style={'width': '100%',
                             },
                        ),
                        
                html.Br(),
                html.B("Choose your y-variable",
                    style={'display': 'inline-block',
                           'vertical-align': 'top',
                           'margin-right': '10px',
                    }),
                
                dcc.Dropdown(
                        id='yvar_logistic',
                        options=[{"label": i, "value": i} for i in []],
                        multi=False, value=None,
                        style={'width': '50%',
                             },
                        ),
                html.Hr(),
                html.Br(),
                html.Button('Run regression', id='btn4', n_clicks=0,
                    style={#'width': '100%',
                            'display': 'inline-block',
                            'margin-right': '3%',
                    },
                    ),
                #html.Button('Download results', id='btn4b', n_clicks=0,
                #    style={#'width': '100%',
                #            'display': 'inline-block',
                #            #'margin-right': '10px',
                #    },
                #    ),
                html.P("", id = 'rt2'),
                ],
                style={'margin-bottom': '0px',
                       'margin': '10px',
                       'width': '98.5%',
                    },
            )
            
#########################################################################################
######################## DASH APP FIGURE FUNCTIONS ######################################
#########################################################################################

def parse_contents(contents, filename, date):
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if '.csv' in filename:
            df = pd.read_csv(
                #io.StringIO(decoded.decode('ISO-8859-1')))
                io.StringIO(decoded.decode('utf-8')))
        elif '.xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        return html.Div([
            'There was an error processing this file.'
        ])

    return df.to_json()



def generate_figure_1():

    return html.Div(
                id="Figure1",
                children=[
                    dcc.Loading(
                        id="loading-fig1",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="figure1",
                            children=[dcc.Graph(id='figure_plot1'),
                                    ],
                                ),
                        ),
                        html.P("", id='fig1txt'),
                    ],
                style={'width': '100%',
                    'display': 'inline-block',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    'margin-right': '10px',
                },
    )
    
def generate_figure_2():

    return html.Div(
                id="Figure2",
                children=[
                    dcc.Loading(
                        id="loading-fig2",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="figure2",
                            children=[dcc.Graph(id="figure_plot2"),
                                    ],
                                ),
                        ),
                    html.P("", id='fig2txt'),
                    ],
                style={'width': '100%',
                    'display': 'inline-block',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    'margin-right': '10px',
                },
    )


def generate_figure_3():

    return html.Div(
                id="Figure3",
                children=[
                    dcc.Loading(
                        id="loading-fig3",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="figure3",
                            children=[dcc.Graph(id="figure_plot3"),
                                    ],
                                ),
                        ),
                    html.P("", id='fig3txt')
                    ],
                style={'width': '100%',
                    'display': 'inline-block',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    'margin-right': '10px',
                },
    )



def generate_figure_4a():

    return html.Div(
                id="Figure4a",
                children=[
                    html.H6("Receiver Operating Characteristic (ROC curve)",
                        style={'width': '73%', 'display': 'inline-block',}),
                    html.I(className="fas fa-question-circle fa-lg", id="target_roc",
                        style={'display': 'inline-block', 'width': '10%', 'color':'#bfbfbf'},),
                    dbc.Tooltip("ROCs reveal the tradeoff between capturing a fraction of actual positives (1's) and missclassifying negatives (0's). The true positive rate (TPR) is the fraction of actual positives that were correctly classified. The false positive rate (FPR) is the fraction of actual negatives (0's) that were misclassified. ROCs do not reveal the reliability of predictions (precision).",
                        target="target_roc", style = {'font-size': 12},),
                    dcc.Loading(
                        id="loading-fig4a",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="figure4a",
                            children=[dcc.Graph(id="figure_plot4a"),
                                    ],
                                ),
                        ),
                    html.P("", id='fig4atxt'),
                    ],
                style={'width': '45%',
                    'display': 'inline-block',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    'margin-right': '8%',
                },
    )

def generate_figure_4b():

    return html.Div(
                id="Figure4b",
                children=[
                    html.H6("Precision-recall curve (PRC)",
                        style={'width': '45%', 'display': 'inline-block',}),
                    html.I(className="fas fa-question-circle fa-lg", id="target_prc",
                        style={'display': 'inline-block', 'width': '10%', 'color':'#bfbfbf'},),
                    dbc.Tooltip("PRCs reveal the tradeoff between correctly classifying actual positives (1's) and capturing a substantial fraction of positives. Precision is the fraction of positive predictions that were correct. Recall is another name for the TPR. A good ROC is misleading if the PRC is weak.",
                        target="target_prc", style = {'font-size': 12},),
                    dcc.Loading(
                        id="loading-fig4b",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="figure4b",
                            children=[dcc.Graph(id="figure_plot4b"),
                                    ],
                                ),
                        ),
                    html.P("", id='fig4btxt'),
                    ],
                style={'width': '45%',
                    'display': 'inline-block',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    'margin-right': '10px',
                },
    )


#########################################################################################
######################### DASH APP TABLE FUNCTIONS ######################################
#########################################################################################

def generate_table_1():

    return html.Div(
                id="Table1",
                children=[
                    html.Br(),
                    dcc.Loading(
                        id="loading-table1",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="table1",
                            children=[html.Div(id="table_plot1"),
                                    ]),
                            )
                    ],
                style={'width': '100%',
                    'display': 'inline-block',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    'margin-right': '10px',
                },
    )
    

def generate_table_2():

    return html.Div(
                id="Table2",
                children=[
                    html.Br(),
                    dcc.Loading(
                        id="loading-table2",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="table2",
                            children=[html.Div(id="table_plot2"),
                                    ]),
                            )
                    ],
                style={'width': '100%',
                    'display': 'inline-block',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    'margin-right': '10px',
                },
    )

def generate_table_3a():

    return html.Div(
                id="Table3a",
                children=[
                    html.Br(),
                    dcc.Loading(
                        id="loading-table3a",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="table3a",
                            children=[html.Div(id="table_plot3a"),
                                    ]),
                            ),
                    html.P("", id='tab3atxt'),
                    ],
                style={'width': '100%',
                    'display': 'inline-block',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    'margin-right': '10px',
                },
    )
    
def generate_table_3b():

    return html.Div(
                id="Table3b",
                children=[
                    html.Br(),
                    dcc.Loading(
                        id="loading-table3b",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="table3b",
                            children=[html.Div(id="table_plot3b"),
                                    ]),
                            ),
                    html.P("", id='tab3btxt'),
                    ],
                style={'width': '100%',
                    'display': 'inline-block',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    'margin-right': '10px',
                },
    )


def generate_table_4a():

    return html.Div(
                id="Table4a",
                children=[
                    html.Br(),
                    dcc.Loading(
                        id="loading-table4a",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="table4a",
                            children=[html.Div(id="table_plot4a"),
                                    ]),
                            ),
                    html.P("", id='tab4atxt'),
                    ],
                style={'width': '100%',
                    'display': 'inline-block',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    'margin-right': '10px',
                },
    )

def generate_table_4b():

    return html.Div(
                id="Table4b",
                children=[
                    html.Br(),
                    dcc.Loading(
                        id="loading-table4b",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="table4b",
                            children=[html.Div(id="table_plot4b"),
                                    ]),
                            ),
                    html.P("", id='tab4btxt'),
                    ],
                style={'width': '100%',
                    'display': 'inline-block',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
                    'margin-right': '10px',
                },
    )
    
#########################################################################################
################################# DASH APP LAYOUT #######################################
#########################################################################################


app.layout = html.Div([
    
    dcc.Store(id='main_df', storage_type='memory'),
    dcc.Store(id='df_models_logistic', storage_type='memory'),
    dcc.Store(id='df_equations', storage_type='memory'),
    dcc.Store(id='df1_summary_logistic', storage_type='memory'),
    
    html.Div(
        id='cat_vars',
        style={'display': 'none'}
        ),
    html.Div(
        id='di_numerical_vars',
        style={'display': 'none'}
        ),
    html.Div(
        id='placeholder1',
        style={'display': 'none'}
        ),
    html.Div(
        id='placeholder2',
        style={'display': 'none'}
        ),
    html.Div(
        id='placeholder3',
        style={'display': 'none'}
        ),
    html.Div(
        id='placeholder4',
        style={'display': 'none'}
        ),
    html.Div(
        id='placeholder5',
        style={'display': 'none'}
        ),
    html.Div(
        id='placeholder6',
        style={'display': 'none'}
        ),
    html.Div(
        id='placeholder7',
        style={'display': 'none'}
        ),
    html.Div(
        id='VARIABLES',
        style={'display': 'none'}
        ),
    html.Div(
        id='ITERATIONS',
        style={'display': 'none'}
        ),
    
    html.Div(
            style={'background-color': '#f9f9f9'},
            id="banner1",
            className="banner",
            children=[#html.Img(src=app.get_asset_url("RUSH_full_color.jpg"), 
                      #         style={'textAlign': 'left'}),
                        html.Img(src=app.get_asset_url("plotly_logo.png"), 
                               style={'textAlign': 'right'}),
                      ],
        ),
    
    html.Div(
            id="top-column1",
            className="ten columns",
            children=[description_card1()],
            style={'width': '95.3%',
                    'display': 'inline-block',
                    'border-radius': '15px',
                    'box-shadow': '1px 1px 1px grey',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
            },
        ),
    
    html.Div(
            id="left-column1",
            className="one columns",
            children=[control_card_upload()],
            style={'width': '24%',
                    'display': 'inline-block',
                    'border-radius': '15px',
                    'box-shadow': '1px 1px 1px grey',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
            },
        ),
    
    html.Div(
            id="right-column1",
            className="one columns",
            children=[
                html.Div(
                id="Data-Table1",
                children=[dcc.Loading(
                    id="data-table1",
                    type="default",
                    fullscreen=False,
                    children=html.Div(id="data_table1",
                        children=[html.H5("Data Table", style={'display': 'inline-block',
                                                                'width': '11.5%'},),
                                  html.I(className="fas fa-question-circle fa-lg", id="target_DataTable",
                                      style={'display': 'inline-block', 'width': '3%', 'color':'#99ccff'},
                                      ),
                                  dbc.Tooltip("This app tries to detect categorical variables. Example: A variable with values 'male' and 'female' will be converted two variables: 'sex:male' and 'sex:female'. The app will also detect which numerical variables only have values 0 and 1, and will allow them to be treated as categorical.", target="target_DataTable",
                                        style = {'font-size': 12},
                                        ),
                                  html.P("", id='rt4'),
                                  html.Hr(),
                                  html.Div(id='data_table_plot1'),
                                ]))],
                            ),
                ],
                style={'width': '69.3%',
                        'height': '328px',
                        'display': 'inline-block',
                        'border-radius': '15px',
                        'box-shadow': '1px 1px 1px grey',
                        'background-color': '#f0f0f0',
                        'padding': '10px',
                        'margin-bottom': '10px',
                    },
            ),
    
    html.Div(
            id="left-column2",
            className="two columns",
            children=[control_card1(),
                      generate_figure_1(),
                      generate_table_1()],
            style={'width': '95.3%',
                    'display': 'inline-block',
                    'border-radius': '15px',
                    'box-shadow': '1px 1px 1px grey',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
            },
        ),
    html.Div(
        id="left-column3",
        className="two columns",
        children=[control_card2(),
                  generate_figure_2(),
                  #generate_table_2b(),
                  ],
        style={'width': '95.3%',
                'display': 'inline-block',
                'border-radius': '15px',
                'box-shadow': '1px 1px 1px grey',
                'background-color': '#f0f0f0',
                'padding': '10px',
                'margin-bottom': '10px',
        },
    ),
    
    html.Div(
        id="left-column4",
        className="two columns",
        children=[control_card3(),
                  generate_figure_3(),
                  generate_table_3a(),
                  generate_table_3b(),
                  ],
        style={'width': '95.3%',
                'display': 'inline-block',
                'border-radius': '15px',
                'box-shadow': '1px 1px 1px grey',
                'background-color': '#f0f0f0',
                'padding': '10px',
                'margin-bottom': '10px',
        },
    ),
    
    html.Div(
        id="left-column5",
        className="two columns",
        children=[control_card4(),
                  generate_figure_4a(),
                  generate_figure_4b(),
                  generate_table_4a(),
                  generate_table_4b(),
                  ],
        style={'width': '95.3%',
                'display': 'inline-block', #'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw',
                'border-radius': '15px',
                'box-shadow': '1px 1px 1px grey',
                'background-color': '#f0f0f0',
                'padding': '10px',
                'margin-bottom': '10px',
        },
    ),
    
    html.Div(
            id="bottom-column1",
            className="ten columns",
            children=[description_card_final()],
            style={'width': '95.3%',
                    'display': 'inline-block',
                    'border-radius': '15px',
                    'box-shadow': '1px 1px 1px grey',
                    'background-color': '#f0f0f0',
                    'padding': '10px',
                    'margin-bottom': '10px',
            },
        ),

])





#########################################################################################
############################    Callbacks   #############################################
#########################################################################################


@app.callback([Output('main_df', 'data'),
               Output('cat_vars', 'children'),
               Output('di_numerical_vars', 'children'),
               Output('rt4', 'children')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')],
            )
def update_output1(list_of_contents, list_of_names, list_of_dates):

    error_string = "Error: Your file was not processed. Only upload csv or Excel files. Ensure there are only rows, columns, and column headers, and that your file contains data. Excel files must only have one sheet and no special formatting (frozen panes, etc.)."
    if list_of_contents is None or list_of_names is None or list_of_dates is None:
        return None, None, None, ""
    
    if list_of_contents is not None:
        children = 0
        df = 0
        try:
            children = [parse_contents(c, n, d) for c, n, d in zip([list_of_contents], [list_of_names], [list_of_dates])]
        except:
            return None, None, None, error_string
        
        try:
            df = children[0]
        except:
            return None, None, None, error_string
            
        try:
            df = pd.read_json(df)
        except:
            return None, None, None, error_string
            
        if df.shape[0] < 2 or df.shape[1] < 2:
            return None, None, None, error_string
        
        df.columns = df.columns.str.strip()
        df.dropna(how='all', axis=1, inplace=True)
        df.dropna(how='all', axis=0, inplace=True)
        
        if df.shape[0] > 10000:
            #df = df.sample(n = 10000, axis=0, replace=False)
            df = df.sample(n = 10000, axis=0, replace=False, random_state=0)
            
        df.dropna(how='all', axis=1, inplace=True)
        df.dropna(how='all', axis=0, inplace=True)
        df = df.loc[:, df.nunique() != 1]
        
        #dwb_col = df.columns.str.replace('\s+', '_')
        #df.columns = dwb_col
        
        df = df.replace(',','', regex=True)
        
        cat_vars = []
        dichotomous_numerical_vars = []
        variables = list(df)
        for i in variables:
            ls = df[i].tolist()
            if all(isinstance(item, str) for item in ls) is True:
                cat_vars.append(i)
            
            else:
                df[i] = pd.to_numeric(df[i], errors='coerce')
                
            if len(list(set(ls))) == 2 and all(isinstance(item, str) for item in ls) is False:
                dichotomous_numerical_vars.append(i)

        df.dropna(how='all', axis=0, inplace=True)
        #df = df._get_numeric_data()
        #df = df.loc[(df.sum(axis=1) != 0), (df.sum(axis=0) != 0)]
        
        return df.to_json(), cat_vars, dichotomous_numerical_vars, ""
    


@app.callback(Output('data_table_plot1', 'children'),
              [Input('main_df', 'data')],
            )
def update_data_table1(main_df):
        
    if main_df is None:
        cols = ['variable 1', 'variable 2', 'variable 3']
        df_table = pd.DataFrame(columns=cols)
        df_table['variable 1'] = [np.nan]*7
        df_table['variable 2'] = [np.nan]*7
        df_table['variable 3'] = [np.nan]*7
        
        dashT = dash_table.DataTable(
            columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in df_table.columns],
            data = df_table.to_dict('records'),
            editable=False,
            #filter_action="native",
            sort_action="native",
            sort_mode="multi",
            #column_selectable="single",
            #row_selectable="multi",
            row_deletable=False,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            #page_action='none',
            page_current= 0,
            page_size= 5,
            style_table={'overflowX': 'scroll',
                         #'overflowY': 'auto',
                         #'height': '415px',
                         },
            style_cell={'textOverflow': 'auto',
                        'textAlign': 'left',
                        'minWidth': '140px',
                        'width': '140px',
                        'maxWidth': '220px',
                        },
        )
        return dashT
        
    else:
        main_df = pd.read_json(main_df)
        
        dashT = dash_table.DataTable(
            columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in main_df.columns],
            data = main_df.to_dict('records'),
            editable=False,
            #filter_action="native",
            sort_action="native",
            sort_mode="multi",
            #column_selectable="single",
            #row_selectable="multi",
            row_deletable=False,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            #page_action='none',
            page_current= 0,
            page_size= 6,
            style_table={'overflowX': 'scroll',
                         #'overflowY': 'auto',
                         #'height': '415px',
                         },
            style_cell={
                'height': 'auto',
                'textAlign': 'left',
                'minWidth': '140px', #'width': '140px',# 'maxWidth': '220px',
                #'whiteSpace': 'normal',
            }
        )
        
        return dashT
        
        


@app.callback([Output('var-select1', 'options'),
               Output('var-select1', 'value')],
              [Input('main_df', 'data'),
               Input('cat_vars', 'children')],
            )
def update_output2(df, cat_vars):
    try:
        df = pd.read_json(df)
        ls = sorted(list(set(list(df))))
        options = [{"label": i, "value": i} for i in ls]
        ls = [ls[0]]
        return options, ls
    
    except:
        return [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], ['Nothing uploaded']


@app.callback([Output('var-select2', 'options'),
               Output('var-select2', 'value')],
              [Input('main_df', 'data'),
               Input('cat_vars', 'children')],
            )
def update_output3(df, cat_vars):
    
    try:
        df = pd.read_json(df)
        ls = sorted(list(set(list(df))))
        options = [{"label": i, "value": i} for i in ls]
        ls = [ls[0]]
        return options, ls
    
    except:
        return [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], ['Nothing uploaded']


@app.callback([Output('xvar', 'options'),
               Output('xvar', 'value'),
               Output('yvar', 'options'),
               Output('yvar', 'value')],
              [Input('main_df', 'data'),
               Input('cat_vars', 'children')],
            )
def update_select_vars1(df, cat_vars):
    try:
        df = pd.read_json(df)
        df.drop(labels=cat_vars, axis=1, inplace=True)
        ls = sorted(list(set(list(df))))
        options = [{"label": i, "value": i} for i in ls]
        return options, ls, options, ls
    
    except:
        return [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], ['Nothing uploaded'], [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], ['Nothing uploaded']



@app.callback([Output('xvar2', 'options'),
               Output('xvar2', 'value'),
               Output('yvar2', 'options'),
               Output('yvar2', 'value')],
              [Input('main_df', 'data'),
               Input('cat_vars', 'children')],
            )
def update_select_vars2(df, cat_vars):
    try:
        df = pd.read_json(df)
        df.drop(labels=cat_vars, axis=1, inplace=True)
        ls = sorted(list(set(list(df))))
        options = [{"label": i, "value": i} for i in ls]
        return options, ls, options, ls
    
    except:
        return [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], ['Nothing uploaded'], [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], ['Nothing uploaded']
        


@app.callback([Output('xvar3', 'options'),
               Output('xvar3', 'value'),
               Output('yvar3', 'options'),
               Output('yvar3', 'value')],
              [Input('main_df', 'data'),
               Input('cat_vars', 'children')],
            )
def update_select_vars3(df, cat_vars):
    # variables for multiple linear regression 
    try:
        df = pd.read_json(df)
        ls1 = sorted(list(set(list(df))))
        options1 = [{"label": i, "value": i} for i in ls1]
        df.drop(labels=cat_vars, axis=1, inplace=True)
                
        ls2 = sorted(list(set(list(df))))
        options2 = [{"label": i, "value": i} for i in ls2]
        
        return options1, ls1, options2, ls2
        
    except:
        return [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], ['Nothing uploaded'], [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], ['Nothing uploaded']
        
        
@app.callback([Output('xvar_logistic', 'options'),
                Output('xvar_logistic', 'value'),
                Output('yvar_logistic', 'options'),
                Output('yvar_logistic', 'value')],
                [Input('main_df', 'data'),
                Input('cat_vars', 'children'),
                Input('di_numerical_vars', 'children')],
            )
def update_select_vars4(df, cat_vars, di_num_vars):
    # variables for multiple logistic regression 
    try:
        df = pd.read_json(df)
        tdf = df.copy(deep=True)
        print('len(cat_vars):', len(cat_vars))
        
        #tdf = df.drop(labels=cat_vars, axis=1, inplace=False)
        ls1 = sorted(list(set(list(tdf))))
        options1 = [{"label": i, "value": i} for i in ls1]
        
        tdf = df.filter(items=cat_vars + di_num_vars, axis=1)
        tdf, dropped, cat_vars_ls = dummify(tdf, cat_vars, False)
        ls2 = sorted(list(set(list(tdf))))
        options2 = [{"label": i, "value": i} for i in ls2]
                
        return options1, ls1, options2, ls2

    except:
        return [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], ['Nothing uploaded'], [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], ['Nothing uploaded']
  


@app.callback([Output('figure_plot1', 'figure'),
               Output('table_plot1', 'children'),
               Output('rt0', 'children'),
               Output('fig1txt', 'children')],
               [Input('btn1', 'n_clicks')],
               [State('main_df', 'data'),
                State('cat_vars', 'children'),
                State('xvar', 'value'),
                State('yvar', 'value')],
            )
def run_simple_regressions(n_clicks, df, cat_vars, xvars, yvars):
    
    if df is None or xvars is None or yvars is None or len(xvars) == 0 or len(yvars) == 0:
            return {}, {}, "", ""
    elif len(xvars) == 1 and len(yvars) == 1 and xvars[0] == yvars[0]:
            return {}, {}, "Error: Your x-variable and y-variable cannot be the same.", ""
    
    else:
        df = pd.read_json(df)
        df.drop(labels=cat_vars, axis=1, inplace=True)
        vars_ = xvars + yvars
        vars_ = list(set(vars_))
        df = df.filter(items=vars_, axis=1)
        
        if df.shape[0] == 0:
            return {}, {}, "Error: There are no rows in the data because of the variables you chose.", ""
            
        else:
            models = []
            coefs = []
            eqns = []
            r2s = []
            #slopes = []
            intercepts = []
            pvals = []
            #rvals = []
            aics = []
            ns = []
            Yvars = []
            Xvars = []
            llf_ls = []
            Xs = []
            Ys = []
            PredY = []
            
            for yvar in yvars:
                for xvar in xvars:
                    
                    if xvar == yvar:
                        continue
                        
                    tdf = df.filter(items=[yvar, xvar], axis=1)
                    tdf.replace([np.inf, -np.inf], np.nan, inplace=True)
                    tdf.dropna(how='any', inplace=True)
                        
                    y_o = tdf[yvar].values.tolist()
                    x_o = tdf[xvar].values.tolist()
                    x_o, y_o = zip(*sorted(zip(x_o, y_o)))
                    
                    x_o = np.array(x_o)
                    y_o = np.array(y_o)
                    
                    #Create single dimension
                    x = x_o[:, np.newaxis]
                    y = y_o[:, np.newaxis]

                    inds = x.ravel().argsort()  # Sort x values and get index
                    x = x.ravel()[inds].reshape(-1, 1)
                    y = y[inds] #Sort y according to x sorted index
                    
                    for model_type in ['linear', 'quadratic', 'cubic']:
                        d = int()
                        if model_type == 'linear': d = 1
                        elif model_type == 'quadratic': d = 2
                        elif model_type == 'cubic': d = 3
                            
                        polynomial_features = PolynomialFeatures(degree = d)
                        xp = polynomial_features.fit_transform(x)
                            
                        model = sm.OLS(y, xp).fit()
                        ypred = model.predict(xp)
                        ypred = ypred.tolist()
                        
                        intercepts.append(model.params[0])
                        coefs.append(model.params[1:])
                        
                        poly_coefs = model.params[1:].tolist()
                        poly_coefs.reverse()
                        
                        poly_exponents = list(range(1, len(poly_coefs)+1))
                        poly_exponents.reverse()
                        
                        eqn = 'y = '
                        for i, p in enumerate(poly_coefs):
                            exp = poly_exponents[i]
                            
                            if exp == 1:
                                exp = 'x'
                            elif exp == 2:
                                exp = 'x²'
                            elif exp == 3:
                                exp = 'x³'
                            
                            if i == 0:
                                p = myround(p)
                                eqn = eqn + str(p) + exp
                                
                            else:
                                if p >= 0:
                                    p = myround(p)
                                    eqn = eqn + ' + ' + str(p) + exp
                                else:
                                    p = myround(p)
                                    eqn = eqn + ' - ' + str(np.abs(p)) + exp
                        
                        b = model.params[0]
                        if b >= 0:
                            b = myround(b)
                            eqn = eqn + ' + ' + str(b)
                        else:
                            b = myround(b)
                            eqn = eqn + ' - ' + str(np.abs(b))
                            
                        eqns.append(eqn)
                        
                        r2 = model.rsquared_adj
                        if r2 < 0:
                            r2 = 0
                        r2_adj = model.rsquared_adj
                        if r2_adj < 0:
                            r2_adj = 0
                            
                        aic = model.aic
                        #bic = model.bic
                        fp = model.f_pvalue
                        llf = model.llf
                        
                        Yvars.append(yvar)
                        Xvars.append(xvar)
                        models.append(model_type)
                        r2s.append(np.round(r2_adj, 3))
                        pvals.append(np.round(fp, 3))
                        aics.append(np.round(aic, 3))
                        llf_ls.append(np.round(llf, 5))
                        ns.append(len(x))
                        Xs.append(x_o)
                        Ys.append(y_o)
                        PredY.append(ypred)
            
            del df
            cols = ['y', 'x', 'Model', 'r-square', 'p-value', 'intercept', 'coefficients', 'AIC', 'log-likelihood']
            df_models = pd.DataFrame(columns=cols)
            df_models['y'] = Yvars
            df_models['x'] = Xvars
            df_models['Model'] = models
            df_models['r-square'] = r2s
            df_models['p-value'] = pvals
            df_models['AIC'] = aics
            df_models['log-likelihood'] = llf_ls
            df_models['ys'] = Ys
            df_models['xs'] = Xs
            df_models['pred_y'] = PredY
            df_models['intercept'] = intercepts
            df_models['coefficients'] = coefs
            df_models['equation'] = eqns
            df_models['n'] = ns
            
            #df_models = df_models.replace('_', ' ', regex=True)
            #for col in list(df_models):
            #    col2 = col.replace("_", " ")
            #    df_models.rename(columns={col: col2})
            #df_models.reset_index(drop=True, inplace=True)
            
            
            ###################### Figure ########################
            fig_data = []
                
            df_models['label'] = df_models['y'] + ' vs ' + df_models['x']
            tdf = df_models[df_models['Model'] == 'cubic']
            tdf.sort_values(by='r-square', inplace=True, ascending=False)
            tdf = tdf.head(10)
            labels = tdf['label'].tolist()
            del tdf

            used = []
            models = ['linear', 'quadratic', 'cubic']
            for i, label in enumerate(labels):
                
                if label in used:
                    continue
                used.append(label)
                    
                clr = "#" + "%06x" % random.randint(0, 0xFFFFFF)
                    
                for model in models:
                    tdf = df_models[(df_models['label'] == label) & (df_models['Model'] == model)]
                    obs_y = tdf['ys'].tolist()
                    obs_y = obs_y[0]
                    obs_x = tdf['xs'].tolist()
                    obs_x = obs_x[0]
                    pred_y = tdf['pred_y'].tolist()
                    pred_y = pred_y[0]
                    r2 = tdf['r-square'].tolist()
                    r2 = r2[0]
                
                    obs_x, obs_y, pred_y = zip(*sorted(zip(obs_x, obs_y, pred_y)))

                    if model == 'linear':
                        fig_data.append(
                            go.Scatter(
                                x = obs_x,
                                y = obs_y,
                                mode = "markers",
                                name = label,
                                opacity = 0.75,
                                marker = dict(size=10,
                                            color=clr)
                            )
                        )
                        
                    line_dict = dict(color = clr, width = 2)
                    if model == 'quadratic':
                        line_dict = dict(color = clr, width = 2, dash='dash')
                    elif model == 'cubic':
                        line_dict = dict(color = clr, width = 2, dash='dot')
                            
                    fig_data.append(
                        go.Scatter(
                            x = obs_x,
                            y = pred_y,
                            mode = "lines",
                            name = model + ': r2 = <sup>'+str(np.round(r2, 3))+'</sup>',
                            opacity = 0.75,
                            line = line_dict,
                        )
                    )
                            
            figure = go.Figure(
                data = fig_data,
                layout = go.Layout(
                    xaxis = dict(
                        title = dict(
                            text = "<b>X</b>",
                            font = dict(
                                family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                                " Helvetica, Arial, sans-serif",
                                size = 18,
                            ),
                        ),
                        #rangemode="tozero",
                        #zeroline=True,
                        showticklabels = True,
                    ),
                                
                    yaxis = dict(
                        title = dict(
                            text = "<b>Y</b>",
                            font = dict(
                                family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                                " Helvetica, Arial, sans-serif",
                                size = 18,
                                        
                            ),
                        ),
                        #rangemode="tozero",
                        #zeroline=True,
                        showticklabels = True,
                    ),
                                
                    margin = dict(l=60, r=30, b=10, t=40),
                    showlegend = True,
                    height = 400,
                    paper_bgcolor = "rgb(245, 247, 249)",
                    plot_bgcolor = "rgb(245, 247, 249)",
                ),
            )
            
            ################## Table ######################
            
            df_models = df_models[df_models['x'].isin(xvars)]
            df_models = df_models[df_models['y'].isin(yvars)]
            df_table = df_models.filter(items=['y', 'x', 'Model', 'r-square', 'p-value', 'n', 'equation'])
            df_table.sort_values(by='r-square', inplace=True, ascending=False)
            
            dashT = dash_table.DataTable(
                columns=[{"name": i, "id": i, "deletable": False, "selectable": False} for i in df_table.columns],
                data = df_table.to_dict('records'),
                editable=False,
                #filter_action="native",
                sort_action="native",
                sort_mode="multi",
                #column_selectable="single",
                #row_selectable="multi",
                row_deletable=False,
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                #page_action='none',
                page_current= 0,
                page_size= 6,
                style_table={'overflowX': 'scroll',
                             #'overflowY': 'auto',
                             #'height': '415px',
                             },
                style_cell={
                    'height': 'auto',
                    'textAlign': 'left',
                    'minWidth': '140px', #'width': '140px',# 'maxWidth': '220px',
                    'whiteSpace': 'normal',
                }
            )
    
            del df_models
            #del df_table
            
            txt = "Linear models are based on linear regression. Nonlinear regression is more appropriate when the relationship between x and y is curved. Quadratic models account for 1 curve. Cubic models account for 2 curves. When interpreting the r-squares, ask whether a curvier model results in meaningfully greater improvement. If not, the simpler model will be less likely to over-fit the data."
            return figure, dashT, "", txt
            
    

    
@app.callback([Output('figure_plot2', 'figure'),
                Output('rt3', 'children'),
                Output('fig2txt', 'children')],
                [Input('btn2', 'n_clicks')],
                [State('xvar2', 'value'),
                 State('yvar2', 'value'),
                 State('x_transform', 'value'),
                 State('y_transform', 'value'),
                 State('model2', 'value'),
                 State('main_df', 'data')],
            )
def update_figure2(n_clicks, xvar, yvar, x_transform, y_transform, model, df):
        
        if df is None or xvar is None or yvar is None or xvar == yvar or isinstance(yvar, list) is True or isinstance(yvar, list) is True:
            
            if df is None:
                return {}, "", ""
            
            elif (isinstance(xvar, list) is True or xvar is None) & (isinstance(yvar, list) is True or yvar is None):
                return {}, "Error: You need to select some variables.", ""
            
            elif isinstance(yvar, list) is True or yvar is None:
                return {}, "Error: You need to select a y-variable.", ""
            
            elif isinstance(xvar, list) is True or xvar is None:
                return {}, "Error: You need to select an x-variable.", ""
            
            elif xvar == yvar and xvar is not None:
                return {}, "Error: Your x-variable and y-variable are the same. Ensure they are different.", ""
            else:
                return {}, "", ""
            
        else:
            df = pd.read_json(df)
            df = df.filter(items=[xvar, yvar], axis=1)
            
            if x_transform == 'log10':
                df[xvar] = np.log10(df[xvar])
            elif x_transform == 'square root':
                df[xvar] = np.sqrt(df[xvar])
            
            if y_transform == 'log10':
                df[yvar] = np.log10(df[yvar])
            elif y_transform == 'square root':
                df[yvar] = np.sqrt(df[yvar])
                
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(how='any', inplace=True)
                
            y_o = df[yvar].values.tolist()
            x_o = df[xvar].values.tolist()
            x_o, y_o = zip(*sorted(zip(x_o, y_o)))
            
            x_o = np.array(x_o)
            y_o = np.array(y_o)
            
            #Create single dimension
            x = x_o[:, np.newaxis]
            y = y_o[:, np.newaxis]

            inds = x.ravel().argsort()  # Sort x values and get index
            x = x.ravel()[inds].reshape(-1, 1)
            y = y[inds] #Sort y according to x sorted index
            
            d = int()
            if model == 'linear': d = 1
            elif model == 'quadratic': d = 2
            elif model == 'cubic': d = 3
            
            polynomial_features = PolynomialFeatures(degree = d)
            xp = polynomial_features.fit_transform(x)
                
            model = sm.OLS(y, xp).fit()
            ypred = model.predict(xp)
            ypred = ypred.tolist()
            
            poly_coefs = model.params[1:].tolist()
            poly_coefs.reverse()
            
            poly_exponents = list(range(1, len(poly_coefs)+1))
            poly_exponents.reverse()
            
            eqn = 'y = '
            for i, p in enumerate(poly_coefs):
                exp = poly_exponents[i]
                
                if exp == 1:
                    exp = 'x'
                elif exp == 2:
                    exp = 'x²'
                elif exp == 3:
                    exp = 'x³'
                
                if i == 0:
                    p = myround(p)
                    eqn = eqn + str(p) + exp
                    
                else:
                    if p >= 0:
                        p = myround(p)
                        eqn = eqn + ' + ' + str(p) + exp
                    else:
                        p = myround(p)
                        eqn = eqn + ' - ' + str(np.abs(p)) + exp
            
            b = model.params[0]
            if b >= 0:
                b = myround(b)
                eqn = eqn + ' + ' + str(b)
            else:
                b = myround(b)
                eqn = eqn + ' - ' + str(np.abs(b))
                
            r2 = model.rsquared_adj
            r2_adj = model.rsquared_adj
            aic = model.aic
            bic = model.bic
            fp = model.f_pvalue
            llf = model.llf
            
            st, data, ss2 = summary_table(model, alpha=0.05)
            fittedvalues = data[:, 2]
            predict_mean_se  = data[:, 3]
            predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
            predict_ci_low, predict_ci_upp = data[:, 6:8].T
            
            outlier_y = []
            outlier_x = []
            nonoutlier_y = []
            nonoutlier_x = []
            for i, yi in enumerate(y_o):
                if yi > predict_ci_upp[i] or yi < predict_ci_low[i]:
                    outlier_y.append(yi)
                    outlier_x.append(x_o[i])
                else:
                    nonoutlier_y.append(yi)
                    nonoutlier_x.append(x_o[i])
                    
            fig_data = []
            
            clr = "#3399ff"
            #x, y, ypred = zip(*sorted(zip(x, y, ypred)))
            
            fig_data.append(go.Scatter(
                                x = nonoutlier_x,
                                y = nonoutlier_y,
                                name = 'Non-outliers',
                                mode = "markers",
                                opacity = 0.75,
                                marker = dict(size=10,
                                            color=clr)
                            )
                        )
                        
            fig_data.append(go.Scatter(
                    x = outlier_x,
                    y = outlier_y,
                    name = 'Outliers',
                    mode = "markers",
                    opacity = 0.75,
                    marker = dict(size=10,
                                color="#ff0000")
                )
            )
            
            fig_data.append(
                        go.Scatter(
                            x = x_o,
                            y = ypred,
                            mode = "lines",
                            name = 'fitted: r2 = <sup>'+str(np.round(r2, 3))+'</sup>',
                            opacity = 0.75,
                            line = dict(color = clr, width = 2),
                        )
                    )
            
            fig_data.append(
                go.Scatter(
                    x = x_o,
                    y = predict_mean_ci_upp,
                    mode = "lines",
                    name = 'upper 95 CI',
                    opacity = 0.75,
                    line = dict(color = clr, width = 2, dash='dash'),
                )
            )
            
            fig_data.append(
                go.Scatter(
                    x = x_o,
                    y = predict_mean_ci_low,
                    mode = "lines",
                    name = 'lower 95 CI',
                    opacity = 0.75,
                    line = dict(color = clr, width = 2, dash='dash'),
                )
            )
            
            fig_data.append(
                go.Scatter(
                    x = x_o,
                    y = predict_ci_upp,
                    mode = "lines",
                    name = 'upper 95 PI',
                    opacity = 0.75,
                    line = dict(color = clr, width = 2, dash='dot'),
                )
            )
            
            fig_data.append(
                go.Scatter(
                    x = x_o,
                    y = predict_ci_low,
                    mode = "lines",
                    name = 'lower 95 PI',
                    opacity = 0.75,
                    line = dict(color = clr, width = 2, dash='dot'),
                )
            )

            if x_transform == 'log10':
                xvar = 'log<sub>10</sub>(' + xvar + ')'
            if x_transform == 'square root':
                xvar = 'square root of ' + xvar
                
            if y_transform == 'log10':
                yvar = 'log<sub>10</sub>(' + yvar + ')'
            if y_transform == 'square root':
                yvar = 'square root of ' + yvar
                
            figure = go.Figure(
                data = fig_data,
                layout = go.Layout(
                    xaxis = dict(
                        title = dict(
                            text = "<b>" + xvar + "</b>",
                            font = dict(
                                family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                                " Helvetica, Arial, sans-serif",
                                size = 18,
                            ),
                        ),
                        #rangemode="tozero",
                        #zeroline=True,
                        showticklabels = True,
                    ),
                                
                    yaxis = dict(
                        title = dict(
                            text = "<b>" + yvar + "</b>",
                            font = dict(
                                family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                                " Helvetica, Arial, sans-serif",
                                size = 18,
                                        
                            ),
                        ),
                        #rangemode="tozero",
                        #zeroline=True,
                        showticklabels = True,
                    ),
                                
                    margin = dict(l=60, r=30, b=10, t=40),
                    showlegend = True,
                    height = 400,
                    paper_bgcolor = "rgb(245, 247, 249)",
                    plot_bgcolor = "rgb(245, 247, 249)",
                ),
            )

            txt = "Confidence intervals (CI) pertain to the data, reflecting confidence in the average y-value across the x-axis. Prediction intervals (PI) pertain to the model. Points outside the prediction interval are unlikely to be explained by the model, and are labeled outliers."
            return figure, "", txt



@app.callback([Output('figure_plot3', 'figure'),
               Output('table_plot3a', 'children'),
               Output('table_plot3b', 'children'),
               Output('rt1', 'children'),
               Output('placeholder5', 'children'),
               Output('fig3txt', 'children'),
               Output('tab3atxt', 'children'),
               Output('tab3btxt', 'children')],
              [Input('btn3', 'n_clicks')],
              [State('xvar3', 'value'),
               State('yvar3', 'value'),
               State('main_df', 'data'),
               State('cat_vars', 'children'),
               State('rfecv', 'value')],
        )
def update_figure3_table3(n_clicks, xvars, yvar, df, cat_vars, rfe_val):
    
                        
    cols = ['Model information', 'Model statistics']
    df_table1 = pd.DataFrame(columns=cols)
    df_table1['Model information'] = [np.nan]*10
    df_table1['Model statistics'] = [np.nan]*10
    
    dashT1 = dash_table.DataTable(
        columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in df_table1.columns],
        data = df_table1.to_dict('records'),
        editable=False,
        #filter_action="native",
        sort_action="native",
        sort_mode="multi",
        #column_selectable="single",
        #row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        #page_action='none',
        page_current= 0,
        page_size= 6,
        style_table={'overflowX': 'scroll',
                     #'overflowY': 'auto',
                     #'height': '415px',
                     },
        style_cell={'textOverflow': 'auto',
                    'textAlign': 'left',
                    'minWidth': '140px',
                    'width': '140px',
                    'maxWidth': '220px',
                    },
    )
    
    cols = ['Parameter', 'coef', 'std err', 'z', 'P>|z|', '[0.025]', '[0.975]', 'VIF']
    df_table2 = pd.DataFrame(columns=cols)
    df_table2['Parameter'] = [np.nan]*10
    df_table2['coef'] = [np.nan]*10
    df_table2['std err'] = [np.nan]*10
    df_table2['z'] = [np.nan]*10
    df_table2['P>|z|'] = [np.nan]*10
    df_table2['[0.025]'] = [np.nan]*10
    df_table2['VIF'] = [np.nan]*10
    
    dashT2 = dash_table.DataTable(
        columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in df_table2.columns],
        data = df_table2.to_dict('records'),
        editable=False,
        #filter_action="native",
        sort_action="native",
        sort_mode="multi",
        #column_selectable="single",
        #row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        #page_action='none',
        page_current= 0,
        page_size= 6,
        style_table={'overflowX': 'scroll',
                     #'overflowY': 'auto',
                     #'height': '415px',
                     },
        style_cell={'textOverflow': 'auto',
                    'textAlign': 'left',
                    'minWidth': '140px',
                    'width': '140px',
                    'maxWidth': '220px',
                    },
    )
    
    if df is None:
        return {}, dashT1, dashT2, "", [0], "", "", ""
    
    elif yvar is None and xvars is None:
        return {}, dashT1, dashT2, "", [0], ""
    
    elif xvars is None or len(xvars) < 2:
        return {}, dashT1, dashT2, "Error: Select two or more x-variables", [0], "", "", ""
        
    elif yvar is None:
        return {}, dashT1, dashT2, "Error: Select a y-variable", [0], "", "", ""
    
    elif (isinstance(yvar, list) is True) & (xvars is None or len(xvars) < 2):
        return {}, dashT1, dashT2, "Error: Select a y-variable and 2 or more x-variables", [0], "", "", ""
    
    elif isinstance(yvar, list) is True:
        return {}, dashT1, dashT2, "Error: Select a y-variable", [0], "", "", ""
    
    elif xvars is None or len(xvars) < 2:
        return {}, dashT1, dashT2, "Error: Select two or more x-variables", [0], "", "", ""
    
    df = pd.read_json(df)
    
    if yvar not in list(df):
        return {}, dashT1, dashT2, "Error: Choose a y-variable", [0], "", "", ""
        
    if yvar in xvars:
        xvars.remove(yvar)
        if len(xvars) == 0:
            return {}, dashT1, dashT2, "Error: Multiple regression requires 2 or more x-variables. You chose one and it's the same as your y-variable", [0], "", "", ""
        elif len(xvars) == 1:
            return {}, dashT1, dashT2, "Error: Multiple regression requires 2 or more x-variables. You chose two but one is the same as your y-variable", [0], "", "", ""
    
    if len(xvars) < 2 and yvar is None:
        return {}, dashT1, dashT2, "Error: Multiple regression requires 2 or more x-variables and a y-variable.", [0], "", "", ""
        
    elif len(xvars) < 2:
        return {}, dashT1, dashT2, "Error: Multiple regression requires 2 or more x-variables.", [0], "", "", ""
                        
    else:
            
        vars = [yvar] + xvars
        df = df.filter(items=vars, axis=1)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        #df.dropna(how='any', inplace=True)
                            
        #Conduct multiple regression
        y_train, y_pred, df1_summary, df2_summary, supported_features, unsupported, colors = run_MLR(df, xvars, yvar, cat_vars, rfe_val)
        
        if len(y_train) == 0:
            rt1 = "Error: Your regression could not run. Your y-values contain no data."
            return {}, dashT1, dashT2, rt1, [0], "", "", ""
            
        y_train = y_train.tolist()
        y_pred = y_pred.tolist()
        
        y_pred_outliers = []
        y_pred_nonoutliers = []
        y_train_outliers = []
        y_train_nonoutliers = []
        
        for i, clr in enumerate(colors):
            if clr == "#ff0000":
                y_pred_outliers.append(y_pred[i])
                y_train_outliers.append(y_train[i])
            elif clr == "#3399ff":
                y_pred_nonoutliers.append(y_pred[i])
                y_train_nonoutliers.append(y_train[i])
        
        fig_data = []
        
        miny = min([min(y_train), min(y_pred)])
        maxy = max([max(y_train), max(y_pred)])
        
        fig_data.append(go.Scatter(x = y_pred_nonoutliers, y = y_train_nonoutliers, name = 'Non-outliers',
                mode = "markers", opacity = 0.75, marker = dict(size=10, color="#3399ff")))
        
        fig_data.append(go.Scatter(x = y_pred_outliers, y = y_train_outliers, name = 'Outliers',
                mode = "markers", opacity = 0.75, marker = dict(size=10, color="#ff0000")))
        
        fig_data.append(go.Scatter(x = [miny, maxy], y = [miny, maxy], name = '1:1',
            mode = "lines", opacity = 0.75, line = dict(color = "#595959", width = 1, dash='dash'),))
                            
        figure = go.Figure(data = fig_data,
            layout = go.Layout(
                xaxis = dict(title = dict(
                        text = "<b>" + 'Predicted ' + yvar + "</b>",
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif", size = 18,),), showticklabels = True,),
                                            
                yaxis = dict(title = dict(
                        text = "<b>" + yvar + "</b>",
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif",size = 18,),),showticklabels = True,),
                                            
                margin = dict(l=60, r=30, b=10, t=40), showlegend = True, height = 400,
                paper_bgcolor = "rgb(245, 247, 249)", plot_bgcolor = "rgb(245, 247, 249)",
            ),
        )

        dashT1 = dash_table.DataTable(
            columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in df1_summary.columns],
            data = df1_summary.to_dict('records'),
            editable=False,
            #filter_action="native",
            sort_action="native",
            sort_mode="multi",
            #column_selectable="single",
            #row_selectable="multi",
            row_deletable=False,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            #page_action='none',
            page_current= 0,
            page_size= 10,
            style_table={'overflowX': 'scroll',
                         #'overflowY': 'auto',
                         #'height': '415px',
                         },
            style_cell={
                'height': 'auto',
                'textAlign': 'left',
                'minWidth': '140px', #'width': '140px',# 'maxWidth': '220px',
                #'whiteSpace': 'normal',
            }
        )
        del df
        #del df1_summary
        
        df2_summary['index'] = df2_summary.index
        df2_summary = df2_summary.astype(str)
        col_names = list(df2_summary)
        
        df3 = pd.DataFrame(columns=col_names)
        df3[col_names[0]] = [col_names[0]] + df2_summary[col_names[0]].tolist()
        df3[col_names[1]] = [col_names[1]] + df2_summary[col_names[1]].tolist()
        df3[col_names[2]] = [col_names[2]] + df2_summary[col_names[2]].tolist()
        df3[col_names[3]] = [col_names[3]] + df2_summary[col_names[3]].tolist()
        c1 = df3[col_names[3]] + ' ' + df3[col_names[0]]
        c2 = df3[col_names[1]] + ' ' + df3[col_names[2]]
        #del df2
        
        c1 = c1.tolist()
        c1.remove(c1[0])
        c1.remove(c1[2])
        c1.remove(c1[2])
        c2 = c2.tolist()
        c2.remove(c2[-1])
        c2.remove(c2[-1])
        c2.remove(c2[-1])
        
        df4 = pd.DataFrame(columns=['Model information', 'Model statistics'])
        df4['Model information'] = c1
        df4['Model statistics'] = c2
        del df2_summary
        
        
        dashT2 = dash_table.DataTable(
            columns=[{"name": i, "id": i, "deletable": False, "selectable": False} for i in df4.columns],
            data = df4.to_dict('records'),
            editable=False,
            #filter_action="native",
            #sort_action="native",
            #sort_action=None,
            #sort_mode="multi",
            #column_selectable="single",
            #row_selectable="multi",
            #row_deletable=False,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            #page_action='none',
            page_current= 0,
            page_size= 10,
            style_table={'overflowX': 'scroll',
                         #'overflowY': 'auto',
                         #'height': '415px',
                         },
            style_cell={
                'height': 'auto',
                'textAlign': 'left',
                #'minWidth': '200px', #'width': '140px',# 'maxWidth': '220px',
                #'whiteSpace': 'normal',
            }
        )
        
        txt1 = "This plot allows you to interpret whether the model tended to over-predict or under-predict values. If a point is above the 1:1 line, then the observed value is greater than the predicted. If the relationship is curved and performance is weak, then try rescaling some of your variables (using log, square root, etc.) and/or centering them (using z-scores)."
        
        txt2 = "Adjusted R-square accounts for sample size and the number of x-variables used."
        
        txt3 = "The variance inflation factor (VIF) measures multicollinearity. VIF > 5 indicates that an x-variable is significantly correlated with one or more other x-variables. VIF > 10 indicates severe multicollinearity, which can lead to overfitting and inaccurate parameter estimates. If your VIF's are high, trying removing some of those variables."
        
        return figure, dashT2, dashT1, "", [1], txt1, txt2, txt3



            
@app.callback([Output('figure_plot4a', 'figure'),
                Output('figure_plot4b', 'figure'),
                Output('table_plot4a', 'children'),
                Output('table_plot4b', 'children'),
                Output('rt2', 'children'),
                Output('placeholder7', 'children'),
                Output('tab4atxt', 'children'),
                Output('tab4btxt', 'children'),
                Output('fig4atxt', 'children'),
                Output('fig4btxt', 'children'),
                ],
                [Input('btn4', 'n_clicks')],
                [State('main_df', 'data'),
                State('xvar_logistic', 'value'),
                State('yvar_logistic', 'value'),
                State('cat_vars', 'children')],
            )
def update_figure4_table4(n_clicks, main_df, xvars, yvar, cat_vars):
    
    figure = go.Figure(data=[go.Table(
                    header=dict(values=[],
                            fill_color='#b3d1ff',
                            align='left'),
                            ),
                        ],
                )
    figure.update_layout(title_font=dict(size=14,
                        color="rgb(38, 38, 38)",
                        ),
                        margin=dict(l=10, r=10, b=10, t=0),
                        paper_bgcolor="#f0f0f0",
                        plot_bgcolor="#f0f0f0",
                        height=400,
                        )
                        
    cols = ['Parameter', 'coef', 'std err', 'z', 'P>|z|', '[0.025]', '[0.975]', 'VIF']
    df_table = pd.DataFrame(columns=cols)
    df_table['Parameter'] = [np.nan]*10
    df_table['coef'] = [np.nan]*10
    df_table['std err'] = [np.nan]*10
    df_table['z'] = [np.nan]*10
    df_table['P>|z|'] = [np.nan]*10
    df_table['[0.025]'] = [np.nan]*10
    df_table['VIF'] = [np.nan]*10
    
    dashT1 = dash_table.DataTable(
        columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in df_table.columns],
        data = df_table.to_dict('records'),
        editable=False,
        #filter_action="native",
        sort_action="native",
        sort_mode="multi",
        #column_selectable="single",
        #row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        #page_action='none',
        page_current= 0,
        page_size= 10,
        style_table={'overflowX': 'scroll',
                     #'overflowY': 'auto',
                     #'height': '415px',
                     },
        style_cell={'textOverflow': 'auto',
                    'textAlign': 'left',
                    'minWidth': '200px',
                    'width': '200px',
                    'maxWidth': '260px',
                    },
    )
    
    
    cols = ['Target', 'Predicted', 'variable 1', 'variable 2', 'variable 3']
    df_table = pd.DataFrame(columns=cols)
    df_table['Target'] = [np.nan]*10
    df_table['Predicted'] = [np.nan]*10
    df_table['variable 1'] = [np.nan]*10
    df_table['variable 1'] = [np.nan]*10
    df_table['variable 1'] = [np.nan]*10
    
    dashT2 = dash_table.DataTable(
        columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in df_table.columns],
        data = df_table.to_dict('records'),
        editable=False,
        #filter_action="native",
        sort_action="native",
        sort_mode="multi",
        #column_selectable="single",
        #row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        #page_action='none',
        page_current= 0,
        page_size= 10,
        style_table={'overflowX': 'scroll',
                     #'overflowY': 'auto',
                     #'height': '415px',
                     },
        style_cell={'textOverflow': 'auto',
                'textAlign': 'left',
                'minWidth': '300px',
                'width': '300px',
                'maxWidth': '360px',
                },
    )
    
    if main_df is None:
        return {}, {}, dashT1, dashT2, "", [0], "", "", "", ""
    
    elif yvar is None and xvars is None:
        return {}, {}, dashT1, dashT2, "", [0], "", "", "", ""
    
    elif xvars is None or len(xvars) < 2:
        return {}, {}, dashT1, dashT2, "Error: Select two or more x-variables", [0], "", "", "", ""
        
    elif yvar is None:
        return {}, {}, dashT1, dashT2, "Error: Select a y-variable", [0], "", "", "", ""
    
    elif (isinstance(yvar, list) is True) & (xvars is None or len(xvars) < 2):
        return {}, {}, dashT1, dashT2, "Error: Select a y-variable and 2 or more x-variables", [0], "", "", "", ""
    
    elif isinstance(yvar, list) is True:
        return {}, {}, dashT1, dashT2, "Error: Select a y-variable", [0], "", "", "", ""
    
    elif xvars is None or len(xvars) < 2:
        return {}, {}, dashT1, dashT2, "Error: Select two or more x-variables", [0], "", "", "", ""
    
    main_df = pd.read_json(main_df)
    y_prefix = str(yvar)
    if ':' in yvar:
        y_prefix = yvar[:yvar.index(":")]
    vars_ = [y_prefix] + xvars
    vars_ = list(set(vars_))
    main_df = main_df.filter(items=vars_, axis=1)
    
    vars_ = cat_vars #+ [yvar]
    vars_ = list(set(vars_))
    main_df, dropped, cat_vars_ls = dummify_logistic(main_df, vars_, y_prefix, True)
    
    if yvar not in list(main_df):
        return {}, {}, dashT1, dashT2, "Error: Choose a y-variable", [0], "", "", "", ""
    
    yvals = main_df[yvar].tolist()
    unique_yvals = list(set(yvals))
    print('no of unique vals:', len(unique_yvals))
    if len(unique_yvals) < 2:
        return {}, {}, dashT1, dashT2, "Error: Your chosen y-variable only contains one unique value: " + str(unique_yvals[0]), [0], "", "", "", ""
    
    if y_prefix in xvars:
        xvars.remove(y_prefix)
        if len(xvars) == 1:
            return {}, {}, dashT1, dashT2, "Error: Multiple logistic regression requires 2 or more x-variables. You chose two but one of them contains your y-variable.", [0], "", "", "", ""
    
    y_prefix = y_prefix + ":"
    for i in list(main_df):
        if y_prefix in i and i != yvar:
            main_df.drop(labels=[i], axis=1, inplace=True)
        
    models_df, df1_summary, df2_summary, error, pred_df = run_logistic_regression(main_df, xvars, yvar, cat_vars)
    
    if error == 1:
        error = "Error: The model exceeded the maximum iterations in trying to find a fit. Try a smaller number of x-variables. Tip: eliminate redundant variables, variables of little-to-no interest, or eliminate categorical variables that have many levels (e.g., a column of diagnosis codes may have hundreds of different codes)."
        return {}, {}, dashT1, dashT2, error, [0], "", "", "", ""
        
        
    fpr = models_df['FPR'].tolist()
    fpr = fpr[0]
    tpr = models_df['TPR'].tolist()
    tpr = tpr[0]
    auroc = models_df['auroc'].tolist()
    auroc = auroc[0]
        
    ppv = models_df['PPV'].tolist()
    ppv = ppv[0]
    recall = models_df['Recall'].tolist()
    recall = recall[0]
    auprc = models_df['auprc'].tolist()
    auprc = auprc[0]
    prc_null = models_df['prc_null'].tolist()
    prc_null = prc_null[0]
        
    fig_data = []
    fig_data.append(
        go.Scatter(
            x = fpr,
            y = tpr,
            mode = "lines",
            name = 'AUC = ' + str(np.round(auroc,3)),
            opacity = 0.75,
            line = dict(color = "#0066ff", width = 2),
            )
        )
                    
    fig_data.append(
            go.Scatter(
                x = [0, 1],
                y = [0, 1],
                mode = "lines",
                name = 'Null AUC = 0.5',
                opacity = 0.75,
                line = dict(color = "#737373", width = 1),
            )
        )
                        
    figure1 = go.Figure(
            data = fig_data,
            layout = go.Layout(
                xaxis = dict(
                    title = dict(
                        text = "<b>False positive rate (FPR)</b>",
                        font = dict(
                            family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif",
                            size = 18,
                        ),
                    ),
                    rangemode="tozero",
                    zeroline=True,
                    showticklabels = True,
                ),
                            
                yaxis = dict(
                    title = dict(
                        text = "<b>True positive rate (TPR)</b>",
                        font = dict(
                            family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif",
                            size = 18,
                                    
                        ),
                    ),
                    rangemode="tozero",
                    zeroline=True,
                    showticklabels = True,
                ),
                            
                margin = dict(l=60, r=30, b=10, t=40),
                showlegend = True,
                height = 400,
                paper_bgcolor = "rgb(245, 247, 249)",
                plot_bgcolor = "rgb(245, 247, 249)",
            ),
        )


    fig_data = []
    fig_data.append(
        go.Scatter(
            x = recall,
            y = ppv,
            mode = "lines",
            name = 'AUC = ' + str(np.round(auprc,3)),
            opacity = 0.75,
            line = dict(color = "#0066ff", width = 2),
            )
        )
                    
    fig_data.append(
            go.Scatter(
                x = [0, 1],
                y = [prc_null, prc_null],
                mode = "lines",
                name = 'Null AUC = ' + str(np.round(prc_null, 3)),
                opacity = 0.75,
                line = dict(color = "#737373", width = 1),
            )
        )
                        
    figure2 = go.Figure(
            data = fig_data,
            layout = go.Layout(
                xaxis = dict(
                    title = dict(
                        text = "<b>Recall (TPR)</b>",
                        font = dict(
                            family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif",
                            size = 18,
                        ),
                    ),
                    rangemode="tozero",
                    zeroline=True,
                    showticklabels = True,
                ),
                            
                yaxis = dict(
                    title = dict(
                        text = "<b>Precision</b>",
                        font = dict(
                            family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif",
                            size = 18,
                                    
                        ),
                    ),
                    rangemode="tozero",
                    zeroline=True,
                    showticklabels = True,
                ),
                            
                margin = dict(l=60, r=30, b=10, t=40),
                showlegend = True,
                height = 400,
                paper_bgcolor = "rgb(245, 247, 249)",
                plot_bgcolor = "rgb(245, 247, 249)",
            ),
        )
        
    dashT1 = dash_table.DataTable(
            columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in df1_summary.columns],
            data = df1_summary.to_dict('records'),
            editable=False,
            #filter_action="native",
            sort_action="native",
            sort_mode="multi",
            #column_selectable="single",
            #row_selectable="multi",
            row_deletable=False,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            #page_action='none',
            page_current= 0,
            page_size= 10,
            style_table={'overflowX': 'scroll',
                         #'overflowY': 'auto',
                         #'height': '415px',
                         },
            style_cell={#'textOverflow': 'auto',
                        'textAlign': 'left',
                        #'minWidth': '140px',
                        #'width': '140px',
                        #'maxWidth': '220px',
                        },
        )
        
    
    dashT2 = dash_table.DataTable(
        columns=[{"name": i, "id": i, "deletable": False, "selectable": True} for i in pred_df.columns],
        data = pred_df.to_dict('records'),
        editable=False,
        #filter_action="native",
        sort_action="native",
        sort_mode="multi",
        #column_selectable="single",
        #row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        #page_action='none',
        page_current= 0,
        page_size= 10,
        style_table={'overflowX': 'scroll',
                     #'overflowY': 'auto',
                     #'height': '415px',
                     },
        style_cell={
                    'textOverflow': 'auto',
                    'textAlign': 'left',
                    'minWidth': '140px',
                    #'width': '140px',
                    'maxWidth': '10000px',
                    },
    )
        
    txt1 = "This table pertains to the fitted model, which predicts the probability of an observation being a positive (1). The variance inflation factor (VIF) measures multicollinearity. VIF > 5 indicates that an x-variable is significantly correlated with one or more other x-variables. VIF > 10 indicates severe multicollinearity, which can lead to overfitting and inaccurate parameter estimates. If your VIF's are high, trying removing some of those variables."
    
    txt2 = "This table pertains to results after finding an optimal diagnostic threshold. This threshold determines whether the value of an outcome's probability is counted as a positive (1) or a negative (0). The threshold is found by determining the point on the ROC curve that is closest to the upper left corner."
    
    auroc = np.round(auroc,3)
    t1 = str()
    if auroc < 0.45:
        t1 = " worse than random "
    elif auroc < 0.55:
        t1 = " effectively random "
    elif auroc < 0.65:
        t1 = " very little "
    elif auroc < 0.7:
        t1 = " highly limited "
    elif auroc < 0.75:
        t1 = " limited "
    elif auroc < 0.8:
        t1 = " potentially useful "
    elif auroc < 0.85:
        t1 = " good "
    elif auroc < 0.95:
        t1 = " good-to-excellent "
    else:
        t1 = " outstanding "
        
    txt3 = "AUC = area under the ROC curve. Random 50:50 guesses produce values of 0.5. Your AUC value of " + str(auroc) + " indicates" + t1 + "diagnostic power."
    
    txt4 = "AUC = area under the PRC curve. Random 50:50 guesses produce AUC values equal to the fraction of actual positive outcomes (1's) in the data."
    
    return figure1, figure2, dashT1, dashT2, "", [1], txt1, txt2, txt3, txt4


#########################################################################################
############################# Run the server ############################################
#########################################################################################


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug = False) # modified to run on linux server
