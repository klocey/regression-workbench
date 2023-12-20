from sklearn.preprocessing import PolynomialFeatures

import dash_bootstrap_components as dbc
from dash import dcc, html

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.tools.tools import pinv_extended
import statsmodels.stats.stattools as stattools
import statsmodels.stats.api as sms
import statsmodels.api as sm

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import dash

from plotly import graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
import json

import app_fxns

def generate_figure_single_regression():

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
                    html.P("Confidence intervals (CI) reflect confidence in the mean y-value " +
                           "across the x-axis. Prediction intervals (PI) pertain to the model, " +
                           "where points outside the PI are unlikely to be explained by the " +
                           "model. Note: When running a robust regression, only the observed " +
                           "vs. predicted r\u00B2 value is returned, which usually equals or " +
                           "nearly equals the r\u00B2 of the fitted model.", 
                           ),
                    ],
                style={'width': '100%',
                       'display': 'inline-block',
                       'background-color': '#f0f0f0',
                       'padding': '1%',
                       },
                )


def control_card_single_regression():

    return html.Div(
        id="control-card2",
        children=[
                html.H5("Conduct a single regression for deeper insights",
                        style={'display': 'inline-block', 
                               'width': '35.4%',
                               'margin-right': '1%',
                               },
                        ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="target_select_vars2",
                       style={'display': 'inline-block', 
                              'width': '3%', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("These analyses are based on ordinary least squares regression (OLS)."+
                            "They exclude categorical features, any features suspected of being " +
                            "dates or times, and any numeric features having less than 4 unique " +
                            "values.", 
                            target="target_select_vars2", 
                            style = {'font-size': 12,
                                     },
                            ),
                html.Hr(),
                
                html.Div(
                    id="control-card2a",
                    children=[
                        html.B("Choose a predictor (x) variable",
                            style={'display': 'inline-block',
                                    'vertical-align': 'top',
                               },
                            ),
                        dcc.Dropdown(
                                id='xvar2',
                                options=[{"label": i, "value": i} for i in []],
                                multi=False,
                                optionHeight=65,
                                placeholder='Select a feature',
                                style={'width': '100%',
                                       'display': 'inline-block',
                                     },
                                ),
                            ],
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                                   'width': '20%',
                                   'margin-right': '1%',
                            },
                    ),
                
                html.Div(
                    id="control-card2b",
                    children=[
                        html.B("Choose a data transformation",
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                            }),
                        dcc.Dropdown(
                                id='x_transform',
                                options=[{"label": i, "value": i} for i in ['None', 'log10', 
                                                                            'square root', 
                                                                            'cube root',
                                                                            'squared', 'cubed', 
                                                                            'log-modulo', 
                                                                            'log-shift']
                                         ],
                                multi=False, 
                                value='None',
                                style={'width': '90%',
                                       'display': 'inline-block',
                                     },
                                ),
                        ],
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'margin-right': '2%',
                               'width': '20%',
                               },
                        ),
                    
                html.Div(
                    id="control-card2c",
                    children=[
                        html.B("Choose a response (y) variable",
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                            },
                            ),
                        dcc.Dropdown(
                                id='yvar2',
                                options=[{"label": i, "value": i} for i in []],
                                multi=False,
                                optionHeight=65,
                                placeholder='Select a feature',
                                style={'width': '100%',
                                       'display': 'inline-block',
                                     },
                                ),
                        ],
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'margin-right': '1%',
                               'width': '20%',
                        },
                    ),
                
                html.Div(
                    id="control-card2d",
                    children=[
                        html.B("Choose a data transformation",
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                            },
                            ),
                        dcc.Dropdown(
                                id='y_transform',
                                options=[{"label": i, "value": i} for i in ['None', 'log10', 
                                                                            'square root', 
                                                                            'cube root',
                                                                            'squared', 'cubed', 
                                                                            'log-modulo', 
                                                                            'log-shift']
                                         ],
                                multi=False, 
                                value='None',
                                style={'width': '90%',
                                       'display': 'inline-block',
                                     },
                                ),
                        ],
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'margin-right': '2%',
                               'width': '20%',
                        },
                    ),
                    
                html.Div(
                    id="control-card2e",
                    children=[
                        html.B("Choose a model",
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                            },
                            ),
                        dcc.Dropdown(
                                id='model2',
                                options=[{"label": i, "value": i} for i in ['linear', 'quadratic', 
                                                                            'cubic']
                                         ],
                                multi=False, 
                                value='linear',
                                style={'width': '100%',
                                       'display': 'inline-block',
                                     },
                                ),
                        ],
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'width': '10%',
                        },
                    ),
                
                html.Hr(),
                html.Br(),
                
                dbc.Button('Run regression', 
                            id='btn2', n_clicks=0,
                            style={'width': '20%',
                                'font-size': 12,
                                "background-color": "#2a8cff",
                                'display': 'inline-block',
                                'margin-right': '1%',
                    },
                    ),
                
                dbc.Button("View residuals plot",
                           id="open-single_regression_residuals_plot",
                           style={
                               "background-color": "#2a8cff",
                               'width': '16%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '1%',
                               },
                    ),
                dbc.Modal(
                    [dbc.ModalBody([dcc.Graph(id="residuals_plot1"), 
                                    html.Br(), 
                                    html.P("", id='fig2txt'),
                                    ],
                                   ),
                     dbc.ModalFooter(
                             dbc.Button("Click to Close",
                                id="close-single_regression_residuals_plot",
                                className="ml-auto",
                                style={
                                    "background-color": "#2a8cff",
                                    'width': '30%',
                                    'font-size': 14,
                                    },
                                ),
                             style={
                                 "background-color": "#A0A0A0",
                                 "display": "flex",
                                 "justify-content": "center",
                                 "align-items": "center",
                                 },
                             ),
                     
                            ],
                    id="modal-single_regression_residuals_plot",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="lg",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button("View results table",
                           id="open-centered_single",
                           style={
                               "background-color": "#2a8cff",
                               'width': '16%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '1%',
                               },
                    ),
                
                dbc.Modal(
                    [dbc.ModalBody([html.H5("Results for single regression"),
                                    html.Div(id="single_table_1"), 
                                    html.Div(id="single_table_2"),
                                    html.Br(),
                                    html.P("", id="single_table_txt"),
                                    ],
                                   ),
                     dbc.ModalFooter(
                             dbc.Button("Click to Close",
                                id="close-centered_single", 
                                className="ml-auto",
                                style={
                                    "background-color": "#2a8cff",
                                    'width': '30%',
                                    'font-size': 14,
                                    },
                                ),
                             style={
                                 "background-color": "#A0A0A0",
                                 "display": "flex",
                                 "justify-content": "center",
                                 "align-items": "center",
                                 },
                             ),

                            ],
                    
                    id="modal-centered_single",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="lg",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button('Run robust', 
                            id='btn_robust2', 
                            n_clicks=0,
                            style={'width': '15%',
                                   'font-size': 12,
                                   "background-color": "#2a8cff",
                                   'display': 'inline-block',
                                   'margin-right': '1%',
                                   },
                            ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="robust2",
                       style={'display': 'inline-block', 
                              'width': '3%', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("Outliers can weaken OLS regression. However, clicking on 'RUN " + 
                            "ROBUST' will run a 'Robust Linear Model' via statsmodels. " +
                            "The r-square (for observed vs predicted) of a robust regression " +
                            "will likely be lower than the r-square of OLS regression. This " +
                            "is because the robust model is not chasing outliers. However, " +
                            "for non-outliers (the main trend), the robust model will be more " +
                            "accurate, stable, valid, and useful for predictions than an OLS " +
                            "model. To run regular OLS regression, simply click 'RUN REGRESSION' " +
                            "again.",
                            target="robust2", 
                            style = {'font-size': 12,
                                     'display': 'inline-block',
                                     },
                            ),
                
                html.P("", id='rt3')
                ],
                style={'margin-left': '1%',
                       'width': '98.5%',
                    },
            )


def get_updated_results(n_clicks, robust, xvar, yvar, x_transform, y_transform, model, df):
        
    cols = ['Model information', 'Model statistics']
    df_table1 = pd.DataFrame(columns=cols)
    df_table1['Model information'] = [np.nan]*3
    df_table1['Model statistics'] = [np.nan]*3
            
    dashT1 = dash_table.DataTable(
        data=df_table1.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in df_table1.columns],
            
        page_action='none',
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
                
        style_table={'height': '300px', 
                     'overflowY': 'auto',
                     },
        style_cell={'padding':'5px',
                    'minwidth':'140px',
                    'width':'160px',
                    'maxwidth':'160px',
                    'whiteSpace':'normal',
                    'textAlign': 'center',
                    },
        )
            
    cols = ['Parameter', 'coef', 'std err', 't', 'P>|t|', '[0.025]', '[0.975]']
    df_table2 = pd.DataFrame(columns=cols)
    df_table2['Parameter'] = [np.nan]*3
    df_table2['coef'] = [np.nan]*3
    df_table2['std err'] = [np.nan]*3
    df_table2['t'] = [np.nan]*3
    df_table2['P>|t|'] = [np.nan]*3
    df_table2['[0.025]'] = [np.nan]*3
        
    dashT2 = dash_table.DataTable(
        data=df_table2.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in df_table2.columns],
            
        page_action='none',
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
            
        style_table={'height': '300px', 
                     'overflowY': 'auto',
                     },
        style_cell={'padding':'5px',
                    'minwidth':'140px',
                    'width':'160px',
                    'maxwidth':'160px',
                    'whiteSpace':'normal',
                    'textAlign': 'center',
                    },
    )  
        
    ctx1 = dash.callback_context
    jd1 = json.dumps({'triggered': ctx1.triggered,})
    jd1 = jd1[:50]
    if 'rt4.children' in jd1:
        return {}, "", "", {}, 0, "", dashT1, dashT2
    
    try:
        df = pd.DataFrame(df)
    except:
        return {}, "", "", {}, 0, "", dashT1, dashT2
    
    if df is None or df.empty or xvar is None or yvar is None or xvar == yvar or isinstance(yvar, list) is True or isinstance(yvar, list) is True:
            
        if df is None or df.empty:
            return {}, "", "", {}, 0, "", dashT1, dashT2
            
        elif (isinstance(xvar, list) is True or xvar is None) & (isinstance(yvar, list) is True or yvar is None):
            return {}, "Error: You need to select some variables.", "", {}, 0, "", dashT1, dashT2
        
        elif isinstance(yvar, list) is True or yvar is None:
            return {}, "Error: You need to select a response variable.", "", {}, 0, "", dashT1, dashT2
            
        elif isinstance(xvar, list) is True or xvar is None:
            return {}, "Error: You need to select an predictor variable.", "", {}, 0, "", dashT1, dashT2
            
        elif xvar == yvar and xvar is not None:
            return {}, "Error: Your predictor variable and response variable are the same. Ensure they are different.", "", {}, 0, "", dashT1, dashT2
        else:
            return {}, "", "", {}, 0, "", dashT1, dashT2
            
    else:
        df = df.filter(items=[xvar, yvar], axis=1)
            
        if x_transform == 'log10':
            df[xvar] = np.log10(df[xvar])
            df.rename(columns={xvar: "log<sub>10</sub>(" + xvar + ")"}, inplace=True)
            xvar = "log<sub>10</sub>(" + xvar + ")"
                
        elif x_transform == 'square root':
            df[xvar] = df[xvar]**0.5
            df.rename(columns={xvar: "\u221A(" + xvar + ")"}, inplace=True)
            xvar = "\u221A(" + xvar + ")"
                
        elif x_transform == 'cube root':
            df[xvar] = df[xvar]**(1/3)
            df.rename(columns={xvar: "\u221B(" + xvar + ")"}, inplace=True)
            xvar = "\u221B(" + xvar + ")"
                
        elif x_transform == 'squared':
            df[xvar] = df[xvar]**2
            df.rename(columns={xvar: "(" + xvar + ")\u00B2"}, inplace=True)
            xvar = "(" + xvar + ")\u00B2"
                
        elif x_transform == 'cubed':
            df.rename(columns={xvar: "(" + xvar + ")\u00B3"}, inplace=True)
            xvar = "(" + xvar + ")\u00B3"
            
        elif x_transform == 'log-modulo':
            lmt = np.log10(np.abs(df[xvar]) + 1).tolist()
            for i, val in enumerate(df[xvar].tolist()):
                if val < 0:
                    lmt[i] = lmt[i] * -1
            df[xvar] = lmt  
            df.rename(columns={xvar: "log-modulo(" + xvar + ")"}, inplace=True)
            xvar = "log-modulo(" + xvar + ")"
            
        elif x_transform == 'log-shift':
            df[xvar] = np.log10(df[xvar] + 1).tolist()
            df.rename(columns={xvar: "log-shift(" + xvar + ")"}, inplace=True)
            xvar = "log-shift(" + xvar + ")"
        
        if y_transform == 'log10':
            df[yvar] = np.log10(df[yvar])
            df.rename(columns={yvar: "log<sub>10</sub>(" + yvar + ")"}, inplace=True)
            yvar = "log<sub>10</sub>(" + yvar + ")"
            
        elif y_transform == 'square root':
            df[yvar] = df[yvar]**0.5
            df.rename(columns={yvar: "\u221A(" + yvar + ")"}, inplace=True)
            yvar = "\u221A(" + yvar + ")"
            
        elif y_transform == 'cube root':
            df[yvar] = df[yvar]**(1/3)
            df.rename(columns={yvar: "\u221B(" + yvar + ")"}, inplace=True)
            yvar = "\u221B(" + yvar + ")"
            
        elif y_transform == 'squared':
            df[yvar] = df[yvar]**2
            df.rename(columns={yvar: "(" + yvar + ")\u00B2"}, inplace=True)
            yvar = "(" + yvar + ")\u00B2"
            
        elif y_transform == 'cubed':
            df.rename(columns={yvar: "(" + yvar + ")\u00B3"}, inplace=True)
            yvar = "(" + yvar + ")\u00B3"
            
        elif y_transform == 'log-modulo':
            lmt = np.log10(np.abs(df[yvar]) + 1).tolist()
            for i, val in enumerate(df[yvar].tolist()):
                if val < 0:
                    lmt[i] = lmt[i] * -1
            df[yvar] = lmt  
            df.rename(columns={yvar: "log-modulo(" + yvar + ")"}, inplace=True)
            yvar = "log-modulo(" + yvar + ")"
            
        elif y_transform == 'log-shift':
            df[yvar] = np.log10(df[yvar] + 1).tolist()
            df.rename(columns={yvar: "log-shift(" + yvar + ")"}, inplace=True)
            yvar = "log-shift(" + yvar + ")"
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(how='any', inplace=True)
        #df = app_fxns.remove_nans_optimal(df, yvar)
                
        d = int()
        if model == 'linear': d = 1
        elif model == 'quadratic': d = 2
        elif model == 'cubic': d = 3
            
        if robust == 0:
            y_o = df[yvar].values.tolist()
            x_o = df[xvar].values.tolist()
            x_o, y_o = zip(*sorted(zip(x_o, y_o)))
                
            x_o = np.array(x_o)
            y_o = np.array(y_o)
            
            #Create single dimension
            x = x_o[:, np.newaxis]
            y = y_o[:, np.newaxis]

            # Sort x values and get index
            inds = x.ravel().argsort()  
            x = x.ravel()[inds].reshape(-1, 1)
            #Sort y according to x sorted index
            y = y[inds]
                
            polynomial_features = PolynomialFeatures(degree = d)
            xp = polynomial_features.fit_transform(x)
                    
            model = sm.OLS(y, xp).fit()
            ypred = model.predict(xp)
            ypred = ypred.tolist()
                
        elif robust == 1:
            # Rub Robust Regression
            y_o = df[yvar].values.tolist()
            x_o = df[xvar].values.tolist()
            x_o, y_o = zip(*sorted(zip(x_o, y_o)))
            
            x = np.array(x_o)
            y = np.array(y_o)
                
            # Create polynomial features up to 3rd degree
            # Add a constant for the intercept
                
            if d == 1:
                # For a 1st-degree
                X_poly = sm.add_constant(x) 
            elif d == 2:
                # For a 2nd-degree polynomial (X, X^2)
                X_poly = np.column_stack((x, x**2))  
                X_poly = sm.add_constant(X_poly)
            elif d == 3:
                # For a 3rd-degree polynomial (X, X^2, X^3)
                X_poly = np.column_stack((x, x**2, x**3))  
                X_poly = sm.add_constant(X_poly)
                
            # Fit a robust polynomial regression model
            model = sm.RLM(y, X_poly, M=sm.robust.norms.TukeyBiweight(), 
                           missing='drop').fit()
            ypred = model.fittedvalues
                
        # Jarque-Bera for normally distributed errors
        jarque_bera_test = sms.jarque_bera(model.resid)
        jarque_bera_p = round(jarque_bera_test[1], 4)
            
        # Durbin-Watson for autocorrelation
        dw = stattools.durbin_watson(model.resid)
        durbin_watson = round(dw, 4)
            
        # Breusch-Pagan for heteroskedasticity
        breusch_pagan_test = sms.het_breuschpagan(model.resid, model.model.exog)
        breusch_pagan_p = round(breusch_pagan_test[1],4)
            
        # Harvey-Collier multiplier test for linearity
        if d == 1:
            try:
                skip = 10 #len(model.params)  # bug in linear_harvey_collier
                rr = sms.recursive_olsresiduals(model, skip=skip, alpha=0.95, order_by=None)
                harvey_collier_test = stats.ttest_1samp(rr[3][skip:], 0)
                harvey_collier_p = round(harvey_collier_test[1], 4)
            except:
                harvey_collier_p = np.nan
        
        else:
            harvey_collier_p = np.nan
        
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
                p = round(p, 4)
                eqn = eqn + str(p) + exp
                
            else:
                if p >= 0:
                    p = round(p, 4)
                    eqn = eqn + ' + ' + str(p) + exp
                else:
                    p = round(p, 4)
                    eqn = eqn + ' - ' + str(np.abs(p)) + exp
        
        b = model.params[0]
        if b >= 0:
            b = round(b, 4)
            eqn = eqn + ' + ' + str(b)
        else:
            b = round(b, 4)
            eqn = eqn + ' - ' + str(np.abs(b))
        
        try:
            y = y.flatten().tolist()
        except:
            pass
        op_r2 = app_fxns.obs_pred_rsquare(np.array(y), np.array(ypred))
        try:
            op_r2 = round(op_r2, 4)
        except:
            pass
        if op_r2 < 0:
            op_r2 = 0
            
        if robust == 0:
            r2 = round(model.rsquared, 4)
            r2_adj = round(model.rsquared_adj, 4)
            #aic = round(model.aic, 4)
            #bic = round(model.bic, 4)
            #fp = round(model.f_pvalue, 4)
            #llf = round(model.llf, 4)
            
            st, data, ss2 = summary_table(model, alpha=0.05)
            #fittedvalues = data[:, 2]
            #predict_mean_se  = data[:, 3]
            predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T # confidence interval
            predict_ci_low, predict_ci_upp = data[:, 6:8].T # prediction interval
            
        elif robust == 1:
            r2 = 'N/A'
            r2_adj = 'N/A'
            #aic = 'N/A'
            #bic = 'N/A'
            #fp = 'N/A'
            #llf = 'N/A'
        
            # Calculate the standard error of residuals
            residuals = model.resid
            standard_error_residuals = np.std(residuals)
            
            # Set the desired confidence level
            confidence_level = 0.95
                
            # Calculate the critical t-value
            dof = len(x_o) - model.df_model - 1  # Degrees of freedom
            t_value = stats.t.ppf((1 + confidence_level) / 2, dof)
            
            x_plot = x_o

            # Create empty arrays to store the upper and lower bounds for the confidence intervals
            lower_ci_limit = np.zeros(len(x_plot))
            upper_ci_limit = np.zeros(len(x_plot))
            lower_pi_limit = np.zeros(len(x_plot))
            upper_pi_limit = np.zeros(len(x_plot))
            
            # Calculate confidence intervals for each x value
            for i, x in enumerate(x_plot):
                if d == 1:
                    x_poly = [1, x,]
                elif d == 2:
                    x_poly = [1, x, x**2]
                elif d == 3:
                    x_poly = [1, x, x**2, x**3]
                    
                # Confidence Intervals
                ci_multiplier = t_value * np.sqrt(np.dot(x_poly, 
                                                         np.dot(model.cov_params(), x_poly)))
                ci_interval = ci_multiplier
                lower_ci_limit[i] = model.predict(exog=x_poly) - ci_interval
                upper_ci_limit[i] = model.predict(exog=x_poly) + ci_interval
            
                # Prediction Intervals
                y_pred = np.dot(model.params, x_poly)
                lower_pi_limit[i] = y_pred - t_value * standard_error_residuals
                upper_pi_limit[i] = y_pred + t_value * standard_error_residuals
            
            predict_mean_ci_low = lower_ci_limit
            predict_mean_ci_upp = upper_ci_limit
            
            predict_ci_low = lower_pi_limit
            predict_ci_upp = upper_pi_limit
            
            
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
            
        obs_pred_r2 = app_fxns.obs_pred_rsquare(y_o, ypred)
        obs_pred_r2 = str(np.round(obs_pred_r2, 3))
            
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
                name = 'r<sup>2</sup> (fitted) =' + str(r2) + '<br>r<sup>2</sup> (adjusted) =' + str(r2_adj) + '<br>r<sup>2</sup> (obs vs pred) =' + obs_pred_r2,
                opacity = 0.75,
                line = dict(color = clr, 
                            width = 2),
                )
            )
            
        fig_data.append(
            go.Scatter(
                x = x_o,
                y = predict_mean_ci_upp,
                mode = "lines",
                name = 'upper 95 CI',
                opacity = 0.75,
                line = dict(color = clr, 
                            width = 2, 
                            dash='dash'),
            )
        )
            
        fig_data.append(
            go.Scatter(
                x = x_o,
                y = predict_mean_ci_low,
                mode = "lines",
                name = 'lower 95 CI',
                opacity = 0.75,
                line = dict(color = clr, 
                            width = 2, 
                            dash='dash'),
            )
        )
            
        fig_data.append(
            go.Scatter(
                x = x_o,
                y = predict_ci_upp,
                mode = "lines",
                name = 'upper 95 PI',
                opacity = 0.75,
                line = dict(color = clr, 
                            width = 2, 
                            dash='dot'),
            )
        )
            
        fig_data.append(
            go.Scatter(
                x = x_o,
                y = predict_ci_low,
                mode = "lines",
                name = 'lower 95 PI',
                opacity = 0.75,
                line = dict(color = clr, 
                            width = 2, 
                            dash='dot'),
            )
        )
              
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
                    showticklabels = True,
                ),
                
                margin = dict(l=60, r=30, b=10, t=40),
                showlegend = True,
                height = 400,
                paper_bgcolor = "rgb(245, 247, 249)",
                plot_bgcolor = "rgb(245, 247, 249)",
            ),
        )

            
        txt = "The Jarque-Bera test suggests "
        if jarque_bera_p < 0.05:
            txt += "non-"
        txt += "normally distributed residuals (p = " + str(jarque_bera_p) + "). "
        
        txt += "The Breusch-Pagan test suggests that the residuals are "
        if breusch_pagan_p < 0.05:
            txt += "not "
        txt += "homoskedastic (p = " + str(breusch_pagan_p) + "). "
        
        txt += "The Durbin-Watson statistic indicates "
        if durbin_watson < 1.5 or durbin_watson > 2.5:
            txt += "non-"
        txt += "independent observations (DW = " + str(durbin_watson) + "). "
        
        txt += "The Harvey-Collier test  "
        if d != 1:
            txt += "was not suitable for this analysis. "
        elif np.isnan(harvey_collier_p) == True:
            txt += "failed to execute. "
        elif harvey_collier_p < 0.05:
            txt += "indicates the relationship is not linear (p = " + str(harvey_collier_p) + "). "
        elif harvey_collier_p >= 0.05:
            txt += "indicates the relationship is linear (p = " + str(harvey_collier_p) + "). "
            
            
        #######################################################################
        #################   Residuals Plot   ################################## 
        #######################################################################
            
        fig_data2 = []
        fig_data2.append(go.Scatter(
            x = x_o,
            y = model.resid,
            name = 'residuals',
            mode = "markers",
            opacity = 0.75,
            marker = dict(size=10,
                          color=clr)
            )
            )
            
        res_figure = go.Figure(
            data = fig_data2,
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
                    showticklabels = True,
                ),
                                
                yaxis = dict(
                    title = dict(
                        text = "<b>" + "Residuals" + "</b>",
                        font = dict(
                            family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif",
                            size = 18,
                            
                        ),
                    ),
                    showticklabels = True,
                ),
                
                margin = dict(l=60, r=30, b=10, t=40),
                showlegend = True,
                height = 400,
                paper_bgcolor = "rgb(245, 247, 249)",
                plot_bgcolor = "rgb(245, 247, 249)",
            ),
        )
        
        
        ######################################### Result Tables ###################################
        results_summary = model.summary()
        
        results_as_html1 = results_summary.tables[0].as_html()
        df1_summary = pd.read_html(results_as_html1)[0]
        #df1_summary['index'] = df1_summary.index
        df1_summary = df1_summary.astype(str)
        col_names = list(df1_summary)
        
        df3 = pd.DataFrame(columns=['Model information', 'Model statistics'])
        df3['Model information']  = df1_summary[col_names[0]].astype(str) + ' ' + df1_summary[col_names[1]].astype(str) 
        df3['Model statistics'] = df1_summary[col_names[2]].astype(str) + ' ' + df1_summary[col_names[3]].astype(str) 
        #del df3, df1_summary
        
        dashT1 = dash_table.DataTable(
            data=df3.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df3.columns],
            export_format="csv",
            
            style_table={'height': '250px', 
                         'overflowY': 'auto',
                         },
            style_cell={'padding':'5px',
                        'minwidth':'140px',
                        'width':'160px',
                        'maxwidth':'160px',
                        'whiteSpace':'normal',
                        'textAlign': 'center',
                        },
        )
        
        
        results_as_html2 = results_summary.tables[1].as_html()
        df2_summary = pd.read_html(results_as_html2, header=0)[0]
        df2_summary.rename(columns={"Unnamed: 0": "Parameter"}, inplace=True)
        
        dashT2 = dash_table.DataTable(
            data=df2_summary.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df2_summary.columns],
            export_format="csv",
            
            style_table={'height': '200px', 
                         'overflowY': 'auto',
                         },
            style_cell={'padding':'5px',
                        'minwidth':'140px',
                        'width':'160px',
                        'maxwidth':'160px',
                        'whiteSpace':'normal',
                        'textAlign': 'center',
                        },
        )
        
        return figure, "", txt, res_figure, 0, "", dashT1, dashT2