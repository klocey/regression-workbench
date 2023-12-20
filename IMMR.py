from sklearn.preprocessing import PolynomialFeatures

import statsmodels.stats.stattools as stattools
import statsmodels.stats.api as sms
import statsmodels.api as sm

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import dash

import plotly.graph_objects as go

from scipy import stats
import pandas as pd
import numpy as np
import random
import json

import app_fxns


def generate_outputs_iterative_multi_model_regression():

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
                    'padding': '1%',
                },
    )
 
    




def control_card_iterative_multi_model_regression():

    return html.Div(
        id="control-card1",
        children=[
                html.H5("Explore relationships between multiple features at once",
                        style={'display': 'inline-block', 
                               'width': '41.5%',
                               },
                        ),
                
                html.I(className="fas fa-question-circle fa-lg", id="target_select_vars",
                            style={'display': 'inline-block', 
                                   'width': '3%', 
                                   'color':'#99ccff',
                                   },
                            ),
                
                dbc.Tooltip("As a default, these analyses are based on ordinary least squares " +
                            "regression (OLS). These analyses exclude categorical features, any" +
                            "features suspected of being dates or times, and any numeric features" +
                            "having less than 4 unique values.", 
                            target="target_select_vars", 
                            style = {'font-size': 12,
                                     },
                            ),
                
                html.Hr(),
                
                html.B("Choose one or more x-variables.",
                    style={'display': 'inline-block',
                            'vertical-align': 'top',
                       },
                    ),
                
                dcc.Dropdown(
                        id='xvar',
                        options=[{"label": i, "value": i} for i in []],
                        multi=True, 
                        value=None,
                        style={'width': '100%',
                             },
                        ),
                        
                html.Br(),
                
                html.B("Choose one or more y-variables.",
                    style={'display': 'inline-block',
                           'vertical-align': 'top',
                    },
                ),
                
                dcc.Dropdown(
                        id='yvar',
                        options=[{"label": i, "value": i} for i in []],
                        multi=True, 
                        value=None,
                        style={'width': '100%',
                             },
                        ),
                
                html.Hr(),
                html.Br(),
                
                dbc.Button('Run regressions', 
                            id='btn1', 
                            n_clicks=0,
                            style={'width': '20%',
                                'font-size': 12,
                                "background-color": "#2a8cff",
                                'display': 'inline-block',
                                'margin-right': '2%',
                                },
                ),
                
                dbc.Button("View results table",
                           id="open-iterative_multimodel_ols_table1",
                           style={
                               'background-color': '#2a8cff',
                               'width': '16%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '2%',
                               },
                    ),
                
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id='table_plot1'), 
                                    html.Br(), 
                                    html.P("", id='table1txt'),
                                    ],
                                   ),
                     dbc.ModalFooter(
                             dbc.Button("Click to Close",
                                id="close-iterative_multimodel_ols_table1", 
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
                    id="modal-iterative_multimodel_ols_table1",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    #size="xl",
                    fullscreen=True,
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button('Smart scale', 
                            id='btn_ss', 
                            n_clicks=0,
                            style={'width': '15%',
                                'font-size': 12,
                                "background-color": "#2a8cff",
                                'display': 'inline-block',
                                'margin-right': '2%',
                                },
                ),
                
                html.I(className="fas fa-question-circle fa-lg", 
                       id="ss1",
                       style={'display': 'inline-block', 
                              'width': '3%', 
                              'color':'#99ccff',
                              },
                            ),
                
                dbc.Tooltip("Skewed data can weaken analyses and visualizations. Click on 'Smart" +
                            "Scale' and the app will automatically detect and apply the best " +
                            "scaling for each skewed variable. Smart scaling will not necessarily" +
                            "improve the r-square.  To remove the rescaling just click " +
                            "'SMART SCALE' again.",
                            target="ss1", 
                            style = {'font-size': 12,
                                     'display': 'inline-block',
                                     },
                            ),
                
                dbc.Button('Run robust', 
                            id='btn_robust', 
                            n_clicks=0,
                            style={'width': '15%',
                                'font-size': 12,
                                "background-color": "#2a8cff",
                                'display': 'inline-block',
                                'margin-right': '2%',
                                },
                            ),
                
                html.I(className="fas fa-question-circle fa-lg", 
                       id="robust1",
                       style={'display': 'inline-block', 
                              'width': '3%', 
                              'color':'#99ccff',
                              },
                       ),
                
                dbc.Tooltip("Outliers can weaken OLS regression. However, clicking on 'RUN ROBUST'" + 
                            " will run a 'Robust Linear Model' via statsmodels. The r-square " +
                            "(for observed vs predicted) of a robust regression will likely be " +
                            "lower than the r-square of OLS regression. This is because the " +
                            "robust model is not chasing outliers. However, for non-outliers " +
                            "(the main trend), the robust model will be more accurate, stable, " +
                            "valid, and useful for predictions than an OLS model. To run regular " + 
                            "OLS regression, simply click 'RUN ROBUST' again.",
                            target="robust1", 
                            style = {'font-size': 12,
                                     'display': 'inline-block',
                                     },
                            ),
                
                html.P("", 
                       id='rt0',
                       ),
                ],
        
                style={'margin-left': '1%',
                       'width': '98.5%',
                    },
            )



def get_updated_results(n_clicks, smartscale, robust, df, cat_vars, xvars, yvars):
    
    ctx1 = dash.callback_context
    jd1 = json.dumps({'triggered': ctx1.triggered,})
    jd1 = jd1[:50]
    
    if 'rt4.children' in jd1:
        return {}, {}, "", "", ""
    
    if df is None or xvars is None or yvars is None or len(xvars) == 0 or len(yvars) == 0:
        return {}, {}, "", "", ""
    
    elif len(xvars) == 1 and len(yvars) == 1 and xvars[0] == yvars[0]:
        return [{}, {}, "Error: Your predictor variable and response variable cannot be the same.",
                "", ""]
    
    else:
        df = pd.DataFrame(df)
        if df.empty:
            return {}, {}, "", "", ""
        
        df.drop(labels=cat_vars, axis=1, inplace=True)
        vars_ = xvars + yvars
        vars_ = list(set(vars_))
        df = df.filter(items=vars_, axis=1)
        
        if df.shape[0] == 0:
            return [{}, {}, 
                    "Error: There are no rows in the data because of the variables you chose.",
                    "", ""]
            
        else:
            
            if smartscale % 2 != 0:
                df, xvars, yvars = app_fxns.smart_scale(df, xvars, yvars)

            models = []
            coefs = []
            eqns = []
            r2s = []
            adj_r2s = []
            obs_pred_r2s = []
            intercepts = []
            pvals = []
            bics = []
            aics = []
            ns = []
            Yvars = []
            Yvars_short = []
            Xvars = []
            Xvars_short = []
            llf_ls = []
            Xs = []
            Ys = []
            PredY = []
            
            durbin_watson = []
            breusch_pagan = []
            jarque_bera = []
            harvey_collier = []
            
            
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
                        
                        if robust % 2 == 0:
                            # Run OLS                        
                            polynomial_features = PolynomialFeatures(degree = d)
                            xp = polynomial_features.fit_transform(x)
                            model = sm.OLS(y, xp).fit()
                        
                            ypred = model.predict(xp)
                            ypred = ypred.tolist()
                            
                            r2 = model.rsquared
                            try:
                                r2 = round(r2,4)
                            except:
                                pass
                            if r2 < 0:
                                r2 = 0
                                
                            r2_adj = model.rsquared_adj
                            try:
                                r2_adj = round(r2_adj,4)
                            except:
                                pass
                            if r2_adj < 0:
                                r2_adj = 0
                            
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
                            
                        elif robust % 2 != 0:
                            # Rub Robust Regression
                            y_o = tdf[yvar].values.tolist()
                            x_o = tdf[xvar].values.tolist()
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
                            op_r2 = round(app_fxns.obs_pred_rsquare(y, ypred),4)
                            if op_r2 < 0:
                                op_r2 = 0
                            r2 = 'N/A'
                            r2_adj = 'N/A'
                            
                        # Shapiro-Wilk for normally distributed errors
                        #val, sw_p = stats.shapiro(model.resid)
                        #shapiro_wilk.append(sw_p)
                        
                        # Jarque-Bera for normally distributed errors
                        jb_test = sms.jarque_bera(model.resid)
                        jb_p = round(jb_test[1], 4)
                        jarque_bera.append(jb_p)
                        
                        # Durbin-Watson for autocorrelation
                        dw = stattools.durbin_watson(model.resid)
                        durbin_watson.append(round(dw, 4))
                        
                        # Breusch-Pagan for heteroskedasticity
                        bp_test = sms.het_breuschpagan(model.resid, model.model.exog)
                        bp_p = round(bp_test[1],4)
                        breusch_pagan.append(bp_p)
                        
                        # Harvey-Collier multiplier test for linearity
                        if d == 1:
                            try:
                                skip = 10 #len(model.params)  # bug in linear_harvey_collier
                                rr = sms.recursive_olsresiduals(model, skip=skip, 
                                                                alpha=0.95, order_by=None)
                                hc_test = stats.ttest_1samp(rr[3][skip:], 0)
                                hc_p = round(hc_test[1], 4)
                            except:
                                hc_p = 'Inconclusive'
                        
                        else:
                            hc_p = 'N/A'
                        
                        harvey_collier.append(hc_p)
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
                            
                        eqns.append(eqn)
                        
                        if robust % 2 == 0:
                            aic = round(model.aic,4)
                            bic = round(model.bic,4)
                            fp = round(model.f_pvalue,4)
                            llf = round(model.llf,4)
                            
                        else:
                            aic = 'N/A'
                            bic = 'N/A'
                            fp = 'N/A'
                            llf = 'N/A'
                        
                        Yvars.append(yvar)
                        Xvars.append(xvar)
                        
                        yvar_short = str(yvar)
                        xvar_short = str(xvar)
                        
                        if len(yvar) > 60:
                            yvar_short = yvar[0:30] + ' ... ' + yvar[-30:]
                        if len(xvar) > 60:
                            xvar_short = xvar[0:30] + ' ... ' + xvar[-30:]
                        
                        Yvars_short.append(yvar_short)
                        Xvars_short.append(xvar_short)
                        
                        models.append(model_type)
                        r2s.append(r2)
                        adj_r2s.append(r2_adj)
                        obs_pred_r2s.append(op_r2)
                        pvals.append(fp)
                        bics.append(bic)
                        aics.append(aic)
                        llf_ls.append(llf)
                        ns.append(len(x))
                        Xs.append(x_o)
                        Ys.append(y_o)
                        PredY.append(ypred)
            
            del df
            cols = ['y-variable', 'x-variable', 'Model', 'r-square', 'adj. r-square', 
                    'obs vs. pred r-square', 'p-value', 'intercept', 'coefficients', 'AIC', 
                    'BIC', 'log-likelihood', 'Durbin-Watson', 'Jarque-Bera (p-value)', 
                    'Breusch-Pagan (p-value)', 'Harvey-Collier (p-value)']
            
            df_models = pd.DataFrame(columns=cols)
            df_models['y-variable'] = Yvars
            df_models['x-variable'] = Xvars
            df_models['y-variable (short)'] = Yvars_short
            df_models['x-variable (short)'] = Xvars_short
            df_models['Model'] = models
            df_models['r-square'] = r2s
            df_models['adj. r-square'] = adj_r2s
            df_models['obs vs. pred r-square'] = obs_pred_r2s
            df_models['p-value'] = pvals
            df_models['AIC'] = aics
            df_models['BIC'] = bics
            df_models['log-likelihood'] = llf_ls
            df_models['ys'] = Ys
            df_models['xs'] = Xs
            df_models['pred_y'] = PredY
            df_models['intercept'] = intercepts
            df_models['coefficients'] = coefs
            df_models['equation'] = eqns
            df_models['sample size'] = ns
            df_models['Durbin-Watson'] = durbin_watson
            df_models['Jarque-Bera (p-value)'] = jarque_bera
            df_models['Breusch-Pagan (p-value)'] = breusch_pagan
            df_models['Harvey-Collier (p-value)'] = harvey_collier
            
            ###################### Figure ########################
            
            fig_data = []
            
            df_models['label'] = df_models['y-variable (short)'] + '<br>' + '      vs.'
            df_models['label'] = df_models['label'] + '<br>' + df_models['x-variable (short)']
            
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
                    op_r2 = tdf['obs vs. pred r-square'].tolist()
                    op_r2 = op_r2[0]
                
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
                    
                    if robust % 2 != 0:
                        r2 = round(op_r2, 4)
                        
                    fig_data.append(
                        go.Scatter(
                            x = obs_x,
                            y = pred_y,
                            mode = "lines",
                            name = model + ': r<sup>2</sup> = '+str(r2),
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
            df_models.drop(labels=['x-variable (short)', 'y-variable (short)'], axis=1, inplace=True)
            
            df_models = df_models[df_models['x-variable'].isin(xvars)]
            df_models = df_models[df_models['y-variable'].isin(yvars)]
            
            cols = ['y-variable', 'x-variable', 'sample size', 'Model', 'r-square', 'adj. r-square',
                    'obs vs. pred r-square', 'p-value', 'AIC', 'BIC', 'log-likelihood', 
                    'Durbin-Watson', 'Jarque-Bera (p-value)', 'Breusch-Pagan (p-value)', 
                    'Harvey-Collier (p-value)', 'equation']
            
            df_table = df_models.filter(items=cols)
            df_table.sort_values(by='adj. r-square', inplace=True, ascending=False)
            
            dashT = dash_table.DataTable(
                data=df_table.to_dict('records'),
                columns=[{'id': c, 'name': c} for c in df_table.columns],
                export_format="csv",
                page_action='none',
                sort_action="native",
                sort_mode="multi",
                filter_action="native",
                
                style_table={'height': '500px', 
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
    
            del df_models
            
            txt1 = "This figure displays up to 10 pairs of features sharing the strongest linear "
            txt1 += "and curvilinear (polynomial) relationships. "
            txt1 += "Polynomial regression is useful when relationships are noticeably curved. "
            txt1 += "Quadratic models account for one curve and cubic models account for two. "
            txt1 += "When interpreting performance, consider whether or not a curvier model "
            txt1 += "produces meaningfully greater r\u00B2"

            txt2 = "The Durbin-Watson statistic ranges between 0 and 4. The closer it is to 2, "
            txt2 += "the more independent the observations. "
            txt2 += "Significant Jarque-Bera tests (p < 0.05) indicate non-normality. "
            txt2 += "Significant Breusch-Pagan tests (p < 0.05) indicate heteroskedasticity. "
            txt2 += "Significant Harvey-Collier test (p < 0.05) indicate non-linearity. "
            
            if robust == 1:
                txt2 += "\nNote, outputs of robust regression do not include AIC, BIC, "
                txt2 += "log-likelihood, or p-values from an F-test, or typical r-square and "
                txt2 += "adjusted r-square values. Instead, r-square values for robust regression "
                txt2 += "are based on observed vs predicted, i.e., a linear regression between "
                txt2 += "observed and predicted values with the slope constrained to 1 and the "
                txt2 += "intercept constrained to 0."
            
            return figure, dashT, "", txt1, txt2