from sklearn.linear_model import LinearRegression, LogisticRegression, GammaRegressor
from sklearn.linear_model import TweedieRegressor, PoissonRegressor
from sklearn.feature_selection import RFECV

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import dash

from plotly import graph_objects as go
import pandas as pd
import numpy as np
import json

import app_fxns


def generate_glm_outputs():

    return html.Div(
                children=[
                    dcc.Loading(
                        type="default",
                        fullscreen=False,
                        children=html.Div(
                            children=[dcc.Graph(id='figure_glm'),
                                    ],
                                ),
                        ),
                    html.P("", id='figure_glm_txt')
                    ],
                style={'width': '100%',
                       'display': 'inline-block',
                       'background-color': '#f0f0f0',
                       'padding': '1%',
                },
    )



def control_card_glm():

    return html.Div(
        children=[
                html.H5("Run a Generalized Linear Model",
                        style={'display': 'inline-block', 
                               'margin-right': '1%',
                               },
                        ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="tt_glm1",
                       style={'display': 'inline-block', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("When trying to explain or predict a non-categorical response " +
                            "variable using two or more predictors.", 
                            target="tt_glm1", 
                            style = {'font-size': 12,
                                     },
                            ),
                html.Br(),
                
                html.B("Choose 2 or more predictors",
                    style={'vertical-align': 'top',
                           'margin-right': '1%',
                           'display': 'inline-block',
                       },
                    ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="tt_glm2",
                       style={'display': 'inline-block', 
                              'color':'#bfbfbf',
                              },
                       ),
                dbc.Tooltip("The app will recognize if your response variable occurs in this " +
                            "list of predictors. If it does, the app will ignore it.",
                    target="tt_glm2", 
                    style = {'font-size': 12,
                             },
                    ),
                
                dcc.Dropdown(
                        id='glm_predictors',
                        options=[{"label": i, "value": i} for i in []],
                        multi=True, 
                        value=None,
                        style={'width': '100%',
                             },
                        ),
                
                html.Br(),
                
                html.Div(
                    children = [
                        html.B("Choose your response variable",
                            style={'vertical-align': 'top',
                                   'margin-right': '1%',
                                   'display': 'inline-block', 
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="tt_glm3",
                               style={'display': 'inline-block', 
                                      'color':'#bfbfbf',
                                      },
                               ),
                        dbc.Tooltip("Does not include categorical features or any numerical " +
                                    "feature with less than 4 unique values.", 
                                    target="tt_glm3", 
                                    style = {'font-size': 12,
                                             },
                                    ),
                        dcc.Dropdown(
                                id='glm_response_var',
                                options=[{"label": i, "value": i} for i in []],
                                multi=False, 
                                value=None,
                                optionHeight=65,
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
                
                html.Div(
                    children = [
                        html.B("Choose a model",
                            style={'display': 'inline-block', 
                                   'margin-right': '1%',
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="tt_glm5",
                               style={'display': 'inline-block', 
                                      'color':'#bfbfbf',
                                      },
                               ),
                        
                        dbc.Tooltip(
                            html.Div([
                                html.P("• Gaussian: Gaussian GLMs assume that the response " + 
                                       "variable follows a normal distribution. The response " +
                                       "variable should be continuous."),
                                html.P("• Inverse Gaussian: The response variable should be " +
                                       "greater than 0 and continuous."),
                                html.P("• Gamma: The response variable should be " +
                                       "greater than 0 and continuous."),
                                html.P("• Negative Binomial: The response variable should be a " +
                                       "non-negative integer."),
                                html.P("• Poisson: The response variable should be a " +
                                       "non-negative integer."),
                                html.P("• Tweedie: Tweedie distribution GLMs are versatile, " +
                                       "but the power parameter should be chosen carefully."),
                                ], 
                                style = {'font-size': 14, 
                                         'width': '200%',
                                         'background-color': "#000000",
                                         'text-align': 'left',
                                         },
                                ),
                                target="tt_glm5",
                                style = {#'font-size': 14, 
                                         'width': '30%',
                                         'background-color': "#000000",
                                         'text-align': 'left',
                                         },
                                ),
                        dcc.Dropdown(
                            id='glm_model',
                            options=[{"label": i, "value": i} for i in [#'Binomial', 
                                                                        'Gamma',
                                                                        'Gaussian', 
                                                                        'InverseGaussian',
                                                                        'NegativeBinomial', 
                                                                        'Poisson', 
                                                                        'Tweedie']
                                     ],
                            multi=False, 
                            value='Gaussian',
                            style={'width': '100%', 
                                   'display': 'block',
                                   },
                            ),
                        ],
                        style={'width': '15%',
                               'display': 'inline-block',
                               'margin-right': '5%',
                               },
                    ),
                
                html.Div(
                    children = [
                        html.B("Remove unimportant variables",
                            style={'display': 'inline-block', 
                                   'margin-right': '1%',
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="tt_glm4",
                               style={'display': 'inline-block', 
                                      'color':'#bfbfbf',
                                      },
                               ),
                        dbc.Tooltip("Remove variables that have little-to-no effect on model " +
                                    "performance. Removal is done using recursive feature " +
                                    "elimination with 5-fold cross-validation. Unimportant " +
                                    "features will not be removed if the resulting number of " +
                                    "variables is less 2.", 
                                    target="tt_glm4", 
                                    style = {'font-size': 12,
                                             },
                                    ),
                        dcc.Dropdown(
                            id='rfecv_glm',
                            options=[{"label": i, "value": i} for i in ['Yes', 'No']],
                            multi=False, 
                            value='Yes',
                            style={'width': '60%', 
                                   'display': 'inline-block',
                             },
                            ),
                        ],
                        style={'width': '30%',
                               'display': 'inline-block',
                        },
                    ),
                
                
                html.Br(),
                html.Br(),
                
                dbc.Button('Run GLM', 
                           id='btn_glm', 
                           n_clicks=0,
                    style={'display': 'inline-block',
                           'width': '18%',
                           'font-size': 12,
                           'margin-right': '3%',
                           'background-color': "#2a8cff",
                           },
                    ),
                
                dbc.Button("View parameters table",
                           id="open-glm_parameters_table",
                           style={
                               "background-color": "#2a8cff",
                               'width': '18%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '3%',
                               },
                    ),
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id='glm_params_table'), 
                                    html.Br(), 
                                    html.P("", id='glm_params_txt'),
                                    ],
                                   ),
                     dbc.ModalFooter(dbc.Button("Close", 
                                                id="close-glm_parameters_table", 
                                                className="ml-auto")
                                    ),
                            ],
                    id="modal-glm_parameters_table",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button("View model performance",
                           id="open-glm_performance_table",
                           style={
                               "background-color": "#2a8cff",
                               'width': '18%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '3%',
                               },
                    ),
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id='glm_performance_table'), 
                                    html.Br(), 
                                    html.P("Adjusted R-square accounts for sample size and " +
                                           "the number of predictors used."),
                                    ],
                                   ),
                     dbc.ModalFooter(
                                    dbc.Button("Close", 
                                               id="close-glm_performance_table", 
                                               className="ml-auto")
                                    ),
                            ],
                    id="modal-glm_performance_table",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button('Smart scale', 
                            id='btn_ss_glm', 
                            n_clicks=0,
                            style={
                                'font-size': 12,
                                "background-color": "#2a8cff",
                                'display': 'inline-block',
                                'margin-right': '1%'
                                },
                            ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="tt_ss_glm",
                       style={'display': 'inline-block', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("Skewed data can weaken analyses and visualizations. " +
                            "Click on 'Smart Scale' and the app will automatically detect and " +
                            "apply the best scaling for each skewed variable. Smart scaling " +
                            "will not necessarily improve the r-square. To remove the " +
                            "rescaling just click 'Run GLM'.", 
                            target='tt_ss_glm', 
                            style = {'font-size': 12,
                                     },
                            ),
                html.P("", id = 'rt1_glm'),
                ],
                style={'width': '98.5%',
                       'margin-left': '1%',
                       },
            )




def run_glm(df, xvars, yvar, cat_vars, rfe_val, family):
    """
    Perform Poisson regression using statsmodels.

    Parameters:
    - df_train (pd.DataFrame): The input data as a DataFrame.
    - xvars (list): List of predictor column names.
    - yvar (str): The name of the target column.
    - cat_vars (list): List of categorical variable column names.
    - rfe_val (str): 'Yes' or 'No' to indicate whether to use recursive feature elimination (RFE).

    Returns:
    - y_obs (pd.Series): The target variable.
    - y_pred (pd.Series): Predicted values.
    - df1_summary (pd.DataFrame): Parameter estimates table.
    - df2_summary (pd.DataFrame): Model summary table.
    - supported_features (list): List of supported predictor variables.
    - unsupported (list): List of unsupported predictor variables.
    - colors (list): List of colors used for later processing.
    """
    
    df, dropped, cat_vars_ls = app_fxns.dummify(df, cat_vars)
    
    # Treat data to prevent regression from failing
    if family == 'Poisson':
        pass
        
    #elif family == 'Binomial':
    #    pass
        
    elif family == 'Gamma': 
        df = df[df[yvar] > 0]
        
    elif family == 'Gaussian':
        pass
    
    elif family == 'InverseGaussian':
        pass
    
    elif family == 'NegativeBinomial':
        pass
    
    elif family == 'Tweedie':
        pass
    
    if df.shape[1] < 2:
        return pd.Series(), pd.Series(), pd.DataFrame(), pd.DataFrame(), [], [], []
    
    # Eliminate features with many 0's
    x_vars = list(df)
    try:
        x_vars.remove(yvar)
    except:
        pass
    
    drop = []
    for var in x_vars:
        vals = df[var].tolist()
        if len(vals) < 1:
            drop.append(var)
        else:
            frac_0 = vals.count(0) / len(vals)
            if frac_0 > 0.95:
                drop.append(var)
    
    df.drop(labels=drop, axis=1, inplace=True)
    df = app_fxns.remove_nans_optimal(df, yvar)
    
    y_obs = df.pop(yvar)
    
    supported_features = []
    unsupported = []

    results = [] 
    ranks = []
    xlabs = []
    
    # RUN RFECV
    if family == 'Poisson':
        poisson_model = PoissonRegressor()
        rfecv = RFECV(estimator=poisson_model, cv=5)
        
    #elif family == 'Binomial':
    #    binomial_model = LogisticRegression()
    #    rfecv = RFECV(estimator=binomial_model, cv=5)
        
    elif family == 'Gamma':
        gamma_model = GammaRegressor()
        rfecv = RFECV(estimator=gamma_model, cv=5)
        
    elif family == 'Gaussian':
        gaussian_model = LinearRegression()
        rfecv = RFECV(estimator=gaussian_model, cv=5)
        
    elif family == 'InverseGaussian':
        inverse_gaussian_model = TweedieRegressor(power=0)  # Specify the power parameter for Inverse link
        rfecv = RFECV(estimator=inverse_gaussian_model, cv=5)
        
    elif family == 'NegativeBinomial':
        negative_binomial_model = PoissonRegressor()
        rfecv = RFECV(estimator=negative_binomial_model, cv=5)
        
    elif family == 'Tweedie':
        tweedie_model = TweedieRegressor(power=1.5)  # Adjust the power parameter as needed
        rfecv = RFECV(estimator=tweedie_model, cv=5)
    
    rfecv.fit(df, y_obs)
    
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
                    df.drop(l, axis=1, inplace=True)
                    unsupported.append(l)
                except:
                    pass
                    
        elif len(check) > 0:
            supported_features.extend(ls)
            supported_features = list(set(supported_features))
    
    if len(supported_features) >= 2:
        if rfe_val == 'Yes':
            df = df.filter(items = supported_features, axis=1)
    
    X_train_lm = sm.add_constant(df, has_constant='add')
    
    if family == 'Poisson':
        # Poisson (Log Link)
        #results = sm.GLM(y_obs, X_train_lm, family=sm.families.Poisson(sm.families.links.log())).fit()
        results = sm.GLM(y_obs, X_train_lm, family=sm.families.Poisson()).fit()

    #elif family == 'Binomial': 
        # Binomial (Logit Link)
    #    results = sm.GLM(y_obs, X_train_lm, family=sm.families.Binomial()).fit()

    elif family == 'Gamma':
        # Gamma (Identity Link)
        results = sm.GLM(y_obs, X_train_lm, family=sm.families.Gamma()).fit()

    elif family == 'Gaussian':
        # Gaussian (Identity Link)
        results = sm.GLM(y_obs, X_train_lm, family=sm.families.Gaussian()).fit()

    elif family == 'InverseGaussian':
        # InverseGaussian (Inverse Link)
        results = sm.GLM(y_obs, X_train_lm, family=sm.families.InverseGaussian()).fit()

    elif family == 'NegativeBinomial':
        # NegativeBinomial (Log Link)
        results = sm.GLM(y_obs, X_train_lm, family=sm.families.NegativeBinomial()).fit()

    elif family == 'Tweedie':
        # Tweedie (Power Link), Adjust power parameter as needed
        results = sm.GLM(y_obs, X_train_lm, family=sm.families.Tweedie(var_power=1.5)).fit()

    y_pred = results.predict(X_train_lm)
    results_summary = results.summary()
    
    results_as_html1 = results_summary.tables[1].as_html()
    df1_summary = pd.read_html(results_as_html1, header=0, index_col=None)[0]
    df1_summary.rename(columns={'Unnamed: 0': 'Variable',}, inplace=True)
    
    results_as_html2 = results_summary.tables[0].as_html()
    df2_summary = pd.read_html(results_as_html2, header=None, index_col=None)[0]
    
    for col in list(df2_summary):
        df2_summary.rename(columns={col: str(col)}, inplace=True)
    
    vifs = [variance_inflation_factor(df.values, j) for j in range(df.shape[1])]
    vifs2 = []
    
    xlabs = list(df)
    for p in df1_summary['Variable'].tolist():
        if p == 'const':
            vifs2.append(np.nan)
        else:
            i = xlabs.index(p)
            vif = vifs[i]
            vifs2.append(np.round(vif, 3))
        
    df1_summary['VIF'] = vifs2
    colors = ["#3399ff"] * len(y_obs)
    
    return y_obs, y_pred, df1_summary, df2_summary, supported_features, unsupported, colors



def get_updated_results(n_clicks, smartscale, xvars, yvar, df, cat_vars, rfe_val, glm_model):
    
    cols = ['Model information', 'Model statistics']
    df_table1 = pd.DataFrame(columns=cols)
    df_table1['Model information'] = [np.nan]*10
    df_table1['Model statistics'] = [np.nan]*10
    
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
        return {}, dashT1, dashT2, "", "", "", 0, 0
    
    if df is None:
        return {}, dashT1, dashT2, "", "", "", 0, 0
    
    elif yvar is None and xvars is None:
        return {}, dashT1, dashT2, "", "", "", 0, 0
    
    elif xvars is None or len(xvars) < 1:
        return {}, dashT1, dashT2, "Error: Select one or more predictors", "", "", 0, 0
        
    elif yvar is None:
        return {}, dashT1, dashT2, "Error: Select a reponse variable", "", "", 0, 0
    
    elif (isinstance(yvar, list) is True) & (xvars is None or len(xvars) < 1):
        return [{}, dashT1, dashT2, 
                "Error: Select a response variable and 1 or more predictors", "", "", 0, 0]
    
    elif isinstance(yvar, list) is True:
        return {}, dashT1, dashT2, "Error: Select a response variable", "", "", 0, 0
    
    elif xvars is None or len(xvars) < 2:
        return {}, dashT1, dashT2, "Error: Select two or more predictors", "", "", 0, 0
    
    df = pd.DataFrame(df)
    if df.empty:
        return {}, dashT1, dashT2, "", "", "", 0, 0
    
    if yvar not in list(df):
        return {}, dashT1, dashT2, "Error: Choose a response variable", "", "", 0, 0
        
    if yvar in xvars:
        xvars.remove(yvar)
        if len(xvars) == 0:
            return [{}, dashT1, dashT2, 
                    "Error: GLM requires 2 or more predictors. You chose one and it's the same as your response variable", 
                    "", "", 0, 0]
        
    if len(xvars) < 1 and yvar is None:
        return [{}, dashT1, dashT2, 
                "Error: GLM requires 1 or more predictors and one response variable.", "", "", 0, 0]
        
    elif len(xvars) < 1:
        return [{}, dashT1, dashT2, 
                "Error: GLM requires 1 or more predictors.", "", "", 0, 0]
                        
    else:
        vars_ = [yvar] + xvars
        df = df.filter(items=vars_, axis=1)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = app_fxns.remove_nans_optimal(df, yvar)
        
        if smartscale == 1:
            df, xvars, yvars = app_fxns.smart_scale(df, xvars, [yvar])
            yvar = yvars[0]
        
        #Conduct glm
        ls = run_glm(df, xvars, yvar, cat_vars, rfe_val, family=glm_model)
        y_obs, y_pred, df1_summary, df2_summary, supported_features, unsupported, colors = ls
        
        if len(y_obs) == 0:
            rt1 = "Error: Your regression could not run. Your y-values contain no data."
            return {}, dashT1, dashT2, rt1, "", "", 0, 0
        
        r2_obs_pred = app_fxns.obs_pred_rsquare(y_obs, y_pred)
        r2_obs_pred = round(r2_obs_pred,2)
        
        y_obs = y_obs.tolist()
        y_pred = y_pred.tolist()
        
        y_pred_outliers = []
        y_pred_nonoutliers = []
        y_obs_outliers = []
        y_obs_nonoutliers = []
        
        for i, clr in enumerate(colors):
            if clr == "#ff0000":
                y_pred_outliers.append(y_pred[i])
                y_obs_outliers.append(y_obs[i])
            elif clr == "#3399ff":
                y_pred_nonoutliers.append(y_pred[i])
                y_obs_nonoutliers.append(y_obs[i])
        
        fig_data = []
        
        miny = min([min(y_obs), min(y_pred)])
        maxy = max([max(y_obs), max(y_pred)])
        
        fig_data.append(go.Scatter(x = y_pred_nonoutliers, 
                                   y = y_obs_nonoutliers, 
                                   name = 'Obs vs Pred',
                                   mode = "markers", 
                                   opacity = 0.75, 
                                   marker = dict(size=10, 
                                                 color="#3399ff"),
                                   ),
                        )
        
        #fig_data.append(go.Scatter(x = y_pred_outliers, y = y_obs_outliers, name = 'Outliers',
        #        mode = "markers", opacity = 0.75, marker = dict(size=10, color="#ff0000")))
        
        fig_data.append(go.Scatter(x = [miny, maxy], 
                                   y = [miny, maxy], 
                                   name = '1:1, r<sup>2</sup> = ' + str(r2_obs_pred),
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   line = dict(color = "#595959", 
                                               width = 1, 
                                               dash='dash'),
                                   ),
                        )
                            
        figure = go.Figure(data = fig_data,
            layout = go.Layout(
                xaxis = dict(title = dict(
                        text = "<b>" + 'Predicted:  ' + yvar + "</b>",
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif", 
                            size = 18,
                            ),
                        ), 
                    showticklabels = True,
                    ),
                                            
                yaxis = dict(title = dict(
                        text = "<b>" + 'Observed:  ' + yvar + "</b>",
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif",
                            size = 18,
                            ),
                        ),
                    showticklabels = True,
                    ),
                                            
                margin = dict(l=60, r=30, b=10, t=40), 
                showlegend = True, height = 400,
                paper_bgcolor = "rgb(245, 247, 249)", 
                plot_bgcolor = "rgb(245, 247, 249)",
            ),
        )

        dashT1 = dash_table.DataTable(
            data=df1_summary.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df1_summary.columns],
            export_format="csv",
            page_action='none',
            sort_action="native",
            sort_mode="single",
            filter_action="none",
            
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
        
        df2_summary['Model information']=df2_summary['0'].astype(str)+' '+df2_summary['1'].astype(str)
        df2_summary['Model statistics']=df2_summary['2'].astype(str)+' '+df2_summary['3'].astype(str)
        
        df2_summary.drop(labels=['0', '1', '2', '3'], 
                         axis=1, 
                         inplace=True)
        
        dashT2 = dash_table.DataTable(
            data=df2_summary.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df2_summary.columns],
            export_format="csv",
            page_action='none',
            sort_action="none",
            filter_action="none",
            
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
        
        txt1 = "This plot allows you to interpret patterns in the regression model's success. "
        txt1 += "Example: If points are consistently above the 1:1 line, then the observed values "
        txt1 += "are always greater than the predicted values. If the relationship is curved and "
        txt1 += "performance is weak, then try rescaling some of your variables "
        txt1 += "(via log, square root, etc.)."
        
        txt2 = "The variance inflation factor (VIF) measures multicollinearity. VIF > 5 indicates "
        txt2 += "that a predictor is significantly correlated with one or more other predictors. "
        txt2 += "VIF > 10 indicates severe multicollinearity, which can lead to overfitting and "
        txt2 += "inaccurate parameter estimates. If your VIF's are high, trying removing some of "
        txt2 += "those variables."
        
        del df2_summary
        del df1_summary
        
        return figure, dashT1, dashT2, "", txt1, txt2, 0, 0