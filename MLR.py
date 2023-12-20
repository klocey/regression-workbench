from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import pinv_extended
import statsmodels.api as sm

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import dash

from plotly import graph_objects as go
import pandas as pd
import numpy as np
import json

import app_fxns


def generate_linear_multivariable_outputs():

    return html.Div(
                children=[
                    dcc.Loading(
                        type="default",
                        fullscreen=False,
                        children=html.Div(
                            children=[dcc.Graph(id='figure_multiple_linear_regression'),
                                    ],
                                ),
                        ),
                    html.P("", id='fig3txt')
                    ],
                style={'width': '100%',
                       'display': 'inline-block',
                       'background-color': '#f0f0f0',
                       'padding': '1%',
                       },
                )


def control_card_linear_multivariable():

    return html.Div(
        children=[
                html.H5("Conduct linear forms of multivariable regression",
                        style={'display': 'inline-block', 
                               'margin-right': '1%',
                               },
                        ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="target_select_vars3",
                       style={'display': 'inline-block', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("This analysis is based on ordinary least squares regression and " +
                            "reveals predicted values, outliers, and the significance of " +
                            "individual variables.", 
                            target="target_select_vars3", 
                            style = {'font-size': 12,
                                     },
                            ),
                html.P("When trying to explain or predict a non-categorical response variable " +
                       "using two or more predictors."),
                
                
                html.B("Choose 2 or more predictors",
                    style={'vertical-align': 'top',
                           'display': 'inline-block',
                           'margin-right': '1%',
                       },
                    ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="target_mlrx1",
                       style={'display': 'inline-block', 
                           'color':'#bfbfbf',
                           },
                       ),
                dbc.Tooltip("The app will recognize if your response variable occurs in this " +
                            "list of predictors. If it does, the app will ignore it.",
                    target="target_mlrx1", 
                    style = {'font-size': 12,
                             },
                    ),
                
                dcc.Dropdown(
                        id='xvar3',
                        options=[{"label": i, "value": i} for i in []],
                        multi=True, 
                        value=None,
                        ),
                
                html.Br(),
                
                html.Div(
                    children = [
                        html.B("Choose your response variable",
                            style={'vertical-align': 'top',
                                   'display': 'inline-block', 
                                   'margin-right': '4%',
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="target_y1",
                               style={'display': 'inline-block', 
                                      'color':'#bfbfbf',
                                      },
                               ),
                        dbc.Tooltip("Does not include categorical features or any numerical " +
                                    "feature with less than 4 unique values.",
                            target="target_y1", 
                            style = {'font-size': 12,
                                     },
                            ),
                        dcc.Dropdown(
                                id='yvar3',
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
                              'margin-right': '2%',
                            },
                        ),
                
                html.Div(
                children=[
                    html.B("Choose a transformation",
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'margin-right': '3%',
                        }),
                    html.I(className="fas fa-question-circle fa-lg", 
                           id="transform_mlr_response",
                           style={'display': 'inline-block', 
                                  'color':'#bfbfbf',
                                  },
                           ),
                    dbc.Tooltip("For rescaling your response variable",
                        target="transform_mlr_response",
                        style = {'font-size': 12,
                                 },
                        ),
                    dcc.Dropdown(
                            id='response_transform',
                            options=[{"label": i, "value": i} for i in ['None', 'log10', 
                                                                        'square root', 'cube root',
                                                                        'squared', 'cubed', 
                                                                        'log-modulo', 'log-shift',
                                                                        ]
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
                           'margin-right': '3%',
                           'width': '20%',
                    }),
                
                html.Div(
                    children = [
                        html.B("Choose a regression type",
                            style={'vertical-align': 'top',
                                   'display': 'inline-block', 
                                   'margin-right': '4%',
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="mlr_type",
                               style={'display': 'inline-block', 
                                      'color':'#bfbfbf',
                                      },
                               ),
                        dbc.Tooltip("Does not include categorical features or any numerical " +
                                    "feature with less than 4 unique values.",
                            target="mlr_type", 
                            style = {'font-size': 12,
                                     },
                            ),
                        dcc.Dropdown(
                                id='mlr_model',
                                options=[{"label": i, "value": i} for i in ['OLS', 
                                                                            'Ridge', 
                                                                            'Lasso']],
                                multi=False, 
                                value='OLS',
                                optionHeight=65,
                                style={'width': '90%',
                                       'display': 'inline-block',
                                     },
                                ),
                            ],
                            style={'width': '20%',
                              'display': 'inline-block',
                              'margin-right': '2%',
                            },
                        ),
                
                html.Div(id='choose_ref',
                    children = [
                        html.B("Remove unimportant variables",
                            style={'display': 'inline-block', 
                                   'margin-right': '2%',
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="target_rfe",
                               style={'display': 'inline-block', 
                                      'width': '5%', 
                                      'color':'#bfbfbf',
                                      },
                               ),
                        dbc.Tooltip("Remove variables that have little-to-no effect on model " +
                                    "performance. Removal is done using recursive feature " +
                                    "elimination with 5-fold cross-validation. Unimportant " +
                                    "features will not be removed if the resulting number of " +
                                    "variables is less 2.", 
                                    target="target_rfe", 
                                    style = {'font-size': 12,
                                             },
                                    ),
                        dcc.Dropdown(
                            id='rfecv',
                            options=[{"label": i, "value": i} for i in ['Yes', 'No']],
                            multi=False, 
                            value='Yes',
                            style={'width': '80%', 
                                   'display': 'inline-block',
                                   },
                            ),
                        ],
                        style={'width': '20%',
                               'display': 'inline-block',
                        },
                    ),
                        
                html.Br(),
                html.Br(),
                
                dbc.Button('Run regression', 
                           id='btn3', 
                           n_clicks=0,
                    style={'display': 'inline-block',
                           'width': '18%',
                           'font-size': 12,
                           'margin-right': '2%',
                           "background-color": "#2a8cff",
                           },
                    ),
                
                dbc.Button("View parameters table",
                           id="open-centered3",
                           style={
                               "background-color": "#2a8cff",
                               'width': '18%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '2%',
                               },
                           ),
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id="table_plot3b"), 
                                    html.Br(), 
                                    html.P("", id='tab3btxt')]),
                                    dbc.ModalFooter(
                                            dbc.Button("Click to Close",
                                               id="close-centered3", 
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
                    id="modal-centered3",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    fullscreen=True,
                    #size="xl",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button("View model performance",
                           id="open-centered4",
                           style={
                               "background-color": "#2a8cff",
                               'width': '18%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '2%',
                               },
                    ),
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id="table_plot3a"), 
                                    html.Br(), 
                                    html.P("Adjusted R-square accounts for sample size and the " +
                                           "number of predictors used."),
                                    ],
                                   ),
                     dbc.ModalFooter(
                             dbc.Button("Click to Close",
                                id="close-centered4",  
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
                    id="modal-centered4",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button('Smart scale', 
                            id='btn_ss2', 
                            n_clicks=0,
                            style={
                                'font-size': 12,
                                'background-color': '#2a8cff',
                                'display': 'inline-block',
                                'width': '15%',
                                'margin-right': '1%',
                                },
                            ),
                
                html.I(className="fas fa-question-circle fa-lg", 
                       id="ss2",
                       style={'display': 'inline-block', 
                              'width': '3%', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("Skewed data can weaken analyses and visualizations. Click on " +
                            "'Smart Scale' and the app will automatically detect and apply the " +
                            "best scaling for each skewed variable. Smart scaling will not " +
                            "necessarily improve the r-square. To remove the rescaling just " +
                            "click 'Run Multiple Regression'.", 
                            target="ss2", 
                            style = {'font-size': 12,
                                     },
                            ),
                
                html.P("", id = 'rt1'),
                ],
                style={'width': '98.5%',
                       'margin-left': '1%',
                    },
            )



def run_MLR(df_train, xvars, yvar, cat_vars, rfe_val, mlr_model):

    X_train = df_train.copy(deep=True)
    X_train, dropped, cat_vars_ls = app_fxns.dummify(X_train, cat_vars)
    X_train = app_fxns.remove_nans_optimal(X_train, yvar)
    
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
    y_train = X_train.pop(yvar)
    
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
        
        # elements of ls that are in supported_features
        check = list(set(ls) & set(supported_features)) 
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
    
    model = sm.OLS(y_train, X_train_lm)
    results = None
    
    if mlr_model == 'OLS':
        results = model.fit()

    elif mlr_model == 'Ridge' or mlr_model == 'Lasso':
        '''
        Note: Statsmodels does not provide summary tables for Ridge regression and Lasso regression. 
        Hence, users are not provided with statistics of model performance and other valuable info.
        It's a recognized issue with people repeatedly asking for greater development.
        
        Here, we combine regularization during model development and then refit with OLS for final 
        interpretation and reporting. We then proceed with the manual construction of an OLSResults 
        object using the estimated parameters and normalized covariance parameters.
        
        This approach allows benefits from regularization during model development while providing 
        flexibility to obtain an OLS-like result for further analysis or interpretation. 
        This can be particularly useful when wanting to balance the benefits of regularization with 
        the interpretability and simplicity of OLS results.
        '''
        
        
        '''    
        1. Use the refit argument in the `fit_regularized` function. The `refit` option in the 
        Statsmodels fit_regularized function allows refitting of the model using the estimated 
        parameters from the regularized fit. When refit=True, the function fits the model again 
        using the estimated parameters obtained during the regularization process. This refitting 
        step is done without regularization, effectively providing an OLS fit based on the 
        estimated parameters from the regularized model.
        '''
        
        if mlr_model == 'Ridge':
            # Setting L1_wt=0.0 for pure L2 regularization
            '''
            Use fit_regularized with L1_wt=0.0 to perform pure L2 (Ridge) regularization. 
            The refit=True option is used to obtain an OLS refit using the estimated parameters 
            from the regularization step.
            '''
            
            results = model.fit_regularized(alpha=1.0, L1_wt=0.0, refit=True)
        
        elif mlr_model == 'Lasso':
            # Setting L1_wt=1.0 for pure L1 regularization
            '''
            Use fit_regularized with L1_wt=1.0 to perform pure L1 (Lasso) regularization. 
            The refit=True option is used to obtain an OLS refit using the estimated parameters 
            from the regularization step.
            '''
            results = model.fit_regularized(alpha=1.0, L1_wt=1.0, refit=True)
        
        '''
        2. Use the pinv_extended function from statsmodels.tools.tools to compute the 
        Moore-Penrose pseudo inverse of the design matrix (model.wexog). This pseudo inverse is 
        then used to compute the normalized covariance parameters, which are subsequently used 
        to create a summary object.
        '''
        
        pinv_wexog,_ = pinv_extended(model.wexog)
        normalized_cov_params = np.dot(pinv_wexog, 
                                       np.transpose(pinv_wexog),
                                       )
        results = sm.regression.linear_model.OLSResults(model,
                                                        results.params,
                                                        normalized_cov_params,
                                                        )
        
    results_summary = results.summary()
    y_pred = results.predict(X_train_lm)
    
    #pval_df = results.pvalues
    R2 = results.rsquared_adj
    if R2 < 0: R2 = 0
    
    results_as_html1 = results_summary.tables[1].as_html()
    df1_summary = pd.read_html(results_as_html1, header=0, index_col=0)[0]
    
    results_as_html2 = results_summary.tables[0].as_html()
    df2_summary = pd.read_html(results_as_html2, header=0, index_col=0)[0]
    
    vifs = [variance_inflation_factor(X_train.values, j) for j in range(X_train.shape[1])]
    
    colors = ["#3399ff"]*len(y_train)
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



def get_updated_results(n_clicks, smartscale, xvars, yvar, df, cat_vars, rfe_val, mlr_model, y_transform):
    
                        
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
        return {}, dashT1, dashT2, "", "", "", 0
    
    if df is None:
        return {}, dashT1, dashT2, "", "", "", 0
    
    elif yvar is None and xvars is None:
        return {}, dashT1, dashT2, "", "", "", 0
    
    elif xvars is None or len(xvars) < 2:
        return {}, dashT1, dashT2, "Error: Select two or more predictors", "", "", 0
        
    elif yvar is None:
        return {}, dashT1, dashT2, "Error: Select a reponse variable", "", "", 0
    
    elif (isinstance(yvar, list) is True) & (xvars is None or len(xvars) < 2):
        return [{}, dashT1, dashT2, "Error: Select a response variable and 2 or more predictors", 
                "", 0]
    
    elif isinstance(yvar, list) is True:
        return {}, dashT1, dashT2, "Error: Select a response variable", "", "", 0
    
    elif xvars is None or len(xvars) < 2:
        return {}, dashT1, dashT2, "Error: Select two or more predictors", "", "", 0
    
    df = pd.DataFrame(df)
    if df.empty:
        return {}, dashT1, dashT2, "", "", "", 0
    
    if yvar not in list(df):
        return {}, dashT1, dashT2, "Error: Choose a response variable", "", "", 0
        
    if yvar in xvars:
        xvars.remove(yvar)
        if len(xvars) == 0:
            return [{}, dashT1, dashT2, "Error: Multiple regression requires 2 or more predictors. You chose one and it's the same as your response variable", "", "", 0]
        
        elif len(xvars) == 1:
            return {}, dashT1, dashT2, "Error: Multiple regression requires 2 or more predictors. You chose two but one is the same as your response variable", "", "", 0
    
    if len(xvars) < 2 and yvar is None:
        return {}, dashT1, dashT2, "Error: Multiple regression requires 2 or more predictors and one response variable.", "", "", 0
        
    elif len(xvars) < 2:
        return {}, dashT1, dashT2, "Error: Multiple regression requires 2 or more predictors.", "", "", 0
                        
    else:
        
        vars_ = [yvar] + xvars
        df = df.filter(items=vars_, axis=1)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if smartscale == 1:
            df, xvars, yvars = app_fxns.smart_scale(df, xvars, [yvar], transform_res=False)
            yvar = yvars[0]
        
        if y_transform is None or y_transform == 'None':
            pass
        
        else:
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
            
        #Conduct multiple regression
        ls = run_MLR(df, xvars, yvar, cat_vars, rfe_val, mlr_model)
        y_train, y_pred, df1_summary, df2_summary, supported_features, unsupported, colors = ls 
        
        if len(y_train) == 0:
            rt1 = "Error: Your regression could not run. Your y-values contain no data."
            return {}, dashT1, dashT2, rt1, "", "", 0
        
        r2_obs_pred = app_fxns.obs_pred_rsquare(y_train, y_pred)
        r2_obs_pred = round(r2_obs_pred,2)
        
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
        
        fig_data.append(go.Scatter(x = y_pred_nonoutliers, 
                                   y = y_train_nonoutliers, 
                                   name = 'Obs vs Pred',
                                   mode = "markers", 
                                   opacity = 0.75, 
                                   marker = dict(size=10, 
                                                 color="#3399ff",
                                                 ),
                                   ),
                        )
        
        #fig_data.append(go.Scatter(x = y_pred_outliers, y = y_train_outliers, name = 'Outliers',
        #        mode = "markers", opacity = 0.75, marker = dict(size=10, color="#ff0000")))
        
        fig_data.append(go.Scatter(x = [miny, maxy], 
                                   y = [miny, maxy], 
                                   name = '1:1, r<sup>2</sup> = ' + str(r2_obs_pred),
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   line = dict(color = "#595959", 
                                               width = 1, 
                                               dash='dash',
                                               ),
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
                showlegend = True, 
                height = 400,
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
            sort_mode="multi",
            filter_action="native",
            fixed_rows={'headers': True},
            
            style_table={'overflowY': 'auto',
                         },
            style_cell={'padding':'5px',
                        'minwidth':'140px',
                        'width':'160px',
                        'maxwidth':'160px',
                        'whiteSpace':'normal',
                        'textAlign': 'center',
                        },
        )
        
        del df
        
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
        
        if mlr_model == 'Ridge':
            c1[0] = 'Model: Ridge'
            c1_str = 'Method: Statsmodels does not provide summary tables for Ridge regression. ' 
            c1_str += 'To produce this summary table, OLS was fitted using the normalized '
            c1_str += 'covariance parameters from the Ridge regression model.'
            c1[1] = c1_str
            
        if mlr_model == 'Lasso':
            c1[0] = 'Model: Lasso'
            c1_str = 'Method: Statsmodels does not provide summary tables for Lasso regression. ' 
            c1_str += 'To produce this summary table, OLS was fitted using the normalized '
            c1_str += 'covariance parameters from the Lasso regression model.'
            c1[1] = c1_str
            
        
        df4['Model information'] = c1
        df4['Model statistics'] = c2
        del df2_summary
        
        
        dashT2 = dash_table.DataTable(
            data=df4.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df4.columns],
            export_format="csv",
            page_action='none',
            sort_action="native",
            sort_mode="multi",
            
            style_table={'overflowY': 'auto',
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
        txt1 += "Example: If points are consistently above the 1:1 line, then the observed "
        txt1 += "values are always greater than the predicted values. If the relationship is "
        txt1 += "curved and performance is weak, then try rescaling some of your variables "
        txt1 += "(via log, square root, etc.)."
        
        txt2 = "The variance inflation factor (VIF) measures multicollinearity. VIF > 5 indicates "
        txt2 += "that a predictor is significantly correlated with one or more other predictors. "
        txt2 += "VIF > 10 indicates severe multicollinearity, which can lead to overfitting and "
        txt2 += "inaccurate parameter estimates. If your VIF's are high, trying removing some "
        txt2 += "of those variables."
        
        return figure, dashT2, dashT1, "", txt1, txt2, 0