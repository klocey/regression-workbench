from statsmodels.stats.outliers_influence import variance_inflation_factor

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import dash

from plotly import graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import contextlib
import json
import io

import app_fxns

plt.switch_backend('Agg')



def generate_survival_outputs():

    return html.Div(
                children=[
                    dcc.Loading(
                        type="default",
                        fullscreen=False,
                        children=html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.H5("Time to Event Histograms",
                                            style={'display': 'inline-block', 
                                                   'color': '#505050',
                                                   'font-size': 20,
                                                   'margin-left': '2%',
                                                   'margin-right': '1%',
                                                   },
                                            ),
                                        html.I(className="fas fa-question-circle fa-lg", 
                                               id="time_to_event",
                                               style={'display': 'inline-block', 
                                                      'color':'#bfbfbf',
                                                      },
                                               ),
                                        dbc.Tooltip("This analysis presents 2 histograms representing distributions of your duration variable for your two event classes.",
                                            target="time_to_event", 
                                            style = {'font-size': 12,
                                                     },
                                            ),
                                        dcc.Graph(id='time_to_event_figure',
                                            style={'width': '100%',
                                                   'display': 'inline-block',
                                                   #'background-color': '#f0f0f0',
                                                   'padding': '1%',
                                                   },
                                                ),
                                        ],
                                    style={'width': '49%',
                                           'display': 'inline-block',
                                           },
                                    ),
                                    
                                html.Div(
                                    children=[
                                        html.H5("Kaplan-Meier Survival Curve",
                                            style={'display': 'inline-block', 
                                                   'color': '#505050',
                                                   'font-size': 20,
                                                   'margin-left': '2%',
                                                   'margin-right': '1%',
                                                   },
                                            ),
                                        html.I(className="fas fa-question-circle fa-lg", 
                                               id="kaplan_meier",
                                               style={'display': 'inline-block', 
                                                      'color':'#bfbfbf',
                                                      },
                                               ),
                                        dbc.Tooltip("Kaplan-Meier curves depict the probability " +
                                                    "of survival over time in medical or survival " +
                                                    " analysis, showing cumulative survival rates " +
                                                    "based on observed event occurrences.",
                                            target="kaplan_meier", 
                                            style = {'font-size': 12,
                                                     },
                                            ),
                                        dcc.Graph(id='kaplan_meier_curve_figure',
                                                style={'width': '100%',
                                                       'display': 'inline-block',
                                                       #'background-color': '#f0f0f0',
                                                       'padding': '1%',
                                                       },
                                                ),
                                        ],
                                    style={'width': '49%',
                                           'display': 'inline-block',
                                           },
                                    ),
                                
                                html.Br(),
                                html.Br(),
                                
                                html.Div(
                                    children=[
                                        html.H5("Nelson-Aalen Cumulative Hazard Curve",
                                            style={'display': 'inline-block', 
                                                   'color': '#505050',
                                                   'font-size': 20,
                                                   'margin-left': '2%',
                                                   'margin-right': '1%',
                                                   },
                                            ),
                                        html.I(className="fas fa-question-circle fa-lg", 
                                               id="nelson_aalen",
                                               style={'display': 'inline-block', 
                                                      'color':'#bfbfbf',
                                                      },
                                               ),
                                        dbc.Tooltip("Nelson-Aalen curves illustrate the " +
                                                    "cumulative hazard function in survival " +
                                                    "analysis, representing the cumulative risk " +
                                                    "of an event occurring over time without " +
                                                    "making assumptions about the underlying " +
                                                    "distribution.",
                                            target="nelson_aalen", 
                                            style = {'font-size': 12,
                                                     },
                                            ),
                                        dcc.Graph(id='cumulative_hazard_curve_figure',
                                                  style={'width': '100%',
                                                         'display': 'inline-block',
                                                         #'background-color': '#f0f0f0',
                                                         'padding': '1%',
                                                         },
                                                  ),
                                        ],
                                    style={'width': '49%',
                                           'display': 'inline-block',
                                           },
                                    ),
                                html.Div(
                                    children=[
                                        html.H5("Cox Proportional Hazards Regression w/ " +
                                                "Partial Effects",
                                            style={'display': 'inline-block', 
                                                   'color': '#505050',
                                                   'font-size': 20,
                                                   'margin-left': '2%',
                                                   'margin-right': '1%',
                                                   },
                                            ),
                                        html.I(className="fas fa-question-circle fa-lg", 
                                               id="cox_regression",
                                               style={'display': 'inline-block', 
                                                      'color':'#bfbfbf',
                                                      },
                                               ),
                                        dbc.Tooltip("Cox Proportional Hazards regression models " +
                                                    "the impact of covariates on survival time, " +
                                                    "assuming proportional hazards, offering " +
                                                    "insights into factors influencing survival " +
                                                    "and event occurrence.",
                                            target="cox_regression", 
                                            style = {'font-size': 12,
                                                     },
                                            ),
                                        dcc.Graph(id='survival_regression_figure',
                                                  style={'width': '100%',
                                                         'display': 'inline-block',
                                                         #'background-color': '#f0f0f0',
                                                         'padding': '1%',
                                                         },
                                                  ),
                                        ],
                                    style={'width': '49%',
                                           'display': 'inline-block',
                                           },
                                    ),
                                ],
                            ),
                        ),
                    html.P("", id='survival_fig_txt')
                    ],
                style={'width': '100%',
                       'display': 'inline-block',
                       'background-color': '#f0f0f0',
                       'padding': '1%',
                },
    )



def control_card_survival_regression():

    return html.Div(
        children=[
                html.H5("Conduct Survival Regression",
                        style={'display': 'inline-block', 
                               'margin-right': '1%',
                               },
                        ),
                
                html.P("When trying to understand how different factors influence survival time or" +
                       ", more generally, the time until an event occurs."),
                
                html.Br(),
                
                html.B("Choose 2 or more covariates",
                    style={'vertical-align': 'top',
                           'display': 'inline-block', 
                           'margin-right': '1%',
                       },
                    ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="tt_survival2",
                       style={'display': 'inline-block', 
                              'color':'#bfbfbf',
                              },
                       ),
                dbc.Tooltip("These are the variables that potentially influence your event " +
                            "variable (e.g., age and health-related variables influence the " +
                            "chance of death). The app will recognize if your event variable " +
                            "occurs in the list of covariates. If it does, the app will ignore it.",
                    target="tt_survival2", 
                    style = {'font-size': 12,
                             },
                    ),
                
                dcc.Dropdown(
                        id='survival_predictors',
                        options=[{"label": i, "value": i} for i in []],
                        multi=True, 
                        value=None,
                        style={'width': '100%',
                             },
                        ),
                
                html.Br(),
                
                html.Div(
                    children = [
                        html.B("Choose an event variable",
                            style={'vertical-align': 'top',
                                   'display': 'inline-block',
                                   'margin-right': '2%',
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="tt_survival3e",
                            style={'display': 'inline-block', 
                                   'color':'#bfbfbf'},
                            ),
                        dbc.Tooltip("This would be a binary variable with 1's indicating an " +
                                    "event (e.g., death, stroke, readmission) had happened and " +
                                    "0's indicating the event had not happened.",
                            target="tt_survival3e", 
                            style = {'font-size': 12,
                                     },
                            ),
                        dcc.Dropdown(
                                id='survival_e_var',
                                options=[{"label": i, "value": i} for i in []],
                                multi=False, 
                                value=None,
                                optionHeight=30,
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
                        html.B("Choose a duration variable",
                            style={'vertical-align': 'top',
                                   'display': 'inline-block', 
                                   'margin-right': '2%',
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="tt_survival3d",
                            style={'display': 'inline-block', 
                                   'color':'#bfbfbf',
                                   },
                            ),
                        dbc.Tooltip("This variable would a measure of time during which the " +
                                    "event of interest could happen.",
                            target="tt_survival3d", 
                            style = {'font-size': 12,
                                     },
                            ),
                        dcc.Dropdown(
                                id='survival_d_var',
                                options=[{"label": i, "value": i} for i in []],
                                multi=False, 
                                value=None,
                                optionHeight=30,
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
                        html.B("Examine partial effects of",
                            style={'vertical-align': 'top',
                                   'margin-right': '2%',
                                   'display': 'inline-block', 
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="tt_survival3",
                            style={'display': 'inline-block', 
                                   'color':'#bfbfbf'},
                            ),
                        dbc.Tooltip("Choose one of your covariates to examine its effect on " +
                                    "the survival function.",
                            target="tt_survival3", 
                            style = {'font-size': 12,
                                     },
                            ),
                        dcc.Dropdown(
                                id='survival_partial',
                                options=[{"label": i, "value": i} for i in []],
                                multi=False, 
                                value=None,
                                optionHeight=30,
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
                        html.B("Reduce multicollinearity",
                            style={'vertical-align': 'top',
                                   'display': 'inline-block', 
                                   'margin-right': '2%',
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="tt_survival4",
                                    style={'display': 'inline-block', 
                                           'color':'#bfbfbf',
                                           },
                                    ),
                        dbc.Tooltip("Predictors that are highly correlated with each other can " +
                                    "obscure each other's effect, significance, and cause " +
                                    "survival regression to fail. This 'multicollinearity' " +
                                    "can be dealt with by iteratively removing the predictors " +
                                    "that contribute most to multicollinearity.", 
                                    target="tt_survival4", 
                                    style = {'font-size': 12,
                                             },
                                    ),
                        dcc.Dropdown(
                            id='survival_multicollinear',
                            options=[{"label": i, "value": i} for i in ['Yes', 'No']],
                            multi=False, 
                            value='Yes',
                            style={'width': '100%', 
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
                
                dbc.Button('Run survival regression', 
                           id='btn_survival', 
                           n_clicks=0,
                           style={'display': 'inline-block',
                                  'width': '18%',
                                  'font-size': 12,
                                  'margin-right': '20px',
                                  "background-color": "#2a8cff",
                                  },
                           ),
                
                dbc.Button("View parameters table",
                           id='open-survival_params_table',
                           style={
                               "background-color": "#2a8cff",
                               'width': '18%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '20px',
                               },
                    ),
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id='survival_params_table'), 
                                    html.Br(), 
                                    html.P("", id='survival_params_table_txt'),
                                    ],
                                   ),
                     dbc.ModalFooter(
                                    dbc.Button("Close", 
                                               id='close-survival_params_table', 
                                               className="ml-auto")
                                    ),
                            ],
                    id='modal-survival_params_table',
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button("View model performance",
                           id='open-survival_performance_table',
                           style={
                               "background-color": "#2a8cff",
                               'width': '18%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '20px',
                               },
                    ),
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id='survival_performance_table'), 
                                    html.Br(), 
                                    html.P("Adjusted R-square accounts for sample size and the " +
                                           "number of predictors used."),
                                    ],
                                   ),
                     dbc.ModalFooter(dbc.Button("Close", 
                                                id='close-survival_performance_table', 
                                                className="ml-auto"),
                                    ),
                            ],
                    id='modal-survival_performance_table',
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                dbc.Button('Smart scale', 
                            id='btn_ss_survival', 
                            n_clicks=0,
                            style={'width': '20%',
                                   'font-size': 12,
                                   "background-color": "#2a8cff",
                                   'display': 'inline-block',
                                   'margin-right': '10px',
                                   },
                            ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="ss_survival",
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
                            target="ss_survival", 
                            style = {'font-size': 12,
                                     },
                            ),
                
                html.P("", id = 'rt_survival'),
                ],

        style={'width': '98.5%',
               'margin-left': '1%',
               },
        )




def run_survival(df, xvars, partial_effects_var, cat_vars, rfe_val, duration_var, event_var):

    try:
        partial_effects_var = partial_effects_var.replace(" ", "_")
    except:
        pass  
    
    try:
        duration_var = duration_var.replace(" ", "_")
    except:
        pass
    
    try:
        event_var = event_var.replace(" ", "_")
    except:
        pass  
    
    xvars2 = []
    for v in xvars:
        try:
            v = v.replace(" ", "_")
            xvars2.append(v)
        except:
            pass
    xvars = list(xvars2)
    
    cat_vars2 = []
    for v in cat_vars:
        try:
            v = v.replace(" ", "_")
            cat_vars2.append(v)
        except:
            pass
    cat_vars = list(cat_vars2)
    
    
    labels = list(df)
    for l in labels:
        try:
            l1 = l.replace(" ", "_")
            df.rename(columns={l: l1}, inplace=True)
        except:
            pass
    
    del xvars2, labels
    
    df, dropped, cat_vars_ls = app_fxns.dummify(df, cat_vars)
    
    if df.shape[1] < 2:
        return [], [], [], [], [], [], []
    
    ########## Eliminating features with many 0's ###########
    x_vars = list(df)
    drop = []
    for var in x_vars:
        vals = df[var].tolist()
        frac_0 = vals.count(0)/len(vals)
        if frac_0 > 0.95:
            drop.append(var)
    
    df.drop(labels=drop, axis=1, inplace=True)
    
    ################################################################################################
    ########## Eliminating features using VIF ######################################################
    ################################################################################################
    
    d1 = df[partial_effects_var].tolist()
    d2 = df[duration_var].tolist()
    d3 = df[event_var].tolist()
    df.drop(labels=[partial_effects_var, duration_var, event_var], axis=1, inplace=True)
    
    try:
        x_vars.remove(partial_effects_var)
    except:
        pass
    try:
        x_vars.remove(duration_var)
    except:
        pass
    try:
        x_vars.remove(event_var)
    except:
        pass
    
    if rfe_val == 1 and len(x_vars) > 1:
        while df.shape[1] > 2:
            cols = list(df)
            vifs = [variance_inflation_factor(df.values, j) for j in range(df.shape[1])]
                    
            max_vif = max(vifs)
            if max_vif > 10:
                i = vifs.index(max(vifs))
                col = cols[i]
                df.drop(labels=[col], axis=1, inplace=True)
            else:
                break
    
    df[partial_effects_var] = d1
    df[duration_var] = d2
    df[event_var] = d3
    
    ################################################################################################
    ########## End VIF #############################################################################
    ################################################################################################
    
    
    df.dropna(how='any', axis=0, inplace=True)
    
    
    ################################################################################################
    ########## Run Cox Proportional-Hazards Model  #################################################
    ################################################################################################
    
    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_var, event_col=event_var)
    
    with contextlib.redirect_stdout(io.StringIO()) as f:
        cph.print_summary()

    s = f.getvalue()
    lines = s.split('\n')

    cols = []
    vals = []
    i1 = 0
    i2 = 0
    
    for i, line in enumerate(lines):
        if ' = ' in line:
            lines2 = line.split(' = ')
            cols.append(lines2[0])
            vals.append(lines2[1])
        
        if i1 == 0:
            if '---' in line:
                i1 = int(i)
        
        elif i1 > 0 and i2 == 0:
            if '---' in line:
                i2 = int(i)
            
    #######################    1st DataFrame    ####################################################
    ################################################################################################
                    
    df_1 = pd.DataFrame(columns=['Model information', 'Model statistics'])
    df_1['Model information'] = cols
    df_1['Model statistics'] = vals

    #######################    2nd DataFrame    ####################################################
    ################################################################################################

    data = []
    lines = lines[i1:i2]
    lines = lines[1:]
    i1 = 0
    for i, l in enumerate(lines):
        l = l.strip()
        if i == 0:
            l = 'covariate  ' + l
            data.append(l)
        
        elif i1 == 0:
            if l.isspace() or l == '' or 'covariate' in l:
                i1 = int(i)
        
        elif i1 > 0:
            if l.isspace() or l == '':
                continue
            else:
                data.append(l)

    i_ls = []
    for i, line in enumerate(data):
        if line == 'covariate':
            i_ls.append(i)
        
    i1 = i_ls[0] - 1
    d1 = data[:i1]
    d3 = []
    d2 = data[i1:]
    d4 = []

    for i, l in enumerate(d1):
        if i == 0:
            ls1 = ['coef lower 95%', 'coef upper 95%', 'exp(coef) lower 95%', 
                   'exp(coef) upper 95%', 'cmp to']
            ls2 = ['coef_lower_95%', 'coef_upper_95%', 'exp(coef)_lower_95%', 
                   'exp(coef)_upper_95%', 'cmp_to']
            for li, l1 in enumerate(ls1):
                if l1 in l:
                    l = l.replace(l1, ls2[li])
                
            d3.append(l)
        elif l == 'covariate':
            continue
        else:
            d3.append(l)


    for i, l in enumerate(d2):
        if i == 0:
            l = 'covariate  ' + l
            ls1 = ['coef lower 95%', 'coef upper 95%', 'exp(coef) lower 95%', 
                   'exp(coef) upper 95%', 'cmp to']
            ls2 = ['coef_lower_95%', 'coef_upper_95%', 'exp(coef)_lower_95%', 
                   'exp(coef)_upper_95%', 'cmp_to']
            for li, l1 in enumerate(ls1):
                if l1 in l:
                    l = l.replace(l1, ls2[li])
                
            d4.append(l)
        elif l == 'covariate':
            continue
        else:
            d4.append(l)
        
    # Join the list of strings into a single string
    d3_str = '\n'.join(d3)
    # Use StringIO to create a file-like object
    d3_file = io.StringIO(d3_str)
    # Read the data as a pandas DataFrame, specifying the delimiter and header
    d3 = pd.read_csv(d3_file, delim_whitespace=True, skipinitialspace=True)
    d3.dropna(axis=1, how='all', inplace=True)

    # Join the list of strings into a single string
    d4_str = '\n'.join(d4)
    # Use StringIO to create a file-like object
    d4_file = io.StringIO(d4_str)
    # Read the data as a pandas DataFrame, specifying the delimiter and header
    d4 = pd.read_csv(d4_file, delim_whitespace=True, skipinitialspace=True)
    d4.dropna(axis=1, how='all', inplace=True)

    df_2 = d3.merge(d4, how='outer', on='covariate')
    del d1, d2, d3, d4
    
    #######################    Survival Curves    ##################################################
    ################################################################################################
    
    p10 = np.percentile(df, 10)
    p20 = np.percentile(df, 20)
    p30 = np.percentile(df, 30)
    p40 = np.percentile(df, 40)
    p50 = np.percentile(df, 50)
    p60 = np.percentile(df, 60)
    p70 = np.percentile(df, 70)
    p80 = np.percentile(df, 80)
    p90 = np.percentile(df, 90)
    
    fig = cph.plot_partial_effects_on_outcome(covariates = partial_effects_var, 
                                              values=[p10, p20, p30,
                                                      p40, p50, p60,
                                                      p70, p80, p90], 
                                              cmap='coolwarm')
    
    # Access the data from the plot and store it in lists
    x_values = []
    y_values = []
    
    for line in fig.get_lines():
        x, y = line.get_data()
        x_values.append(x)
        y_values.append(y)
    
    return df_1, df_2, x_values, y_values




def get_updated_results(n_clicks, smartscale, xvars, partial_effects_var, df, cat_vars, rfe_val, 
                        duration_var, event_var):
    
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
        return {},{},{},{}, dashT1, dashT2, "", "", "", 0, 0
    
    if df is None:
        return {},{},{},{}, dashT1, dashT2, "", "", "", 0, 0
    
    #elif yvar is None and xvars is None:
    #    return {}, dashT1, dashT2, "", "", "", 0, 0
    
    #elif yvar is None:
    #    return {}, dashT1, dashT2, "Error: Select a reponse variable", "", "", 0, 0
    
    #elif (isinstance(yvar, list) is True) & (xvars is None or len(xvars) < 2):
    #    return {}, dashT1, dashT2, "Error: Select a response variable and 2 or more predictors", "", "", 0, 0
    
    #elif isinstance(yvar, list) is True:
    #    return {}, dashT1, dashT2, "Error: Select a response variable", "", "", 0, 0
    
    elif xvars is None or len(xvars) < 1:
        return {},{},{},{}, dashT1, dashT2, "Error: Select one or more predictors", "", "", 0, 0
    
    df = pd.DataFrame(df)
    if df.empty:
        return {},{},{},{}, dashT1, dashT2, "", "", "", 0, 0
    
    #if yvar not in list(df):
    #    return {}, dashT1, dashT2, "Error: Choose a response variable", "", "", 0, 0
        
    #if yvar in xvars:
    #    xvars.remove(yvar)
    #    if len(xvars) == 0:
    #        return {}, dashT1, dashT2, "Error: Multiple regression requires 2 or more predictors. You chose one and it's the same as your response variable", "", "", 0, 0
    #    elif len(xvars) == 1:
    #        return {}, dashT1, dashT2, "Error: Multiple regression requires 2 or more predictors. You chose two but one is the same as your response variable", "", "", 0, 0
    
    #if len(xvars) < 2 and yvar is None:
    #    return {}, dashT1, dashT2, "Error: Multiple regression requires 2 or more predictors and one response variable.", "", "", 0, 0
        
    #elif len(xvars) < 2:
    #    return {}, dashT1, dashT2, "Error: Multiple regression requires 2 or more predictors.", "", "", 0, 0
                        
    else:
        vars_ = xvars + [partial_effects_var] + [duration_var] + [event_var]
        vars_ = list(set(vars_))
        df = df.filter(items=vars_, axis=1)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        #if smartscale == 1:
        #    df, xvars, yvars = smart_scale(df, xvars, [yvar])
        #    yvar = yvars[0]
            
        #df.dropna(how='any', inplace=True)
        
        
        ############################################################################################
        ########## Make Time To Event Plot #########################################################
        ############################################################################################
        
        time_to_event_figure = {}
        fig_data = []
        
        # Filter data for each event
        event_0_data = df[df[event_var] == 0][duration_var]
        event_1_data = df[df[event_var] == 1][duration_var]
        
        # Create histograms with outlines
        nm = event_var + ' = 0'
        
        fig_data.append(go.Histogram(x=event_0_data,
                                     opacity=0.7, 
                                     marker=dict(color='#0066ff',
                                                 line=dict(color='#0000e6', 
                                                           width=1.5)),
                                     name=nm))
        nm = event_var + ' = 1'
        fig_data.append(go.Histogram(x=event_1_data,
                                     opacity=0.7, 
                                     marker=dict(color='#ff0000', 
                                                 line=dict(color='#d60000', 
                                                           width=1.5)),
                                     name=nm))
        
        
        tvar = str(event_var)
        if len(tvar) > 14:
            tvar = tvar[:7] + '...' + tvar[7:]
            
        time_to_event_figure = go.Figure(data = fig_data,
            layout = go.Layout(
                xaxis = dict(title = dict(
                        text = "<b>" + duration_var + "</b>",
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif", 
                            size = 18,
                            ),
                        ), 
                    showticklabels = True,
                    ),
                                            
                yaxis = dict(title = dict(
                        text = "<b>Count<b>",
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif", 
                            size = 18,
                            ),
                        ),
                    showticklabels = True,
                    ),
                                            
                margin = dict(l=60, r=30, b=10, t=10), 
                showlegend = True, 
                height = 400,
                paper_bgcolor = "rgb(245, 247, 249)", 
                plot_bgcolor = "rgb(245, 247, 249)",
            ),
        )
        
        # Update layout for better visibility
        time_to_event_figure.update_layout(
                          barmode='overlay',  # Overlay histograms
                          bargap=0.1,  # Gap between bars
                          bargroupgap=0.1,
                          legend=dict(
                              y=0.8,
                              ),
                          )  # Gap between groups of bars
        
        ############################################################################################
        ########## Make Kaplan-Meier Curve #########################################################
        ############################################################################################
        kaplan_meier_curve_figure = {}
        
        KMC_df = pd.DataFrame(columns=['Time', 'Survival Probabilities', 'CI_lower', 'CI_upper'])
        kmf = KaplanMeierFitter()
        
        T = df[duration_var]
        E = df[event_var]
        
        kmf.fit(T, event_observed=E)
        
        # Extract survival function data
        survival_function_data = kmf.survival_function_
        
        # Extract survival probabilities and confidence intervals
        KMC_df['Time'] = survival_function_data.index.tolist()
        KMC_df['Survival Probabilities'] = survival_function_data['KM_estimate'].tolist()
        
        # Extract confidence intervals using the `confidence_interval_` attribute
        CIs = kmf.confidence_interval_
        
        KMC_df['CI_lower'] = CIs['KM_estimate_lower_0.95'].tolist() 
        KMC_df['CI_upper'] = CIs['KM_estimate_upper_0.95'].tolist() 
        
        fig_data = []
        fig_data.append(go.Scatter(x = KMC_df['Time'], 
                                   y = KMC_df['CI_upper'], 
                                   name = 'Upper 95% CI',
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   marker = dict(size=1, 
                                                 color="#66a3ff"),
                                   ),
                        )
        
        fig_data.append(go.Scatter(x = KMC_df['Time'], 
                                   y = KMC_df['Survival Probabilities'], 
                                   name = 'Kaplan-Meier Estimate',
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   marker = dict(size=1, 
                                                 color="#0066ff"),
                                   ),
                        )
        
        fig_data.append(go.Scatter(x = KMC_df['Time'], 
                                   y = KMC_df['CI_lower'], 
                                   name = 'Lower 95% CI',
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   marker = dict(size=1, 
                                                 color="#66ccff"),
                                   ),
                        )
        
        tvar = str(event_var)
        if len(tvar) > 14:
            tvar = tvar[:7] + '...' + tvar[7:]
            
        ytext = "<b>Probability that " + tvar + " <br>has not occurred</b>"
        
        kaplan_meier_curve_figure = go.Figure(data = fig_data,
            layout = go.Layout(
                xaxis = dict(title = dict(
                        text = "<b>" + duration_var + "</b>",
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif", 
                            size = 18,
                            ),
                        ), 
                    showticklabels = True,
                    ),
                                            
                yaxis = dict(title = dict(
                        text = ytext,
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif", 
                            size = 18,
                            ),
                        ),
                    showticklabels = True,
                    ),
                                            
                margin = dict(l=60, r=30, b=10, t=10), 
                showlegend = True, 
                height = 400,
                paper_bgcolor = "rgb(245, 247, 249)", 
                plot_bgcolor = "rgb(245, 247, 249)",
            ),
        )
        kaplan_meier_curve_figure.update_layout(
            legend=dict(
                y=0.8,  # Adjust this value to move the legend vertically
                #traceorder='normal',
            )
        )
        ############################################################################################
        ########## Make Cumulative Hazard Curve ####################################################
        ############################################################################################
        cumulative_hazard_curve_figure = {}
        
        CHC_df = pd.DataFrame(columns=['Time', 'Cumulative Hazard', 'CI_lower', 'CI_upper'])
        naf = KaplanMeierFitter()
        
        T = df[duration_var]
        E = df[event_var]
        
        # Fit Nelson-Aalen estimator
        naf = NelsonAalenFitter()
        naf.fit(T, event_observed=E)
        
        # Extract cumulative hazard function data
        cumulative_hazard_data = naf.cumulative_hazard_
        
        # Extract survival probabilities and confidence intervals
        CHC_df['Time'] = cumulative_hazard_data.index.tolist()
        CHC_df['Cumulative Hazard'] = cumulative_hazard_data['NA_estimate'].tolist()
        
        # Extract confidence intervals using the `confidence_interval_` attribute
        CIs = naf.confidence_interval_
        
        CHC_df['CI_lower'] = CIs['NA_estimate_lower_0.95'].tolist() 
        CHC_df['CI_upper'] = CIs['NA_estimate_upper_0.95'].tolist() 
        
        fig_data = []
        
        fig_data.append(go.Scatter(x = CHC_df['Time'], 
                                   y = CHC_df['CI_upper'], 
                                   name = 'Upper 95% CI',
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   marker = dict(size=1, 
                                                 color="#66a3ff"),
                                   ),
                        )
        
        fig_data.append(go.Scatter(x = CHC_df['Time'], 
                                   y = CHC_df['Cumulative Hazard'], 
                                   name = 'Nelson-Aalen Estimate',
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   marker = dict(size=1, 
                                                 color="#0066ff"),
                                   ),
                        )
        
        fig_data.append(go.Scatter(x = CHC_df['Time'], 
                                   y = CHC_df['CI_lower'], 
                                   name = 'Lower 95% CI',
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   marker = dict(size=1, 
                                                 color="#66ccff"),
                                   ),
                        )
        
        tvar = str(event_var)
        if len(tvar) > 14:
            tvar = tvar[:7] + '...' + tvar[7:]
            
        ytext = "<b>Cumulative risk that " + tvar + " <br>has occurred</b>"
        
        cumulative_hazard_curve_figure = go.Figure(data = fig_data,
            layout = go.Layout(
                xaxis = dict(title = dict(
                        text = "<b>" + duration_var + "</b>",
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif", 
                            size = 18,
                            ),
                        ), 
                    showticklabels = True,
                    ),
                                            
                yaxis = dict(title = dict(
                        text = ytext,
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif", 
                            size = 18,
                            ),
                        ),
                    showticklabels = True,
                    ),
                                            
                margin = dict(l=60, r=30, b=10, t=10), 
                showlegend = True, 
                height = 400,
                paper_bgcolor = "rgb(245, 247, 249)", 
                plot_bgcolor = "rgb(245, 247, 249)",
            ),
        )
        cumulative_hazard_curve_figure.update_layout(
            legend=dict(
                y=0.8,  # Adjust this value to move the legend vertically
                #traceorder='normal',
            )
        )
        
        ############################################################################################
        ############ Conduct Survival Regression ###################################################
        ############################################################################################
        ls = run_survival(df, xvars, partial_effects_var, cat_vars, rfe_val, duration_var, event_var)
        df_1, df_2, x_values, y_values = ls
        ############################################################################################
        
        
        ############################################################################################
        ############ Make Survival Regression Figure ###############################################
        ############################################################################################
        
        fig_data = []
        
        lab = 10
        for i, x in enumerate(x_values):
            if i == len(x_values) - 1:
                break
            y = y_values[i]
            name = str(lab) + 'th percentile'
            
            lab += 10
            fig_data.append(go.Scatter(x = x, 
                                       y = y, 
                                       name = name,
                                       mode = "lines", 
                                       opacity = 0.75, 
                                       marker = dict(size=1, 
                                                     #color="#66ccff",
                                                     ),
                                       ),
                            )
        
        baseline_x = x_values[-1]
        baseline_y = y_values[-1]
        fig_data.append(go.Scatter(x = baseline_x, 
                                   y = baseline_y, 
                                   name = 'baseline',
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   marker = dict(size=1, 
                                                 color="#999999"),
                                   ),
                        )
        
        #if len(y_train) == 0:
        #    rt1 = "Error: Your regression could not run. Your y-values contain no data."
        #    return {}, dashT1, dashT2, rt1, "", "", 0, 0
        
        tvar = str(event_var)
        if len(tvar) > 14:
            tvar = tvar[:7] + '...' + tvar[7:]
            
        xtext = "<b>Probability that " + tvar + " <br>has not occurred</b>"
        
        survival_regression_figure = go.Figure(data = fig_data,
            layout = go.Layout(
                xaxis = dict(title = dict(
                        text = "<b>" + duration_var + "</b>",
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif", 
                            size = 18,
                            ),
                        ), 
                    showticklabels = True,
                    ),
                                            
                yaxis = dict(title = dict(
                        text = xtext,
                        font = dict(family = '"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                            " Helvetica, Arial, sans-serif", 
                            size = 18,
                            ),
                        ),
                    showticklabels = True,
                    ),
                                            
                margin = dict(l=60, r=30, b=10, t=10), 
                showlegend = True, 
                height = 400,
                paper_bgcolor = "rgb(245, 247, 249)", 
                plot_bgcolor = "rgb(245, 247, 249)",
            ),
        )
        survival_regression_figure.update_layout(
            legend=dict(
                y=0.8,  # Adjust this value to move the legend vertically
                #traceorder='normal',
            )
        )
        
        ############################################################################################
        ############ Make DataTables to hold regression results ####################################
        ############################################################################################
        
        dashT1 = dash_table.DataTable(
            data=df_1.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df_1.columns],
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
        
        dashT2 = dash_table.DataTable(
            data=df_2.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df_2.columns],
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
        del df
        del df_1
        del df_2
        
        txt1 = "..."
        txt2 = "..."
        
        return [survival_regression_figure, time_to_event_figure, kaplan_meier_curve_figure, 
                cumulative_hazard_curve_figure, dashT2, dashT1, "", txt1, txt2, 0, 0]