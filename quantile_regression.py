import statsmodels.formula.api as smf

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import dash

from plotly import graph_objects as go
import pandas as pd
import numpy as np
import json

import app_fxns


def generate_figure_quantile_regression():

    return html.Div(
                children=[
                    dcc.Loading(
                        type="default",
                        fullscreen=False,
                        children=html.Div(
                            children=[dcc.Graph(id="figure_quantile_regression"),
                                    ],
                                ),
                        ),
                    html.Br(),
                    html.P("Coefficients of determination (R\u00B2) for quantile regression " +
                           "are Cox-Snell likelihood ratio pseudo R\u00B2 values. They are not " +
                           "directly comparable to the r\u00B2 of ordinary least-squares " +
                           "regression.",
                           ), 
                    ],
                style={'width': '100%',
                       'display': 'inline-block',
                       'background-color': '#f0f0f0',
                       'padding': '1%',
                       },
                )




def control_card_quantile_regression():

    return html.Div(
        id="control-card_quantile_regression",
        children=[
                html.H5("Conduct linear and polynomial quantile regression",
                        style={'display': 'inline-block', 
                               'margin-right': '1%',
                               },
                        ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="target_select_vars2_quant",
                       style={'display': 'inline-block', 
                              'width': '3%', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("Data too noisy or too oddly distributed for ordinary least squares " +
                            "regression? Quantile regression makes no assumptions about the " +
                            "distribution of data and allows you to explore boundaries on the " +
                            "relationships between features.", 
                            target="target_select_vars2_quant", 
                            style = {'font-size': 12,
                                     },
                            ),
                html.Hr(),
                
                html.Div(
                id="control-card2a_quant",
                children=[
                    html.B("Choose a predictor (x) variable",
                        style={'display': 'inline-block',
                                'vertical-align': 'top',
                           },
                        ),
                    dcc.Dropdown(
                            id='xvar2_quant',
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
                        }),
                
                html.Div(
                id="control-card2b_quant",
                children=[
                    html.B("Choose a data transformation",
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                        }),
                    dcc.Dropdown(
                            id='x_transform_quant',
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
                           'margin-right': '2%',
                           'width': '20%',
                    }),
                    
                html.Div(
                id="control-card2c_quant",
                children=[
                    html.B("Choose a response (y) variable",
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                        },
                        ),
                    dcc.Dropdown(
                            id='yvar2_quant',
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
                    }),
                
                html.Div(
                id="control-card2d_quant",
                children=[
                    html.B("Choose a data transformation",
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                        }),
                    dcc.Dropdown(
                            id='y_transform_quant',
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
                           'margin-right': '2%',
                           'width': '20%',
                    },
                    ),
                
                html.Hr(),
                html.Div(
                    id="control-card2e_quant_quantiles",
                    children=[
                        html.B("Choose lower and upper quantiles",
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                            },
                            ),
                        dcc.RangeSlider(
                            id='quantiles',
                            min=1, 
                            max=99, 
                            value=[5, 95], 
                            allowCross=False, 
                            tooltip={"placement": "bottom", 
                                     "always_visible": True},
                            ),
                        ],
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'margin-right': '1%',
                               'width': '20%',
                        },
                    ),
                
                html.Div(
                    id="control-card2e_quant",
                    children=[
                        html.B("Choose a model",
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                            },
                        ),
                        dcc.Dropdown(
                                id='model2_quant',
                                options=[{"label": i, "value": i} for i in ['linear', 
                                                                            'quadratic', 
                                                                            'cubic',
                                                                            ]
                                         ],
                                multi=False, 
                                value='linear',
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
                
                
                dbc.Button('Run regression', 
                            id='btn2_quant', 
                            n_clicks=0,
                            style={'width': '20%',
                                   'font-size': 12,
                                   "background-color": "#2a8cff",
                                   'display': 'inline-block',
                                   'margin-right': '1%',
                                   'margin-top': '1.7%',
                                   },
                            ),
                
                dbc.Button("View results table",
                           id="open-quant_regression_table",
                           style={
                               "background-color": "#2a8cff",
                               'width': '16%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-top': '1.7%',
                               },
                    ),
                dbc.Modal(
                    [dbc.ModalBody([html.H5("Results for upper quantile:"),
                                    html.Div(id="quant_table_1"), 
                                    html.Div(id="quant_table_2"),
                                    html.Br(),
                                    
                                    html.H5("Results for 50th quantile (aka Least Absolute " +
                                            "Deviation Model):"),
                                    html.Div(id="quant_table_3"), 
                                    html.Div(id="quant_table_4"),
                                    html.Br(),
                                    
                                    html.H5("Results for lower quantile:"),
                                    html.Div(id="quant_table_5"), 
                                    html.Div(id="quant_table_6"),
                                    html.Br(),
                                    html.P("", id="quant_table_txt"),
                                    ]),
                                    dbc.ModalFooter(
                                    dbc.Button("Close", 
                                               id="close-quant_regression_table", 
                                               className="ml-auto")
                                    ),
                            ],
                    id="modal-quant_regression_table",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="lg",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                html.P("", id='rt3_quant')
                ],
                style={'width': '98.5%',
                       'margin-left': '1%',
                    },
            )



def get_updated_results(n_clicks, xvar, yvar, x_transform, y_transform, model, df, quantiles):
        
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
        return {}, "", "", dashT1, dashT2, dashT1, dashT2, dashT1, dashT2
    
    if df is None or xvar is None or yvar is None or xvar == yvar or isinstance(yvar, list) is True or isinstance(yvar, list) is True:
            
        if df is None:
            return [{}, "", "", dashT1, dashT2, dashT1, dashT2, dashT1, dashT2]
            
        elif (isinstance(xvar, list) is True or xvar is None) & (isinstance(yvar, list) is True or yvar is None):
            return [{}, "Error: You need to select some variables.", "", 
                    dashT1, dashT2, dashT1, dashT2, dashT1, dashT2]
            
        elif isinstance(yvar, list) is True or yvar is None:
            return [{}, "Error: You need to select a response variable.", "", 
                    dashT1, dashT2, dashT1, dashT2, dashT1, dashT2]
            
        elif isinstance(xvar, list) is True or xvar is None:
            return [{}, "Error: You need to select an predictor variable.", "", 
                    dashT1, dashT2, dashT1, dashT2, dashT1, dashT2]
            
        elif xvar == yvar and xvar is not None:
            return [{}, "Error: Your predictor variable and response variable are the same. Ensure they are different.",
                    "", dashT1, dashT2, dashT1, dashT2, dashT1, dashT2]
        else:
            return [{}, "", "", dashT1, dashT2, dashT1, dashT2, dashT1, dashT2]
            
    else:
        df = pd.DataFrame(df)
        if df.empty:
            return {}, "", "", dashT1, dashT2, dashT1, dashT2, dashT1, dashT2
        
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
            df[xvar] = np.log10(np.abs(df[xvar]) + 1).tolist()
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
        df.dropna(how='any', axis=0, inplace=True)
            
        ql = quantiles[0]
        qh = quantiles[1]
        quantiles = [0.01*ql, 0.5, 0.01*qh]

        formula = int()
        #degree = int()
        if model == 'linear': 
            formula = 'y ~ 1 + x'
            #degree = 1
        elif model == 'quadratic': 
            #degree = 2
            formula = 'y ~ 1 + x + I(x ** 2.0)'
        elif model == 'cubic': 
            #degree = 3
            formula = 'y ~ 1 + x + I(x ** 3.0) + I(x ** 2.0)'
            
        #########################################################################################
        ################## GET QUANTILE PREDICTIONS FOR EACH PLOT ###############################
        #########################################################################################
            
        # Polynomial Quantile regression
        # Least Absolute Deviation (LAD)
        # The LAD model is a special case of quantile regression where q=0.5
        # #res = mod.fit(q=.5)
            
        x, y = (np.array(t) for t in zip(*sorted(zip(df[xvar], df[yvar]))))
        d = {'x': x, 'y': y}
        df = pd.DataFrame(data=d)
        mod = smf.quantreg(formula, df)
        res_all = [mod.fit(q=q) for q in quantiles]
        #res_ols = smf.ols(formula, df).fit()
        
        x_p = np.array(x)
        #df_p = pd.DataFrame({'x': x_p})
        
        y_lo = res_all[0].fittedvalues
        #y_lo_resid = res_all[0].resid
        #obs_lo = y_lo - y_lo_resid
        pr2_lo = str(np.round(res_all[0].prsquared, 3))
        
        y_50 = res_all[1].fittedvalues
        y_50_resid = res_all[1].resid
        obs_50 = y_50 - y_50_resid
        pr2_50 = str(np.round(res_all[1].prsquared, 3))
        r2_50 = str(np.round(app_fxns.obs_pred_rsquare(obs_50, y_50), 3))
        
        y_hi = res_all[2].fittedvalues
        #y_hi_resid = res_all[2].resid
        #obs_hi = y_hi - y_hi_resid
        pr2_hi = str(np.round(res_all[2].prsquared, 3))
        
        #y_ols_predicted = res_ols.predict(df_p)
        
        del df
            
        fig_data = []
        clr = "#3399ff"

        fig_data.append(go.Scatter(
            x = x,
            y = y,
            name = 'Observed',
            mode = "markers",
            opacity = 0.75,
            marker = dict(size=10,
                          color=clr)
            )
        )
            
        qh = str(qh)
        if qh[-1] == '1' and qh != '11':
            tname = qh + 'st quantile, pseudo R\u00B2 = ' + pr2_hi
        elif qh[-1] == '2':
            tname = qh + 'nd quantile, pseudo R\u00B2 = ' + pr2_hi
        elif qh[-1] == '3':
            tname = qh + 'rd quantile, pseudo R\u00B2 = ' + pr2_hi
        else:
            tname = qh + 'th quantile, pseudo R\u00B2 = ' + pr2_hi
            
        fig_data.append(go.Scatter(
            x = x_p,
            y = y_hi,
            name = tname,
            mode = "lines",
            opacity = 0.75,
            line = dict(width=3,
                        dash='dash',
                        color="#0047b3")
            )
        )
            
        fig_data.append(
            go.Scatter(
                x = x_p,
                y = y_50,
                mode = "lines",
                name = 'Median (50th quantile):<br>   pseudo R\u00B2 = ' + pr2_50 + '<br>   obs vs pred r\u00B2 = ' + r2_50,
                opacity = 0.75,
                line = dict(width=3,
                            dash='dash',
                            color="#ff0000")
                )
        )
        
        ql = str(ql)
        if ql[-1] == '1' and ql != '11':
            tname = ql + 'st quantile, R\u00B2 = ' + pr2_lo
        elif ql[-1] == '2':
            tname = ql + 'nd quantile, R\u00B2 = ' + pr2_lo
        elif ql[-1] == '3':
            tname = ql + 'rd quantile, R\u00B2 = ' + pr2_lo
        else:
            tname = ql + 'th quantile, R\u00B2 = ' + pr2_lo
            
        fig_data.append(go.Scatter(
            x = x_p,
            y = y_lo,
            name = tname,
            mode = "lines",
            opacity = 0.75,
            line = dict(width=3,
                        dash='dash',
                        color="#0066ff")
            )
        )
        
        
        ######################################### Lower quantile ###################################
        results_summary = res_all[0].summary()
        
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
        
        ######################################### 50th quantile ###################################
        results_summary = res_all[1].summary()
        
        results_as_html1 = results_summary.tables[0].as_html()
        df1_summary = pd.read_html(results_as_html1)[0]
        #df1_summary['index'] = df1_summary.index
        df1_summary = df1_summary.astype(str)
        col_names = list(df1_summary)
        
        df3 = pd.DataFrame(columns=['Model information', 'Model statistics'])
        df3['Model information']  = df1_summary[col_names[0]].astype(str) + ' ' + df1_summary[col_names[1]].astype(str) 
        df3['Model statistics'] = df1_summary[col_names[2]].astype(str) + ' ' + df1_summary[col_names[3]].astype(str) 
        #del df3, df1_summary
        
        dashT3 = dash_table.DataTable(
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
        
        dashT4 = dash_table.DataTable(
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
        
        
        ######################################### Upper quantile ###################################
        results_summary = res_all[2].summary()
        
        results_as_html1 = results_summary.tables[0].as_html()
        df1_summary = pd.read_html(results_as_html1)[0]
        #df1_summary['index'] = df1_summary.index
        df1_summary = df1_summary.astype(str)
        col_names = list(df1_summary)
        
        df3 = pd.DataFrame(columns=['Model information', 'Model statistics'])
        df3['Model information']  = df1_summary[col_names[0]].astype(str) + ' ' + df1_summary[col_names[1]].astype(str) 
        df3['Model statistics'] = df1_summary[col_names[2]].astype(str) + ' ' + df1_summary[col_names[3]].astype(str) 
        #del df3, df1_summary
        
        dashT5 = dash_table.DataTable(
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
        
        dashT6 = dash_table.DataTable(
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

            
        txt = ""
        '''
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
        '''
            
        return figure, "", txt, dashT1, dashT2, dashT3, dashT4, dashT5, dashT6