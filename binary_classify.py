from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import dash

from plotly import graph_objects as go
from scipy import stats
import pandas as pd
import numpy as np
import json

import app_fxns



def generate_logistic_a():

    return html.Div(
                id="Figure4a",
                children=[
                    html.H6("Receiver Operating Characteristic (ROC curve)",
                        style={'display': 'inline-block',
                               'margin-right': '1%',
                               },
                        ),
                    html.I(className="fas fa-question-circle fa-lg", id="target_roc",
                        style={'display': 'inline-block', 
                               'color':'#bfbfbf',
                               },
                        ),
                    dbc.Tooltip("ROCs reveal the tradeoff between capturing a fraction of " +
                                "actual positives (1's) and missclassifying negatives (0's). " + 
                                "The true positive rate (TPR) is the fraction of actual " +
                                "positives that were correctly classified. The false " +
                                "positive rate (FPR) is the fraction of actual negatives " +
                                "(0's) that were misclassified. ROCs do not reveal the " +
                                "reliability of predictions (precision).",
                        target="target_roc", 
                        style = {'font-size': 12,
                                 },
                        ),
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
                style={'width': '48%',
                       'display': 'inline-block',
                       'background-color': '#f0f0f0',
                       'padding': '1%',
                       'margin-right': '4%',
                       },
                )


def generate_logistic_b():

    return html.Div(
                id="Figure4b",
                children=[
                    html.H6("Precision-recall curve (PRC)",
                        style={'margin-right': '1%',
                               'display': 'inline-block',
                               },
                        ),
                    html.I(className="fas fa-question-circle fa-lg", 
                           id="target_prc",
                           style={'display': 'inline-block',
                                  'color':'#bfbfbf',
                                  },
                           ),
                    dbc.Tooltip("PRCs reveal the tradeoff between correctly classifying actual " + 
                                "positives (1's) and capturing a substantial fraction of " +
                                "positives. Precision is the fraction of positive predictions " +
                                "that were correct. Recall is another name for the TPR. A " +
                                "good ROC is misleading if the PRC is weak.",
                        target="target_prc", 
                        style = {'font-size': 12,
                                 },
                        ),
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
                style={'width': '48%',
                       'display': 'inline-block',
                       'background-color': '#f0f0f0',
                       'padding': '1%',
                       },
                )


def control_card_logistic():

    return html.Div(
        id="control-card4",
        children=[
                html.H5("Conduct Binary Logistic and Probit Regression",
                        style={'display': 'inline-block', 
                               'margin-right': '1%',
                               },
                        ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="BinClass",
                       style={'display': 'inline-block', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("In statistics, logistic and probit regression are used to find " +
                            "explanatory relationships and to understand the significance of " +
                            "variables. In machine learning, they are used to obtain predictions. " +
                            "This app does both.", 
                            target="BinClass", 
                            style = {'font-size': 12,
                                     },
                            ),
                
                html.Br(),
                
                html.P("When trying to explain, predict, or classify a binary variable " +
                       "(1/0, yes/no) using one or more other variables as predictors",
                       style={'display': 'inline-block', 
                              'width': '62%',
                              },
                       ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="target_SLR_vars2",
                       style={'display': 'inline-block', 
                              'width': '3%', 
                              'color':'#bfbfbf',
                              },
                       ),
                dbc.Tooltip("This app takes several efficiency steps when conducting " +
                            "logistic and probit regression, i.e., when using >1 predictor variable. " + 
                            "First, predictors that are 95% zeros will be removed from " +
                            "analysis. Highly multicollinear predictors are also removed " +
                            "during analysis, as are predictors that are perfect correlates " +
                            "of the response variable and any predictor variable that only has " +
                            "one value. If the number of resulting features is greater than " +
                            "100, the app will use cross-validated recursive feature elimination " +
                            "to remove statistically unimportant variables.", 
                            target="target_SLR_vars2", 
                            style = {'font-size': 12,
                                     },
                            ),
                
                html.Br(),
                
                html.B("Choose one or more predictors",
                    style={'display': 'inline-block',
                            'vertical-align': 'top',
                            'margin-right': '1%',
                       },
                    ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="target_select_x",
                       style={'display': 'inline-block', 
                              'color':'#bfbfbf',
                              },
                       ),
                dbc.Tooltip("Any that contain your response variable will be removed from " +
                            "analysis. Example: If one of your predictors is 'sex' and your " +
                            "response variable is 'sex:male', then 'sex' will be removed from " +
                            "your predictors during regression.", 
                            target="target_select_x", 
                            style = {'font-size': 12,
                                     },
                            ),
                
                dcc.Dropdown(
                        id='xvar_logistic',
                        options=[{"label": i, "value": i} for i in []],
                        multi=True, 
                        value=None,
                        style={'width': '100%',
                             },
                        ),
                
                html.Br(),
                
                html.Div(
                    children = [
                        html.B("Choose a response variable",
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                                   'margin-right': '2%',
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="target_select_y",
                               style={'display': 'inline-block', 
                                      'color':'#bfbfbf',
                                      },
                               ),
                        dbc.Tooltip("This is your 'target', the thing you want to predict.", 
                                    target="target_select_y", 
                                    style = {'font-size': 12,
                                             },
                                    ),
                        
                        dcc.Dropdown(
                                id='yvar_logistic',
                                options=[{"label": i, "value": i} for i in []],
                                multi=False, 
                                value=None,
                                optionHeight=65,
                                style={'width': '100%',
                                     },
                                ),
                        ],
                    style={'width': '30%',
                           'display': 'inline-block',
                           'margin-right': '3%',
                           },
                ),
                    
                html.Div(
                    children=[
                        html.B("Choose a model",
                            style={'display': 'inline-block', 
                                   'margin-right': '4%',
                                   },
                            ),
                        html.I(className="fas fa-question-circle fa-lg", 
                               id="tt_bc1",
                               style={'display': 'inline-block', 
                                      'color':'#bfbfbf',
                                      },
                               ),
                        
                        dbc.Tooltip(
                            html.Div([
                                html.P("• Logistic: ..."),
                                html.P("• Probit: ..."),
                                ], 
                                style = {'font-size': 14, 
                                         'width': '200%',
                                         'background-color': "#000000",
                                         'text-align': 'left',
                                         },
                                ),
                                target="tt_bc1",
                                style = {#'font-size': 14, 
                                         'width': '30%',
                                         'background-color': "#000000",
                                         'text-align': 'left',
                                         },
                                ),
                        dcc.Dropdown(
                            id='binary_classifier_model',
                            options=[{"label": i, "value": i} for i in ['Logistic', 'Probit']
                                     ],
                            multi=False, 
                            value='Logistic',
                            style={'width': '100%', 
                                   'display': 'block',
                             },
                            ),
                        ],
                        style={'width': '15%',
                               'display': 'inline-block',
                               'vertical-align': 'bottom',
                               'margin-right': '5%',
                        },
                    ),
                
                html.Br(),
                html.Br(),
                
                dbc.Button('Run Regression', 
                            id='btn4', 
                            n_clicks=0,
                            style={'width': '20%',
                                   'font-size': 12,
                                   "background-color": "#2a8cff",
                                   'display': 'inline-block',
                                   'margin-right': '20px',
                                   },
                            ),
                dbc.Button("View parameters table",
                           id="open-centered5",
                           style={"background-color": "#2a8cff",
                                  'width': '20%',
                                  'font-size': 12,
                                  'display': 'inline-block',
                                  'margin-right': '20px',
                                  },
                           ),
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id="table_plot4a"), 
                                    html.Br(), 
                                    html.P("", id='tab4atxt'),
                                    ],
                                   ),
                     dbc.ModalFooter(
                                    dbc.Button("Close", 
                                               id="close-centered5", 
                                               className="ml-auto"),
                                    ),
                            ],
                    id="modal-centered5",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button("View predictions table",
                           id="open-centered6",
                           style={"background-color": "#2a8cff",
                                  'width': '20%',
                                  'font-size': 12,
                                  'display': 'inline-block',
                                  'margin-right': '20px',
                                  },
                           ),
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id="table_plot4b"), 
                                    html.Br(), 
                                    html.P("", id='tab4btxt'),
                                    ],
                                   ),
                     dbc.ModalFooter(
                                    dbc.Button("Close", 
                                               id="close-centered6", 
                                               className="ml-auto")
                                    ),
                            ],
                    id="modal-centered6",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button('Smart scale', 
                            id='btn_ss3', 
                            n_clicks=0,
                            style={'width': '20%',
                                   'font-size': 12,
                                   "background-color": "#2a8cff",
                                   'display': 'inline-block',
                                   'margin-right': '10px',
                                   },
                            ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="ss3",
                       style={'display': 'inline-block', 
                              'width': '3%', 
                              'color':'#99ccff',
                              },
                       ),
                
                dbc.Tooltip("Skewed data can weaken analyses and visualizations. Click on " +
                            "'Smart Scale' and the app will automatically detect and apply " +
                            "the best scaling for each skewed variable. Smart scaling will " +
                            "not necessarily improve the r-square.  To remove the rescaling " +
                            "just click 'Run Regression'.", 
                            target="ss3", 
                            style = {'font-size': 12,
                                     },
                            ),
                
                html.P("", id = 'rt2'),
                ],
        
                style={'width': '98.5%',
                       'margin-left': '1%',
                    },
            )



def run_binary_classify(df, yvar, cat_vars, classifier_model):
    
    df = app_fxns.remove_nans_optimal(df, yvar)
    coefs = []
    r2s = []
    pvals = []
    aics = []
    llf_ls = []
    PredY = []
    PredProb = []
    Ys = []
    
    y_o = df.filter(items=[yvar], axis=1)
    x_o = df.drop(labels=[yvar], axis=1, inplace=False)
    
    model = 0
    x_o_lm = sm.add_constant(x_o, has_constant='add')
    y_o = y_o[yvar]
    
    try:
        if classifier_model == 'Logistic':
            model = sm.Logit(y_o, x_o_lm).fit(maxiter=30)
            
        elif classifier_model == 'Probit':
            model = sm.Probit(y_o, x_o_lm).fit(maxiter=30)
            
    except:
        error = 'Error: Your model failed to run. One or more of your categorical variables '
        error += 'has too many categories to process or some levels of the categories '
        error += 'do not correspond to enough observations. '
        error += 'Try dropping one or more potentially problematic categorical variables.' 
        return None, None, None, error, None
    
    results_summary = model.summary()
    
    results_as_html1 = results_summary.tables[1].as_html()
    df1_summary = pd.read_html(results_as_html1, header=0, index_col=0)[0]
    
    results_as_html2 = results_summary.tables[0].as_html()
    df2_summary = pd.read_html(results_as_html2, header=0, index_col=0)[0]
    
    #results_as_html2 = results_summary.tables[2].as_html()
    #df2_summary = pd.read_html(results_as_html2, header=0, index_col=0)[0]
    
    x_vars = list(x_o)
    if len(x_vars) > 1:        
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
    
    if len(xlabs) > 1: 
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
    
    ####### ROC CURVE #######################################
    fpr, tpr, thresholds_roc = roc_curve(y_o, ypred, pos_label=1)
    auroc = auc(fpr, tpr)
            
    ####### PRECISION-RECALL CURVE ##############################################
    ppv, recall, thresholds_prc = precision_recall_curve(y_o, ypred, pos_label=1)
    auprc = average_precision_score(y_o, ypred, pos_label=1)
    
    ####### 
    dist1 = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    dist1 = dist1.tolist()
    di1 = dist1.index(np.nanmin(dist1))
    thresholds_roc = thresholds_roc.tolist()
    opt_roc_threshold = thresholds_roc[di1]
    
    dist2 = np.sqrt((ppv - 1)**2 + (recall - 1)**2)
    dist2 = dist2.tolist()
    di2 = dist2.index(np.nanmin(dist2))
    thresholds_prc = thresholds_prc.tolist()
    opt_prc_threshold = thresholds_prc[di2]
    
    opt_threshold = (opt_roc_threshold + opt_prc_threshold)/2
    
    dif = np.abs(np.array(thresholds_roc) - opt_threshold).tolist()
    di = dif.index(np.nanmin(dif))
    opt_tpr = tpr[di]
    opt_fpr = fpr[di]
    
    dif = np.abs(np.array(thresholds_prc) - opt_threshold).tolist()
    di = dif.index(np.nanmin(dif))
    opt_ppv = ppv[di]
    
    df['Predicted probability'] = ypred
    
    ypred2 = []
    for i in ypred:
        if i < opt_threshold:
            ypred2.append(0)
        else:
            ypred2.append(1)
    ypred = list(ypred2)
    
    lab = 'Binary prediction (optimal threshold =' + str(round(opt_threshold, 6)) + ')'
    df[lab] = ypred
    coefs.append(model.params[0])
                    
    pr2 = model.prsquared
    if pr2 < 0:
        pr2 = 0
                    
    aic = model.aic
    #bic = model.bic
    tp = model.pvalues[0]
    llf = model.llf
                    
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
    df_models['pred_prob'] = [PredProb]
    df_models['prc_null'] = [prc_null]
    df_models['optimal_threshold'] = [opt_threshold]
    df_models['optimal_tpr'] = [opt_tpr]
    df_models['optimal_fpr'] = [opt_fpr]
    df_models['optimal_ppv'] = [opt_ppv]
    df_models['coefficients'] = coefs
    
    #df_models = df_models.replace('_', ' ', regex=True)
    #for col in list(df_models):
    #    col2 = col.replace("_", " ")
    #    df_models.rename(columns={col: col2})
            
    df_models.reset_index(drop=True, inplace=True)
    
    #col = df.pop('probability of ')
    #df.insert(0, col.name, col)
    
    col = df.pop('Predicted probability')
    df.insert(0, col.name, col)
    
    col = df.pop(lab)
    df.insert(0, col.name, col)
    
    col = df.pop(yvar)
    df.insert(0, col.name, col)
    
    return df_models, df1_summary, df2_summary, '', df




def get_updated_results(n_clicks, smartscale, main_df, xvars, yvar, cat_vars, classifier_model):
    
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
        data=df_table.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in df_table.columns],
        
        page_action='none',
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        
        style_table={'height': '500px', 
                     'overflowY': 'auto',
                     },
        style_cell={'padding':'5px',
                    'width':'160px',
                    'whiteSpace':'normal',
                    'textAlign': 'center',
                    },
    )
    
    
    cols = ['Target', 'Prediction', 'feature 1', 'feature 2', 'feature 3']
    df_table = pd.DataFrame(columns=cols)
    df_table['Target'] = [np.nan]*10
    df_table['Prediction'] = [np.nan]*10
    df_table['feature 1'] = [np.nan]*10
    df_table['feature 1'] = [np.nan]*10
    df_table['feature 1'] = [np.nan]*10
    
    dashT2 = dash_table.DataTable(
        data=df_table.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in df_table.columns],
        
        page_action='none',
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        
        style_table={'height': '500px', 
                     'overflowY': 'auto',
                     },
        style_cell={'padding':'5px',
                    'width':'160px',
                    'whiteSpace':'normal',
                    'textAlign': 'center',
                    },
    )
    
    ctx1 = dash.callback_context
    jd1 = json.dumps({'triggered': ctx1.triggered,})
    jd1 = jd1[:50]
    
    if 'rt4.children' in jd1:
        return {}, {}, dashT1, dashT2, "", "", "", "", "", 0
    
    if main_df is None:
        return {}, {}, dashT1, dashT2, "", "", "", "", "", 0
    
    elif yvar is None and xvars is None:
        return {}, {}, dashT1, dashT2, "", "", "", "", "", 0
    
    elif xvars is None:
        return [{}, {}, dashT1, dashT2, 
                "Error: Select one or more features for your predictors", "", "", "", "", 0]
        
    elif yvar is None:
        return [{}, {}, dashT1, dashT2, 
                "Error: Select a feature for your response variable", "", "", "", "", 0]
    
    elif (isinstance(yvar, list) is True) & (xvars is None or len(xvars) < 1):
        return [{}, {}, dashT1, dashT2, 
                "Error: Select a feature for your response variable and 1 or more for your predictors", 
                "", "", "", "", 0]
    
    elif isinstance(yvar, list) is True:
        return [{}, {}, dashT1, dashT2, 
                "Error: Select a feature for your response variable", 
                "", "", "", "", 0]
    
    main_df = pd.DataFrame(main_df)
    if main_df.empty:
        return {}, {}, dashT1, dashT2, "", "", "", "", "", 0
    
    y_prefix = str(yvar)
    if ':' in yvar:
        y_prefix = yvar[:yvar.index(":")]
    vars_ = [y_prefix] + xvars
    vars_ = list(set(vars_))
    main_df = main_df.filter(items=vars_, axis=1)
    
    if smartscale == 1:
        main_df, xvars, yvars = app_fxns.smart_scale(main_df, xvars, [yvar])
        yvar = yvars[0]
    
    
    vars_ = cat_vars #+ [yvar]
    vars_ = list(set(vars_))
    main_df, dropped, cat_vars_ls = app_fxns.dummify_logistic(main_df, vars_, y_prefix, True)
    
    # Replace infinite values (np.inf, -np.inf) with numerical NaN values
    main_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    ################################################################################################
    ################# RUN CHECKS TO ENSURE THE DATA ARE FIT FOR MODELING ###########################
    ################################################################################################
    
    if yvar not in list(main_df):
        return [{}, {}, dashT1, dashT2, 
                "Error: Choose a feature for your response variable", 
                "", "", "", "", 0]
    
    unique_yvals = main_df[yvar].unique().tolist()
    unique_usable_yvals = [x for x in unique_yvals if not np.isnan(x)]
    
    if len(unique_usable_yvals) != len(unique_yvals) and len(unique_usable_yvals) == 0:
        error = "Error: After removing NaN values (missing data) from your chosen response "
        error += "variable, it contains no usable values. "
        error += "Please use a different response variable." 
        return [{}, {}, dashT1, dashT2, error, "", "", "", "", 0]
    
    elif len(unique_usable_yvals) != len(unique_yvals) and len(unique_usable_yvals) == 1:
        error = "Error: After removing NaN values (missing data) from your chosen response "
        error += "variable, it only contains one unique value (" + str(unique_usable_yvals[0]) +". "
        error += "Binary classification requires response variables to have two unique values. " 
        error += "Please use a different response variable." 
        return [{}, {}, dashT1, dashT2, error, "", "", "", "", 0]
    
    elif len(unique_yvals) == len(unique_usable_yvals) and len(unique_usable_yvals) == 1:
        error = "Error: Your chosen response variable only contains one unique value (" 
        error += str(unique_usable_yvals[0]) +". "
        error += "Binary classification requires response variables to have two unique values. "
        error += "Please use a different response variable."
        return [{}, {}, dashT1, dashT2, error, "", "", "", "", 0]
    
    y_prefix = y_prefix + ":"
    for i in list(main_df):
        if y_prefix in i and i != yvar:
            main_df.drop(labels=[i], axis=1, inplace=True)
    
    ########## Remove NaNs while minimizing data loss ##############################################
    
    total_nans = 0
    try:
        total_nans = main_df.isnull().sum().sum()
    except:
        pass
    
    if total_nans > 0:
        main_df = app_fxns.remove_nans_optimal(main_df, yvar)
    
    if main_df.shape[0] < 10 and len(main_df[yvar].unique().tolist()) == 1:
        error = "Error: Insufficient data. "
        error += "After optimizing the removal of missing data while minimizing data loss, your "
        error += "dataset only contains " + str(main_df.shape[1]) + " observations and your "
        error += "response variable (" + yvar + ") only contains a single value. Excessive NaNs in "
        error += "in one or more of your predictors caused this. Try "
        error += "using a smaller set of predictors. Once you have a model that runs, try adding "
        error += "more predictors if you wish."
        return [{}, {}, dashT1, dashT2, error, "", "", "", "", 0]                                   
                                     
    elif main_df.shape[0] < 10:
        error = "Error: Insufficient data. "
        error += "After optimizing the removal of missing data while minimizing data loss, your "
        error += "dataset only contains " + str(main_df.shape[1]) + " observations. Excessive NaNs in "
        error += "in one or more of your predictors caused this. Try "
        error += "using a smaller set of predictors. Once you have a model that runs, try adding "
        error += "more predictors if you wish."
        return [{}, {}, dashT1, dashT2, error, "", "", "", "", 0]
        
    elif len(main_df[yvar].unique().tolist()) == 1:
        
        error = "Error: Insufficent values in the response variable. "
        error += "After optimizing the removal of missing data while minimizing data loss, your "
        error += "response variable (" + yvar + ") only contained a single value. Excessive NaNs in "
        error += "in one or more of your predictors caused this. Try "
        error += "using a smaller set of predictors. Once you have a model that runs, try adding "
        error += "more predictors if you wish."
        return [{}, {}, dashT1, dashT2, error, "", "", "", "", 0]         
    
    ########## Eliminate features that only have one value #########################################
    main_df = main_df.loc[:, main_df.nunique() != 1]
    y_o = main_df.filter(items=[yvar], axis=1)
    x_o = main_df.drop(labels=[yvar], axis=1, inplace=False)
    
    if x_o is None or x_o.shape[1] < 1:
        error = "Error: Too few predictors. "
        error += "After optimizing the removal of missing data while minimizing data loss, "
        error += "and after removing predictors containing a single value, your "
        error += "dataset contains no predictors. "
        error += "One or more of your predictors may have caused this. Try "
        error += "using a smaller set of predictors. Once you have a model that runs, try adding "
        error += "more predictors if you wish."
        return [{}, {}, dashT1, dashT2, error, "", "", "", "", 0]
    
    ########## Eliminating features with many 0's ##################################################
    x_vars = list(x_o)
    drop = []
    for var in x_vars:
        vals = x_o[var].tolist()
        frac_0 = vals.count(0)/len(vals)
        frac_1 = vals.count(1)/len(vals)
        
        if frac_0 > 0.99 or frac_1 > 0.99:
            drop.append(var)
    
    x_o.drop(labels=drop, axis=1, inplace=True)
    if x_o is None or x_o.shape[1] < 1:
        error = "Error: Too few predictors. "
        error += "After optimizing the removal of missing data while minimizing data loss, "
        error += "and after removing predictors containing a single value and binary predictors "
        error += "that are nearly all 1's or nearly all 0's, your"
        error += "dataset contains no predictors. "
        error += "Try using a smaller set of predictors and/or grouping some categories of " 
        error += "your categorical variables. Once you have a model that runs, try adding "
        error += "more predictors if you wish."
        return [{}, {}, dashT1, dashT2, error, "", "", "", "", 0]
    
    ########## Eliminate features that are perfectly correlated to the response variable ###########
    perfect_correlates = []
    for xvar in list(x_o):
        x = x_o[xvar].tolist()
        y = y_o[yvar].tolist()
        slope, intercept, r, p, se = stats.linregress(x, y)
        if r**2 == 1.0:
            perfect_correlates.append(xvar)
    x_o.drop(labels=perfect_correlates, axis=1, inplace=True)
    
    if x_o.shape[1] < 1 or x_o is None:
        error = "Error: Insufficient data. "
        error += "After optimizing the removal of missing data while minimizing data loss, and "
        error += "after removing predictors containing a single value as well as predictors that "
        error += "are perfectly correlated to your response variable, you have no predictors left. " 
        error += "The combination of excessive NaNs in one or more of your predictors may have caused this." 
        error += "You should also avoid using predictors that are perfectly (or near perfectly) "
        error += "correlated to your response variable."
        error += "Try using a smaller set of predictors and avoiding perfect correlates. "
        error += "Once you have a model that runs, try adding more predictors, if you wish."
        return None, None, None, error, None
    
    main_df = main_df.filter(items=[yvar] + list(x_o), axis=1)
    
    y_o = main_df.filter(items=[yvar], axis=1)
    x_o = main_df.drop(labels=[yvar], axis=1, inplace=False)
    
    x_vars = list(x_o)
    if len(x_vars) > 1:
        
        ########## RFECV ############
        if x_o.shape[1] > 2:
            model = LogisticRegression()
            try:
                rfecv = RFECV(model, step=1, min_features_to_select=2)
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
                
                x_o = x_o.filter(items = supported_features, axis=1)
            except:
                pass
            
        ########## Eliminating features using vif ###########
        
        while x_o.shape[1] > 2:
            cols = list(x_o)
            vifs = [variance_inflation_factor(x_o.values, j) for j in range(x_o.shape[1])]
                
            max_vif = max(vifs)
            if max_vif > 10:
                i = vifs.index(max(vifs))
                col = cols[i]
                x_o.drop(labels=[col], axis=1, inplace=True)
            else:
                break
        
        
    
    main_df = main_df.filter(items=[yvar] + list(x_o), axis=1)
    
    ############################## END CHECKS ######################################################
    ################################################################################################
    
    error = ''
    
    ls = run_binary_classify(main_df, yvar, cat_vars, classifier_model)
    models_df, df1_summary, df2_summary, error, pred_df = ls
    
    if error != '':
        return {}, {}, dashT1, dashT2, error, "", "", "", "", 0
    
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
    
    opt_threshold = models_df['optimal_threshold'].iloc[0]
    opt_threshold = 'Optimal threshold: ' + str(round(opt_threshold, 6))
    opt_tpr = models_df['optimal_tpr'].iloc[0]
    opt_fpr = models_df['optimal_fpr'].iloc[0]
    opt_ppv = models_df['optimal_ppv'].iloc[0]
    
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
            x = [opt_fpr],
            y = [opt_tpr],
            mode = "markers",
            name = 'optimum',
            text = [opt_threshold],
            marker = dict(color = "#0066ff", size = 20),
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
            x = [opt_tpr],
            y = [opt_ppv],
            mode = "markers",
            name = 'optimum',
            text = [opt_threshold],
            marker = dict(color = "#0066ff", size = 20),
        )
    )
                
    fig_data.append(
            go.Scatter(
                x = [0, 1],
                y = [prc_null, prc_null],
                mode = "lines",
                name = 'Null AUC = ' + str(np.round(prc_null, 3)),
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
        data=df1_summary.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in df1_summary.columns],
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
        data=pred_df.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in pred_df.columns],
        export_format="csv",
        page_action='native',
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        page_size=100,
        
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
        
    txt1 = "This table pertains to the fitted model. This model predicts the probability of an "
    txt1 += "observation being a positive (1) instead of a negative (0). All this is before "
    txt1 += "applying a diagnositic threshold, i.e., the point where we count an estimated "
    txt1 += "probability as a 0 or a 1. The variance inflation factor (VIF) measures "
    txt1 += "multicollinearity. VIF > 5 indicates that a predictor variable is significantly "
    txt1 += "correlated with one or more other predictors. VIF > 10 indicates severe "
    txt1 += "multicollinearity, which can lead to overfitting and inaccurate parameter "
    txt1 += "estimates. If your VIF's are high, trying removing some of those variables."
    
    txt2 = "This table pertains to results after finding an optimal diagnostic threshold. "
    txt2 += "This threshold determines whether the value of an outcome's probability is counted "
    txt2 += "as a positive (1) or a negative (0). The threshold is found by determining the point "
    txt2 += "on the ROC curve that is closest to the upper left corner."
    
    auroc = np.round(auroc, 3)
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
    elif auroc < 0.9:
        t1 = " good-to-excellent "
    elif auroc < 0.95:
        t1 = " excellent "
    else:
        t1 = " outstanding "
        
    txt3 = "AUC = area under the ROC curve, i.e., average true positive rate across diagnostic "
    txt3 += "thresholds. Random 50:50 guesses produce values of 0.5. Your AUC value of "
    txt3 += str(auroc) + " indicates" + t1 + "diagnostic power."
    
    txt4 = "AUC = area under the PRC curve, i.e., average precision across diagnostic thresholds. "
    txt4 += "Random 50:50 guesses produce AUC values that equal the fraction of positive outcomes "
    txt4 += "(1's) in the data. "
    
    p = auprc/prc_null
    auprc_t = str(round(auprc, 3))
    if p > 1.5 and auprc > 0.9:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. Your model seems to be highly precise, "
        txt4 += "especially when compared to the null expectation."
    
    elif p > 1.5 and auprc > 0.8:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is moderate-to-high, "
        txt4 += "especially when compared to the null expectation."
    
    elif p > 1.5 and auprc > 0.7:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is moderately useful, "
        txt4 += "especially when compared to the null expectation."
    
    elif p > 1.5 and auprc > 0.6:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is low-to-moderate, "
        txt4 += "especially when compared to the null expectation."
    
    elif p > 1.5 and auprc > 0.5:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is, however, "
        txt4 += "barely better than a coin toss."
    
    elif p > 1.5:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is, however, "
        txt4 += "no better than a coin toss."

    
    elif p > 1.25 and auprc > 0.9:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. Your model seems to be highly precise, "
        txt4 += "particularly regarding to the null expectation."

    elif p > 1.25 and auprc > 0.8:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is moderate-to-high, "
        txt4 += "particularly regarding to the null expectation."
    
    elif p > 1.25 and auprc > 0.7:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is moderately useful, "
        txt4 += "particularly regarding to the null expectation."

    elif p > 1.25 and auprc > 0.6:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is low-to-moderate, "
        txt4 += " particularly in regard to the null expectation."
    
    elif p > 1.25 and auprc > 0.5:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is, "
        txt4 += "however, barely better than a coin toss."

    elif p > 1.25:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is, "
        txt4 += "however, worse than or equal to a coin toss."
    
    elif p > 1.1 and auprc > 0.9:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. Your model seems to be highly precise, "
        txt4 += "but maybe not much more than the null expectation."

    elif p > 1.1 and auprc > 0.8:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is moderate-to-high, "
        txt4 += "but maybe not much more than the null expectation."
    
    elif p > 1.1 and auprc > 0.7:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is moderately useful, "
        txt4 += "but maybe not much more than the null expectation."
    
    elif p > 1.1 and auprc > 0.6:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is low-to-moderate, "
        txt4 += "but maybe not much more than the null expectation."
    
    elif p > 1.1 and auprc > 0.5:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is, however, "
        txt4 += "barely better than a coin toss."
    
    elif p > 1.1:
        txt4 += "Your AUC was " + auprc_t + ", which is " + str(round(p, 3))
        txt4 += " times greater than the null. The precision of your model is, however, "
        txt4 += "worse than or equal to a coin toss."
    
    
    elif p > 1 and auprc > 0.9:
        txt4 += "Your AUC of " + auprc_t + " was nearly equal to the null, "
        txt4 += "even though your model seems to be highly precise."
        
    elif p > 1 and auprc > 0.8:
        txt4 += "Your AUC of " + auprc_t + " was nearly equal to the null, "
        txt4 += "even though the precision of your model is moderate-to-high."
    
    elif p > 1 and auprc > 0.7:
        txt4 += "Your AUC of " + auprc_t + " was nearly equal to the null, "
        txt4 += "even though the precision of your model is moderately useful."
    
    elif p > 1 and auprc > 0.6:
        txt4 += "Your AUC of " + auprc_t + " was nearly equal to the null and your model "
        txt4 += "model is low-to-moderately precise."
    
    elif p > 1 and auprc > 0.5:
        txt4 += "Your AUC of " + auprc_t
        txt4 += " was nearly equal to the null and the precision of your model is low."
    
    elif p == 1:
        txt4 = txt4 + "Your AUC of " + auprc_t + " equalled the null."
      
    elif p < 1:
        txt4 = txt4 + "Your AUC of " + auprc_t + " was worse than the null."
    
    return figure1, figure2, dashT1, dashT2, "", txt1, txt2, txt3, txt4, 0