from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import dash_bootstrap_components as dbc
from dash import dash_table
from dash import dcc, html
import dash

from plotly import graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import base64
import json
import io

import app_fxns


def generate_decision_tree_outputs():

    return html.Div(
                children=[
                    dcc.Loading(
                        type="default",
                        fullscreen=False,
                        children=html.Div(
                            children=[
                                dcc.Graph(id='figure_decision_tree_obs_pred',
                                          style={'width': '90%',
                                                 'display': 'inline-block',
                                                 #'background-color': '#f0f0f0',
                                                 'padding': '1%',
                                                 },
                                          ),
                                ],
                                        
                            style={
                                #"background-color": "#A0A0A0",
                                "display": "flex",
                                "justify-content": "center",
                                "align-items": "center",
                                },
                            ),
                        ),
                    dcc.Markdown("", id = 'text-dec_tree_reg1'),
                    dcc.Markdown("", id = 'text-dec_tree_reg2'),
                    ],
                style={'width': '100%',
                       'display': 'inline-block',
                       'background-color': '#f0f0f0',
                       'padding': '1%',
                       },
                )


def control_card_decision_tree_regression():

    return html.Div(
        children=[
                html.H5("Conduct Decision Tree Regression",
                        style={'display': 'inline-block', 
                               'margin-right': '1%',
                               },
                        ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="target_dec_tree_reg",
                       style={'display': 'inline-block', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("...", 
                            target="target_dec_tree_reg", 
                            style = {'font-size': 12,
                                     },
                            ),
                html.P("..."),
                
                
                html.B("Choose 2 or more predictors",
                    style={'vertical-align': 'top',
                           'display': 'inline-block',
                           'margin-right': '1%',
                       },
                    ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="target_dec_tree_reg2",
                       style={'display': 'inline-block', 
                           'color':'#bfbfbf',
                           },
                       ),
                dbc.Tooltip("The app will recognize if your response variable occurs in this " +
                            "list of predictors. If it does, the app will ignore it.",
                    target="target_dec_tree_reg2", 
                    style = {'font-size': 12,
                             },
                    ),
                
                dcc.Dropdown(
                        id='xvar_dec_tree_reg',
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
                               id="target_dec_tree_reg3",
                               style={'display': 'inline-block', 
                                      'color':'#bfbfbf',
                                      },
                               ),
                        dbc.Tooltip("Does not include categorical features or any numerical " +
                                    "feature with less than 4 unique values.",
                            target="target_dec_tree_reg3", 
                            style = {'font-size': 12,
                                     },
                            ),
                        dcc.Dropdown(
                                id='yvar_dec_tree_reg',
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
                
                html.Br(),
                html.Br(),
                
                dbc.Button('Run regression', 
                           id='btn_dec_tree_reg', 
                           n_clicks=0,
                    style={'display': 'inline-block',
                           'width': '18%',
                           'font-size': 12,
                           'margin-right': '2%',
                           "background-color": "#2a8cff",
                           },
                    ),
                
                dbc.Button("View results table",
                           id="open-dec_tree_reg_table_params",
                           style={
                               "background-color": "#2a8cff",
                               'width': '18%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '2%',
                               },
                           ),
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id="dec_tree_reg_table_params"), 
                                    html.Br(), 
                                    dcc.Markdown("", id='text-dec_tree_reg3'),
                                    ],
                                   ),
                     dbc.ModalFooter(
                                    dbc.Button("Close",
                                               id="close-dec_tree_reg_table_params", 
                                               className="ml-auto"),
                                    ),
                            ],
                    id="modal-dec_tree_reg_table_params",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button("View Decision Tree",
                           id="open-dec_tree_reg_table_perf",
                           style={
                               "background-color": "#2a8cff",
                               'width': '18%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '2%',
                               },
                    ),
                dbc.Modal(
                    [dbc.ModalBody([
                        html.Div(
                            id='figure_decision_tree',
                            children=[
                            html.Img(
                                src='',
                                style={'width': '100%', 
                                       'height': 'auto'},
                                ),
                            ],
                            style={'pointer-events': 'none'},
                            ),
                        dcc.Markdown("", id='text-dec_tree_reg4',
                                     style={'margin-right': '5%',
                                            'margin-left': '5%',
                                            },
                                     ),
                        html.Br(),
                        ],
                        ),
                        
                     dbc.ModalFooter(
                                    dbc.Button("Close", 
                                    id="close-dec_tree_reg_table_perf", 
                                    className="ml-auto")
                                    ),
                            ],
                    id="modal-dec_tree_reg_table_perf",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    fullscreen=True,
                    keyboard=True,
                    fade=True,
                    backdrop=False,
                    ),
                
                ],
                style={'width': '98.5%',
                       'margin-left': '1%',
                    },
            )




def run_decision_tree_regression(df, xvars, yvar, cat_vars):
    # Use your dummify function
    df, dropped, cat_var_ls = app_fxns.dummify(df, cat_vars, dropone=False)
    df = app_fxns.remove_nans_optimal(df, yvar)
    
    # Separate target variable and features
    y = df.filter(items=[yvar], axis=1)
    X = df.drop(labels=[yvar], axis=1)
    del df  # deleted to save memory

    if X.shape[1] < 2:
        return [], [], [], [], [], [], []

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=42)

    # Create the decision tree model
    model = DecisionTreeRegressor(max_depth=3)

    # Define hyperparameter grid
    param_grid = {
        #'max_depth': [None, 5, 10, 15],
        #'min_samples_split': [2, 5, 10],
        #'min_samples_leaf': [1, 2, 4],
        #'max_features': ['auto', 'sqrt', 'log2', None],
        'ccp_alpha': [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
    }

    # Hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_hyper_params = grid_search.best_params_

    # Train the model using the best hyperparameters
    model.set_params(**best_hyper_params)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Get feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame(columns=['Variable', 'Importance'])
    importance_df['Variable'] = list(X_train)
    importance_df['Importance'] = importances
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    
    # Access the decision tree structure
    tree_structure_data = model.tree_
    
    # Get the number of nodes
    n_nodes = model.tree_.node_count
    
    # Initialize arrays to store the level of each node
    node_levels = np.zeros(n_nodes, dtype=int)
    
    # Initialize queue for breadth-first traversal
    queue = [(0, 0)]  # (node_index, level)
    
    while queue:
        node_index, level = queue.pop(0)
        node_levels[node_index] = level
    
        # Get the child nodes
        left_child = model.tree_.children_left[node_index]
        right_child = model.tree_.children_right[node_index]
    
        # Add child nodes to the queue
        if left_child != right_child:  # Internal node
            queue.append((left_child, level + 1))
            queue.append((right_child, level + 1))
    
    # Count the number of internal nodes at each level
    internal_nodes_per_level = np.bincount(node_levels)[1:]
    internal_nodes_per_level = internal_nodes_per_level.tolist()
    
    try:
        index = max(range(len(internal_nodes_per_level)), key=lambda i: (internal_nodes_per_level[i], i))
        
        fs = (10/(index + 1.8)) + 1
    except:
        fs = 6
        
    
    plt.figure(figsize=(8,3))
    
    # Create a file-like buffer to receive the output figure
    buf = io.BytesIO()

    plot_tree(model, filled=True, feature_names=X.columns, rounded=True, fontsize=fs)
    plt.savefig(buf, format='png', dpi=300)
    plt.close()

    # Convert the buffer to a byte array
    buf.seek(0)
    img_bytes = buf.read()

    # Convert the byte array to base64 for Plotly
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Create a Plotly figure with the image
    tree_diagram = {
        'data': [go.Image(source=f'data:image/png;base64,{img_base64}')],
        'layout': {'title': 'Decision Tree'},
    }

    # Get mean squared error
    model_mse = mean_squared_error(y_test, y_pred)

    # Get R-squared
    r_square = r2_score(y_test, y_pred)

    return y_train, y_test, y_pred, best_hyper_params, importance_df, tree_diagram, tree_structure_data, model_mse, r_square, list(X_train)



def get_updated_results(n_clicks, xvars, yvar, df, cat_vars):
    
    cols = ['Variable', 'Importance']
    df_table1 = pd.DataFrame(columns=cols)
    df_table1['Variable'] = [np.nan]*10
    df_table1['Importance'] = [np.nan]*10
    
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
    
    ctx1 = dash.callback_context
    jd1 = json.dumps({'triggered': ctx1.triggered,})
    jd1 = jd1[:50]
    if 'rt4.children' in jd1:
        return [{}, {}, dashT1, "", "", "", ""]
    
    if df is None:
        return [{}, {}, dashT1, "", "", "", ""]
    
    elif yvar is None and xvars is None:
        return [{}, {}, dashT1, "", "", "", ""]
    
    elif xvars is None or len(xvars) < 2:
        return [{}, {}, dashT1, "Error: Select two or more predictors", "", "", ""]
        
    elif yvar is None:
        return [{}, {}, dashT1, "Error: Select a reponse variable", "", "", ""]
    
    elif (isinstance(yvar, list) is True) & (xvars is None or len(xvars) < 2):
        return [{}, {}, dashT1, "Error: Select a response variable and 2 or more predictors", 
                "", "", ""]
    
    elif isinstance(yvar, list) is True:
        return [{}, {}, dashT1, "Error: Select a response variable", "", "", ""]
    
    elif xvars is None or len(xvars) < 2:
        return [{}, {}, dashT1, "Error: Select two or more predictors", "", "", ""]
    
    df = pd.DataFrame(df)
    if df.empty:
        return {}, {}, dashT1, "", "", "", ""
    
    if yvar not in list(df):
        return [{}, {}, dashT1, "Error: Choose a response variable", "", "", ""]
        
    if yvar in xvars:
        xvars.remove(yvar)
        if len(xvars) == 0:
            str_ = "Error: Decision Tree regression requires 2 or more predictors. "
            str_ += "You chose one and it's the same as your response variable"
            return [{}, {}, dashT1, str_, "","",""]
        
        elif len(xvars) == 1:
            str_ = "Error: Decision Tree regression requires 2 or more predictors. "
            str_ += "You chose two but one is the same as your response variable"
            return [{}, {}, dashT1, str_, "","",""]
    
    if len(xvars) < 2 and yvar is None:
        str_ =  "Error: Decision Tree regression requires 2 or more predictors and one "
        str_ += "response variable."
        return [{}, {}, dashT1, str_, "","",""]
        
    elif len(xvars) < 2:
        return [{}, {}, dashT1, 
                "Error: Decision Tree regression requires 2 or more predictors.", "","",""]
                        
    else:
        
        vars_ = [yvar] + xvars
        df = df.filter(items=vars_, axis=1)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        cat_vars = [element for element in cat_vars if element in xvars]    
                            
        #Conduct decision tree regression
        ls = run_decision_tree_regression(df, xvars, yvar, cat_vars)
        y_train, y_test, y_pred, best_hyper_params, importance_df, tree_diagram, tree_data, model_mse, r_square, xlabs = ls 
        
        lab = list(y_test)
        lab = lab[0]
        y_test = y_test[lab].tolist()
        y_test = np.array(y_test)
        
        if len(y_test) == 0:
            rt1 = "Error: Your regression could not run. Your y-values contain no data."
            return {}, {}, dashT1, rt1, "", "", ""
            
        r2_obs_pred = app_fxns.obs_pred_rsquare(y_test, y_pred)
        r2_obs_pred = round(r2_obs_pred,2)
        
        y_test = y_test.tolist()
        y_pred = y_pred.tolist()
        
        fig_data = []
        
        miny = min([min(y_test), min(y_pred)])
        maxy = max([max(y_test), max(y_pred)])
        
        fig_data.append(go.Scatter(x = y_pred, 
                                   y = y_test, 
                                   name = 'Obs vs Pred',
                                   mode = "markers", 
                                   opacity = 0.75, 
                                   marker = dict(size=10, 
                                                 color="#3399ff",
                                                 ),
                                   ),
                        )
        
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

        ############################################################################################
        ############ CREATE TEXT OUTPUTS ###########################################################
        ############################################################################################
        
        txt1 = "The figure above allows you to interpret patterns in the regression model's "
        txt1 += "success. For example, if points are consistently above the 1:1 line, then all of "
        txt1 += "the observed values were greater than all of the predicted values; the model "
        txt1 += "would have consistently under-predicted all observed values."
        
        # Create a baseline model that predicts the mean of the target variable
        mean_baseline = np.mean(y_train)
        y_pred_baseline = np.full_like(y_test, mean_baseline)
        # Calculate the MSE of the baseline model
        mse_baseline = mean_squared_error(y_test, y_pred_baseline)
        
        txt2 = "Values predicted by the Decision Tree regression explained " 
        txt2 += str(round(100*r_square, 2)) + "% of variation in " + yvar + ". "
        #txt2 += "The modeling used cost complexity pruning (ccp) to prevent the decision tree " 
        #txt2 += "from overfitting your data. This optimization technique produced a ccp value of  
        
        if model_mse < mse_baseline:
            txt2 += "Additionally, the baseline mean-squared error (MSE) "
            txt2 += "was higher than your model's MSE (" + str(round(mse_baseline,2))
            txt2 += " vs " + str(round(model_mse,2)) + "). This indicates that your model provides "
            txt2 += "better predictions than a model that simply guesses the average value. "
            txt2 += "In other words, the fitted model added value by making more accurate predictions "
            txt2 += "than a naive baseline."
            
        elif model_mse >= mse_baseline:
            txt2 += "Additionally, the baseline mean-squared error (MSE) "
            txt2 += "was less than or equal to your model's MSE (" + str(round(mse_baseline,2))
            txt2 += " vs " + str(round(model_mse,2)) + "). This indicates that your model fails to "
            txt2 += "provides better predictions than a model that simply guesses the average value."
            
        txt3 = "The importance of a feature is determined by how much it helps reduce "
        txt3 += "uncertainty in the model. This reduction in uncertainty is measured using " 
        txt3 += "a metric known as Gini Importance. A feature's importance is then calculated " 
        txt3 += "as the total reduction in uncertainty brought by that feature. The more that a " 
        txt3 += "feature decreases uncertainty, the more important it is."
        
        
        dp1_feature = tree_data.feature[0]
        dp1_threshold = tree_data.threshold[0]
        f1 = xlabs[dp1_feature]
        
        dp2_feature = tree_data.feature[1]
        dp2_threshold = tree_data.threshold[1]
        f2 = xlabs[dp2_feature]
        
        txt4 =  "**How to interpret the tree diagram:**\n"
        txt4 += "The decision tree visualizes the logical sequence of conditions that the "
        txt4 += "model used to make predictions based on the predictor variables. "
        txt4 += "The top box, known as the root node, represents the entire set of observations. "
        txt4 += "It's also the first decision point. "
        txt4 += "This point examines the value of **" + f1 + "** for each observation, "
        txt4 += "determining for each, whether they are less than or equal to **"
        txt4 += str(round(dp1_threshold,1)) + "**. If true, we follow the left arrow; if false, we "
        txt4 += "follow the right arrow. The subsequent decision points (boxes) "
        txt4 += "represent more specific conditions and branches that further segment the data. "
        txt4 += "Each box contains information such "
        txt4 += "as the squared error, the **number of samples** (i.e., observations), and the "
        txt4 += "predicted **value** for the response variable at that node."
        
        #print('Best hyper parameters:')
        #print(best_hyper_params, '\n')
        
        
        ############################################################################################
        ############### END TEXT OUTPUTS ###########################################################
        ############################################################################################
        
        
        tree_image_data = tree_diagram['data'][0]['source'].split(",")[1]
        
        # Update the image in the callback
        image_component = html.Img(
            src=f'data:image/png;base64,{tree_image_data}',
            style={'width': '100%', 'height': 'auto'}  # Adjust image size as needed
        )
        
        dashT1 = dash_table.DataTable(
            data = importance_df.to_dict('records'),
            columns = [{'id': c, 'name': c} for c in importance_df.columns],
            
            page_action='none',
            sort_action="native",
            sort_mode="single",
            filter_action="native",
            
            style_table={#'height': '300px', 
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
        
        return figure, image_component, dashT1, txt1, txt2, txt3, txt4

        



