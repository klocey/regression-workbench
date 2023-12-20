import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import binary_classify
import DecisionTree
import GLM
import IMMR
import MLR
import quantile_regression
import Single_InDepth
import survival
import choose_data

statsmodels_tags = ["AIDS ","Down","WHO ","abortion","adolescent","adult","anxiety","arthritis",
                    "auditory","autism","behavior","biopsy","birth","blood","body","breast",
                    "cancer","cardiovascular","cell","child","cholera","cirrhosis","clinical",
                    "coma","consumption","contagious","contraceptive","coronary","covid","deaths",
                    "dengue","dependency","deprivation","diabetes","diarrhaea","disease","doctor",
                    "donation","drug","eating","ebola","efficacy","exercise","feet","fertility",
                    "food","freedom","gestation","headaches","health","healthcare","hepatocellular",
                    "hiv","illness","indomethacin","infant","insurance","liver","male","malignant",
                    "mammogram","marrow","medicaid","medical","medicare","medpar","melanoma",
                    "migraine","neuro","nutrition","obesity","obstetrics","organ","pancreatic",
                    "parents","periodontal","physician","prenatal","prostate","prothrombin",
                    "quarantine","radiation","recovery","remission","reporting","respiratory",
                    "risk","serum","sleep","smallpox","smoking","society","stress","syndrome",
                    "transplant","urine","vaccine","weight","wheeze","women"]


####################################################################################################
####################      DASH APP CONTROL CARDS       #############################################
####################################################################################################


def description_card1():
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.H3("Regression workbench", 
                            style={'textAlign': 'left', 
                                   'margin-left': '2%', 
                                   'color': '#2a8cff',
                                   }
                    ),
                    dcc.Markdown("Discover relationships within data using the most common tool " +
                                 "of statistical analysis. This open-source analytical " +
                                 "application offers simple and sophisticated forms of " +
                                 "regression, automated analyses and optimizations, and provides " +
                                 "user guidance and interpretive outputs. Use the web application" +
                                 " or download the " +
                                 "[source code] (https://github.com/klocey/regression-workbench)" +
                                 " and run it locally.",
                                 style={'textAlign': 'left', 
                                        'margin-left': '3%',
                                        },
                                 ),
                    html.Br(), 
                    control_card_upload1(),
                    inspect_data_table(),
                    control_card_choose_reg1(),
                ],
                style={ 'width': '99%', 
                       'display': 'inline-block',
                       },
                ),
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
                        },
                    ),
            html.P("Kenneth J. Locey, PhD. Senior clinical data scientist. Center for Quality, " +
                   "Safety and Value Analytics. Rush University Medical Center.",
                    style={
                        'textAlign': 'left',
                        },
                    ),
            html.H5("Testers",
                    style={
                        'textAlign': 'left',
                        },
                    ),
            html.P("Ryan Schipfer. Senior clinical data scientist. Center for Quality, Safety " +
                   "and Value Analytics. Rush University Medical Center.",
                    style={
                        'textAlign': 'left',
                        },
                    ),
            html.P("Brittnie Dotson. Clinical data scientist. Center for Quality, Safety and " +
                   "Value Analytics. Rush University Medical Center.",
                    style={
                        'textAlign': 'left',
                        },
                    ),
        ],
    )


def control_card_upload1():
    
    return html.Div(
        id="control-card-upload1",
        children=[
            dbc.Button("1. load a dataset",
                       id="open-centered-controlcard_load",
                       style={
                           "background-color": "#2a8cff",
                           'width': '99%',
                           'font-size': 16,
                           'display': 'inline-block',            
                           },
                ),
            dbc.Modal(
                [dbc.ModalBody([
                    html.Div(
                            id="left-column1a",
                            className="one columns",
                            children=[control_card_upload1a()],
                            style={'width': '46.0%',
                                   'display': 'block',
                                   'border-radius': '15px',
                                   'box-shadow': '1px 1px 1px grey',
                                   'background-color': '#f0f0f0',
                                   'padding': '10px',
                                   'margin-bottom': '10px',
                            },
                        ),
                    html.Div(
                            id="left-column1b",
                            className="one columns",
                            children=[control_card_upload1b()],
                            style={'width': '46.0%',
                                    'display': 'block',
                                    'border-radius': '15px',
                                    'box-shadow': '1px 1px 1px grey',
                                    'background-color': '#f0f0f0',
                                    'padding': '10px',
                                    'margin-bottom': '10px',
                            },
                        ),
                    html.Br(), 
                    ],
                    ),
                dbc.ModalFooter(
                        dbc.Button("Close", 
                                   id="close-centered-controlcard_load", 
                                   className="ml-auto")
                        ),
                ],
                id="modal-centered-controlcard_load",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
                ],
        style={
            'width': '28%',
            'margin-left': '5%',
            'margin-bottom': '1%',
            'display': 'inline-block',
            },
        )


def control_card_upload1a():
    
    return html.Div(
        id="control-card-upload1a",
        children=[
            html.H5("Option 1. Upload your own data", 
                    style={'display': 'inline-block',
                           'margin-right': '1%',
                           },
                    ),
            html.I(className="fas fa-question-circle fa-lg", 
                   id="target1a",
                   style={
                       'display': 'inline-block', 
                       'width': '5%', 
                       'color':'#99ccff',
                       },
                ),
            dbc.Tooltip("Uploaded should have a simple format: rows, columns, and one row of " +
                        "column headers. Headers should not have commas or colons. Data should " +
                        "not have mixed types (10% and 10cm have numeric and non-numeric " +
                        "characters).", 
                        target="target1a",
                        style = {
                            'font-size': 14,
                            },
                        ),
            html.P("This app only accepts .csv files. Data are deleted when the app is " +
                   "refreshed, closed, or when another file is uploaded. Still, do not " +
                   "upload sensitive data.",
                   ),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a CSV File', 
                           style={'color':'#2c8cff', 
                                  "text-decoration": "underline",
                                  },
                           ),
                    ],
                    ),
                style={
                    'lineHeight': '34px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                },
                multiple=False,
            ),
            ],
        )


def control_card_upload1b():
    
    return html.Div(
        children=[
            html.H5("Option 2. Select a healthcare dataset", 
                    style={'display': 'inline-block',
                           'margin-right': '5%',
                           },
                    ),
            html.P("These preprocessed healthcare datasets are derived from publicly available " +
                   "data provided in peer-reviewed publications or by the Centers for Medicare " +
                   "and Medicaid Services."),
            dbc.Button("Select a dataset",
                       id="open-centered-controlcard",
                       style={
                           "background-color": "#2a8cff",
                           'width': '95%',
                           'font-size': 12,
                           },
                ),
            dbc.Modal(
                [dbc.ModalBody([choose_data.control_card_choose_data(), 
                                html.Br(), 
                                ],
                               ),
                        ],
                id="modal-centered-controlcard",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
                ],
        )


def level_grouper_control_card():
    
    return html.Div(
        children=[
            html.H5('Group Levels',
                   style={'textAlign': 'left', 
                          'font-size': 28,
                          'color': '#ffffff',
                          },
                   ),
            html.Br(),
            
            html.B("Choose a categorical variable",
                style={'display': 'block',
                       'font-size': 16,
                        'vertical-align': 'top',
                        'color': '#ffffff',
                   },
                ),
            dcc.Dropdown(
                    id='cat_var_group',
                    options=[{"label": i, "value": i} for i in []],
                    multi=False, 
                    value=None,
                    style={'width': '50%',
                         },
                    ),
            html.Br(),
            
            html.B("Choose one or more levels",
                style={'display': 'block',
                       'font-size': 16,
                       'vertical-align': 'top',
                       'color': '#ffffff',
                },
            ),
            dcc.Dropdown(
                    id='level_vars',
                    options=[{"label": i, "value": i} for i in []],
                    multi=True, 
                    value=None,
                    style={'width': '99%',
                         },
                    ),
            html.Br(),
            
            html.B("Create a new level OR group the level(s) you chose into an existing level",
                style={'display': 'block',
                       'font-size': 16,
                       'vertical-align': 'top',
                       'color': '#ffffff',
                },
            ),
            dcc.Input(
                id='new_level_name',
                type='text',
                placeholder='Type the name of an existing level or create a new level',
                style={'width': '51%',
                    'font-size': 16,
                    'display': 'block',
                    },
            ),
            html.Br(),
            
            dbc.Button('Group', 
                        id='level-group-btn1', 
                        n_clicks=0,
                        style={'width': '20%',
                            'font-size': 14,
                            "background-color": "#2a8cff",
                            'margin-right': '2%',
                            },
            ),
            html.Br(),
            html.H5("", id='group_text',
            style={'textAlign': 'left',
                   'font-size': 16,
                   'margin-left': '3%',
                   'color': '#ffffff'}),
            dcc.Interval(id='group-interval-component', interval=3000, n_intervals=0)
            ],
        )
    

def statsmodels_data_table():
    return html.Div(
        id='statsmodels_data_table_div', 
        className="ten columns",
        children=[
            
            dash_table.DataTable(
                id='statsmodels_data_table',
                columns=[{
                    'name': 'Column {}'.format(i),
                    'id': 'column-{}'.format(i),
                    'deletable': False,
                    'renamable': False,
                    'selectable': False,
                    } for i in range(1, 4)],
                                        
                data=None,
                is_focused=True,
                #virtualization=True,
                #editable=True,
                page_action='native',
                page_size=120,
                filter_action='native',
                sort_action='native',
                #row_deletable=True,
                #column_selectable='multi',
                #export_format='xlsx',
                #export_headers='display',
                fixed_rows={'headers': True},
                                  
                style_header={'padding':'1px', 
                              #'width':'250px', 
                              #'minWidth':'250px', 
                              #'maxWidth':'250px', 
                              'textAlign': 'center', 
                              'overflowX': 'auto',
                              },
                style_data = {'padding':'5px', 
                              #'width':'250px', 
                              #'minWidth':'250px', 
                              #'maxWidth':'250px', 
                              'textAlign': 'center', 
                              'overflowX': 'auto',
                              },
                style_table={#'height': '120px', 
                             'overflowX': 'auto',
                             },
                style_cell_conditional=[
                    {'if': {'column_id': 'id'}, 'width': '60%'},
                    {'if': {'column_id': 'No. of rows'}, 'width': '20%'},
                    {'if': {'column_id': 'No. of columns'}, 'width': '20%'},
                ],
                ),
            
            ],
        style={'display':'block',
               'width': '93%',
               "background-color": "#696969",
               }
        )


def data_table():
    return html.Div(
        id='Data-Table1', 
        className="ten columns",
        children=[html.H5("Data Table", 
                          style={'display': 'inline-block', 
                                 'color': '#FFFFFF',
                                 'font-size': 28,
                                 'width': '10.5%',
                                 },
                    ),
            html.I(className="fas fa-question-circle fa-lg", 
                   id="target_DataTable",
                   style={'display': 'inline-block', 
                          'width': '3%', 
                          'color':'#99ccff',
                          'vertical-align': 'center',
                          },
                   ),
            dbc.Tooltip("Use this table to ensure your data loaded as expected and to delete any " +
                        "rows or select, delete, and rename any columns. There are no limits on " +
                        "dataset size when running the application locally. But, when using the " +
                        "web application, any dataset containing more than 5K rows or 50 columns " +
                        "will be randomly sampled to meet those constraints.", 
                        target="target_DataTable",
                        style = {'font-size': 12, 
                                #'display': 'inline-block',
                            },
                    ),
                                      
            html.P("", id='rt4',
                   style={'color': '#FFFFFF', 
                          'font-size': 18,
                          },
                   ),
                                  
            dash_table.DataTable(
                id='data_table',
                columns=[{
                    'name': 'Column {}'.format(i),
                    'id': 'column-{}'.format(i),
                    'deletable': True,
                    'renamable': True,
                    'selectable': True,
                    } for i in range(1, 9)],
                                        
                data=None,
                #virtualization=True,
                editable=True,
                page_action='native',
                page_size=100,
                filter_action='native',
                sort_action='native',
                #row_deletable=True,
                column_selectable='multi',
                export_format='xlsx',
                export_headers='display',
                #fixed_rows={'headers': True},
                
                style_header={'padding':'1px', 
                              'width':'250px', 
                              'minWidth':'250px', 
                              'maxWidth':'250px', 
                              'textAlign': 'center', 
                              'overflowX': 'auto',
                              },
                style_data = {'padding':'5px', 
                              'width':'250px', 
                              'minWidth':'250px', 
                              'maxWidth':'250px', 
                              'textAlign': 'center', 
                              'overflowX': 'auto',
                              },
                style_table={'height': '120px', 
                             'overflowX': 'auto',
                             },
                ),
            
            ],
        style={'display':'block',
               'width': '95%',
               "background-color": "#696969",
               }
        )


def inspect_data_table():
    
    return html.Div(
        children=[
            dbc.Button("2. inspect your data",
                       id="open-inspect_main_datatable",
                       style={
                           "background-color": "#2a8cff",
                           'width': '99%',
                           'font-size': 16,
                           'display': 'inline-block',
                           },
                ),
            dbc.Modal([
                dbc.ModalBody([
                        html.Div(
                                id="left-column_inspect_table",
                                className="one columns",
                                children=[data_table()],
                                style={'width': '100%'},
                            ),
                        html.Br(),
                        ],
                        style={'width': '100%',
                               "background-color": "#696969",
                               },
                        ),
                    
                dbc.ModalFooter([
                        dbc.Button('Group Levels',
                                   id="open-level-group",
                                   style={
                                       "background-color": "#2a8cff",
                                       'width': '30%',
                                       'font-size': 16,
                                       'display': 'inline-block',
                                       },
                            ),
                        
                        dbc.Modal([
                            dbc.ModalBody([
                                    html.Div(
                                            id="level-group",
                                            className="one columns",
                                            children=[level_grouper_control_card()],
                                            style={'width': '100%'},
                                        ),
                                    html.Br(),
                                    ],
                                    style={'width': '100%',
                                           "background-color": "#A0A0A0",
                                           },
                                    ),
                            dbc.ModalFooter(
                                    dbc.Button("Click to Close",
                                       id="close-level-group", 
                                       className="ml-auto",
                                       style={
                                           "background-color": "#2a8cff",
                                           'width': '30%',
                                           'font-size': 16,
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
                            id="modal-level-group",
                            
                            is_open=False,
                            centered=True,
                            autoFocus=True,
                            size="xl",
                            keyboard=True,
                            fade=True,
                            backdrop=True,
                            
                            ),
                        
                        
                        dbc.Button("Click to Close", 
                                   id="close-inspect_main_datatable", 
                                   className="ml-auto",
                                   style={
                                       "background-color": "#2a8cff",
                                       'width': '30%',
                                       'font-size': 16,
                                       'display': 'inline-block',
                                       },
                                   ),
                        ],
                        style={
                            "background-color": "#696969",
                            "display": "flex",
                            "justify-content": "center",
                            "align-items": "center",
                            },
                        ),
                    ],
                
                id="modal-inspect_main_datatable",
                is_open=False,
                centered=True,
                autoFocus=True,
                fullscreen=True,
                keyboard=True,
                fade=True,
                ),
                ],
        style={'width': '28%', 
               'display': 'inline-block',
               'margin-left': '3%',
               'margin-bottom': '1%',
               },
        )


def control_card_choose_reg1():
    
    return html.Div(
        children=[
            dbc.Button("3. choose an analysis",
                       id="open-choose_regression",
                       style={
                           "background-color": "#2a8cff",
                           'width': '99%',
                           'font-size': 16,
                           'display': 'inline-block',
                           },
                ),
            dbc.Modal(
                [dbc.ModalBody([
                    html.Div(
                            id="left-column_choose_reg2",
                            className="one columns",
                            children=[control_card_choose_reg2()],
                            style={'width': '100%',
                                   'display': 'inline-block',
                            },
                        ),
                    
                    html.Br(), 
                    ],
                    ),
                dbc.ModalFooter(
                        dbc.Button("Close", 
                                   id="close-choose_regression", 
                                   className="ml-auto",
                                   style={"background-color": "#2a8cff",
                                          'width': '30%',
                                          'font-size': 14,
                                          },
                                   ),
                        style={ "background-color": "#696969",
                            "display": "flex",
                            "justify-content": "center",  # Center horizontally
                            "align-items": "center",  # Center vertically)
                            },
                        ),
                ],
                id="modal-choose_regression",
                is_open=False,
                centered=True,
                autoFocus=True,
                size="xl",
                keyboard=True,
                fade=True,
                backdrop=True,
                ),
                ],
        style={'width': '28%', 
               'display': 'inline-block',
               'margin-left': '3%',
               'margin-bottom': '2%',
               },
        )

def control_card_choose_reg2():
    return html.Div(
        children=[    
            html.Div(
                children=[
                    html.H5("Iterative Multi-Model Regressions", 
                            style={'display': 'inline-block', 
                                   },
                            ),
                    dcc.Markdown("Automate the search for **linear and non-linear** 1-to-1 " +
                                 "relationships. Apply optimized data transformations and choose " +
                                 "between the classic ordinary least squares (**OLS**) approach " +
                                 "or run **Robust regression** to reduce the influence of outliers.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dbc.Button("Run iterative multi-model analysis",
                               id='open-iterative_ols',
                               style={"background-color": "#2a8cff",
                                      'width': '92.5%',
                                      'font-size': 12,
                                      'display': 'inline-block',
                                      },
                        ),
                    dbc.Modal([
                        dbc.ModalBody([
                            html.Div(
                                id="left-column2",
                                className="two columns",
                                children=[IMMR.control_card_iterative_multi_model_regression(),
                                          IMMR.generate_outputs_iterative_multi_model_regression(),
                                          ],
                                style={'width': '95.3%',
                                        'display': 'block',
                                        'border-radius': '15px',
                                        'box-shadow': '1px 1px 1px grey',
                                        'background-color': '#f0f0f0',
                                        'padding': '10px',
                                        'margin-bottom': '10px',
                                        },
                                    ),                        
                                ]),
                            dbc.ModalFooter(
                                dbc.Button("Click to Close", 
                                           id='close-iterative_ols', 
                                           className="ml-auto",
                                           style={
                                               "background-color": "#2a8cff",
                                               'width': '30%',
                                               'font-size': 14,
                                               },
                                           ),
                                style={
                                    "background-color": "#696969",
                                    "display": "flex",
                                    "justify-content": "center",
                                    "align-items": "center",
                                    },
                                ),
                        ],
                    id="modal-iterative_ols",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    fullscreen=True,
                    keyboard=True,
                    fade=True,
                    ),
                ],
                style={'display': 'inline-block', 
                       'width': '45%',
                       'margin-right': '7%',
                       'margin-left': '1%',
                       },
            ),
            
            html.Div(
                children=[
                    html.H5("In-depth Bivariate Regression", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("Focus on the relationship between 2 variables using **linear " +
                                 "and polynomial** forms of **OLS regression** and **Robust " +
                                 "regression**. Perform data transformations, build confidence " +
                                 "intervals and prediction intervals, and identify outliers.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dbc.Button("Run in-depth bivariate regression",
                               id='open-single_ols',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
                        ),
                    dbc.Modal([
                        dbc.ModalBody([
                            html.Div(
                                id="left-column3",
                                className="two columns",
                                children=[Single_InDepth.control_card_single_regression(),
                                          Single_InDepth.generate_figure_single_regression(),
                                          ],
                                style={'width': '95.3%',
                                        'border-radius': '15px',
                                        'box-shadow': '1px 1px 1px grey',
                                        'background-color': '#f0f0f0',
                                        'padding': '10px',
                                        'margin-bottom': '10px',
                                },
                            ),                    
                            ]),
                            dbc.ModalFooter(
                                dbc.Button("Click to Close", 
                                           id='close-single_ols', 
                                           className="ml-auto",
                                           style={
                                               "background-color": "#2a8cff",
                                               'width': '30%',
                                               'font-size': 14,
                                               },
                                           ),
                                style={
                                    "background-color": "#696969",
                                    "display": "flex",
                                    "justify-content": "center",
                                    "align-items": "center",
                                    },
                                ),
                        ],
                    id="modal-single_ols",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    fullscreen=True,
                    keyboard=True,
                    fade=True,
                    ),
                ],
                style={'display': 'inline-block', 
                       'width': '45%',
                       }
            ),
            
            html.Br(),
            html.Br(),
            html.Br(),
            
            html.Div(
                children=[
                    html.H5("Quantile regression", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("Abandon constraints of linear regression and examine the upper " +
                                 "and lower bounds of a 1-to-1 relationship, or any area in " +
                                 "between. The Rush Regression Workbench extends quantile " +
                                 "regression to curvilinear relationships and the use of automated, " +
                                 "optimized forms of data scaling and feature selection.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dbc.Button("Run quantile regression",
                               id='open-quant_reg',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
                        ),
                    dbc.Modal([
                        dbc.ModalBody([
                            html.Div(
                                id="left-column3_quant",
                                className="two columns",
                                children=[quantile_regression.control_card_quantile_regression(),
                                          quantile_regression.generate_figure_quantile_regression(),
                                          ],
                                style={'width': '95.3%',
                                        'border-radius': '15px',
                                        'box-shadow': '1px 1px 1px grey',
                                        'background-color': '#f0f0f0',
                                        'padding': '10px',
                                        'margin-bottom': '10px',
                                },
                            ),           
                            ],
                            ),
                            dbc.ModalFooter(
                                dbc.Button("Click to Close", 
                                           id='close-quant_reg', 
                                           className="ml-auto",
                                           style={
                                               "background-color": "#2a8cff",
                                               'width': '30%',
                                               'font-size': 14,
                                               },
                                           ),
                                style={
                                    "background-color": "#696969",
                                    "display": "flex",
                                    "justify-content": "center",
                                    "align-items": "center",
                                    },
                                ),
                        ],
                    id="modal-quant_reg",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    fullscreen=True,
                    keyboard=True,
                    fade=True,
                    ),
                ],
                style={'display': 'inline-block', 
                       'width': '45%',
                       'margin-right': '7%',
                       'margin-left': '1%',
                       },
            ),
            
            html.Div(
                children=[
                    html.H5("Multivariable Forms of Linear Regression", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("Understand how multiple predictors collectively influence " +
                                 "another variable. Take the OLS approach or use **Ridge " +
                                 "regression** and **Lasso regression** to reduce statistical " +
                                 "issues that arise when using many predictors. Automated, " +
                                 "optimized forms of data scaling and feature selection are included.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dbc.Button("Run multivariable forms of linear regression",
                               id='open-multi_reg',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
                        ),
                    dbc.Modal([
                        dbc.ModalBody([
                            html.Div(
                                id="left-column4",
                                className="two columns",
                                children=[MLR.control_card_linear_multivariable(),
                                          MLR.generate_linear_multivariable_outputs(),
                                          ],
                                style={'width': '95.3%',
                                        'display': 'block',
                                        'border-radius': '15px',
                                        'box-shadow': '1px 1px 1px grey',
                                        'background-color': '#f0f0f0',
                                        'padding': '10px',
                                        'margin-bottom': '10px',
                                },
                            ),
                            ],
                            ),
                            dbc.ModalFooter(
                                dbc.Button("Click to Close", 
                                           id='close-multi_reg', 
                                           className="ml-auto",
                                           style={
                                               "background-color": "#2a8cff",
                                               'width': '30%',
                                               'font-size': 14,
                                               },
                                           ),
                                style={
                                    "background-color": "#696969",
                                    "display": "flex",
                                    "justify-content": "center",
                                    "align-items": "center",
                                    },
                                ),
                        ],
                    id="modal-multi_reg",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    fullscreen=True,
                    keyboard=True,
                    fade=True,
                    ),
                ],
                style={'display': 'inline-block', 
                       'width': '45%',
                       },
            ),
            
            html.Br(),
            html.Br(),
            html.Br(),
            
            html.Div(
                children=[
                    html.H5("Binary Classification", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("Use **logistic regression** and **probit regression** to " + 
                                 "classify outcomes (e.g., readmission/non-readmission, " +
                                 "positive/negative) based on one or more predictor variables. " +
                                 "Leverage machine learning, automated optimized data scaling, " +
                                 "and diagostic curves.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dbc.Button("Run Binary Classification",
                               id='open-logistic_reg',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
                        ),
                    dbc.Modal([
                        dbc.ModalBody([
                            html.Div(
                                id="left-column5",
                                className="two columns",
                                children=[binary_classify.control_card_logistic(),
                                          binary_classify.generate_logistic_a(),
                                          binary_classify.generate_logistic_b(),
                                          ],
                                style={'width': '95.3%',
                                        'display': 'block',
                                        'border-radius': '15px',
                                        'box-shadow': '1px 1px 1px grey',
                                        'background-color': '#f0f0f0',
                                        'padding': '10px',
                                        'margin-bottom': '10px',
                                },
                            ),
                            ]),
                            dbc.ModalFooter(
                                dbc.Button("Click to Close", 
                                           id='close-logistic_reg', 
                                           className="ml-auto",
                                           style={
                                               "background-color": "#2a8cff",
                                               'width': '30%',
                                               'font-size': 14,
                                               },
                                           ),
                                style={
                                    "background-color": "#696969",
                                    "display": "flex",
                                    "justify-content": "center",
                                    "align-items": "center",
                                    },
                                ),
                        ],
                    id="modal-logistic_reg",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    fullscreen=True,
                    keyboard=True,
                    fade=True,
                    ),
                ],
                style={'display': 'inline-block', 
                       'width': '45%',
                       'margin-right': '7%',
                       'margin-left': '1%',
                       },
            ),
            
            html.Div(
                children=[
                    html.H5("Generalized Linear Models (GLMs)", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("GLMs reveal how multiple predictors influence a single " +
                                 "response variable and accommodate data distributions that " +
                                 "simpler regression models do not. Choose from **Poisson**, " +
                                 "**Negative Binomial**, **Gamma**, **Gaussian**, " +
                                 "**Inverse Gaussian**, and **Tweedie** models.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dbc.Button("Run Generalized Linear Modeling",
                               id='open-glm',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
                        ),
                    dbc.Modal([
                        dbc.ModalBody([
                            html.Div(
                                id="left-column6",
                                className="two columns",
                                children=[GLM.control_card_glm(),
                                          GLM.generate_glm_outputs(),
                                          ],
                                style={'width': '95.3%',
                                        'display': 'block',
                                        'border-radius': '15px',
                                        'box-shadow': '1px 1px 1px grey',
                                        'background-color': '#f0f0f0',
                                        'padding': '10px',
                                        'margin-bottom': '10px',
                                },
                            ),
                            ]),
                            dbc.ModalFooter(
                                dbc.Button("Click to Close", 
                                           id='close-glm', 
                                           className="ml-auto",
                                           style={
                                               "background-color": "#2a8cff",
                                               'width': '30%',
                                               'font-size': 14,
                                               },
                                           ),
                                style={
                                    "background-color": "#696969",
                                    "display": "flex",
                                    "justify-content": "center",
                                    "align-items": "center",
                                    },
                                ),
                        ],
                    id="modal-glm",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    fullscreen=True,
                    keyboard=True,
                    fade=True,
                    ),
                ],
                style={'display': 'inline-block', 
                       'width': '45%',
                       },
            ),
            
            html.Br(),
            html.Br(),
            html.Br(),
            
            html.Div(
                children=[
                    html.H5("Survival Regression", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("Survival regression is used in so-called survival " +
                                 "analysis or 'time-to-event' analysis. It is used to model the " +
                                 "time until an event of interest, " +
                                 "considering the impact of covariates on the hazard rate, " +
                                 "a measure of risk.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dbc.Button("Run Survival Regression",
                               id='survival_reg_btn',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
                        ),
                    dbc.Modal([
                        dbc.ModalBody([
                            html.Div(
                                className="two columns",
                                children=[survival.control_card_survival_regression(),
                                          survival.generate_survival_outputs(),
                                          ],
                                style={'width': '95.3%',
                                        'display': 'block',
                                        'border-radius': '15px',
                                        'box-shadow': '1px 1px 1px grey',
                                        'background-color': '#f0f0f0',
                                        'padding': '10px',
                                        'margin-bottom': '10px',
                                },
                            ),
                            ],
                            ),
                            dbc.ModalFooter(
                                dbc.Button("Click to Close", 
                                           id='close-survival', 
                                           className="ml-auto",
                                           style={
                                               "background-color": "#2a8cff",
                                               'width': '30%',
                                               'font-size': 14,
                                               },
                                           ),
                                style={
                                    "background-color": "#696969",
                                    "display": "flex",
                                    "justify-content": "center",
                                    "align-items": "center",
                                    },
                                ),
                        ],
                    id='modal-survival',
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    fullscreen=True,
                    keyboard=True,
                    fade=True,
                    ),
                ],
                style={'display': 'inline-block', 
                       'width': '45%',
                       'margin-right': '7%',
                       'margin-left': '1%',
                       },
            ),
            
            
            html.Div(
                children=[
                    html.H5("Decision Tree Regression", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("Decision tree regression models data by recursively dividing it " + 
                                 "into segments and predicting outcomes based on the mean of the " +
                                 "target variable within each segment, providing insights in a " +
                                 "structured manner.",

                                 style={'width': '94.1%',
                                        },
                                 ),
                    dbc.Button("Run Decision Tree Regression",
                               id='open-decision_tree',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
                        ),
                    dbc.Modal([
                        dbc.ModalBody([
                            html.Div(
                                className="two columns",
                                children=[DecisionTree.control_card_decision_tree_regression(),
                                          DecisionTree.generate_decision_tree_outputs(),
                                          ],
                                style={'width': '95.3%',
                                        'display': 'block',
                                        'border-radius': '15px',
                                        'box-shadow': '1px 1px 1px grey',
                                        'background-color': '#f0f0f0',
                                        'padding': '10px',
                                        'margin-bottom': '10px',
                                },
                            ),
                            ],
                            ),
                            dbc.ModalFooter(
                                dbc.Button("Click to Close", 
                                           id='close-decision_tree', 
                                           className="ml-auto",
                                           style={
                                               "background-color": "#2a8cff",
                                               'width': '30%',
                                               'font-size': 14,
                                               },
                                           ),
                                style={
                                    "background-color": "#696969",
                                    "display": "flex",
                                    "justify-content": "center",
                                    "align-items": "center",
                                    },
                                ),
                        ],
                    id='modal-decision_tree',
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    fullscreen=True,
                    keyboard=True,
                    fade=True,
                    ),
                ],
                style={'display': 'inline-block', 
                       'width': '45%',
                       },
            ),
            ],
        )





