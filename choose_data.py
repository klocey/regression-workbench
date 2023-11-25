import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table


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


def control_card_choose_data():
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.H5("Hospital Cost Reports", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("Each year, thousands of US hospitals submit cost reports to " +
                                 "the Centers for Medicare and Medicaid Services. The data " +
                                 "provided here are derived from a recently developed open-source" +
                                 "[project](https://github.com/klocey/HCRIS-databuilder/tree/master)" +
                                 " and [application] (https://hcris-app.herokuapp.com/) " +
                                 " for analyzing hospital cost report data. See the associated " +
                                 "peer-reviewed [publication]" +
                                 "(https://www.sciencedirect.com/science/article/pii/S2772442523001417)" +
                                 " in Healthcare Analytics for details.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dcc.Dropdown(
                            id='hcris-year',
                            options=[{"label": i, "value": i} for i in ['2023', '2022', '2021', 
                                                                        '2020', '2019', '2018', 
                                                                        '2017', '2016', '2015', 
                                                                        '2014', '2013', '2012',
                                                                        '2011', '2010',
                                                                        ]
                                     ],
                            multi=False, 
                            value=None,
                            placeholder='Choose a federal fiscal year',
                            style={'width': '96.2%',
                                   'margin-bottom': '1%',
                                 },
                            ),
                    dbc.Button("Load Cost Report dataset",
                               id='hcris',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
                        ),
                    html.Div(id='button-output'),
                ],
                style={'display': 'inline-block', 
                       'width': '45%',
                       'margin-right': '7%',
                       'margin-left': '1%',
                       }
            ),
            
            html.Div(
                children=[
                    html.H5("Healthcare-Associated Infections", 
                            style={'display': 'inline-block', 
                                   },
                            ),
                    dcc.Markdown("Healthcare-Associated Infections (HAIs) measures provide data" +
                                 "on inpatient infections among individual hospitals. HAIs can " +
                                 "relate to devices, surgical procedures, or the spread of " +
                                 "bacterial infections. The data provided here are curated and " +
                                 "compiled versions of [data] " +
                                 "(https://data.cms.gov/provider-data/dataset/77hc-ibv8) " +
                                 "offered by the Centers for Medicare and Medicaid Services.",
                                 style={'width': '94.1%',
                                        }),
                    dcc.Dropdown(
                            id='hais-year',
                            options=[{"label": i, "value": i} for i in ['2023', '2022', '2021', 
                                                                        '2020', '2019', '2018', 
                                                                        '2017', '2016', '2015', 
                                                                        '2014']
                                     ],
                            multi=False,
                            value=None,
                            placeholder='Choose a federal fiscal year',
                            style={'width': '96.2%',
                                   'margin-bottom': '1%',
                                 },
                            ),
                    dbc.Button("Load HAI dataset",
                               id='hais',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
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
                    html.H5("Complications and Deaths", 
                            style={'display': 'inline-block', 
                                   },
                            ),
                    dcc.Markdown("This data set includes provider-level data for the hip/knee " +
                                 "complication measure, the CMS Patient Safety Indicators, and " +
                                 "30-day death rates. The data provided here are curated and " +
                                 "compiled versions of " +
                                 "[data](https://data.cms.gov/provider-data/dataset/ynj2-r877) " +
                                 "offered by the Centers for Medicare and Medicaid Services.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dcc.Dropdown(
                            id='c_and_d-year',
                            options=[{"label": i, "value": i} for i in ['2023', '2022', '2021', 
                                                                        '2020', '2019', '2018', 
                                                                        '2017', '2016', '2015', 
                                                                        '2014']
                                     ],
                            multi=False, 
                            value=None,
                            placeholder='Choose a federal fiscal year',
                            style={'width': '96.2%',
                                   'margin-bottom': '10px',
                                 },
                            ),
                    dbc.Button("Load complications and deaths dataset",
                               id='c_and_d',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
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
                    html.H5("Hospital Acquired Conditions Reduction Program (HACRP)", 
                            style={'display': 'inline-block', 
                                   'margin-right': '10px',
                                   },
                            ),
                    dcc.Markdown("CMS reduces Medicare fee-for-service payments by 1% for " +
                                 "hospitals that rank in the worst-performing quartile of " +
                                 "total hospital-acquired condition (HAC) scores. The data " +
                                 "provided here are curated and compiled versions of " +
                                 "[data](https://data.cms.gov/provider-data/dataset/yq43-i98g) " +
                                 "offered by the Centers for Medicare and Medicaid Services.",
                                 style={'width': '94.1%'},
                                 ),
                    dcc.Dropdown(
                            id='hacrp-year',
                            options=[{"label": i, "value": i} for i in ['2023', '2022', '2021', 
                                                                        '2020', '2019', '2018', 
                                                                        '2017', '2016', '2015']
                                     ],
                            multi=False, 
                            value=None,
                            placeholder='Choose a federal fiscal year',
                            style={'width': '96.2%',
                                   'margin-bottom': '10px',
                                 },
                            ),
                    dbc.Button("Load HACRP dataset",
                               id='hacrp',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
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
                    html.H5("Hospital Readmissions Reduction Program (HRRP)", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("CMS reduces Medicare payments for hospitals with excess " +
                                 "readmissions, which are measured by the ratio of a " +
                                 "hospital's predicted rate of readmissions for heart attack " +
                                 "(AMI), heart failure (HF), pneumonia, chronic obstructive " +
                                 "pulmonary disease (COPD), hip/knee replacement (THA/TKA), " +
                                 "and coronary artery bypass graft surgery (CABG) to an " +
                                 "expected rate, based on an average hospital with similar " +
                                 "patients. The data provided here are curated and compiled " +
                                 "versions of " +
                                 "[data](https://data.cms.gov/provider-data/dataset/9n3s-kdb3)" +
                                 "offered by the Centers for Medicare and Medicaid Services.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dcc.Dropdown(
                            id='hrrp-year',
                            options=[{"label": i, "value": i} for i in ['2023', '2022', '2021', 
                                                                        '2020', '2019', '2018', 
                                                                        '2017', '2016', '2015', 
                                                                        '2014', '2013']
                                     ],
                            multi=False, 
                            value=None,
                            placeholder='Choose a federal fiscal year',
                            style={'width': '96.2%',
                                   'margin-bottom': '10px',
                                 },
                            ),
                    dbc.Button("Load HRRP dataset",
                               id='hrrp',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
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
                    html.H5("Payment and Value of Care", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("The Medicare Spending Per Beneficiary (MSPB or “Medicare " +
                                 "hospital spending per patient”) measure shows whether " +
                                 "Medicare spends more, less, or about the same on an episode " +
                                 "of care for a Medicare patient treated in a specific " +
                                 "inpatient hospital compared to how much Medicare spends on " +
                                 "an episode of care across all inpatient hospitals " +
                                 "nationally. The data provided here are curated and compiled " +
                                 "versions of " +
                                 "[data](https://data.cms.gov/provider-data/dataset/c7us-v4mf) " +
                                 "offered by the Centers for Medicare and Medicaid Services.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dcc.Dropdown(
                            id='p_and_v-year',
                            options=[{"label": i, "value": i} for i in ['2023', '2022', '2021', 
                                                                        '2020', '2019', '2018', 
                                                                        '2017', '2016', '2015']
                                     ],
                            multi=False, 
                            value=None,
                            placeholder='Choose a federal fiscal year',
                            style={'width': '96.2%',
                                   'margin-bottom': '10px',
                                 },
                            ),
                    dbc.Button("Load Payment and Value of Care dataset",
                               id='p_and_v',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
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
                    html.H5("Timely and Effective Care", 
                            style={'display': 'inline-block', 
                                   'margin-right': '10px',
                                   },
                            ),
                    dcc.Markdown("The measures of timely and effective care, also known as " +
                                 "process of care measures, show how often or how quickly " +
                                 "hospitals provide care that research shows gets the best " +
                                 "results for patients with certain conditions, and how " + 
                                 "hospitals use outpatient medical imaging tests (like CT " +
                                 "Scans and MRIs). The data provided here are curated and " +
                                 "compiled versions of " +
                                 "[data](https://data.cms.gov/provider-data/dataset/yv7e-xc69)" +
                                 "offered by the Centers for Medicare and Medicaid Services.",
                                 style={'width': '94.1%'},
                                 ),
                    dcc.Dropdown(
                            id='t_and_e-year',
                            options=[{"label": i, "value": i} for i in ['2023', '2022', '2021', 
                                                                        '2020', '2019', '2018', 
                                                                        '2017', '2016', '2015', 
                                                                        '2014',
                                                                        ]
                                     ],
                            multi=False, 
                            value=None,
                            placeholder='Choose a federal fiscal year',
                            style={'width': '96.2%',
                                   'margin-bottom': '10px',
                                 },
                            ),
                    dbc.Button("Load Timely and Effective Care dataset",
                               id='t_and_e',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
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
                    html.H5("Unplanned Visits", 
                            style={'display': 'inline-block', 
                                   'margin-right': '10px'},
                            ),
                    dcc.Markdown("This data set includes hospital-level data for the hospital " +
                                 "return days (or excess days in acute care [EDAC]) measures, " +
                                 "the unplanned readmissions measures, and measures of unplanned " +
                                 "hospital visits after outpatient procedures. The data provided " +
                                 "here are curated and compiled versions of " +
                                 "[data](https://data.cms.gov/provider-data/dataset/632h-zaca)" +
                                 "offered by the Centers for Medicare and Medicaid Services.",
                                 style={'width': '94.1%'},
                                 ),
                    dcc.Dropdown(
                            id='unplanned_visits-year',
                            options=[{"label": i, "value": i} for i in ['2023', '2022', '2021', 
                                                                        '2020', '2019', '2018']
                                     ],
                            multi=False, 
                            value=None,
                            placeholder='Choose a federal fiscal year',
                            style={'width': '96.2%',
                                   'margin-bottom': '10px',
                                 },
                            ),
                    dbc.Button("Load Unplanned Visits dataset",
                               id='unplanned_visits',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
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
                    html.H5("Outpatient Imaging Efficiency", 
                            style={'display': 'inline-block', 
                                   'margin-right': '10px',
                                   },
                            ),
                    dcc.Markdown("These measures give you information about hospitals' use of " +
                                 "medical imaging tests for outpatients. Examples of medical " + 
                                 "imaging tests include CT scans and MRIs. The data provided " +
                                 "here are curated and compiled versions of " +
                                 "[data](https://data.cms.gov/provider-data/dataset/632h-zaca) " +
                                 "offered by the Centers for Medicare and Medicaid Services.",
                                 style={'width': '94.1%'},
                                 ),
                    dcc.Dropdown(
                            id='imaging-year',
                            options=[{"label": i, "value": i} for i in ['2023', '2022', '2021', '2020',
                                                                        '2019', '2018', '2017', '2016', 
                                                                        '2015', '2014',
                                                                        ]
                                     ],
                            multi=False, 
                            value=None,
                            placeholder='Choose a federal fiscal year',
                            style={'width': '96.2%',
                                   'margin-bottom': '10px',
                                 },
                            ),
                    dbc.Button("Load Outpatient Imaging Efficiency dataset",
                               id='imaging',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
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
                    html.H5("Statsmodels Datasets", 
                            style={'display': 'inline-block', 
                                   'margin-right': '10px',
                                   },
                            ),
                    dcc.Markdown("The python-based Statsmodels library provides hundreds of data sets (data and meta-data) for use in examples, tutorials, model testing, etc. The Rush Regression Workbench makes 100 of these datasets and their documentation available. Each of these datasets is healthcare-based and contains 900 to >10K rows (observations).",
                                 style={'width': '94.1%'},
                                 ),
                    dbc.Button("Explore & Load Statsmodels Data",
                               id='open-statsmodels',
                               style={
                                   "background-color": "#2a8cff",
                                   'width': '92.5%',
                                   'font-size': 12,
                                   'display': 'inline-block',
                                   },
                        ),
                    dbc.Modal(
                        [dbc.ModalBody([
                            
                            html.Div(
                                    className="one columns",
                                    children=[
                                        html.H5("Expore & Load Statsmodels Healthcare Datasets", 
                                                style={'color': '#FFFFFF',
                                                       'font-size': 28,
                                                       },
                                                    ),
                                        html.Hr(),
                                        html.P("Choose one or more tags", 
                                                style={'color': '#FFFFFF',
                                                       'font-size': 20,
                                                       },
                                                    ),
                                        dcc.Dropdown(
                                            id='statsmodels_tags',
                                            options=[{"label": i, "value": i} for i in statsmodels_tags],
                                            multi=True, 
                                            value=statsmodels_tags,
                                            placeholder=None,
                                            style={'margin-bottom': '10px',
                                                   'width':'99%',
                                                 },
                                            ),
                                        html.Br(),
                                        
                                        html.H5("Data Table", 
                                                style={'display': 'inline-block', 
                                                       'color': '#FFFFFF',
                                                       'font-size': 28,
                                                       'margin-right': '5%',
                                                       'margin-left': '2%',
                                                       },
                                                ),
                                        
                                        dbc.Button("View MetaData",
                                                   id='open-statsmodels_data_doc',
                                                   style={
                                                       'display': 'inline-block', 
                                                       "background-color": "#2a8cff",
                                                       'font-size': 16,
                                                       'width':'25%',
                                                       'margin-right': '5%',
                                                       },
                                            ),
                                        dbc.Modal([
                                            
                                            dbc.ModalFooter(
                                                    dbc.Button("Click to Close", 
                                                               id='close-statsmodels_data_doc', 
                                                               className="ml-auto",
                                                               style={
                                                                   "background-color": "#2a8cff",
                                                                   'width': '30%',
                                                                   'font-size': 14,
                                                                   },
                                                               ),
                                                    style={
                                                        "background-color": "#989898",
                                                        "display": "flex",
                                                        "justify-content": "center",
                                                        "align-items": "center",
                                                        },
                                                    ),
                                            
                                            dbc.ModalBody([
                                                html.Div(
                                                    className="two columns",
                                                    children=[dcc.Markdown("Click on a dataset in the 'id' column of the datatable.", id='statsmodels_data_doc',
                                                                     ),
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
                                            
                                            ],
                                        id="modal-statsmodels_data_doc",
                                        is_open=False,
                                        centered=True,
                                        autoFocus=True,
                                        size='xl',
                                        #fullscreen=True,
                                        keyboard=True,
                                        fade=True,
                                        ),
                                        
                                        dbc.Button("Load Statsmodels Dataset",
                                                   id='load_statsmodels_dataset',
                                                   style={
                                                       'display': 'inline-block', 
                                                       "background-color": "#2a8cff",
                                                       'font-size': 16,
                                                       'width':'25%',
                                                       },
                                            ),
                                        statsmodels_data_table(),
                                        
                                        ],
                                    style={'width': '100%',
                                           "background-color": "#696969",
                                           #"display": "flex",
                                           "justify-content": "center",  # Center horizontally
                                           "align-items": "center",  # Center vertically)
                                    },
                                ),
                            
                            html.Br(), 
                            ],
                            style={'width': '100%',
                                   "background-color": "#696969",
                                   #"display": "flex",
                                   "justify-content": "center",  # Center horizontally
                                   "align-items": "center",  # Center vertically)
                            },
                            ),
                        dbc.ModalFooter(
                                dbc.Button("Close", 
                                           id="close-statsmodels", 
                                           className="ml-auto",
                                           style={"background-color": "#2a8cff",
                                                  'width': '30%',
                                                  'font-size': 14,
                                                  },
                                           ),
                                style={"background-color": "#696969",
                                    "display": "flex",
                                    "justify-content": "center",  # Center horizontally
                                    "align-items": "center",  # Center vertically)
                                    },
                                ),
                        ],
                        id="modal-statsmodels",
                        is_open=False,
                        centered=True,
                        autoFocus=True,
                        fullscreen=True,
                        keyboard=True,
                        fade=True,
                        backdrop=True,
                        
                        style={#"background-color": "#696969",
                               "display": "flex",
                               "justify-content": "center",  # Center horizontally
                               "align-items": "center",  # Center vertically)
                            },
                        ),
                    
                ],
                style={'display': 'inline-block', 
                       'width': '45%',
                       },
            ),
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

