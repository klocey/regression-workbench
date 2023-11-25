from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import dcc, html
import dash

import statsmodels.api as sm

import pandas as pd
import warnings
import json
import os

import quantile_regression
import binary_classify
import Single_InDepth
import control_cards
import DecisionTree
import app_fxns
import survival
import IMMR
import MLR
import GLM


####################################################################################################
#################################      CONFIG APP      #############################################
####################################################################################################


FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"
chriddyp = 'https://codepen.io/chriddyp/pen/bWLwgP.css'

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME, chriddyp]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
server = app.server

xvars = ['Nothing uploaded']
yvar = 'Nothing uploaded'


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

statsmodels_df = pd.read_csv('statsmodels_df.csv')
statsmodels_df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
statsmodels_df = statsmodels_df.rename(columns={'Dataset': 'id'})
statsmodels_df.set_index('id', inplace=True, drop=False)


####################################################################################################
##############################        DASH APP LAYOUT        #######################################
####################################################################################################


app.layout = html.Div([
    
    dcc.Store(id='main_df', storage_type='memory'),
    
    html.Div(
        id='reset',
        style={'display': 'none'}
        ),
    html.Div(
        id='cat_vars',
        style={'display': 'none'}
        ),
    html.Div(
        id='di_numerical_vars',
        style={'display': 'none'}
        ),
    
    html.Div(
            style={'background-color': '#f9f9f9'},
            id="banner1",
            className="banner",
            children=[html.Img(src=app.get_asset_url("RUSH_full_color.jpg"), 
                               style={'textAlign': 'left'},
                               ),
                      html.Img(src=app.get_asset_url("plotly_logo.png"), 
                               style={'textAlign': 'right'},
                               ),
                      ],
        ),
    
    html.Div(
            id="top-column1",
            className="ten columns",
            children=[control_cards.description_card1()],
            style={'width': '95.3%',
                   'display': 'inline-block',
                   'border-radius': '15px',
                   'box-shadow': '1px 1px 1px grey',
                   'background-color': '#f0f0f0',
                   'margin-bottom': '1%',
                   },
            ),
    
    html.Div(
            className="ten columns",
            children=[
                dbc.Carousel(
                    items=[
                        {"key": "1", "src": "/assets/images_for_ap/a1.png"},
                        {"key": "2", "src": "/assets/images_for_ap/a2.png"},
                        {"key": "3", "src": "/assets/images_for_ap/a3.png"},
                        {"key": "4", "src": "/assets/images_for_ap/a4.png"},
                        {"key": "5", "src": "/assets/images_for_ap/a5.png"},
                        {"key": "6", "src": "/assets/images_for_ap/a6.png"},
                        {"key": "7", "src": "/assets/images_for_ap/a7.png"},
                        {"key": "8", "src": "/assets/images_for_ap/a8.png"},
                    ],
                    controls=True,
                    indicators=True,
                    interval=5000,
                    ride="carousel",
                    style={"maxWidth": "100%"},
                    
                ),
                ],
            style={'width': '95%',
                   'display': 'block',
                   'margin-left': '2%',
                   'margin-right': '1%',
                   'margin-top': '1%',
                   'margin-bottom': '1%',
                   },
            ),
    
    html.Div(
            id="bottom-column1",
            className="ten columns",
            children=[control_cards.description_card_final()],
            style={'width': '95.3%',
                   'display': 'inline-block',
                   'border-radius': '15px',
                   'box-shadow': '1px 1px 1px grey',
                   'background-color': '#f0f0f0',
                   'padding': '1%',
            },
        ),

])


####################################################################################################
############################          Callbacks         ############################################
####################################################################################################

#############################      Modals      #####################################################

@app.callback(
    Output("modal-single_regression_residuals_plot", "is_open"),
    [Input("open-single_regression_residuals_plot", "n_clicks"), 
     Input("close-single_regression_residuals_plot", "n_clicks")],
    [State("modal-single_regression_residuals_plot", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-quant_regression_table", "is_open"),
    [Input("open-quant_regression_table", "n_clicks"), 
     Input("close-quant_regression_table", "n_clicks")],
    [State("modal-quant_regression_table", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_quant(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered_single", "is_open"),
    [Input("open-centered_single", "n_clicks"), 
     Input("close-centered_single", "n_clicks")],
    [State("modal-centered_single", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_single(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-iterative_multimodel_ols_table1", "is_open"),
    [Input("open-iterative_multimodel_ols_table1", "n_clicks"), 
     Input("close-iterative_multimodel_ols_table1", "n_clicks")],
    [State("modal-iterative_multimodel_ols_table1", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_c2(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-iterative_ols', "is_open"),
    [Input('open-iterative_ols', "n_clicks"), 
     Input('close-iterative_ols', "n_clicks")],
    [State('modal-iterative_ols', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_iterative_ols(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-single_ols', "is_open"),
    [Input('open-single_ols', "n_clicks"), 
     Input('close-single_ols', "n_clicks")],
    [State('modal-single_ols', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_single_ols(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-quant_reg', "is_open"),
    [Input('open-quant_reg', "n_clicks"), 
     Input('close-quant_reg', "n_clicks")],
    [State('modal-quant_reg', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_quant_reg(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-multi_reg', "is_open"),
    [Input('open-multi_reg', "n_clicks"), 
     Input('close-multi_reg', "n_clicks")],
    [State('modal-multi_reg', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_multi_reg(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-dec_tree_reg_table_perf', "is_open"),
    [Input('open-dec_tree_reg_table_perf', "n_clicks"), 
     Input('close-dec_tree_reg_table_perf', "n_clicks")],
    [State('modal-dec_tree_reg_table_perf', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_dec_tree_reg_table_perf(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-dec_tree_reg_table_params', "is_open"),
    [Input('open-dec_tree_reg_table_params', "n_clicks"), 
     Input('close-dec_tree_reg_table_params', "n_clicks")],
    [State('modal-dec_tree_reg_table_params', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_dec_tree_reg_table_params(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-logistic_reg', "is_open"),
    [Input('open-logistic_reg', "n_clicks"), 
     Input('close-logistic_reg', "n_clicks")],
    [State('modal-logistic_reg', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_logistic_reg(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-glm', "is_open"),
    [Input('open-glm', "n_clicks"), 
     Input('close-glm', "n_clicks")],
    [State('modal-glm', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_glm(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered-controlcard", "is_open"),
    [Input("open-centered-controlcard", "n_clicks"), 
     Input('main_df', 'data'),
     ],
    [State("modal-centered-controlcard", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_cc(n1, df, is_open):
    
    ctx1 = dash.callback_context
    jd1 = json.dumps({'triggered': ctx1.triggered,})
    jd1 = jd1[:50]
    
    if n1:# or df == 1:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered-controlcard_load", "is_open"),
    [Input("open-centered-controlcard_load", "n_clicks"), 
     Input("close-centered-controlcard_load", "n_clicks"),
     Input('main_df', 'data')],
    [State("modal-centered-controlcard_load", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_ccl(n1, n2, df, is_open):
    ctx1 = dash.callback_context
    jd1 = json.dumps({'triggered': ctx1.triggered,})
    jd1 = jd1[:50]
    
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-choose_regression", "is_open"),
    [Input("open-choose_regression", "n_clicks"), 
     Input("close-choose_regression", "n_clicks")],
    [State("modal-choose_regression", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_cc_choose_reg(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-inspect_main_datatable", "is_open"),
    [Input("open-inspect_main_datatable", "n_clicks"), 
     Input("close-inspect_main_datatable", "n_clicks")],
    [State("modal-inspect_main_datatable", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_inspect_main_datatable(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-level-collapse", "is_open"),
    [Input("open-level-collapse", "n_clicks"), 
     Input("close-level-collapse", "n_clicks")],
    [State("modal-level-collapse", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_level_collapse(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered-controlcard1", "is_open"),
    [Input("open-centered-controlcard1", "n_clicks"), 
     Input("close-centered-controlcard1", "n_clicks")],
    [State("modal-centered-controlcard1", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_cc1(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered3", "is_open"),
    [Input("open-centered3", "n_clicks"), 
     Input("close-centered3", "n_clicks")],
    [State("modal-centered3", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_c3(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-glm_parameters_table", "is_open"),
    [Input("open-glm_parameters_table", "n_clicks"), 
     Input("close-glm_parameters_table", "n_clicks")],
    [State("modal-glm_parameters_table", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_glm_params_table(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-glm_performance_table", "is_open"),
    [Input("open-glm_performance_table", "n_clicks"), 
     Input("close-glm_performance_table", "n_clicks")],
    [State("modal-glm_performance_table", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_glm_performance_table(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered4", "is_open"),
    [Input("open-centered4", "n_clicks"), 
     Input("close-centered4", "n_clicks")],
    [State("modal-centered4", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_c4(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered5", "is_open"),
    [Input("open-centered5", "n_clicks"), 
     Input("close-centered5", "n_clicks")],
    [State("modal-centered5", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_c5(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal-centered6", "is_open"),
    [Input("open-centered6", "n_clicks"), 
     Input("close-centered6", "n_clicks")],
    [State("modal-centered6", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_c6(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-survival', "is_open"),
    [Input('survival_reg_btn', "n_clicks"), 
     Input('close-survival', "n_clicks")],
    [State('modal-survival', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_survival(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-decision_tree', "is_open"),
    [Input('open-decision_tree', "n_clicks"), 
     Input('close-decision_tree', "n_clicks")],
    [State('modal-decision_tree', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_decision_tree(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-survival_params_table', "is_open"),
    [Input('open-survival_params_table', "n_clicks"), 
     Input('close-survival_params_table', "n_clicks")],
    [State('modal-survival_params_table', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_survival_params_table(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-survival_performance_table', "is_open"),
    [Input('open-survival_performance_table', "n_clicks"), 
     Input('close-survival_performance_table', "n_clicks")],
    [State('modal-survival_performance_table', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_survival_performance_table(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-statsmodels', "is_open"),
    [Input('open-statsmodels', "n_clicks"), 
     Input('close-statsmodels', "n_clicks")],
    [State('modal-statsmodels', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_statsmodals(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-statsmodels_data_doc', "is_open"),
    [Input('open-statsmodels_data_doc', "n_clicks"), 
     Input('close-statsmodels_data_doc', "n_clicks")],
    [State('modal-statsmodels_data_doc', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_statsmodals_data_doc(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-lifelines', "is_open"),
    [Input('open-lifelines', "n_clicks"), 
     Input('close-lifelines', "n_clicks")],
    [State('modal-lifelines', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_lifelines(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


####################################################################################################
###############  Buttons, Main DataFrames, Main DataTable   ########################################
####################################################################################################


@app.callback(Output('statsmodels_data_doc', 'children'),
              [Input('statsmodels_data_table', 'active_cell'),
               Input('statsmodels_data_table', 'data'),
              ],
              )
def update_statmodels_data_doc(row, df):
    
    if row is None:
        raise PreventUpdate
    
    df = pd.DataFrame(df)
    if df.empty:
        return 'Your data table is empty'
    
    dataset = row['row_id']
    doc = statsmodels_df[statsmodels_df['id'] == dataset]['docs'].iloc[0]
    
    return doc
    


@app.callback([Output('main_df', 'data'),
               Output('rt4', 'children')],
              [Input('upload-data', 'contents'),
               Input('hcris', 'n_clicks'),
               Input('hais', 'n_clicks'),
               Input('hacrp', 'n_clicks'),
               Input('hrrp', 'n_clicks'),
               Input('c_and_d', 'n_clicks'),
               Input('p_and_v', 'n_clicks'),
               Input('t_and_e', 'n_clicks'),
               Input('unplanned_visits', 'n_clicks'),
               Input('imaging', 'n_clicks'),
               Input('load_statsmodels_dataset', 'n_clicks'),
               ],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified'),
               State('hcris-year', 'value'),
               State('hais-year', 'value'),
               State('hacrp-year', 'value'),
               State('hrrp-year', 'value'),
               State('c_and_d-year', 'value'),
               State('p_and_v-year', 'value'),
               State('t_and_e-year', 'value'),
               State('unplanned_visits-year', 'value'),
               State('imaging-year', 'value'),
               State('statsmodels_data_table', 'active_cell'),
               ], 
            )
def update_main_DataFrame(list_of_contents, hcris, hais, hacrp, hrrp, c_and_d, p_and_v, 
                          t_and_e, unplanned_visits, imaging, statsmodels_data, 
                          file_name, list_of_dates, hcris_yr, hais_yr, hacrp_yr, hrrp_yr, 
                          c_and_d_yr, p_and_v_yr, t_and_e_yr, unplanned_visits_yr, 
                          imaging_yr, statsmodels_row):

    ctx1 = dash.callback_context
    jd1 = json.dumps({'triggered': ctx1.triggered,})
    jd1 = jd1[:50]
    
    labs = ['hcris', 'hais', 'hacrp', 'hrrp', 'c_and_d', 'p_and_v', 
            't_and_e', 'unplanned_visits', 'imaging', 'load_statsmodels_data']
    
    yrs = [hcris_yr, hais_yr, hacrp_yr, hrrp_yr, c_and_d_yr, p_and_v_yr, 
           t_and_e_yr, unplanned_visits_yr, imaging_yr]
    
    urls = ['https://github.com/klocey/HCRIS-databuilder/raw/master/filtered_datasets/HCRIS_filtered_',
            'https://github.com/klocey/hospitals-data-archive/raw/main/dataframes/filtered_files/HAIs/HAIs_',
            'https://github.com/klocey/hospitals-data-archive/raw/main/dataframes/filtered_files/Hospital_Acquired_Conditions_Reduction_Program/Hospital_Acquired_Conditions_Reduction_Program_',
            'https://github.com/klocey/hospitals-data-archive/raw/main/dataframes/filtered_files/Hospital_Readmissions_Reduction_Program/Hospital_Readmissions_Reduction_Program_',
            'https://github.com/klocey/hospitals-data-archive/raw/main/dataframes/filtered_files/Complications_and_Deaths/Complications_and_Deaths_',
            'https://github.com/klocey/hospitals-data-archive/raw/main/dataframes/filtered_files/Payment_and_Value_of_Care/Payment_and_Value_of_Care_',
            'https://github.com/klocey/hospitals-data-archive/raw/main/dataframes/filtered_files/Timely_and_Effective_Care/Timely_and_Effective_Care_',
            'https://github.com/klocey/hospitals-data-archive/raw/main/dataframes/filtered_files/Unplanned_Visits/Unplanned_Visits_',
            'https://github.com/klocey/hospitals-data-archive/raw/main/dataframes/filtered_files/Outpatient_Imaging_Efficiency/Outpatient_Imaging_Efficiency_',
            ]
    
    select_preprocessed = 'n'
    for i, lab in enumerate(labs):
        if lab in jd1:
            select_preprocessed = 'y'
            
            if lab == 'load_statsmodels_data':
                dataset = statsmodels_row['row_id']
                package = statsmodels_df[statsmodels_df['id'] == dataset]['package'].iloc[0]
                item	 = statsmodels_df[statsmodels_df['id'] == dataset]['item'].iloc[0]
                df = sm.datasets.get_rdataset(item, package).data
                
            else:
                try:
                    url = urls[i] + yrs[i] + '.csv'
                    df = pd.read_csv(url)
                except:
                    raise PreventUpdate
            break
    
    if select_preprocessed == 'n':
        
        if list_of_contents is None or file_name is None or list_of_dates is None: 
            # uploaded file contains nothing
            return None, ""
        
        elif file_name[-4:] != '.csv': 
            # uploaded file does not have the .csv extension
            error_string = "Error: This application only accepts the universally useful "
            error_string += "CSV file type. Ensure that your file has the .csv extension " 
            error_string += "and is correctly formatted."
            return None, error_string
        
        elif list_of_contents is not None:
            error_string = "Error: Your .csv file was not processed. "
            error_string += "Ensure there are only rows, columns, and one row of column headers. "
            error_string += "Make sure your file contains enough data to analyze."
            children = 0
            df = 0
            
            # Attempt to parse the content
            try: 
                children = [app_fxns.parse_contents(c, n, d) for c, n, d in zip([list_of_contents], 
                                                                       [file_name], 
                                                                       [list_of_dates])]
            except: 
                return None, error_string
            
            # Attempt to assign contents to an object
            try: 
                df = children[0]
            except: 
                return None, error_string
             
            # Attempt to read the object as a pandas dataframe
            try: 
                df = pd.read_json(df)
            except: 
                return None, error_string
    
    # Check for variables named 'Unnamed' and removed them
    var_ls = list(df)
    ls1 = []
    for i in var_ls:
        if 'Unnamed' in i:
            ls1.append(i)
    df.drop(labels=ls1, axis=1, inplace=True)
    del ls1, var_ls
    
    # Check for whether the dataframe contains a trivial amount of data
    try:
        if df.shape[0] < 4 or df.shape[1] < 2: 
            error_string = "Error: Your .csv file was not processed. "
            error_string += "Ensure there are only rows, columns, and one row of column headers. "
            error_string += "Make sure your file contains enough data to analyze."
            return None, error_string
    except:
        error_string = "Error: Your .csv file was not processed. "
        error_string += "Ensure there are only rows, columns, and one row of column headers. "
        error_string += "Make sure your file contains enough data to analyze."
        return None, error_string
    
    df.columns = df.columns.str.strip() # remove leading and trailing whitespaces
    #df.columns = df.columns.str.replace(":", " ") # replace colons with white spaces
    df = df.replace(',',' ', regex=True) # replace commas with white spaces
    df = df.replace({None: 'None'}) # replace None objects with string objects of 'None'
    df = df.replace({'?': 0}) # replace question marks with 0 integer values
    df.dropna(how='all', axis=1, inplace=True) # drop all columns having no data
    df.dropna(how='all', axis=0, inplace=True) # drop all rows having no data
    
    # If the dataframe contains >5000 rows or >50 columns, sample at random to meet those constraints
    if os.environ.get('DEPLOYMENT_ENV', 'local') != 'local':  
        if df.shape[0] > 10000: df = df.sample(n = 10000, 
                                               axis=0, replace=False, random_state=0)
        if df.shape[1] > 50: df = df.sample(n = 50, 
                                            axis=1, replace=False, random_state=0)
    
    df.dropna(how='all', axis=1, inplace=True) # drop all columns having no data
    df.dropna(how='all', axis=0, inplace=True) # drop all rows having no data
    df = df.loc[:, df.nunique() != 1] # drop all columns containing only one unique value
    
    variables = list(df)
    
    ############################################################################################
        
    # Attempt to detect datetime features based on their label.
    # This is done because python's datetime library can easily convert numeric data to datetime
    # objects (meaning it's no use to ask whether a feature can be converted to datetime).
    datetime_ls1 = [' date ', ' DATE ', ' Date ', ' date', ' DATE', ' Date', '_date_', '_DATE_', 
                    '_Date_', '_date', '_DATE', '_Date', ',date', ',DATE', ',Date', ';date', 
                    ';DATE', ';Date', '-date', '-DATE', '-Date', ':date', ':DATE', ':Date']
    datetime_ls2 = ['date', 'DATE', 'Date'] 
    
       
    for i in variables:            
        if i in datetime_ls2:
            df.drop(labels = [i], axis=1, inplace=True)
            continue
            
        else:
            for j in datetime_ls1:
                if j in i:
                    df.drop(labels = [i], axis=1, inplace=True)
                    break
        
    ############################################################################################
    ############################################################################################
        
    # A final check to dump any row containing no data
    df.dropna(how='all', axis=0, inplace=True)
    return df.to_json(), ""



@app.callback(Output('collapse_text', 'style'),
              [Input('collapse-interval-component', 'n_intervals')]
              )
def hide_text(n):
    if n >= 2:
        return {'display': 'none'}
    else:
        return {'textAlign': 'left',
                'margin-left': '3%',
                'color': '#ffffff'}
    
    
@app.callback(Output('level_vars', 'options'),
              [Input('cat_var_collapse', 'value')],
              [State('data_table', 'data')],
              )
def update_level_vars_options(cat_var, data):
    
    if data is None:
        return [{"label": i, "value": i} for i in []]
        
    elif cat_var is None:
        data = pd.DataFrame(data)
        ls = [{"label": i, "value": i} for i in []]
        return ls
    else:
        data = pd.DataFrame(data)
        ls = sorted(data[cat_var].unique().tolist())
        ls = [{"label": i, "value": i} for i in ls]
        return ls


@app.callback([Output('cat_var_collapse', 'options'),
               Output('cat_var_collapse', 'value'),
               ],
              [Input("open-level-collapse", 'n_clicks'),
               Input('collapse_text', 'children'),
               ],
              [State('cat_vars', 'children'),
               State('data_table', 'selected_columns'),
               ],
            )
def update_cat_vars_for_collapse(n_clicks, trigger, cat_vars, selected_cols):
    
    if selected_cols is None or cat_vars is None:
        raise PreventUpdate
        
    else:
        cat_vars = sorted([element for element in cat_vars if element in selected_cols])
        
        if cat_vars is None or cat_vars == []:
            return [], None
        
        options = [{"label": i, "value": i} for i in cat_vars]
        return options, cat_vars[0]


@app.callback([Output('statsmodels_data_table', 'data'),
               Output('statsmodels_data_table', 'columns'),
               ],
              [Input('statsmodels_tags', 'value'),
               ]
              )
def update_statsmodels_data_table(tags):
    
    if tags is None:
        raise PreventUpdate
    
    f_df = statsmodels_df.drop(labels=['docs', 'package', 'item'], axis=1)
    f_df = f_df[f_df['id'].str.contains('|'.join(tags), case=True)]
    
    data = f_df.to_dict('records')
    
    columns = [{'id': c, 
                'name': c, 
                } for c in f_df.columns]
    
    return data, columns


@app.callback([Output('data_table', 'data'),
               Output('data_table', 'columns'),
               Output('data_table', 'style_table'),
               Output('Data-Table1', 'style'),
               Output('cat_vars', 'children'),
               Output('di_numerical_vars', 'children'),
               Output('collapse_text', 'children'),
               Output('collapse-interval-component', 'n_intervals'),
               Output('data_table', 'selected_columns'),
               ],
              [Input('main_df', 'data'),
               Input('level-collapse-btn1', 'n_clicks'),
               Input('data_table', 'columns'),
               Input('data_table', 'selected_columns'),
               ],
              [State('rt4', 'children'),
               State('data_table', 'data'),
               State('cat_var_collapse', 'value'),
               State('level_vars', 'value'),
               State('new_level_name', 'value'),
               ],
            )
def update_main_DataTable(df, btn, Dcols, selected_cols, rt4, data, cat_var, level_vars, new_name):
    
    if df is None:
        raise PreventUpdate
        
    else:
        ctx1 = dash.callback_context
        jd1 = json.dumps({'triggered': ctx1.triggered,})
        jd1 = jd1[:50]
        out_text = ""
        
        if 'level-collapse-btn1' not in jd1:
            df = pd.read_json(df)
            if df.empty:
                raise PreventUpdate
        
        ############################################################################################
        ################# COLLAPSE SELECTED LEVELS FOR SELECTED CATEGORICAL VARIABLE ###############
        
        elif 'level-collapse-btn1' in jd1:
        
            if cat_var is None or level_vars is None:
                raise PreventUpdate
            else:
                df = pd.DataFrame(data)
                
                ls = df[cat_var].tolist()
                ls = [new_name if i in level_vars else i for i in ls]
                df[cat_var] = ls
                out_text = "Level collapse complete! Continue collapsing levels or click the button below to close this window."
        
        
        ############################################################################################
        ################# UPDATE CHANGES IN SELECTED COLUMNS AND VARIABLE NAMES ####################
        
        if 'data_table' in jd1:
            names = [column['name'] for column in Dcols]
            ids = [column['id'] for column in Dcols]
            
            df = pd.DataFrame(data)
            df = df.filter(items=names, axis=1)
            
            if selected_cols is None:
                pass
            elif len(selected_cols) > 0:
                s_cols2 = []
                for col in selected_cols:
                    i = ids.index(col)
                    new_name = names[i]
                    s_cols2.append(new_name)
                    
                selected_cols = list(s_cols2)
            
            df.columns = names
            
        ############################################################################################
        ########## DETECT VARIABLES THAT ARE NUMERIC, CATEGORICAL, OR POTENTIALLY BOTH #############
        
        ct, cat_vars, dichotomous_numerical_vars = 1, [], []
            
        variables = list(df)
        for i in variables:
            if 'Unnamed' in i:
                new_lab = 'Unnamed ' + str(ct)
                df.rename(columns={i: new_lab}, inplace=True)
                i = new_lab
                ct += 1
            
            # 1. Convert df[i] to numeric and coerce all non-numbers to np.nan 
            df['temp'] = pd.to_numeric(df[i], errors='coerce')
            # 2. Replace all the np.nan's in df['temp'] with values in df[i]
            df['temp'].fillna(df[i], inplace=True)
            # 3. Replace df[i] with df['temp']
            df[i] = df['temp'].copy(deep=True)
            # 4. Drop df['temp']
            df.drop(labels=['temp'], axis=1, inplace=True)
            
            ls = df[i].unique()
                
            if all(isinstance(item, str) for item in ls) is True:
                # if all items are strings, then call the feature categorical
                cat_vars.append(i)
                
            else:
                # else call the feature numeric
                df[i] = pd.to_numeric(df[i], errors='coerce')
                
            if len(ls) == 2 and all(isinstance(item, str) for item in ls) is False:
                dichotomous_numerical_vars.append(i)
        
        
        data = df.to_dict('records')
        
        columns = [{'id': c, 
                    'name': c, 
                    'deletable': True, 
                    'renamable': True, 
                    'selectable': True} for c in df.columns]
        
        style_table={'overflowX': 'auto', 
                     'overflowY': 'auto',
                     }
        
        style = {'width': '100%',
                 'height': '50%', 
                 "display": "block",
                 }
        
        cat_vars = sorted(cat_vars)
        
        if selected_cols is None or 'main_df.data' in jd1:
            selected_cols = list(df)
        
        if 'level-collapse-btn1' in jd1:
            return [data, columns, style_table, style, cat_vars, 
                    dichotomous_numerical_vars, out_text, 0, selected_cols]
        
        return [data, columns, style_table, style, cat_vars, 
                    dichotomous_numerical_vars, out_text, 0, selected_cols]




####################################################################################################
###################  Update Variables For Models  ##################################################
####################################################################################################



@app.callback([Output('xvar', 'options'),
               Output('xvar', 'value'),
               Output('yvar', 'options'),
               Output('yvar', 'value')],
              [Input('data_table', 'data'), 
               Input('data_table', 'selected_columns'),
               Input('cat_vars', 'children')],
            )
def update_variables_for_iterative_multimodel_ols(df, selected_cols, cat_vars):
    try:
        df = pd.DataFrame(df)
        if df is None or df.empty:
            return [[{"label": 'Nothing loaded', "value": 'Nothing loaded'}], 
                    ['Nothing loaded'], [{"label": 'Nothing loaded', "value": 'Nothing loaded'}], 
                    ['Nothing loaded']]
        
        for l in cat_vars:
            try:
                df.drop(labels=[l], axis=1, inplace=True)
            except:
                pass
        df = df.filter(items=selected_cols, axis=1)
        
        drop_vars = []
        for f in list(df):
            if len(df[f].unique()) < 4:
                drop_vars.append(f)
        df.drop(labels=drop_vars, axis=1, inplace=True)
        
        ls1 = sorted(list(set(list(df))))
        ls2 = list(ls1)
        options = [{"label": i, "value": i} for i in ls1]
        
        if len(ls1) > 4:
            ls1 = ls1[:2]
            ls2 = ls2[-2:]
        
        return options, ls1, options, ls2
    
    except:
        return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded']]


@app.callback([Output('xvar2', 'options'),
               Output('xvar2', 'value'),
               Output('xvar2', 'optionHeight'),
               Output('yvar2', 'options'),
               Output('yvar2', 'value'),
               Output('yvar2', 'optionHeight'),],
              [Input('data_table', 'data'),
               Input('data_table', 'selected_columns'),
               Input('cat_vars', 'children')],
            )
def update_variables_for_in_depth_single_OLS(df, selected_cols, cat_vars):
    optionHeight = 30
    try:
        df = pd.DataFrame(df)
        if df is None or df.empty:
            return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight, 
                    [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight]
        
        for l in cat_vars:
            try:
                df.drop(labels=[l], axis=1, inplace=True)
            except:
                pass
            
        df = df.filter(items=selected_cols, axis=1)
        
        drop_vars = []
        for f in list(df):
            if len(df[f].unique()) < 4:
                drop_vars.append(f)
        df.drop(labels=drop_vars, axis=1, inplace=True)
        
        ls = sorted(list(set(list(df))))
        
        lens = []
        for l in ls:
            lens.append(len(l))
        maxl = max(lens)
        
        if maxl < 40:
            optionHeight = 50
        elif maxl < 50:
            optionHeight = 60
        elif maxl < 60:
            optionHeight = 70
        elif maxl < 80:
            optionHeight = 90
        elif maxl < 100:
            optionHeight = 110
        else:
            optionHeight = 140
            
        options = [{"label": i, "value": i} for i in ls]
        return options, ls, optionHeight, options, ls, optionHeight
    
    except:
        return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight, 
                [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight]
        

@app.callback([Output('xvar2_quant', 'options'),
               Output('xvar2_quant', 'value'),
               Output('xvar2_quant', 'optionHeight'),
               Output('yvar2_quant', 'options'),
               Output('yvar2_quant', 'value'),
               Output('yvar2_quant', 'optionHeight')],
              [Input('data_table', 'data'),
               Input('data_table', 'selected_columns'),
               Input('cat_vars', 'children')],
            )
def update_variables_for_quantile_regression(df, selected_cols, cat_vars):
    optionHeight = 30
    try:
        df = pd.DataFrame(df)
        if df is None or df.empty:
            return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight, 
                    [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight]
        
        for l in cat_vars:
            try:
                df.drop(labels=[l], axis=1, inplace=True)
            except:
                pass
        df = df.filter(items=selected_cols, axis=1)
        
        drop_vars = []
        for f in list(df):
            if len(df[f].unique()) < 4:
                drop_vars.append(f)
        df.drop(labels=drop_vars, axis=1, inplace=True)
        
        ls = sorted(list(set(list(df))))
        
        lens = []
        for l in ls:
            lens.append(len(l))
        maxl = max(lens)
        
        if maxl < 40:
            optionHeight = 50
        elif maxl < 50:
            optionHeight = 60
        elif maxl < 60:
            optionHeight = 70
        elif maxl < 80:
            optionHeight = 90
        elif maxl < 100:
            optionHeight = 110
        else:
            optionHeight = 140
            
        options = [{"label": i, "value": i} for i in ls]
        return options, ls, optionHeight, options, ls, optionHeight
    
    except:
        return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight, 
                [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight]


@app.callback([Output('xvar3', 'options'),
               Output('xvar3', 'value'),
               Output('yvar3', 'options'),
               Output('yvar3', 'value'),
               Output('yvar3', 'optionHeight')],
              [Input('data_table', 'data'),
               Input('data_table', 'selected_columns'),
               Input('cat_vars', 'children')],
            )
def update_variables_for_multiple_linear_regression(df, selected_cols, cat_vars):
    optionHeight = 30
    try:
        df = pd.DataFrame(df)
        if df is None or df.empty:
            return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], 
                    [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight]
        
        df = df.filter(items=selected_cols, axis=1)
        ls1 = sorted(list(set(list(df))))
        options1 = [{"label": i, "value": i} for i in ls1]
        
        for l in cat_vars:
            try:
                df.drop(labels=[l], axis=1, inplace=True)
            except:
                pass
        
        drop_vars = []
        for f in list(df):
            if len(df[f].unique()) < 4:
                drop_vars.append(f)
        df.drop(labels=drop_vars, axis=1, inplace=True)
        
        ls2 = sorted(list(set(list(df))))
        lens = []
        for l in ls2:
            lens.append(len(l))
        maxl = max(lens)
        
        if maxl < 40:
            optionHeight = 50
        elif maxl < 50:
            optionHeight = 60
        elif maxl < 60:
            optionHeight = 70
        elif maxl < 80:
            optionHeight = 80
        elif maxl < 100:
            optionHeight = 100
        else:
            optionHeight = 120
        options2 = [{"label": i, "value": i} for i in ls2]
        
        return options1, ls1, options2, ls2, optionHeight
        
    except:
        return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], 
                [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight]


@app.callback([Output('glm_predictors', 'options'),
               Output('glm_predictors', 'value'),
               Output('glm_response_var', 'options'),
               Output('glm_response_var', 'value'),
               Output('glm_response_var', 'optionHeight')],
              [Input('data_table', 'data'),
               Input('data_table', 'selected_columns'),
               Input('cat_vars', 'children')],
            )
def update_variables_for_glm(df, selected_cols, cat_vars):
    optionHeight = 30
    try:
        df = pd.DataFrame(df)
        if df is None or df.empty:
            return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], 
                    [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight]
        
        df = df.filter(items=selected_cols, axis=1)
        ls1 = sorted(list(set(list(df))))
        options1 = [{"label": i, "value": i} for i in ls1]
        
        for l in cat_vars:
            try:
                df.drop(labels=[l], axis=1, inplace=True)
            except:
                pass
        
        drop_vars = []
        for f in list(df):
            if len(df[f].unique()) < 4:
                drop_vars.append(f)
        df.drop(labels=drop_vars, axis=1, inplace=True)
        
        ls2 = sorted(list(set(list(df))))
        lens = []
        for l in ls2:
            lens.append(len(l))
        maxl = max(lens)
        
        if maxl < 40:
            optionHeight = 50
        elif maxl < 50:
            optionHeight = 60
        elif maxl < 60:
            optionHeight = 70
        elif maxl < 80:
            optionHeight = 80
        elif maxl < 100:
            optionHeight = 100
        else:
            optionHeight = 120
        options2 = [{"label": i, "value": i} for i in ls2]
        
        return options1, ls1, options2, ls2, optionHeight
        
    except:
        return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], 
                [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight]

        
@app.callback([Output('xvar_logistic', 'options'),
                Output('xvar_logistic', 'value'),
                Output('yvar_logistic', 'options'),
                Output('yvar_logistic', 'value'),
                Output('yvar_logistic', 'optionHeight')],
                [Input('data_table', 'data'),
                 Input('data_table', 'selected_columns'),
                Input('cat_vars', 'children'),
                Input('di_numerical_vars', 'children')],
            )
def update_variables_for_logistic_regression(df, selected_cols, cat_vars, di_num_vars):
    optionHeight = 30
    try:
        df = pd.DataFrame(df)
        if df is None or df.empty:
            return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], 
                    [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight]
        
        df = df.filter(items=selected_cols, axis=1)
        tdf = df.copy(deep=True)
        
        ls1 = sorted(list(set(list(tdf))))
        options1 = [{"label": i, "value": i} for i in ls1]
        
        tdf = df.filter(items=cat_vars + di_num_vars, axis=1)
        tdf, dropped, cat_vars_ls = app_fxns.dummify(tdf, cat_vars, False)
        ls2 = sorted(list(set(list(tdf))))
        
        lens = []
        for l in ls2:
            lens.append(len(l))
        maxl = max(lens)
        
        if maxl < 40:
            optionHeight = 30
        elif maxl < 50:
            optionHeight = 30
        elif maxl < 60:
            optionHeight = 40
        elif maxl < 80:
            optionHeight = 50
        elif maxl < 100:
            optionHeight = 60
        else:
            optionHeight = 80
            
        options2 = [{"label": i, "value": i} for i in ls2]
                
        return options1, ls1, options2, ls2, optionHeight

    except:
        return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], 
                [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight]
  

@app.callback([Output('survival_predictors', 'options'),
               Output('survival_predictors', 'value'),
               Output('survival_partial', 'options'),
               Output('survival_partial', 'value'),
               Output('survival_partial', 'optionHeight'),
               Output('survival_d_var', 'options'),
               Output('survival_d_var', 'value'),
               Output('survival_d_var', 'optionHeight'),
               Output('survival_e_var', 'options'),
               Output('survival_e_var', 'value'),
               Output('survival_e_var', 'optionHeight')],
              [Input('data_table', 'data'),
               Input('data_table', 'selected_columns'),
               Input('cat_vars', 'children'),
               Input('di_numerical_vars', 'children')],
            )
def update_variables_for_survival_regression(df, selected_cols, cat_vars, di_num_vars):
    optionHeight = 30
    try:
        df = pd.DataFrame(df)
        if df is None or df.empty:
            return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], 
                    [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight,
                    [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight,
                    [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight,
                    ]
        
        df = df.filter(items=selected_cols, axis=1)
        tdf = df.copy(deep=True)
        
        ls1 = sorted(list(set(list(tdf))))
        options1 = [{"label": i, "value": i} for i in ls1]
        
        tdf = df.filter(items=cat_vars + di_num_vars, axis=1)
        tdf, dropped, cat_vars_ls = app_fxns.dummify(tdf, cat_vars, False)
        ls2 = sorted(list(set(list(tdf))))
        
        nonD_vars = list(set(cat_vars + di_num_vars))
        ls3 = [element for element in ls1 if element not in nonD_vars]
        options3 = [{"label": i, "value": i} for i in ls3]
        
        lens = []
        for l in ls2:
            lens.append(len(l))
        maxl = max(lens)
        
        if maxl < 40:
            optionHeight = 30
        elif maxl < 50:
            optionHeight = 30
        elif maxl < 60:
            optionHeight = 40
        elif maxl < 80:
            optionHeight = 50
        elif maxl < 100:
            optionHeight = 60
        else:
            optionHeight = 80
            
        options2 = [{"label": i, "value": i} for i in ls2]
        
        
        return [options1, ls1, 
                options1, ls1, optionHeight, 
                options3, ls3, optionHeight, 
                options2, ls2, optionHeight]

    except:
        return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], 
                [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight,
                [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight,
                [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight,
                ]
    

@app.callback([Output('xvar_dec_tree_reg', 'options'),
               Output('xvar_dec_tree_reg', 'value'),
               Output('yvar_dec_tree_reg', 'options'),
               Output('yvar_dec_tree_reg', 'value'),
               Output('yvar_dec_tree_reg', 'optionHeight')],
              [Input('data_table', 'data'),
               Input('data_table', 'selected_columns'),
               Input('cat_vars', 'children')],
            )
def update_variables_for_decision_tree_regression(df, selected_cols, cat_vars):
    optionHeight = 30
    try:
        df = pd.DataFrame(df)
        if df is None or df.empty:
            return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], 
                    [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight]
        
        df = df.filter(items=selected_cols, axis=1)
        ls1 = sorted(list(set(list(df))))
        options1 = [{"label": i, "value": i} for i in ls1]
        
        for l in cat_vars:
            try:
                df.drop(labels=[l], axis=1, inplace=True)
            except:
                pass
        
        drop_vars = []
        for f in list(df):
            if len(df[f].unique()) < 4:
                drop_vars.append(f)
        df.drop(labels=drop_vars, axis=1, inplace=True)
        
        ls2 = sorted(list(set(list(df))))
        lens = []
        for l in ls2:
            lens.append(len(l))
        maxl = max(lens)
        
        if maxl < 40:
            optionHeight = 50
        elif maxl < 50:
            optionHeight = 60
        elif maxl < 60:
            optionHeight = 70
        elif maxl < 80:
            optionHeight = 80
        elif maxl < 100:
            optionHeight = 100
        else:
            optionHeight = 120
        options2 = [{"label": i, "value": i} for i in ls2]
        
        return options1, ls1, options2, ls2, optionHeight
        
    except:
        return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], 
                [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight]
    

####################################################################################################
######################      Update Models      #####################################################
####################################################################################################


@app.callback([Output('figure_plot1', 'figure'),
               Output('table_plot1', 'children'),
               Output('rt0', 'children'),
               Output('fig1txt', 'children'),
               Output('table1txt', 'children'),
               ],
               [Input('btn1', 'n_clicks'),
                Input('btn_ss', 'n_clicks'),
                Input('btn_robust', 'n_clicks'),
                Input('rt4', 'children')],
               [State('data_table', 'data'),
                State('cat_vars', 'children'),
                State('xvar', 'value'),
                State('yvar', 'value')],
            )
def update_simple_regressions(n_clicks, smartscale, robust, reset, df, cat_vars, xvars, yvars):
    
    return IMMR.get_updated_results(n_clicks, smartscale, robust, df, cat_vars, xvars, yvars)

    
@app.callback([Output('figure_plot2', 'figure'),
                Output('rt3', 'children'),
                Output('fig2txt', 'children'),
                Output('residuals_plot1', 'figure'),
                Output('btn_robust2', 'n_clicks'),
                Output('single_table_txt', 'children'),
                Output('single_table_1', 'children'),
                Output('single_table_2', 'children'),],
                [Input('btn2', 'n_clicks'),
                 Input('btn_robust2', 'n_clicks'),
                 Input('rt4', 'children')],
                [State('xvar2', 'value'),
                 State('yvar2', 'value'),
                 State('x_transform', 'value'),
                 State('y_transform', 'value'),
                 State('model2', 'value'),
                 State('data_table', 'data')],
            )
def update_single_regression(n_clicks, robust, reset, xvar, yvar, x_transform, 
                             y_transform, model, df):
        
    return Single_InDepth.get_updated_results(n_clicks, robust, xvar, yvar, x_transform, 
                                              y_transform, model, df)


@app.callback([Output('figure_quantile_regression', 'figure'),
                Output('rt3_quant', 'children'),
                Output('quant_table_txt', 'children'),
                Output('quant_table_5', 'children'),
                Output('quant_table_6', 'children'),
                Output('quant_table_3', 'children'),
                Output('quant_table_4', 'children'),
                Output('quant_table_1', 'children'),
                Output('quant_table_2', 'children'),
                ],
                [Input('btn2_quant', 'n_clicks'),
                 Input('rt4', 'children')],
                [State('xvar2_quant', 'value'),
                 State('yvar2_quant', 'value'),
                 State('x_transform_quant', 'value'),
                 State('y_transform_quant', 'value'),
                 State('model2_quant', 'value'),
                 State('data_table', 'data'),
                 State('quantiles', 'value')],
            )
def update_quantile_regression(n_clicks, reset, xvar, yvar, x_transform, y_transform, model, 
                               df, quantiles):
        
    return quantile_regression.get_updated_results(n_clicks, xvar, yvar, x_transform, y_transform, 
                                                   model, df, quantiles)


@app.callback([Output('figure_multiple_linear_regression', 'figure'),
               Output('table_plot3a', 'children'),
               Output('table_plot3b', 'children'),
               Output('rt1', 'children'),
               Output('fig3txt', 'children'),
               Output('tab3btxt', 'children'),
               Output('btn_ss2', 'n_clicks')],
              [Input('btn3', 'n_clicks'),
               Input('btn_ss2', 'n_clicks'),
               Input('rt4', 'children')],
              [State('xvar3', 'value'),
               State('yvar3', 'value'),
               State('data_table', 'data'),
               State('cat_vars', 'children'),
               State('rfecv', 'value'),
               State('mlr_model', 'value'),
               ]
        )
def update_linear_multivariable_regression(n_clicks, smartscale, reset, xvars, yvar, df, cat_vars, 
                                           rfe_val, mlr_model):
    
    return MLR.get_updated_results(n_clicks, smartscale, xvars, yvar, df, cat_vars, 
                                   rfe_val, mlr_model)
    

@app.callback([Output('figure_plot4a', 'figure'),
                Output('figure_plot4b', 'figure'),
                Output('table_plot4a', 'children'),
                Output('table_plot4b', 'children'),
                Output('rt2', 'children'),
                Output('tab4atxt', 'children'),
                Output('tab4btxt', 'children'),
                Output('fig4atxt', 'children'),
                Output('fig4btxt', 'children'),
                Output('btn_ss3', 'n_clicks'),
                ],
                [Input('btn4', 'n_clicks'),
                 Input('btn_ss3', 'n_clicks'),
                 Input('rt4', 'children')],
                [State('data_table', 'data'),
                State('xvar_logistic', 'value'),
                State('yvar_logistic', 'value'),
                State('cat_vars', 'children'),
                State('binary_classifier_model', 'value')],
            )
def update_logistic_regression(n_clicks, smartscale, reset, main_df, xvars, yvar, cat_vars, 
                               classifier_model):
    
    return binary_classify.get_updated_results(n_clicks, smartscale, main_df, xvars, yvar, cat_vars, 
                                               classifier_model)


@app.callback([Output('figure_glm', 'figure'),
               Output('glm_params_table', 'children'),
               Output('glm_performance_table', 'children'),
               Output('rt1_glm', 'children'),
               Output('figure_glm_txt', 'children'),
               Output('glm_params_txt', 'children'),
               Output('btn_glm', 'n_clicks'),
               Output('btn_ss_glm', 'n_clicks')],
              [Input('btn_glm', 'n_clicks'),
               Input('btn_ss_glm', 'n_clicks'),
               Input('rt4', 'children')],
              [State('glm_predictors', 'value'),
               State('glm_response_var', 'value'),
               State('data_table', 'data'),
               State('cat_vars', 'children'),
               State('rfecv_glm', 'value'),
               State('glm_model', 'value'),
               ],
        )
def update_glm(n_clicks, smartscale, reset, xvars, yvar, df, cat_vars, rfe_val, glm_model):
    return GLM.get_updated_results(n_clicks, smartscale, xvars, yvar, df, cat_vars, rfe_val, glm_model)


@app.callback([Output('survival_regression_figure', 'figure'),
               Output('time_to_event_figure', 'figure'),
               Output('kaplan_meier_curve_figure', 'figure'),
               Output('cumulative_hazard_curve_figure', 'figure'),
               Output('survival_params_table', 'children'),
               Output('survival_performance_table', 'children'),
               Output('rt_survival', 'children'),
               Output('survival_fig_txt', 'children'),
               Output('survival_params_table_txt', 'children'),
               Output('btn_survival', 'n_clicks'),
               Output('btn_ss_survival', 'n_clicks')],
              [Input('btn_survival', 'n_clicks'),
               Input('btn_ss_survival', 'n_clicks'),
               Input('rt4', 'children')],
              [State('survival_predictors', 'value'),
               State('survival_partial', 'value'),
               State('data_table', 'data'),
               State('cat_vars', 'children'),
               State('survival_multicollinear', 'value'),
               State('survival_d_var', 'value'),
               State('survival_e_var', 'value')],
        )
def update_survival_regression(n_clicks, smartscale, reset, xvars, 
                          partial_effects_var, df, cat_vars, rfe_val, duration_var, event_var):
    
    return survival.get_updated_results(n_clicks, smartscale, xvars, partial_effects_var, df, 
                                        cat_vars, rfe_val, duration_var, event_var)
    

@app.callback([Output('figure_decision_tree_obs_pred', 'figure'),
               Output('figure_decision_tree', 'children'),
               Output('dec_tree_reg_table_params', 'children'),
               Output('text-dec_tree_reg1', 'children'),
               Output('text-dec_tree_reg2', 'children'),
               Output('text-dec_tree_reg3', 'children'),
               Output('text-dec_tree_reg4', 'children'),
               ],
              [Input('btn_dec_tree_reg', 'n_clicks'),
               Input('rt4', 'children'),
               ],
              [State('xvar_dec_tree_reg', 'value'),
               State('yvar_dec_tree_reg', 'value'),
               State('data_table', 'data'),
               State('cat_vars', 'children'),
               ]
        )
def update_decision_tree_regression(n_clicks, reset, xvars, yvar, df, cat_vars):
    return DecisionTree.get_updated_results(n_clicks, xvars, yvar, df, cat_vars)
    

####################################################################################################
#############################      Run the server      #############################################
####################################################################################################


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug = True) # modified to run on linux server
