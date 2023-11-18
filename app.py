from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LinearRegression, LogisticRegression, GammaRegressor
from sklearn.linear_model import TweedieRegressor, PoissonRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson
import statsmodels.stats.stattools as stattools
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.api as sm

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import dash

import plotly.graph_objects as go
from scipy import stats
import pandas as pd
import numpy as np
import contextlib
import warnings
import random
import base64
import json
import io
import os

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

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
###########################      CUSTOM FUNCTIONS      #############################################
####################################################################################################


def obs_pred_rsquare(obs, pred):
    '''
    Determines the proportion of variability in a data set accounted for by a model
    In other words, this determines the proportion of variation explained by the 1:1 line
    in an observed-predicted plot.
    
    Used in various peer-reviewed publications:
        1. Locey, K.J. and White, E.P., 2013. How species richness and total abundance 
        constrain the distribution of abundance. Ecology letters, 16(9), pp.1177-1185.
        2. Xiao, X., McGlinn, D.J. and White, E.P., 2015. A strong test of the maximum 
        entropy theory of ecology. The American Naturalist, 185(3), pp.E70-E80.
        3. Baldridge, E., Harris, D.J., Xiao, X. and White, E.P., 2016. An extensive 
        comparison of species-abundance distribution models. PeerJ, 4, p.e2823.
    '''
    r2 = 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)
    return r2


def smart_scale(df, predictors, responses):
    
    '''
    Skewness generally comes in two forms:
    1. Positive skew: Data with many small values and few large values.
    2. Negative skew: Date with many large values and few small values.
    
    Significantly skewed data can invalidate or obsure regression results by causing outliers
    (extreme values in reponse variables) and leverage points (extreme values in predictor variables)
    to exert a biased influence on the analysis.
    
    The smart_scale function loops through each data feature in the input dataframe 'df' and conducts
    a skewness test using scipy's skewtest function:
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html#scipy.stats.skewtest
    If a feature is significantly skewed, the smart_scale function will loop through various data 
    transformations and attempt to find the one that brings the skewness closest to zero.
    '''
    
    for i in list(df):
        stat, pval = float(), float()
        try: stat, pval = stats.skewtest(df[i], nan_policy='omit')
        except: continue
        
        if pval >= 0.05:
            continue
        
        else:
            # Based on the Fisher-Pearson coefficient
            skewness = stats.skew(df[i], nan_policy='omit')       
            best_skew = float(skewness)
            best_lab = str(i)
            t_vals = df[i].tolist()
            
            if np.nanmin(df[i]) < 0: 
                
                # log-modulo transformation
                lmt = np.log10(np.abs(df[i]) + 1).tolist()
                for j, val in enumerate(df[i].tolist()):
                    if val < 0: lmt[j] = lmt[j] * -1
                
                new_skew = stats.skew(lmt, nan_policy='omit')
                if np.abs(new_skew) < best_skew:
                    best_skew = np.abs(new_skew)
                    best_lab = 'log<sub>mod</sub>(' + i + ')'
                    t_vals = lmt
                
                # cube root transformation
                crt = df[i]**(1/3)
                new_skew = stats.skew(crt, nan_policy='omit')
                if np.abs(new_skew) < best_skew:
                    best_skew = np.abs(new_skew)
                    best_lab = '\u221B(' + i + ')'
                    t_vals = crt
                    
                # cube transformation
                ct = df[i]**3
                new_skew = stats.skew(ct, nan_policy='omit')
                if np.abs(new_skew) < best_skew:
                    best_skew = np.abs(new_skew)
                    best_lab = '(' + i + ')\u00B3'
                    t_vals = ct
                
            elif np.nanmin(df[i]) == 0:
                
                # log-shift transformation
                lmt = np.log10(df[i] + 1).tolist()
                new_skew = stats.skew(lmt, nan_policy='omit')
                if np.abs(new_skew) < best_skew:
                    best_skew = np.abs(new_skew)
                    best_lab = 'log-shift(' + i + ')'
                    t_vals = lmt
                    
                # square root transformation
                srt = df[i]**(1/2)
                new_skew = stats.skew(srt, nan_policy='omit')
                if np.abs(new_skew) < best_skew:
                    best_skew = np.abs(new_skew)
                    best_lab = '\u221A(' + i + ')'
                    t_vals = srt
                    
                # square transformation
                st = df[i]**2
                new_skew = stats.skew(st, nan_policy='omit')
                if np.abs(new_skew) < best_skew:
                    best_skew = np.abs(new_skew)
                    best_lab = '(' + i + ')\u00B2'
                    t_vals = st
                
            elif np.nanmin(df[i]) > 0:
                lt = np.log10(df[i])
                new_skew = stats.skew(lt, nan_policy='omit')
                if np.abs(new_skew) < best_skew:
                    best_skew = np.abs(new_skew)
                    best_lab = 'log(' + i + ')'
                    t_vals = lt
                    
                # square root transformation
                srt = df[i]**(1/2)
                new_skew = stats.skew(srt, nan_policy='omit')
                if np.abs(new_skew) < best_skew:
                    best_skew = np.abs(new_skew)
                    best_lab = '\u221A(' + i + ')'
                    t_vals = srt
                    
                # square transformation
                st = df[i]**2
                new_skew = stats.skew(st, nan_policy='omit')
                if np.abs(new_skew) < best_skew:
                    best_skew = np.abs(new_skew)
                    best_lab = '(' + i + ')\u00B2'
                    t_vals = st
                
            df[i] = list(t_vals)
            
        df.rename(columns={i: best_lab}, inplace=True)
        
        if i in predictors:
            predictors.remove(i)
            predictors.append(best_lab)
        if i in responses:
            responses.remove(i)
            responses.append(best_lab)
        
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df, predictors, responses


def dummify(df, cat_vars, dropone=True):
    
    '''
    Convert categorical features to binary dummy variables. 
    
    df: input dataframe containing all numerical and categorical features
    
    cat_vars: a list of categorical features
    
    dropone: Indicates whether or not to drop one level from each categorical feature, as when
        conducting linear or logistic multivariable regression.
        
    Note: In the event that a categorical feature contains more than 10 levels, only the 10
        most common levels are retained. If this happens, then the dropone argument can be ignored
        as its function (to prevent perfect multicollinearity) will be redundant.
    
    '''
    
    dropped = []
    cat_var_ls = []
    
    interxn = list(set(cat_vars) & set(list(df)))
    
    for i in interxn:
        labs = list(set(df[i].tolist()))
        df[i] = df[i].replace(r"^ +| +$", r"", regex=True)
        
        subsample = 0
        one_hot = pd.get_dummies(df[i])
        if one_hot.shape[1] > 10:
            subsample = 1
            one_hot = one_hot[one_hot.sum().sort_values(ascending=False).index[:10]]
            
        one_hot = one_hot.add_prefix(i + ':')
        ls2 = list(one_hot)
        
        if dropone == True and subsample == 0:
            nmax = 0
            lab = 0
            for ii in ls2:
                x = one_hot[ii].tolist()
                n = x.count(1)
                if n > nmax:
                    nmax = int(n)
                    lab = ii
            
            one_hot.drop(labels=[lab], axis = 1, inplace=True)
            dropped.append(lab)
             
        labs = list(one_hot)
        cat_var_ls.append(labs)
        df = df.join(one_hot)
        df.drop(labels=[i], axis = 1, inplace=True)
        
    return df, dropped, cat_var_ls
            

def dummify_logistic(df, cat_vars, y_prefix, dropone=True):

    '''
    Convert categorical features to binary dummy variables. 
    
    df: input dataframe containing all numerical and categorical features
    
    cat_vars: a list of categorical features
    
    y_prefix: the category of the feature that was chosen as the response variable
    
    dropone: Indicates whether or not to drop one level from each categorical feature, as when
        conducting linear or logistic multivariable regression.
        
    Note: In the event that a categorical feature contains more than 10 levels, only the 10
        most common levels are retained. If this happens, then the dropone argument can be ignored
        as its function (to prevent perfect multicollinearity) will be redundant.
    
    '''
    
    dropped = []
    cat_var_ls = []
    
    interxn = list(set(cat_vars) & set(list(df)))
    
    for i in interxn:
        labs = list(set(df[i].tolist()))
        df[i] = df[i].replace(r"^ +| +$", r"", regex=True)
        
        subsample = 0
        one_hot = pd.get_dummies(df[i])
        if one_hot.shape[1] > 10:
            subsample = 1
            one_hot = one_hot[one_hot.sum().sort_values(ascending=False).index[:10]]
            
        one_hot = one_hot.add_prefix(i + ':')
        ls2 = list(one_hot)
        
        if dropone == True and subsample == 0 and i != y_prefix:
            nmax = 0
            lab = 0
            for ii in ls2:
                x = one_hot[ii].tolist()
                n = x.count(1)
                if n > nmax:
                    nmax = int(n)
                    lab = ii
            
            one_hot.drop(labels=[lab], axis = 1, inplace=True)
            dropped.append(lab)
            
        labs = list(one_hot)
        cat_var_ls.append(labs)
        df = df.join(one_hot)
        df.drop(labels=[i], axis = 1, inplace=True)
        
    return df, dropped, cat_var_ls


def run_MLR(df_train, xvars, yvar, cat_vars, rfe_val):

    X_train = df_train.copy(deep=True)
    
    X_train, dropped, cat_vars_ls = dummify(X_train, cat_vars)
    
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
    X_train.dropna(how='any', inplace=True)
    
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
    results = sm.OLS(y_train, X_train_lm).fit()
    
    y_pred = results.predict(X_train_lm)
    #pval_df = results.pvalues
    R2 = results.rsquared_adj
    if R2 < 0: R2 = 0
    
    results_summary = results.summary()
    
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
    

def run_logistic_regression(df, xvars, yvar, cat_vars, classifier_model):
    
    coefs = []
    r2s = []
    pvals = []
    aics = []
    llf_ls = []
    PredY = []
    PredProb = []
    Ys = []
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how='any', inplace=True)
    
    if yvar in list(df): 
        pass
    else:
        return None, None, None, 3, None
    
    df = df.loc[:, df.nunique() != 1]
    
    if yvar in list(df): 
        pass
    else:
        return None, None, None, 2, None

    y_o = df[yvar]
    x_o = df.drop(labels=[yvar], axis=1, inplace=False)
    
    ########## Eliminating features that are perfectly correlated to the response variable #########
    perfect_correlates = []
    for xvar in list(x_o):
        x = x_o[xvar].tolist()
        y = y_o.tolist()
        slope, intercept, r, p, se = stats.linregress(x, y)
        if r**2 == 1.0:
            perfect_correlates.append(xvar)
    x_o.drop(labels=perfect_correlates, axis=1, inplace=True)
    
    ########## Eliminating features that only have one value ###########
    singularities = []
    for xvar in list(x_o):
        x = len(list(set(x_o[xvar].tolist())))
        if x == 1:
            singularities.append(xvar)
    x_o.drop(labels=singularities, axis=1, inplace=True)
    
    ########## Eliminating features with many 0's ###########
    x_vars = list(x_o)
    drop = []
    for var in x_vars:
        vals = x_o[var].tolist()
        frac_0 = vals.count(0)/len(vals)
        if frac_0 > 0.95:
            drop.append(var)
    
    x_o.drop(labels=drop, axis=1, inplace=True)
    x_vars = list(x_o)
    
    if len(x_vars) > 1:
        ########## Eliminating features using vif ###########
        while x_o.shape[1] > 100:
            cols = list(x_o)
            vifs = [variance_inflation_factor(x_o.values, j) for j in range(x_o.shape[1])]
                
            max_vif = max(vifs)
            if max_vif > 10:
                i = vifs.index(max(vifs))
                col = cols[i]
                x_o.drop(labels=[col], axis=1, inplace=True)
            else:
                break
        
        ########## RFECV ############
        if x_o.shape[1] > 100:
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
    
    model = 0
    x_o_lm = sm.add_constant(x_o, has_constant='add')
    
    try:
        if classifier_model == 'Logistic':
            model = sm.Logit(y_o, x_o_lm).fit(maxiter=30)
        elif classifier_model == 'Probit':
            model = sm.Probit(y_o, x_o_lm).fit(maxiter=30)
        
    except:
        return None, None, None, 1, None
    
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
    
    x_vars = list(x_o)
    if len(x_vars) > 1: 
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
    
    #df['predicted probability'] = ypred
    
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
                    
    #m_coefs = model.params[0].tolist()
    #m_coefs.reverse()
                            
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
    
    return df_models, df1_summary, df2_summary, 0, df


def run_glm(df_train, xvars, yvar, cat_vars, rfe_val, family):
    """
    Perform Poisson regression using statsmodels.

    Parameters:
    - df_train (pd.DataFrame): The input data as a DataFrame.
    - xvars (list): List of predictor column names.
    - yvar (str): The name of the target column.
    - cat_vars (list): List of categorical variable column names.
    - rfe_val (str): 'Yes' or 'No' to indicate whether to use recursive feature elimination (RFE).

    Returns:
    - y_train (pd.Series): The target variable.
    - y_pred (pd.Series): Predicted values.
    - df1_summary (pd.DataFrame): Parameter estimates table.
    - df2_summary (pd.DataFrame): Model summary table.
    - supported_features (list): List of supported predictor variables.
    - unsupported (list): List of unsupported predictor variables.
    - colors (list): List of colors used for later processing.
    """
    
    X_train = df_train.copy(deep=True)
    X_train, dropped, cat_vars_ls = dummify(X_train, cat_vars)
    
    if X_train.shape[1] < 2:
        return pd.Series(), pd.Series(), pd.DataFrame(), pd.DataFrame(), [], [], []
    
    # Eliminate features with many 0's
    x_vars = list(X_train)
    drop = []
    for var in x_vars:
        vals = X_train[var].tolist()
        frac_0 = vals.count(0) / len(vals)
        if frac_0 > 0.95:
            drop.append(var)
    
    X_train.drop(labels=drop, axis=1, inplace=True)
    X_train.dropna(how='any', inplace=True)
    y_train = X_train.pop(yvar)
    
    supported_features = []
    unsupported = []

    results = [] 
    ranks = []
    xlabs = []
    
    # RUN RFECV
    if family == 'Poisson':
        poisson_model = PoissonRegressor()
        rfecv = RFECV(estimator=poisson_model, cv=5)
        
    elif family == 'Binomial':
        binomial_model = LogisticRegression()
        rfecv = RFECV(estimator=binomial_model, cv=5)
        
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
    
    rfecv.fit(X_train, y_train)
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
    
    if family == 'Poisson':
        # Poisson (Log Link)
        results = sm.GLM(y_train, X_train_lm, family=sm.families.Poisson(sm.families.links.log())).fit()

    elif family == 'Binomial': 
        # Binomial (Logit Link)
        results = sm.GLM(y_train, X_train_lm, family=sm.families.Binomial(sm.families.links.logit())).fit()

    elif family == 'Gamma':
        # Gamma (Identity Link)
        results = sm.GLM(y_train, X_train_lm, family=sm.families.Gamma(sm.families.links.identity())).fit()

    elif family == 'Gaussian':
        # Gaussian (Identity Link)
        results = sm.GLM(y_train, X_train_lm, family=sm.families.Gaussian(sm.families.links.identity())).fit()

    elif family == 'InverseGaussian':
        # InverseGaussian (Inverse Link)
        results = sm.GLM(y_train, X_train_lm, family=sm.families.InverseGaussian(sm.families.links.inverse_power())).fit()

    elif family == 'NegativeBinomial':
        # NegativeBinomial (Log Link)
        results = sm.GLM(y_train, X_train_lm, family=sm.families.NegativeBinomial(sm.families.links.log())).fit()

    elif family == 'Tweedie':
        # Tweedie (Power Link), Adjust power parameter as needed
        results = sm.GLM(y_train, X_train_lm, family=sm.families.Tweedie(sm.families.links.power(1.5))).fit()
    
    y_pred = results.predict(X_train_lm)
    results_summary = results.summary()
    
    results_as_html1 = results_summary.tables[1].as_html()
    df1_summary = pd.read_html(results_as_html1, header=0, index_col=None)[0]
    df1_summary.rename(columns={'Unnamed: 0': 'Variable',}, inplace=True)
    
    results_as_html2 = results_summary.tables[0].as_html()
    df2_summary = pd.read_html(results_as_html2, header=None, index_col=None)[0]
    
    for col in list(df2_summary):
        df2_summary.rename(columns={col: str(col)}, inplace=True)
    
    vifs = [variance_inflation_factor(X_train.values, j) for j in range(X_train.shape[1])]
    vifs2 = []
    
    xlabs = list(X_train)
    for p in df1_summary['Variable'].tolist():
        if p == 'const':
            vifs2.append(np.nan)
        else:
            i = xlabs.index(p)
            vif = vifs[i]
            vifs2.append(np.round(vif, 3))
        
    df1_summary['VIF'] = vifs2
    colors = ["#3399ff"] * len(y_train)
    
    return y_train, y_pred, df1_summary, df2_summary, supported_features, unsupported, colors


def run_cox(df, xvars, partial_effects_var, cat_vars, rfe_val, duration_var, event_var):

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
    
    df, dropped, cat_vars_ls = dummify(df, cat_vars)
    
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
    
    # Fit the Cox Proportional-Hazards model using lifelines
    
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
            
    ################################################################################################
    #######################    1st DataFrame    ####################################################
    ################################################################################################
                    
    df_1 = pd.DataFrame(columns=['Model information', 'Model statistics'])
    df_1['Model information'] = cols
    df_1['Model statistics'] = vals

    ################################################################################################
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
    
    ################################################################################################
    #######################    Survival Curves    ##################################################
    ################################################################################################
    
    p10 = np.percentile(df, 10)
    p50 = np.percentile(df, 50)
    p90 = np.percentile(df, 90)
    
    fig = cph.plot_partial_effects_on_outcome(covariates = partial_effects_var, 
                                              values=[p10, p50, p90], cmap='coolwarm')
    
    # Access the data from the plot and store it in lists
    x_values = []
    y_values = []
    
    for line in fig.get_lines():
        x, y = line.get_data()
        x_values.append(x)
        y_values.append(y)
    
    '''
    fig = cph.plot_cumulative_hazard()
    # Access the data from the plot and store it in lists
    x_values_haz = []
    y_values_haz = []
    
    for line in fig.get_lines():
        x, y = line.get_data()
        x_values_haz.append(x)
        y_values_haz.append(y)
    
    
    '''
    
    return df_1, df_2, x_values, y_values


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
                [dbc.ModalBody([control_card_choose_data(), 
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


def level_collapser_control_card():
    
    return html.Div(
        children=[
            html.H5('Collapse Levels',
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
                    id='cat_var_collapse',
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
            
            html.B("Create a new level OR collapse the level(s) you chose into an existing level",
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
            
            dbc.Button('Collapse', 
                        id='level-collapse-btn1', 
                        n_clicks=0,
                        style={'width': '20%',
                            'font-size': 14,
                            "background-color": "#2a8cff",
                            'margin-right': '2%',
                            },
            ),
            html.Br(),
            html.H5("", id='collapse_text',
            style={'textAlign': 'left',
                   'font-size': 16,
                   'margin-left': '3%',
                   'color': '#ffffff'}),
            dcc.Interval(id='collapse-interval-component', interval=3000, n_intervals=0)
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
                                      
            html.P("", id='rt4'),
                                  
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
                row_deletable=True,
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
                        dbc.Button('Collapse Levels',
                                   id="open-level-collapse",
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
                                            id="level-collapse",
                                            className="one columns",
                                            children=[level_collapser_control_card()],
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
                                       id="close-level-collapse", 
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
                            id="modal-level-collapse",
                            
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
    

def control_card_choose_reg2():
    return html.Div(
        children=[    
            html.Div(
                children=[
                    html.H5("Iterative Multi-Model Regressions", 
                            style={'display': 'inline-block', 
                                   },
                            ),
                    dcc.Markdown("Save tons of time by automating the search for simple 1-to-1 " +
                                 "relationships among variables with multiple models, OLS and " +
                                 "Robust regression, and optimized data transformations.",
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
                                children=[control_card_iterative_multi_model_regression(),
                                          generate_outputs_iterative_multi_model_regression(),
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
                    dcc.Markdown("Examine the relationship between 2 variables. Use OLS, Robust " +
                                 "regression, data transformations, linear and polynomial " +
                                 "models, and build confidence intervals, prediction intervals, " +
                                 "and identify outliers.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dbc.Button("Run in-depth OLS regression",
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
                                children=[control_card_single_regression(),
                                          generate_figure_single_regression(),
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
                    dcc.Markdown("Extend linear regression to examine upper limits, lower " +
                                 "limits, or any quantile of a response variable in relation to " +
                                 "a predictor. Or, go even futher with polynomial quantile " +
                                 "regression.",
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
                                children=[control_card_quantile_regression(),
                                          generate_figure_quantile_regression(),
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
                    html.H5("Multiple Linear Regression", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("Get insights into how 2 or more predictors collectively " +
                                 "influence a response variable. Include automated, optimized " +
                                 "data scaling and the removal of unimportant variables.",
                                 style={'width': '94.1%',
                                        },
                                 ),
                    dbc.Button("Run multiple linear regression",
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
                                children=[control_card_multiple_regression(),
                                          generate_multiple_regression_outputs(),
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
                                children=[control_card_logistic(),
                                          generate_logistic_a(),
                                          generate_logistic_b(),
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
                    html.H5("Generalized Linear Modeling (GLM)", 
                            style={'display': 'inline-block',
                                   },
                            ),
                    dcc.Markdown("GLMs reveal how 2 or more predictors influence a response " +
                                 "variable. GLMs accommodate data distributions that simpler " +
                                 "regression models don't. Our GLMs included **Poisson**, " +
                                 "**Binomial**, **Negative Binomial**, **Gamma**, **Gaussian**, " +
                                 "and other families of models.",
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
                                children=[control_card_glm(),
                                          generate_glm_outputs(),
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
                               id='cox_reg_btn',
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
                                children=[control_card_cox_regression(),
                                          generate_cox_outputs(),
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
                                           id='close-cox', 
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
                    id='modal-cox',
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
            ],
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
                         dbc.Button("Close", 
                                    id="close-iterative_multimodel_ols_table1", 
                                    className="ml-auto",
                                    ),
                         ),
                    ],
                    id="modal-iterative_multimodel_ols_table1",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
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


def control_card_single_regression():

    return html.Div(
        id="control-card2",
        children=[
                html.H5("Conduct a single regression for deeper insights",
                        style={'display': 'inline-block', 
                               'width': '35.4%',
                               'margin-right': '1%',
                               },
                        ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="target_select_vars2",
                       style={'display': 'inline-block', 
                              'width': '3%', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("These analyses are based on ordinary least squares regression (OLS)."+
                            "They exclude categorical features, any features suspected of being " +
                            "dates or times, and any numeric features having less than 4 unique " +
                            "values.", 
                            target="target_select_vars2", 
                            style = {'font-size': 12,
                                     },
                            ),
                html.Hr(),
                
                html.Div(
                    id="control-card2a",
                    children=[
                        html.B("Choose a predictor (x) variable",
                            style={'display': 'inline-block',
                                    'vertical-align': 'top',
                               },
                            ),
                        dcc.Dropdown(
                                id='xvar2',
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
                                   'width': '20%',
                                   'margin-right': '1%',
                            },
                    ),
                
                html.Div(
                    id="control-card2b",
                    children=[
                        html.B("Choose a data transformation",
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                            }),
                        dcc.Dropdown(
                                id='x_transform',
                                options=[{"label": i, "value": i} for i in ['None', 'log10', 
                                                                            'square root', 
                                                                            'cube root',
                                                                            'squared', 'cubed', 
                                                                            'log-modulo', 
                                                                            'log-shift']
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
                    
                html.Div(
                    id="control-card2c",
                    children=[
                        html.B("Choose a response (y) variable",
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                            },
                            ),
                        dcc.Dropdown(
                                id='yvar2',
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
                        },
                    ),
                
                html.Div(
                    id="control-card2d",
                    children=[
                        html.B("Choose a data transformation",
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                            },
                            ),
                        dcc.Dropdown(
                                id='y_transform',
                                options=[{"label": i, "value": i} for i in ['None', 'log10', 
                                                                            'square root', 
                                                                            'cube root',
                                                                            'squared', 'cubed', 
                                                                            'log-modulo', 
                                                                            'log-shift']
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
                    
                html.Div(
                    id="control-card2e",
                    children=[
                        html.B("Choose a model",
                            style={'display': 'inline-block',
                                   'vertical-align': 'top',
                            },
                            ),
                        dcc.Dropdown(
                                id='model2',
                                options=[{"label": i, "value": i} for i in ['linear', 'quadratic', 
                                                                            'cubic']
                                         ],
                                multi=False, 
                                value='linear',
                                style={'width': '100%',
                                       'display': 'inline-block',
                                     },
                                ),
                        ],
                        style={'display': 'inline-block',
                               'vertical-align': 'top',
                               'width': '10%',
                        },
                    ),
                
                html.Hr(),
                html.Br(),
                
                dbc.Button('Run regression', 
                            id='btn2', n_clicks=0,
                            style={'width': '20%',
                                'font-size': 12,
                                "background-color": "#2a8cff",
                                'display': 'inline-block',
                                'margin-right': '1%',
                    },
                    ),
                
                dbc.Button("View residuals plot",
                           id="open-single_regression_residuals_plot",
                           style={
                               "background-color": "#2a8cff",
                               'width': '16%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '1%',
                               },
                    ),
                dbc.Modal(
                    [dbc.ModalBody([dcc.Graph(id="residuals_plot1"), 
                                    html.Br(), 
                                    html.P("", id='fig2txt'),
                                    ],
                                   ),
                     dbc.ModalFooter(
                                    dbc.Button("Close", 
                                               id="close-single_regression_residuals_plot", 
                                               className="ml-auto")
                                    ),
                            ],
                    id="modal-single_regression_residuals_plot",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="lg",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button("View results table",
                           id="open-centered_single",
                           style={
                               "background-color": "#2a8cff",
                               'width': '16%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '1%',
                               },
                    ),
                
                dbc.Modal(
                    [dbc.ModalBody([html.H5("Results for single regression"),
                                    html.Div(id="single_table_1"), 
                                    html.Div(id="single_table_2"),
                                    html.Br(),
                                    html.P("", id="single_table_txt"),
                                    ],
                                   ),
                     dbc.ModalFooter(
                                    dbc.Button("Close", 
                                               id="close-centered_single", 
                                               className="ml-auto")
                                    ),
                            ],
                    
                    id="modal-centered_single",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="lg",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button('Run robust', 
                            id='btn_robust2', 
                            n_clicks=0,
                            style={'width': '15%',
                                   'font-size': 12,
                                   "background-color": "#2a8cff",
                                   'display': 'inline-block',
                                   'margin-right': '1%',
                                   },
                            ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="robust2",
                       style={'display': 'inline-block', 
                              'width': '3%', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("Outliers can weaken OLS regression. However, clicking on 'RUN " + 
                            "ROBUST' will run a 'Robust Linear Model' via statsmodels. " +
                            "The r-square (for observed vs predicted) of a robust regression " +
                            "will likely be lower than the r-square of OLS regression. This " +
                            "is because the robust model is not chasing outliers. However, " +
                            "for non-outliers (the main trend), the robust model will be more " +
                            "accurate, stable, valid, and useful for predictions than an OLS " +
                            "model. To run regular OLS regression, simply click 'RUN REGRESSION' " +
                            "again.",
                            target="robust2", 
                            style = {'font-size': 12,
                                     'display': 'inline-block',
                                     },
                            ),
                
                html.P("", id='rt3')
                ],
                style={'margin-left': '1%',
                       'width': '98.5%',
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


def control_card_multiple_regression():

    return html.Div(
        children=[
                html.H5("Conduct multiple linear regression",
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
                              'margin-right': '5%',
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
                            style={'width': '50%', 
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
                
                dbc.Button('Run multiple regression', 
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
                                    dbc.Button("Close", 
                                               id="close-centered3", 
                                               className="ml-auto")
                                    ),
                            ],
                    id="modal-centered3",
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
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
                                    dbc.Button("Close", 
                                    id="close-centered4", 
                                    className="ml-auto")
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
    

def control_card_cox_regression():

    return html.Div(
        children=[
                html.H5("Conduct Cox Proportional-Hazards Regression",
                        style={'display': 'inline-block', 
                               'margin-right': '1%',
                               },
                        ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="tt_cox1",
                       style={'display': 'inline-block', 
                              'color':'#99ccff',
                              },
                       ),
                dbc.Tooltip("This analysis is based on ordinary least squares regression and " +
                            "reveals predicted values, outliers, and the significance of " +
                            "individual variables.", 
                            target="tt_cox1", 
                            style = {'font-size': 12,
                                     },
                            ),
                html.P("When trying to understand how different factors influence the time " +
                       "until an event occurs."),
                
                html.Br(),
                
                html.B("Choose 2 or more covariates",
                    style={'vertical-align': 'top',
                           'margin-right': '10px',
                           'display': 'inline-block', 
                           'margin-right': '1%',
                       },
                    ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="tt_cox2",
                       style={'display': 'inline-block', 
                              'color':'#bfbfbf',
                              },
                       ),
                dbc.Tooltip("These are the variables that potentially influence your event " +
                            "variable (e.g., age and health-related variables influence the " +
                            "chance of death). The app will recognize if your event variable " +
                            "occurs in the list of covariates. If it does, the app will ignore it.",
                    target="tt_cox2", 
                    style = {'font-size': 12,
                             },
                    ),
                
                dcc.Dropdown(
                        id='cox_predictors',
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
                               id="tt_cox3e",
                            style={'display': 'inline-block', 
                                   'color':'#bfbfbf'},
                            ),
                        dbc.Tooltip("This would be a binary variable with 1's indicating an " +
                                    "event (e.g., death, stroke, readmission) had happened and " +
                                    "0's indicating the event had not happened.",
                            target="tt_cox3e", 
                            style = {'font-size': 12,
                                     },
                            ),
                        dcc.Dropdown(
                                id='cox_e_var',
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
                               id="tt_cox3d",
                            style={'display': 'inline-block', 
                                   'color':'#bfbfbf',
                                   },
                            ),
                        dbc.Tooltip("This variable would a measure of time during which the " +
                                    "event of interest could happen.",
                            target="tt_cox3d", 
                            style = {'font-size': 12,
                                     },
                            ),
                        dcc.Dropdown(
                                id='cox_d_var',
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
                               id="tt_cox3",
                            style={'display': 'inline-block', 
                                   'color':'#bfbfbf'},
                            ),
                        dbc.Tooltip("Choose one of your covariates to examine its effect on " +
                                    "the survival function.",
                            target="tt_cox3", 
                            style = {'font-size': 12,
                                     },
                            ),
                        dcc.Dropdown(
                                id='cox_partial',
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
                               id="tt_cox4",
                                    style={'display': 'inline-block', 
                                           'color':'#bfbfbf',
                                           },
                                    ),
                        dbc.Tooltip("Predictors that are highly correlated with each other can " +
                                    "obscure each other's effect, significance, and cause " +
                                    "survival regression to fail. This 'multicollinearity' " +
                                    "can be dealt with by iteratively removing the predictors " +
                                    "that contribute most to multicollinearity.", 
                                    target="tt_cox4", 
                                    style = {'font-size': 12,
                                             },
                                    ),
                        dcc.Dropdown(
                            id='cox_multicollinear',
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
                
                dbc.Button('Run Cox regression', 
                           id='btn_cox', 
                           n_clicks=0,
                           style={'display': 'inline-block',
                                  'width': '18%',
                                  'font-size': 12,
                                  'margin-right': '20px',
                                  "background-color": "#2a8cff",
                                  },
                           ),
                
                dbc.Button("View parameters table",
                           id='open-cox_params_table',
                           style={
                               "background-color": "#2a8cff",
                               'width': '18%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '20px',
                               },
                    ),
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id='cox_params_table'), 
                                    html.Br(), 
                                    html.P("", id='cox_params_table_txt'),
                                    ],
                                   ),
                     dbc.ModalFooter(
                                    dbc.Button("Close", 
                                               id='close-cox_params_table', 
                                               className="ml-auto")
                                    ),
                            ],
                    id='modal-cox_params_table',
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                
                dbc.Button("View model performance",
                           id='open-cox_performance_table',
                           style={
                               "background-color": "#2a8cff",
                               'width': '18%',
                               'font-size': 12,
                               'display': 'inline-block',
                               'margin-right': '20px',
                               },
                    ),
                dbc.Modal(
                    [dbc.ModalBody([html.Div(id='cox_performance_table'), 
                                    html.Br(), 
                                    html.P("Adjusted R-square accounts for sample size and the " +
                                           "number of predictors used."),
                                    ],
                                   ),
                     dbc.ModalFooter(dbc.Button("Close", 
                                                id='close-cox_performance_table', 
                                                className="ml-auto"),
                                    ),
                            ],
                    id='modal-cox_performance_table',
                    is_open=False,
                    centered=True,
                    autoFocus=True,
                    size="xl",
                    keyboard=True,
                    fade=True,
                    backdrop=True,
                    ),
                dbc.Button('Smart scale', 
                            id='btn_ss_cox', 
                            n_clicks=0,
                            style={'width': '20%',
                                   'font-size': 12,
                                   "background-color": "#2a8cff",
                                   'display': 'inline-block',
                                   'margin-right': '10px',
                                   },
                            ),
                html.I(className="fas fa-question-circle fa-lg", 
                       id="ss_cox",
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
                            target="ss_cox", 
                            style = {'font-size': 12,
                                     },
                            ),
                
                html.P("", id = 'rt_cox'),
                ],

        style={'width': '98.5%',
               'margin-left': '1%',
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
                dbc.Tooltip("...", 
                            target="tt_glm1", 
                            style = {'font-size': 12,
                                     },
                            ),
                html.P("When trying to explain or predict a non-categorical response " + 
                       "variable using two or more predictors."),
                
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
                                html.P("• Binomial: Ensure that the response variable is a " +
                                       "non-negative integer."),
                                html.P("• Gaussian: Gaussian GLMs assume that the response " + 
                                       "variable follows a normal distribution. The response " +
                                       "variable should be continuous."),
                                html.P("• Inverse Gaussian: The response variable should be " +
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
                            options=[{"label": i, "value": i} for i in ['Binomial', 
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


####################################################################################################
###################      DASH APP REGRESSION OUTPUT FUNCTIONS      #################################
####################################################################################################
    
    
def parse_contents(contents, filename, date):
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except:
            df = pd.read_csv(io.StringIO(decoded.decode('ISO-8859-1')))
            
    except Exception as e:
        return html.Div([
            'There was an error processing this file.'
        ])

    return df.to_json()


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
 
   
def generate_figure_single_regression():

    return html.Div(
                id="Figure2",
                children=[
                    dcc.Loading(
                        id="loading-fig2",
                        type="default",
                        fullscreen=False,
                        children=html.Div(id="figure2",
                            children=[dcc.Graph(id="figure_plot2"),
                                    ],
                                ),
                        ),
                    html.P("Confidence intervals (CI) reflect confidence in the mean y-value " +
                           "across the x-axis. Prediction intervals (PI) pertain to the model, " +
                           "where points outside the PI are unlikely to be explained by the " +
                           "model. Note: When running a robust regression, only the observed " +
                           "vs. predicted r\u00B2 value is returned, which usually equals or " +
                           "nearly equals the r\u00B2 of the fitted model.", 
                           ),
                    ],
                style={'width': '100%',
                       'display': 'inline-block',
                       'background-color': '#f0f0f0',
                       'padding': '1%',
                       },
                )


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


def generate_multiple_regression_outputs():

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


def generate_cox_outputs():

    return html.Div(
                children=[
                    dcc.Loading(
                        type="default",
                        fullscreen=False,
                        children=html.Div(
                            children=[dcc.Graph(id='cox_regression_figure'),
                                    ],
                                ),
                        ),
                    html.P("", id='cox_fig_txt')
                    ],
                style={'width': '100%',
                       'display': 'inline-block',
                       'background-color': '#f0f0f0',
                       'padding': '1%',
                },
    )


####################################################################################################
##############################        DASH APP LAYOUT        #######################################
####################################################################################################


app.layout = html.Div([
    
    dcc.Store(id='main_df', storage_type='memory'),
    
    html.Div(
        id='cat_vars',
        style={'display': 'none'}
        ),
    html.Div(
        id='di_numerical_vars',
        style={'display': 'none'}
        ),
    html.Div(
        id='VARIABLES',
        style={'display': 'none'}
        ),
    html.Div(
        id='ITERATIONS',
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
            children=[description_card1()],
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
            children=[description_card_final()],
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
    Output('modal-cox', "is_open"),
    [Input('cox_reg_btn', "n_clicks"), 
     Input('close-cox', "n_clicks")],
    [State('modal-cox', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_cox(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-cox_params_table', "is_open"),
    [Input('open-cox_params_table', "n_clicks"), 
     Input('close-cox_params_table', "n_clicks")],
    [State('modal-cox_params_table', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_cox_params_table(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('modal-cox_performance_table', "is_open"),
    [Input('open-cox_performance_table', "n_clicks"), 
     Input('close-cox_performance_table', "n_clicks")],
    [State('modal-cox_performance_table', "is_open")],
    prevent_initial_call=True,
)
def toggle_modal_cox_performance_table(n1, n2, is_open):
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
def toggle_modal_statsmodals(n1, n2, is_open):
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

###############  Buttons, Main DataFrames, Main DataTable   ########################################

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
            error_string = "Ensure there are only rows, columns, and one row of column headers. "
            error_string = "Make sure your file contains enough data to analyze."
            children = 0
            df = 0
            
            # Attempt to parse the content
            try: 
                children = [parse_contents(c, n, d) for c, n, d in zip([list_of_contents], 
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
    if df.shape[0] < 4 or df.shape[1] < 2: 
        return None, None, None, error_string
        
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
        ########## DETECT VARIABLE THAT ARE NUMERIC, CATEGORICAL, OR POTENTIALLY BOTH ##############
        
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





###################  Update Variables For Models  ##################################################

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
def update_select_variables_for_in_depth_single_OLS(df, selected_cols, cat_vars):
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
        tdf, dropped, cat_vars_ls = dummify(tdf, cat_vars, False)
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
  

@app.callback([Output('cox_predictors', 'options'),
               Output('cox_predictors', 'value'),
               Output('cox_partial', 'options'),
               Output('cox_partial', 'value'),
               Output('cox_partial', 'optionHeight'),
               Output('cox_d_var', 'options'),
               Output('cox_d_var', 'value'),
               Output('cox_d_var', 'optionHeight'),
               Output('cox_e_var', 'options'),
               Output('cox_e_var', 'value'),
               Output('cox_e_var', 'optionHeight')],
              [Input('data_table', 'data'),
               Input('data_table', 'selected_columns'),
               Input('cat_vars', 'children')],
            )
def update_variables_for_cox_regression(df, selected_cols, cat_vars):
    optionHeight = 30
    try:
        df = pd.DataFrame(df)
        if df is None or df.empty:
            return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight, 
                    [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                    ['Nothing uploaded'], optionHeight, 
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
        
        return [options1, ls1, options2, ls2, optionHeight, options1, 
                ls1, optionHeight, options1, ls1, optionHeight]
        
    except:
        return [[{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight, 
                [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight, 
                [{"label": 'Nothing uploaded', "value": 'Nothing uploaded'}], 
                ['Nothing uploaded'], optionHeight]
    

######################      Update Models      #####################################################


@app.callback([Output('figure_plot1', 'figure'),
               Output('table_plot1', 'children'),
               Output('rt0', 'children'),
               Output('fig1txt', 'children'),
               Output('table1txt', 'children'),
               ],
               [Input('upload-data', 'contents'),
                Input('hcris', 'children'),
                Input('hais', 'children'),
                Input('btn1', 'n_clicks'),
                Input('btn_ss', 'n_clicks'),
                Input('btn_robust', 'n_clicks')],
               [State('data_table', 'data'),
                State('cat_vars', 'children'),
                State('xvar', 'value'),
                State('yvar', 'value')],
            )
def update_simple_regressions(contents1, contents2, contents3, n_clicks, smartscale, 
                              robust, df, cat_vars, xvars, yvars):
    
    ctx1 = dash.callback_context
    jd1 = json.dumps({'triggered': ctx1.triggered,})
    jd1 = jd1[:50]
    
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
                df, xvars, yvars = smart_scale(df, xvars, yvars)

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
                            
                            op_r2 = obs_pred_rsquare(np.array(y), np.array(ypred))
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
                            op_r2 = round(obs_pred_rsquare(y, ypred),4)
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

    
@app.callback([Output('figure_plot2', 'figure'),
                Output('rt3', 'children'),
                Output('fig2txt', 'children'),
                Output('residuals_plot1', 'figure'),
                Output('btn_robust2', 'n_clicks'),
                Output('single_table_txt', 'children'),
                Output('single_table_1', 'children'),
                Output('single_table_2', 'children'),],
                [Input('upload-data', 'contents'),
                 Input('hcris', 'children'),
                 Input('hais', 'children'),
                 Input('btn2', 'n_clicks'),
                 Input('btn_robust2', 'n_clicks')],
                [State('xvar2', 'value'),
                 State('yvar2', 'value'),
                 State('x_transform', 'value'),
                 State('y_transform', 'value'),
                 State('model2', 'value'),
                 State('data_table', 'data')],
            )
def update_single_regression(contents1, content2, contents3, n_clicks, robust, 
                             xvar, yvar, x_transform, y_transform, model, df):
        
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
    if jd1 == '{"triggered": [{"prop_id": "upload-data.contents",':
        return {}, "", "", {}, 0, "", dashT1, dashT2
    
    try:
        df = pd.DataFrame(df)
    except:
        return {}, "", "", {}, 0, "", dashT1, dashT2
    
    if df is None or df.empty or xvar is None or yvar is None or xvar == yvar or isinstance(yvar, list) is True or isinstance(yvar, list) is True:
            
        if df is None or df.empty:
            return {}, "", "", {}, 0, "", dashT1, dashT2
            
        elif (isinstance(xvar, list) is True or xvar is None) & (isinstance(yvar, list) is True or yvar is None):
            return {}, "Error: You need to select some variables.", "", {}, 0, "", dashT1, dashT2
        
        elif isinstance(yvar, list) is True or yvar is None:
            return {}, "Error: You need to select a response variable.", "", {}, 0, "", dashT1, dashT2
            
        elif isinstance(xvar, list) is True or xvar is None:
            return {}, "Error: You need to select an predictor variable.", "", {}, 0, "", dashT1, dashT2
            
        elif xvar == yvar and xvar is not None:
            return {}, "Error: Your predictor variable and response variable are the same. Ensure they are different.", "", {}, 0, "", dashT1, dashT2
        else:
            return {}, "", "", {}, 0, "", dashT1, dashT2
            
    else:
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
            df[xvar] = np.log10(df[xvar] + 1).tolist()
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
        df.dropna(how='any', inplace=True)
                
        d = int()
        if model == 'linear': d = 1
        elif model == 'quadratic': d = 2
        elif model == 'cubic': d = 3
            
        if robust == 0:
            y_o = df[yvar].values.tolist()
            x_o = df[xvar].values.tolist()
            x_o, y_o = zip(*sorted(zip(x_o, y_o)))
                
            x_o = np.array(x_o)
            y_o = np.array(y_o)
            
            #Create single dimension
            x = x_o[:, np.newaxis]
            y = y_o[:, np.newaxis]

            # Sort x values and get index
            inds = x.ravel().argsort()  
            x = x.ravel()[inds].reshape(-1, 1)
            #Sort y according to x sorted index
            y = y[inds]
                
            polynomial_features = PolynomialFeatures(degree = d)
            xp = polynomial_features.fit_transform(x)
                    
            model = sm.OLS(y, xp).fit()
            ypred = model.predict(xp)
            ypred = ypred.tolist()
                
        elif robust == 1:
            # Rub Robust Regression
            y_o = df[yvar].values.tolist()
            x_o = df[xvar].values.tolist()
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
                
        # Jarque-Bera for normally distributed errors
        jarque_bera_test = sms.jarque_bera(model.resid)
        jarque_bera_p = round(jarque_bera_test[1], 4)
            
        # Durbin-Watson for autocorrelation
        dw = stattools.durbin_watson(model.resid)
        durbin_watson = round(dw, 4)
            
        # Breusch-Pagan for heteroskedasticity
        breusch_pagan_test = sms.het_breuschpagan(model.resid, model.model.exog)
        breusch_pagan_p = round(breusch_pagan_test[1],4)
            
        # Harvey-Collier multiplier test for linearity
        if d == 1:
            try:
                skip = 10 #len(model.params)  # bug in linear_harvey_collier
                rr = sms.recursive_olsresiduals(model, skip=skip, alpha=0.95, order_by=None)
                harvey_collier_test = stats.ttest_1samp(rr[3][skip:], 0)
                harvey_collier_p = round(harvey_collier_test[1], 4)
            except:
                harvey_collier_p = np.nan
        
        else:
            harvey_collier_p = np.nan
        
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
        
        try:
            y = y.flatten().tolist()
        except:
            pass
        op_r2 = obs_pred_rsquare(np.array(y), np.array(ypred))
        try:
            op_r2 = round(op_r2, 4)
        except:
            pass
        if op_r2 < 0:
            op_r2 = 0
            
        if robust == 0:
            r2 = round(model.rsquared, 4)
            r2_adj = round(model.rsquared_adj, 4)
            #aic = round(model.aic, 4)
            #bic = round(model.bic, 4)
            #fp = round(model.f_pvalue, 4)
            #llf = round(model.llf, 4)
            
            st, data, ss2 = summary_table(model, alpha=0.05)
            #fittedvalues = data[:, 2]
            #predict_mean_se  = data[:, 3]
            predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T # confidence interval
            predict_ci_low, predict_ci_upp = data[:, 6:8].T # prediction interval
            
        elif robust == 1:
            r2 = 'N/A'
            r2_adj = 'N/A'
            #aic = 'N/A'
            #bic = 'N/A'
            #fp = 'N/A'
            #llf = 'N/A'
        
            # Calculate the standard error of residuals
            residuals = model.resid
            standard_error_residuals = np.std(residuals)
            
            # Set the desired confidence level
            confidence_level = 0.95
                
            # Calculate the critical t-value
            dof = len(x_o) - model.df_model - 1  # Degrees of freedom
            t_value = stats.t.ppf((1 + confidence_level) / 2, dof)
            
            x_plot = x_o

            # Create empty arrays to store the upper and lower bounds for the confidence intervals
            lower_ci_limit = np.zeros(len(x_plot))
            upper_ci_limit = np.zeros(len(x_plot))
            lower_pi_limit = np.zeros(len(x_plot))
            upper_pi_limit = np.zeros(len(x_plot))
            
            # Calculate confidence intervals for each x value
            for i, x in enumerate(x_plot):
                if d == 1:
                    x_poly = [1, x,]
                elif d == 2:
                    x_poly = [1, x, x**2]
                elif d == 3:
                    x_poly = [1, x, x**2, x**3]
                    
                # Confidence Intervals
                ci_multiplier = t_value * np.sqrt(np.dot(x_poly, 
                                                         np.dot(model.cov_params(), x_poly)))
                ci_interval = ci_multiplier
                lower_ci_limit[i] = model.predict(exog=x_poly) - ci_interval
                upper_ci_limit[i] = model.predict(exog=x_poly) + ci_interval
            
                # Prediction Intervals
                y_pred = np.dot(model.params, x_poly)
                lower_pi_limit[i] = y_pred - t_value * standard_error_residuals
                upper_pi_limit[i] = y_pred + t_value * standard_error_residuals
            
            predict_mean_ci_low = lower_ci_limit
            predict_mean_ci_upp = upper_ci_limit
            
            predict_ci_low = lower_pi_limit
            predict_ci_upp = upper_pi_limit
            
            
        outlier_y = []
        outlier_x = []
        nonoutlier_y = []
        nonoutlier_x = []
        for i, yi in enumerate(y_o):
            if yi > predict_ci_upp[i] or yi < predict_ci_low[i]:
                outlier_y.append(yi)
                outlier_x.append(x_o[i])
            else:
                nonoutlier_y.append(yi)
                nonoutlier_x.append(x_o[i])
                
        fig_data = []
            
        clr = "#3399ff"
            
        obs_pred_r2 = obs_pred_rsquare(y_o, ypred)
        obs_pred_r2 = str(np.round(obs_pred_r2, 3))
            
        fig_data.append(go.Scatter(
            x = nonoutlier_x,
            y = nonoutlier_y,
            name = 'Non-outliers',
            mode = "markers",
            opacity = 0.75,
            marker = dict(size=10,
                          color=clr)
            )
            )
                        
        fig_data.append(go.Scatter(
            x = outlier_x,
            y = outlier_y,
            name = 'Outliers',
            mode = "markers",
            opacity = 0.75,
            marker = dict(size=10,
                          color="#ff0000")
            )
        )
            
        fig_data.append(
            go.Scatter(
                x = x_o,
                y = ypred,
                mode = "lines",
                name = 'r<sup>2</sup> (fitted) =' + str(r2) + '<br>r<sup>2</sup> (adjusted) =' + str(r2_adj) + '<br>r<sup>2</sup> (obs vs pred) =' + obs_pred_r2,
                opacity = 0.75,
                line = dict(color = clr, 
                            width = 2),
                )
            )
            
        fig_data.append(
            go.Scatter(
                x = x_o,
                y = predict_mean_ci_upp,
                mode = "lines",
                name = 'upper 95 CI',
                opacity = 0.75,
                line = dict(color = clr, 
                            width = 2, 
                            dash='dash'),
            )
        )
            
        fig_data.append(
            go.Scatter(
                x = x_o,
                y = predict_mean_ci_low,
                mode = "lines",
                name = 'lower 95 CI',
                opacity = 0.75,
                line = dict(color = clr, 
                            width = 2, 
                            dash='dash'),
            )
        )
            
        fig_data.append(
            go.Scatter(
                x = x_o,
                y = predict_ci_upp,
                mode = "lines",
                name = 'upper 95 PI',
                opacity = 0.75,
                line = dict(color = clr, 
                            width = 2, 
                            dash='dot'),
            )
        )
            
        fig_data.append(
            go.Scatter(
                x = x_o,
                y = predict_ci_low,
                mode = "lines",
                name = 'lower 95 PI',
                opacity = 0.75,
                line = dict(color = clr, 
                            width = 2, 
                            dash='dot'),
            )
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

            
        txt = "The Jarque-Bera test suggests "
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
            
            
        #######################################################################
        #################   Residuals Plot   ################################## 
        #######################################################################
            
        fig_data2 = []
        fig_data2.append(go.Scatter(
            x = x_o,
            y = model.resid,
            name = 'residuals',
            mode = "markers",
            opacity = 0.75,
            marker = dict(size=10,
                          color=clr)
            )
            )
            
        res_figure = go.Figure(
            data = fig_data2,
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
                        text = "<b>" + "Residuals" + "</b>",
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
        
        
        ######################################### Result Tables ###################################
        results_summary = model.summary()
        
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
        
        return figure, "", txt, res_figure, 0, "", dashT1, dashT2


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
                [Input('upload-data', 'contents'),
                 Input('hcris', 'children'),
                 Input('hais', 'children'),
                 Input('btn2_quant', 'n_clicks')],
                [State('xvar2_quant', 'value'),
                 State('yvar2_quant', 'value'),
                 State('x_transform_quant', 'value'),
                 State('y_transform_quant', 'value'),
                 State('model2_quant', 'value'),
                 State('data_table', 'data'),
                 State('quantiles', 'value')],
            )
def update_quantile_regression(contents1, contents2, contents3, n_clicks, xvar, 
                               yvar, x_transform, y_transform, model, df, quantiles):
        
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
    if jd1 == '{"triggered": [{"prop_id": "upload-data.contents",':
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
        r2_50 = str(np.round(obs_pred_rsquare(obs_50, y_50), 3))
        
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


@app.callback([Output('figure_multiple_linear_regression', 'figure'),
               Output('table_plot3a', 'children'),
               Output('table_plot3b', 'children'),
               Output('rt1', 'children'),
               Output('fig3txt', 'children'),
               Output('tab3btxt', 'children'),
               Output('btn_ss2', 'n_clicks')],
              [Input('upload-data', 'contents'),
               Input('hcris', 'children'),
               Input('hais', 'children'),
               Input('btn3', 'n_clicks'),
               Input('btn_ss2', 'n_clicks')],
              [State('xvar3', 'value'),
               State('yvar3', 'value'),
               State('data_table', 'data'),
               State('cat_vars', 'children'),
               State('rfecv', 'value')],
        )
def update_multiple_regression(contents1, contents2, contents3, n_clicks, smartscale, xvars, 
                               yvar, df, cat_vars, rfe_val):
    
                        
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
    if jd1 == '{"triggered": [{"prop_id": "upload-data.contents",':
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
            df, xvars, yvars = smart_scale(df, xvars, [yvar])
            yvar = yvars[0]
            
        #df.dropna(how='any', inplace=True)
                            
        #Conduct multiple regression
        ls = run_MLR(df, xvars, yvar, cat_vars, rfe_val)
        y_train, y_pred, df1_summary, df2_summary, supported_features, unsupported, colors = ls 
        
        if len(y_train) == 0:
            rt1 = "Error: Your regression could not run. Your y-values contain no data."
            return {}, dashT1, dashT2, rt1, "", "", 0
        
        r2_obs_pred = obs_pred_rsquare(y_train, y_pred)
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
            
            style_table={'height': '225px', 
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
                [Input('upload-data', 'contents'),
                 Input('hcris', 'children'),
                 Input('hais', 'children'),
                 Input('btn4', 'n_clicks'),
                 Input('btn_ss3', 'n_clicks')],
                [State('data_table', 'data'),
                State('xvar_logistic', 'value'),
                State('yvar_logistic', 'value'),
                State('cat_vars', 'children'),
                State('binary_classifier_model', 'value')],
            )
def update_logistic_regression(contents1, contents2, contents3, n_clicks, smartscale, 
                               main_df, xvars, yvar, cat_vars, classifier_model):
    
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
    
    if jd1 == '{"triggered": [{"prop_id": "upload-data.contents",':
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
    
    elif (isinstance(yvar, list) is True) & (xvars is None or len(xvars) < 2):
        return [{}, {}, dashT1, dashT2, 
                "Error: Select a feature for your response variable and 2 or more for your predictors", 
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
        main_df, xvars, yvars = smart_scale(main_df, xvars, [yvar])
        yvar = yvars[0]
    
    
    vars_ = cat_vars #+ [yvar]
    vars_ = list(set(vars_))
    main_df, dropped, cat_vars_ls = dummify_logistic(main_df, vars_, y_prefix, True)
    
    if yvar not in list(main_df):
        return [{}, {}, dashT1, dashT2, 
                "Error: Choose a feature for your response variable", 
                "", "", "", "", 0]
    
    yvals = main_df[yvar].tolist()
    unique_yvals = list(set(yvals))
    if len(unique_yvals) < 2:
        return [{}, {}, dashT1, dashT2, 
                "Error: Your chosen response variable only contains one unique value: " + str(unique_yvals[0]), 
                "", "", "", "", 0]
    
    #if y_prefix in xvars:
    #    xvars.remove(y_prefix)
    #    if len(xvars) == 1:
    #        return [{}, {}, dashT1, dashT2, "Error: Multiple logistic regression requires 2 or more predictors. You chose two but one of them contains your response variable.", [0], "", "", "", "", 0]
    
    y_prefix = y_prefix + ":"
    for i in list(main_df):
        if y_prefix in i and i != yvar:
            main_df.drop(labels=[i], axis=1, inplace=True)
    
    ls = run_logistic_regression(main_df, xvars, yvar, cat_vars, classifier_model)
    models_df, df1_summary, df2_summary, error, pred_df = ls
    
    if error == 1 and smartscale == 1:
        error = "Error: The model exceeded the maximum iterations in trying to find a fit. "
        error += "Try running w/out Smart Scale. You might also have one or more severely "
        error += "redundant (multicollinear) predictors included. Try removing one or more "
        error += "potentially problematic predictors. Tip: Eliminate redundant predictors, "
        error += "predictors of little-to-no interest, or categorical variables with many levels, "
        error += "e.g., a column of diagnosis codes may have hundreds of different codes."
        return {}, {}, dashT1, dashT2, error, "", "", "", "", 0
    
    if error == 1 and smartscale == 0:
        error = "Error: The model exceeded the maximum iterations in trying to find a fit. "
        error += "Try using Smart Scale. You might also have one or more severely redundant "
        error += "(multicollinear) predictors included. Try removing one or more potentially "
        error += "problematic predictors. Tip: Eliminate redundant predictors, predictors of "
        error += "little-to-no interest, or categorical variables with many levels, e.g., a "
        error += "column of diagnosis codes may have hundreds of different codes."
        return {}, {}, dashT1, dashT2, error, "", "", "", "", 0
    
    if error == 2:
        error = "Error: The response variable must contain two unique values (e.g., 0 and 1). "
        error += "However, after removing rows with missing data, your chosen y-variable "
        error += "(" + yvar + ") only contains one unique value. Rows with missing data are "
        error += "removed because the logistic model will fail if the dataset contains any "
        error += "missing data. Try starting with fewer predictors. Once you have a working model, "
        error += "then try adding more to achieve better performance."
        return {}, {}, dashT1, dashT2, error, "", "", "", "", 0
    
    if error == 3:
        error = "Error: After removing rows with missing data, your dataset no longer contains "
        error += "any rows matching your chosen y-variable (" + yvar + "). Rows with missing data "
        error += "are removed because the logistic model will fail if the dataset contains any "
        error += "missing data. Try starting with fewer predictors. Once you have a working model, "
        error += "then try adding more to achieve better performance."
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


@app.callback([Output('figure_glm', 'figure'),
               Output('glm_params_table', 'children'),
               Output('glm_performance_table', 'children'),
               Output('rt1_glm', 'children'),
               Output('figure_glm_txt', 'children'),
               Output('glm_params_txt', 'children'),
               Output('btn_glm', 'n_clicks'),
               Output('btn_ss_glm', 'n_clicks')],
              [Input('upload-data', 'contents'),
               Input('hcris', 'children'),
               Input('hais', 'children'),
               Input('btn_glm', 'n_clicks'),
               Input('btn_ss_glm', 'n_clicks')],
              [State('glm_predictors', 'value'),
               State('glm_response_var', 'value'),
               State('data_table', 'data'),
               State('cat_vars', 'children'),
               State('rfecv_glm', 'value'),
               State('glm_model', 'value'),
               ],
        )
def update_glm(contents1, contents2, contents3, n_clicks, smartscale, 
               xvars, yvar, df, cat_vars, rfe_val, glm_model):
    
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
    if jd1 == '{"triggered": [{"prop_id": "upload-data.contents",':
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
        df.dropna(how='any', axis=0, inplace=True)
        
        if smartscale == 1:
            df, xvars, yvars = smart_scale(df, xvars, [yvar])
            yvar = yvars[0]
        
        # Treat data to prevent regression from failing
        if glm_model == 'Poisson':
            pass
            
        elif glm_model == 'Binomial':
            pass
            
        elif glm_model == 'Gamma':
            df = df[df[yvar] > 0]
            for var in xvars:
                df = df[df[var] > 0]
            df.dropna(how='any', inplace=True)    
            pass
        
        elif glm_model == 'Gaussian':
            pass
        
        elif glm_model == 'InverseGaussian':
            pass
        
        elif glm_model == 'NegativeBinomial':
            pass
        
        elif glm_model == 'Tweedie':
            pass
        
        #Conduct glm
        ls = run_glm(df, xvars, yvar, cat_vars, rfe_val, family=glm_model)
        y_train, y_pred, df1_summary, df2_summary, supported_features, unsupported, colors = ls
        
        if len(y_train) == 0:
            rt1 = "Error: Your regression could not run. Your y-values contain no data."
            return {}, dashT1, dashT2, rt1, "", "", 0, 0
        
        r2_obs_pred = obs_pred_rsquare(y_train, y_pred)
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
                                                 color="#3399ff"),
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
        
        del df2_summary
        del df1_summary
        
        txt1 = "This plot allows you to interpret patterns in the regression model's success. "
        txt1 = "Example: If points are consistently above the 1:1 line, then the observed values "
        txt1 = "are always greater than the predicted values. If the relationship is curved and "
        txt1 = "performance is weak, then try rescaling some of your variables "
        txt1 = "(via log, square root, etc.)."
        
        txt2 = "The variance inflation factor (VIF) measures multicollinearity. VIF > 5 indicates "
        txt2 = "that a predictor is significantly correlated with one or more other predictors. "
        txt2 = "VIF > 10 indicates severe multicollinearity, which can lead to overfitting and "
        txt2 = "inaccurate parameter estimates. If your VIF's are high, trying removing some of "
        txt2 = "those variables."
        
        return figure, dashT1, dashT2, "", txt1, txt2, 0, 0


@app.callback([Output('cox_regression_figure', 'figure'),
               Output('cox_params_table', 'children'),
               Output('cox_performance_table', 'children'),
               Output('rt_cox', 'children'),
               Output('cox_fig_txt', 'children'),
               Output('cox_params_table_txt', 'children'),
               Output('btn_cox', 'n_clicks'),
               Output('btn_ss_cox', 'n_clicks')],
              [Input('upload-data', 'contents'),
               Input('hcris', 'children'),
               Input('hais', 'children'),
               Input('btn_cox', 'n_clicks'),
               Input('btn_ss_cox', 'n_clicks')],
              [State('cox_predictors', 'value'),
               State('cox_partial', 'value'),
               State('data_table', 'data'),
               State('cat_vars', 'children'),
               State('cox_multicollinear', 'value'),
               State('cox_d_var', 'value'),
               State('cox_e_var', 'value')],
        )
def update_cox_regression(contents1, contents2, contents3, n_clicks, smartscale, xvars, 
                          partial_effects_var, df, cat_vars, rfe_val, duration_var, event_var):
    
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
    if jd1 == '{"triggered": [{"prop_id": "upload-data.contents",':
        return {}, dashT1, dashT2, "", "", "", 0, 0
    
    if df is None:
        return {}, dashT1, dashT2, "", "", "", 0, 0
    
    #elif yvar is None and xvars is None:
    #    return {}, dashT1, dashT2, "", "", "", 0, 0
    
    #elif yvar is None:
    #    return {}, dashT1, dashT2, "Error: Select a reponse variable", "", "", 0, 0
    
    #elif (isinstance(yvar, list) is True) & (xvars is None or len(xvars) < 2):
    #    return {}, dashT1, dashT2, "Error: Select a response variable and 2 or more predictors", "", "", 0, 0
    
    #elif isinstance(yvar, list) is True:
    #    return {}, dashT1, dashT2, "Error: Select a response variable", "", "", 0, 0
    
    elif xvars is None or len(xvars) < 1:
        return {}, dashT1, dashT2, "Error: Select one or more predictors", "", "", 0, 0
    
    df = pd.DataFrame(df)
    if df.empty:
        return {}, dashT1, dashT2, "", "", "", 0, 0
    
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
                            
        #Conduct cox regression
        ls = run_cox(df, xvars, partial_effects_var, cat_vars, rfe_val, duration_var, event_var)
        df_1, df_2, x_values, y_values = ls
        
        fig_data = []
        
        x10 = x_values[0]
        y10 = y_values[0]
        fig_data.append(go.Scatter(x = x10, 
                                   y = y10, 
                                   name = '10th percentile',
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   marker = dict(size=1, 
                                                 color="#66ccff"),
                                   ),
                        )
        
        x50 = x_values[1]
        y50 = y_values[1]
        fig_data.append(go.Scatter(x = x50, 
                                   y = y50, 
                                   name = '50th percentile',
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   marker = dict(size=1, 
                                                 color="#008ae6"),
                                   ),
                        )
        
        x90 = x_values[2]
        y90 = y_values[2]
        fig_data.append(go.Scatter(x = x90, 
                                   y = y90, 
                                   name = '90th percentile',
                                   mode = "lines", 
                                   opacity = 0.75, 
                                   marker = dict(size=1, 
                                                 color="#0052cc"),
                                   ),
                        )
        
        baseline_x = x_values[3]
        baseline_y = y_values[3]
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
        
        figure = go.Figure(data = fig_data,
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
                                            
                margin = dict(l=60, r=30, b=10, t=40), 
                showlegend = True, 
                height = 400,
                paper_bgcolor = "rgb(245, 247, 249)", 
                plot_bgcolor = "rgb(245, 247, 249)",
            ),
        )

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
        
        return figure, dashT2, dashT1, "", txt1, txt2, 0, 0


####################################################################################################
#############################      Run the server      #############################################
####################################################################################################


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug = True) # modified to run on linux server
