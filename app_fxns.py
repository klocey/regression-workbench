from dash import html
from scipy import stats
import pandas as pd
import numpy as np
import base64
import io

####################################################################################################
###########################      CUSTOM FUNCTIONS      #############################################
####################################################################################################

def remove_nans_optimal(df, yvar):
    
    df['index'] = list(range(df.shape[0]))
    y_df = df.filter(items=['index', yvar], axis=1)
    mod_df = df.drop(labels=[yvar], axis=1)
    
    total_nans = mod_df.isnull().sum().sum()
    
    while total_nans > 0:
        # Find the column with the most NaNs
        col_with_most_nans = mod_df.isnull().sum().idxmax()
        if col_with_most_nans == yvar:
            continue
        
        # Drop the column and declare a temporary dataframe (tdf)
        tdf = mod_df.drop(labels=[col_with_most_nans], axis=1)
        # Then drop the rows containing NaNs
        tdf.dropna(how='any', axis=0, inplace=True)
        # Get the size
        size1 = tdf.size
        
        # Declare a new temporary dataframe (tdf) and drop all rows containing NaNs
        tdf = mod_df.dropna(how='any', axis=0)
        # Get the size
        size2 = tdf.size
        
        if size1 <= size2:
            # Drop rows
            mod_df.dropna(how='any', axis=0, inplace=True)
            total_nans = mod_df.isnull().sum().sum()
            
        elif size1 > size2:
            # Drop the column
            mod_df.drop(labels=[col_with_most_nans], axis=1, inplace=True)
            total_nans = mod_df.isnull().sum().sum()
    
    total_nans = mod_df.isnull().sum().sum()
    
    df = y_df.merge(mod_df, how='outer', on='index')
    df.dropna(how='any', axis=0, inplace=True)
    
    total_nans = df.isnull().sum().sum()
    df.drop(labels=['index'], axis=1, inplace=True)
    
    return df


def nan_analysis(dataframe):
    # 1. The percent of rows with NaN values.
    percent_nan_rows = (dataframe.isnull().sum(axis=1) / len(dataframe.columns)) * 100

    # 2. The number and names of columns with NaN values.
    nan_columns = dataframe.columns[dataframe.isnull().any()]
    num_nan_columns = len(nan_columns)

    # 3. The number and names of columns without NaN values.
    non_nan_columns = dataframe.columns[dataframe.notnull().all()]
    num_non_nan_columns = len(non_nan_columns)

    return percent_nan_rows, num_nan_columns, nan_columns, num_non_nan_columns, non_nan_columns


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





