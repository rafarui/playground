import pandas as pd
import numpy as np
from functions import utils
from pathlib import Path
import gc
gc.enable()
import time
from contextlib import contextmanager

na_values=['XNA','XPA','XAP','nan']
true_values = ['Y','F','Yes','True']
false_values  = ['N','M','No','False']


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, categorical_columns=None, nan_as_category = True):
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def raw_train_test(base_dir):
    application_train = pd.read_csv(base_dir/'application_train.csv', 
                                    converters={'date':pd.to_datetime}, 
                                    na_values=na_values, true_values=true_values, false_values=false_values)
    application_test = pd.read_csv(base_dir/'application_test.csv',  
                                   converters={'date':pd.to_datetime}, 
                                   na_values=na_values, true_values=true_values, false_values=false_values)
    
    print('Initial test ')
    print(application_test.shape, len(set(application_test.columns)))
    print('Initial train ')
    print(application_train.shape, len(set(application_train.columns)))
    
    #for col in yes_no_cols:
    #    application_train[col] = application_train[col].astype(bool)
    #    application_test[col] = application_test[col].astype(bool)
        
    application_train['NAME_CONTRACT_TYPE'] = application_train['NAME_CONTRACT_TYPE'].map({'Cash loans':1, "Revolving loans":0})
    application_test['NAME_CONTRACT_TYPE'] = application_test['NAME_CONTRACT_TYPE'].map({'Cash loans':1, "Revolving loans":0})
    
        
    y = application_train[['TARGET','SK_ID_CURR']]
    del application_train['TARGET']
    
    application_train['_type'] = 'train'
    application_test['_type'] = 'test'
    
    data = pd.concat([application_train, application_test], ignore_index=True)
    
    # Days 365.243 values -> nan
    data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    yes_no_cols = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','EMERGENCYSTATE_MODE']
    
    for col in yes_no_cols:
        data[col] = data[col].map({True:1, False:0})
    
    return data, y

def prepare_train_test(base_dir, encode_categories ='binary'):

    data, y = raw_train_test(base_dir)
    inc_by_org = data[['AMT_INCOME_TOTAL', 
                   'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    
    docs = [_f for _f in data.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in data.columns 
            if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    #new_features
    data['NEW_CREDIT_TO_ANNUITY_RATIO'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']
    data['NEW_CREDIT_TO_GOODS_RATIO'] = data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']
    data['NEW_DOC_IND_KURT'] = data[docs].kurtosis(axis=1)
    data['NEW_LIVE_IND_SUM'] = data[live].sum(axis=1)
    data['NEW_INC_PER_CHLD'] = data['AMT_INCOME_TOTAL'] / (1 + data['CNT_CHILDREN'])
    data['NEW_INC_BY_ORG'] = data['ORGANIZATION_TYPE'].map(inc_by_org)
    data['NEW_EMPLOY_TO_BIRTH_RATIO'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    data['NEW_ANNUITY_TO_INCOME_RATIO'] = data['AMT_ANNUITY'] / (1 + data['AMT_INCOME_TOTAL'])
    data['NEW_SOURCES_PROD'] = data['EXT_SOURCE_1'] * data['EXT_SOURCE_2'] * data['EXT_SOURCE_3']
    data['NEW_EXT_SOURCES_MEAN'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    data['NEW_SCORES_STD'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    data['NEW_SCORES_STD'] = data['NEW_SCORES_STD'].fillna(data['NEW_SCORES_STD'].mean())
    data['NEW_CAR_TO_BIRTH_RATIO'] = data['OWN_CAR_AGE'] / data['DAYS_BIRTH']
    data['NEW_CAR_TO_EMPLOY_RATIO'] = data['OWN_CAR_AGE'] / data['DAYS_EMPLOYED']
    data['NEW_PHONE_TO_BIRTH_RATIO'] = data['DAYS_LAST_PHONE_CHANGE'] / data['DAYS_BIRTH']
    data['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = data['DAYS_LAST_PHONE_CHANGE'] / data['DAYS_EMPLOYED']
    data['NEW_CREDIT_TO_INCOME_RATIO'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    
    categorical_feats = [
        f for f in data.columns if (data[f].dtype == 'object') and (f != '_type')
    ]
    
    not_cat = [col for col in data.columns if col not in categorical_feats]
    
    
    if encode_categories =='OHE':
        df, cat_cols = one_hot_encoder(data, categorical_feats)
    elif encode_categories == 'binary':
        df = [utils.create_binary(data[[col]].fillna('__unknown__'),col) for col in categorical_feats]
        df =  pd.concat(df, axis=1)
        df =  pd.concat([data[not_cat],df], axis=1)
    else:
        raise
    
    
    application_train = df[df['_type'] =='train']
    application_test = df[df['_type'] =='test']
    
    del application_train['_type']
    del application_test['_type']
    del df
    gc.collect()
    
    print('Final test ')
    print(application_test.shape, len(set(application_test.columns)))
    print('Final train ')
    print(application_train.shape, len(set(application_train.columns)))
    return application_train.reset_index(drop=True) , application_test.reset_index(drop=True), y, categorical_feats


def raw_previous_application(base_dir, unique_curr):
    previous_application = pd.read_csv(base_dir/'previous_application.csv', na_values=na_values,
                                       true_values=true_values, false_values=false_values)
    print(previous_application.shape)
    print('remove Canceled, it seems that there is always a new application after those canceled')
    previous_application = previous_application[previous_application.NAME_CONTRACT_STATUS != 'Canceled'] 
    print(previous_application.shape)

    print('Only in test/train set')
    previous_application = previous_application[previous_application.SK_ID_CURR.isin(unique_curr)]
    print('Only in test/train set, ',previous_application.shape)
    
    # Days 365.243 values -> nan
    previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    return previous_application


def prepare_previous_application(base_dir, unique_curr):

    previous_application = raw_previous_application(base_dir, unique_curr)


    previous_application['APP_CREDIT_PERC'] = previous_application['AMT_APPLICATION'] / previous_application['AMT_CREDIT']


    previous_application, cat_cols = one_hot_encoder(previous_application, 
                                                     nan_as_category= True)


    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = previous_application.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e1 + "_" + e2.upper() for e1,e2 in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = previous_application[previous_application['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left')
    # Previous Applications: Refused Applications - only numerical features
    refused = previous_application[previous_application['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left')
    del refused, refused_agg, approved, approved_agg, previous_application
    gc.collect()
        
    
    #prev_count = previous_application[['SK_ID_CURR', 'SK_ID_PREV',
    #                                   'NAME_CONTRACT_STATUS']].groupby(['SK_ID_CURR',
    #                                                                     'NAME_CONTRACT_STATUS']).count()
    #prev_count =prev_count.unstack(level=1,fill_value =0)
    #prev_count.columns = ['prev_approved','prev_refused','prev_unused']
    #prev_count['prev_total'] = prev_count.sum(axis = 1)
    
    
    print('Final, ')
    print(prev_agg.shape, len(set(prev_agg.columns)))
    return prev_agg



def prepare_bureau(base_dir):
    bureau_balance = pd.read_csv(base_dir/'bureau_balance.csv', na_values=na_values,
                                 true_values=true_values, false_values=false_values)
    
    bureau = pd.read_csv(base_dir/'bureau.csv', na_values=na_values,
                         true_values=true_values, false_values=false_values)
    
    bureau_balance, bb_cat = one_hot_encoder(bureau_balance)
    bureau, bureau_cat = one_hot_encoder(bureau)
    
    print('bureau_balance shape : ', bureau_balance.shape, '- unique', bureau_balance.SK_ID_BUREAU.nunique())

    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() 
                               for e in bb_agg.columns.tolist()])
    bureau = bureau.merge(bb_agg, how='left', right_index=True, left_on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb_aggregations, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: 
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat: 
        cat_aggregations[cat + "_MEAN"] = ['mean']
        
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() 
                                   for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() 
                                   for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() 
                                   for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg
    
# Preprocess POS_CASH_balance.csv
def prepare_pos_cash(base_dir):
    pos = pd.read_csv(base_dir/'POS_CASH_balance.csv', na_values=na_values,
                          true_values=true_values, false_values=false_values)
    
    print('pos_cash shape : ', pos.shape)

    pos['DIFF_CTN_INSTALLMENT'] = pos['CNT_INSTALMENT_FUTURE'] - pos['CNT_INSTALMENT'] 
    pos, cat_cols = one_hot_encoder(pos)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'DIFF_CTN_INSTALLMENT' : ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() 
                                for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    print('pos_cash shape : ', pos_agg.shape)
    return pos_agg



def raw_installments_payments(base_dir,unique_curr):
    ins = pd.read_csv(base_dir/'installments_payments.csv', na_values=na_values,
                      true_values=true_values, false_values=false_values)
    ins = ins[ins.SK_ID_CURR.isin(unique_curr)]
    
    return ins


# Preprocess installments_payments + previous_application
def prepare_open_application(base_dir,unique_curr):
    

    previous_application = raw_previous_application(base_dir, unique_curr)
    previous_application= previous_application[previous_application.DAYS_TERMINATION.isna()]
    previous_application= previous_application[previous_application.NAME_CONTRACT_STATUS == 'Approved']
    previous_application= previous_application[previous_application.NAME_CONTRACT_TYPE != 'Revolving loans']
    print('Shape open aplication',previous_application.shape)
    
    unique_prev = previous_application.SK_ID_PREV.unique()
    
    ins = raw_installments_payments(base_dir,unique_curr)
    ins = ins[ins.SK_ID_PREV.isin(unique_prev)]
    
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    # Features: Perform aggregations
    aggregations = {
        'DPD': ['max', 'mean', 'sum','min','std' ],
        'DBD': ['max', 'mean', 'sum','min','std'],
        'PAYMENT_PERC': ['max', 'mean', 'sum','min','std'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum','min','std'],
        'AMT_INSTALMENT': ['sum'],
        'AMT_PAYMENT': ['sum'],
        'NUM_INSTALMENT_NUMBER':['max']
    }

    ins_agg = ins.groupby('SK_ID_PREV').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_PREV_' + e[0] + "_" + e[1].upper() 
                                for e in ins_agg.columns.tolist()])
    
    
    
    ins_agg= ins_agg.join(previous_application.set_index('SK_ID_PREV')[['SK_ID_CURR',
                                                                        'AMT_CREDIT', 'CNT_PAYMENT']])
    
    ins_agg['DIFF_DUE_PREV'] = ins_agg['INSTAL_PREV_AMT_PAYMENT_SUM'] - ins_agg['AMT_CREDIT']
    ins_agg['RATIO_DUE_PREV'] = ins_agg['INSTAL_PREV_AMT_PAYMENT_SUM']/ins_agg['AMT_CREDIT']
    ins_agg['RATIO_NUM_INST'] = ins_agg['INSTAL_PREV_NUM_INSTALMENT_NUMBER_MAX']/ins_agg['CNT_PAYMENT']
    ins_agg['DIFF_NUM_INST'] = ins_agg['CNT_PAYMENT'] - ins_agg['INSTAL_PREV_NUM_INSTALMENT_NUMBER_MAX']
    print('Shape open aplication',previous_application.shape)
    
    ins_agg.reset_index(drop=True, inplace=True)
    print(ins_agg.SK_ID_CURR.nunique(), ins_agg.shape)
    ins_agg = ins_agg.groupby('SK_ID_CURR').mean()
    print('Shape open aplication',ins_agg.shape)
    return ins_agg

# Preprocess installments_payments.csv
def prepare_installments_payments(base_dir,unique_curr):

    ins = raw_installments_payments(base_dir,unique_curr)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum','min','std' ],
        'DBD': ['max', 'mean', 'sum','min','std'],
        'PAYMENT_PERC': [ 'max','mean',  'var','min','std'],
        'PAYMENT_DIFF': [ 'max','mean', 'var','min','std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum','min','std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum','std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum','std']
    }

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() 
                                for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    print('installments shape : ', ins_agg.shape)
    return ins_agg

# Preprocess credit_card_balance.csv
def prepare_credit_card_balance(base_dir):
    cc = pd.read_csv(base_dir/'credit_card_balance.csv', na_values=na_values,
                     true_values=true_values, false_values=false_values)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() 
                               for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    print('credit card shape : ', cc_agg.shape)
    return cc_agg