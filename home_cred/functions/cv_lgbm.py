import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from lightgbm import LGBMClassifier



def train_model_LGBM(data_, y_ ,test_, folds_, ignore_cols_ = None ):

    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])

    feature_importance_df = pd.DataFrame()

    if ignore_cols_ is not None:
        feats = [f for f in data_.columns if f not in ignore_cols_]
    else:
        feats = list(data_.columns)

    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]

        clf = LGBMClassifier(
            #n_jobs=20,
            #random_state=42,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.95,
            subsample=0.85,
            max_depth=8,
            reg_alpha=0.0415,
            reg_lambda=0.0735,
            min_split_gain=0.022,
            min_child_weight=40,
            silent=-1,
            verbose=50, )

        clf.fit(
            trn_x,
            trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric='auc',
            verbose=50,
            early_stopping_rounds=100 
        )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats],
            num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' %
              (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))


    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    test_['TARGET'] = sub_preds
    
    ids = data_.SK_ID_CURR
    df_oof_preds = pd.DataFrame({'SK_ID_CURR':ids, 'TARGET':y_, 'PREDICTION':oof_preds})
    df_oof_preds = df_oof_preds[['SK_ID_CURR', 'TARGET', 'PREDICTION']]

    return oof_preds, df_oof_preds, test_[['SK_ID_CURR', 'TARGET'
                             ]], feature_importance_df, roc_auc_score(y_, oof_preds)
        
