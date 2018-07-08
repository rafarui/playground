import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

_columns_description = pd.read_csv('/home/ubuntu//off/home-credit-default-risk/data/'+'HomeCredit_columns_description.csv', encoding='latin')

def cols_desc(col):
    desc = _columns_description.set_index('Row').loc[col,['Description','Table']]
    if not isinstance(desc,pd.Series):
        for i,col in desc.iterrows():
            print(f"{col.Table[:-4]} - {col.Description}")
    else:
        print(f"{desc.Table[:-4]} - {desc.Description}")
        
        
def create_binary(df, col):
    assert isinstance(col,str)

    uniques = pd.DataFrame(df[col].unique(), columns=[col])
    enc = ce.BinaryEncoder(verbose=1, cols=[col])
    uniques = pd.concat([uniques,enc.fit_transform(uniques)], axis=1)
    return df.merge(uniques, how = 'left', on = col)


def display_importances(feature_importance_df_, save_result = False):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(
            by="importance", ascending=False)[:50].index

    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    if save_result:
        plt.savefig('lgbm_importances-01.png')


def display_roc_curve(y_, oof_preds_, folds_idx_, save_result = False):
    # Plot ROC curves
    plt.figure(figsize=(6, 6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    plt.plot(
        [0, 1], [0, 1],
        linestyle='--',
        lw=2,
        color='r',
        label='Luck',
        alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(
        fpr,
        tpr,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_result:
        plt.savefig('roc_curve-01.png')


def display_precision_recall(y_, oof_preds_, folds_idx_, save_result = False):
    # Plot ROC curves
    plt.figure(figsize=(6, 6))

    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(
        precision,
        recall,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Recall / Precision')
    plt.legend(loc="best")
    plt.tight_layout()
    if save_result:
        plt.savefig('recall_precision_curve-01.png')
    
