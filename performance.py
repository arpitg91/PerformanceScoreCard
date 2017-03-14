import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib import use
use("Agg")
import matplotlib.pyplot as plt
import csv,argparse
import warnings
warnings.filterwarnings("ignore")
# For plotting inline in notebook
# %matplotlib inline 
def get_roc(df,score,target,title,plot=1):
    df1 = df[[score,target]].dropna()
    fpr, tpr, thresholds = roc_curve(df1[target], df1[score])
    ks=np.abs(tpr-fpr)
    if plot==1:
    # Plot ROC curve
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label='AUC=%0.2f KS=%0.2f' %(auc(fpr, tpr),ks.max()))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(b=True, which='both', color='0.65',linestyle='-')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title+'Receiver Operating Characteristic')
        plt.legend(loc="lower right")
    return auc(fpr, tpr),np.max(np.abs(tpr-fpr)),thresholds[ks.argmax()]
def get_cum_gains(df,score,target,title):
    df1 = df[[score,target]].dropna()
    fpr, tpr, thresholds = roc_curve(df1[target], df1[score])
    ppr=(tpr*df[target].sum()+fpr*(df[target].count()-df[target].sum()))/df[target].count()
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(ppr, tpr, label='')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
    plt.xlabel('%Population')
    plt.ylabel('%Target')
    plt.title(title+'Cumulative Gains Chart')
    plt.legend(loc="lower right")

    plt.subplot(1,2,2)
    plt.plot(ppr, tpr/ppr, label='')
    plt.plot([0, 1], [1, 1], 'k--')
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
    plt.xlabel('%Population')
    plt.ylabel('Lift')
    plt.title(title+'Lift Curve')
def get_precision_recall(df,score,target,title):
    precision, recall, _ = precision_recall_curve(df[target], df[score])
    roc_pr = average_precision_score(df[target], df[score])
    # Plot ROC curve
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label='Precision-Recall curve (AUC = %0.2f)' % roc_pr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title+"Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
def get_deciles_analysis(df,score,target):
    df1 = df[[score,target]].dropna()
    _,bins = pd.qcut(df1[score],10,retbins=True)
    bins[0] -= 0.001
    bins[-1] += 0.001
    bins_labels = ['%d.(%0.2f,%0.2f]'%(9-x[0],x[1][0],x[1][1]) for x in enumerate(zip(bins[:-1],bins[1:]))]
    bins_labels[0] = bins_labels[0].replace('(','[')
    df1['Decile']=pd.cut(df1[score],bins=bins,labels=bins_labels)
    df1['Population']=1
    df1['Zeros']=1-df1[target]
    df1['Ones']=df1[target]
    summary=df1.groupby(['Decile'])[['Ones','Zeros','Population']].sum()
    summary=summary.sort_index(ascending=False)
    summary['TargetRate']=summary['Ones']/summary['Population']
    summary['CumulativeTargetRate']=summary['Ones'].cumsum()/summary['Population'].cumsum()
    summary['TargetsCaptured']=summary['Ones'].cumsum()/summary['Ones'].sum()
    return summary
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifile", help="Input file")
    parser.add_argument("--d", help="Delimiter. Default: Comma")
    parser.add_argument("--score", help="Score Column. Default: score")
    parser.add_argument("--target", help="Target Column. Default: target")
    parser.add_argument("--tag", help="Target Column. Default: performance")
    parser.add_argument("--title", help="Target Column. Default: None")
    
    args = parser.parse_args()
    infile = args.ifile
    score = args.score if args.score else 'score'
    target = args.target if args.target else 'target'
    tag = args.tag if args.tag else 'performance'
    delimiter = args.d if args.d else ','
    title = args.title+':' if args.title else ''
    
    score_card=pd.read_csv(infile,delimiter=delimiter,usecols=[score,target])
    auc,ks,ks_score=get_roc(score_card,score,target,title)
    plt.savefig('%s_roc.png'%tag)
    get_cum_gains(score_card,score,target,title)
    plt.savefig('%s_cum_gains.png'%tag)
    get_precision_recall(score_card,score,target,title)
    plt.savefig('%s_precision_recall.png'%tag)
    decile_analysis=get_deciles_analysis(score_card,score,target)
    decile_analysis.to_csv('%s_decile_analysis.csv'%tag)
    pd.Series([auc,ks,ks_score],index=['auc','ks','ks_score']).to_csv('%s_summary.csv'%tag)
    