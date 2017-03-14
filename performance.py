import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib import use
use("Agg")
import matplotlib.pyplot as plt

def get_roc(df,score,target):
    fpr, tpr, thresholds = roc_curve(df[target], df[score])
    results = pd.DataFrame(np.reshape(np.hstack([tpr,fpr,thresholds]),(fpr.shape[0],3)),columns = ["tpr","fpr","threshholds"])
    results.to_csv("auc_curve.csv",index=False)
    print results.describe()
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    plt.savefig("myfig1.png")
# Plot ROC curve

def get_precision_recall(df,score,target):
    precision, recall, _ = precision_recall_curve(df[target], df[score])
    roc_pr = average_precision_score(df[target], df[score])
    print "Area under the Precision-Recall curve : %f" % roc_pr
    # Plot ROC curve
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve (AUC = %0.2f)' % roc_pr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.show()
    plt.savefig("myfig2.png")

#ROC, AUC, KS, GINI, Bivariate, Precision Recall, Sensitivity and specificity

def get_ks(df,score,target):
    fpr, tpr, thresholds = roc_curve(df[target], df[score])
    roc_auc = auc(fpr, tpr)
    data = df.copy(deep=True)
    ascending = (data[score].corr(data[target]) < 0)
    dsd = data.groupby(score)[target].agg(["sum","count" ])
    dsd.sort_index( ascending = ascending, inplace=True)
    dsd["ones"] = dsd["sum"]
    dsd["zeros"] = dsd["count"] - dsd["ones"]
    dsd["TP"] = dsd["ones"].cumsum()
    dsd["FP"] = dsd["zeros"].cumsum()
    dsd["FN"] = dsd["ones"].sum() - dsd["ones"]
    dsd["TN"] = dsd["zeros"].sum() - dsd["zeros"]
    dsd["TPR"] = dsd["TP"] /dsd["ones"].sum()
    dsd["FPR"] = dsd["FP"] /dsd["zeros"].sum()
    dsd["FNR"] = dsd["FN"] /dsd["ones"].sum()
    dsd["TNR"] = dsd["TN"] /dsd["zeros"].sum()
    KS = np.max(dsd["TPR"] - dsd["FPR"])
    KS_Score = dsd.index.values[np.argmax(dsd["TPR"] - dsd["FPR"])]
    TPR_KS = dsd["TPR"].values[np.argmax(dsd["TPR"] - dsd["FPR"])]
    FPR_KS = dsd["FPR"].values[np.argmax(dsd["TPR"] - dsd["FPR"])]
    print "AUC = %0.3f, KS = %0.3f at score %.3f, TPR = %0.3f, FPR = %0.3f "%(roc_auc, KS, KS_Score, TPR_KS, FPR_KS)
    plt.savefig("myfig3.png")
    plt.clf()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.plot(dsd["FPR"], dsd["TPR"], label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.annotate ('', [FPR_KS,FPR_KS], [FPR_KS, TPR_KS], arrowprops={'arrowstyle':'<->'})
    plt.annotate('KS = %0.2f'%KS, xy=(FPR_KS+ 0.005, (3*TPR_KS + 2*FPR_KS)/5), xycoords = 'data',xytext = (3, 0), textcoords = 'offset points')
    plt.legend(loc="lower right")
    plt.figure()
    plt.show()
    plt.savefig("myfig4.png")

def get_cum_gains(df,score,target):
    data = df.copy(deep=True)
    ascending = (data[score].corr(data[target]) < 0)
    dsd = data.groupby(score)[target].agg(["sum","count" ])
    dsd.sort_index( ascending = ascending, inplace=True)
    dsd["ones"] = dsd["sum"]
    dsd["zeros"] = dsd["count"] - dsd["ones"]
    dsd["TP"] = dsd["ones"].cumsum()
    dsd["FP"] = dsd["zeros"].cumsum()
    dsd["FN"] = dsd["ones"].sum() - dsd["ones"]
    dsd["TN"] = dsd["zeros"].sum() - dsd["zeros"]
    dsd["TOT"] = dsd["count"].cumsum()
    dsd["TPR"] = dsd["TP"] /np.float(dsd["ones"].sum())
    dsd["FPR"] = dsd["FP"] /np.float(dsd["zeros"].sum())
    dsd["FNR"] = dsd["FN"] /np.float(dsd["ones"].sum())
    dsd["TNR"] = dsd["TN"] /np.float(dsd["zeros"].sum())
    dsd["TOTR"] = dsd["TOT"] /np.float(dsd["count"].sum())
    dsd["LIFT"] = dsd["TPR"] /dsd["TOTR"]
    plt.clf()
    plt.ylabel('Cumulative Gains')
    plt.xlabel('Population Rate')
    plt.title('Cumulative Gains Chart')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])  
    plt.plot(dsd["FPR"], dsd["TPR"])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.figure()
    plt.show()
    plt.clf()
    plt.ylabel('Lift')
    plt.xlabel('Population Rate')
    plt.title('Lift Curve')
    plt.ylim([0.99, dsd["LIFT"].max()])
    plt.xlim([0.0, 1.0])  
    plt.plot(dsd["FPR"], dsd["LIFT"])
    plt.plot([0, 1], [1, 1], 'k--')
    plt.figure()
    plt.show()
    plt.savefig("myfig4.png")

if __name__ == "__main__":
    mean = np.array([1,1])
    cov = np.array([[1,0.3],[0.3,1]])
    score_card = np.random.multivariate_normal(mean,cov,10000)
    score_card = pd.DataFrame(np.concatenate((score_card,np.reshape(1*(score_card[:,1] > 0.999),(10000,1))),axis=1),columns=["Score","Raw Target","Target"])
    
    score = "Score"
    target = "Target"
    
    ascending = (score_card[score].corr(score_card[target]) < 0)
    score_card.sort(columns = score, ascending = ascending)
    #get_roc(score_card,score,target)
    get_precision_recall(score_card,score,target)
    get_ks(score_card,score,target)
    get_cum_gains(score_card,score,target)
    