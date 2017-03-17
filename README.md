# PerformanceScoreCard

Python script to automate the performance of scorecard. The script creates the charts for the following charts and profiles:

- ROC, AUC, KS, Score@KS
- Cumulative Gains Chart
- Precision Recall Chart
- Deciles Profile
- Confusion Matrix
- Histogram of classes
- Quadtractic weighted kappa
- Linear weighted kappa
- Kappa
- Mean quadratic weighted kappa
- Weighted mean quadratic weighted kappa
    
## Functions
The fuctions can be called from another python script to get individual performances.
    
> get_roc(df,score,target,title,plot=1)
> get_cum_gains(df,score,target,title)
> get_precision_recall(df,score,target,title)
> get_deciles_analysis(df,score,target)
> confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
> histogram(ratings, min_rating=None, max_rating=None):
> quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
> linear_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
> kappa(rater_a, rater_b, min_rating=None, max_rating=None):
> mean_quadratic_weighted_kappa(kappas, weights=None):
> weighted_mean_quadratic_weighted_kappa(solution, submission):

