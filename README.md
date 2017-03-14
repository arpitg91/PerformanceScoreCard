# PerformanceScoreCard

Python script to automate the performance of scorecard. The script creates the charts for the following charts and profiles:
    ROC, AUC, KS, Score@KS
    Cumulative Gains Chart
    Precision Recall Chart
    Deciles Profile

Functions:
    The fuctions can be called from another python script to get individual performances.
        ROC: get_roc(df,score,target,title,plot=1)
        Gains Chart: get_cum_gains(df,score,target,title)
        Precision, Recall: get_precision_recall(df,score,target,title)
        Deciles Profile: get_deciles_analysis(df,score,target)

Arguments:
  --ifile IFILE    Input file
  --d D            Delimiter.     Default: Comma
  --score SCORE    Score  Column. Default: score
  --target TARGET  Target Column. Default: target
  --tag TAG        Target Column. Default: performance
  --title TITLE    Target Column. Default: None
