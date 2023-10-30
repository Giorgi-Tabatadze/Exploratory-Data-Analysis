def unistats(df):
  import pandas as pd
  output_df = pd.DataFrame(columns=["Count", "Missing", "Unique", "Dtype", "Numeric", "Mode" ,"Mean",  "Min", "25", "Median", "75%", "Max", "Std", "Skew", "Kurt"])
  for col in df:
    if pd.api.types.is_numeric_dtype(df[col]):
      output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].dtype, pd.api.types.is_numeric_dtype(df[col]), df[col].mode().values[0], 
      df[col].mean(), df[col].min(), df[col].quantile(0.25), df[col].median(), df[col].quantile(0.25), df[col].min(),df[col].std(), df[col].skew(),df[col].kurt()]
    else:
      output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].dtype, pd.api.types.is_numeric_dtype(df[col]),
      df[col].mode().values[0] ,"-", "-", "-","-", "-", "-","-", "-","-" ]
  return output_df.sort_values(by=["Numeric","Skew", "Unique"], ascending=False)



def anova(df, feature, label): 
  import pandas as pd
  import numpy as np
  from scipy import stats

  groups = df[feature].unique()

  df_grouped = df.groupby(feature)
  group_labels = []

  for g in groups:
    g_list = df_grouped.get_group(g)
    group_labels.append(g_list[label])

  return stats.f_oneway(*group_labels)



def bivstats(df, label):
  from scipy import stats
  import pandas as pd
  import numpy as np

  output_df = pd.DataFrame(columns=["Stat", "+/-", "Effect size", "p-value"])

  for col in df:
    if not col == label:
      if df[col].isnull().sum() == 0:
        if pd.api.types.is_numeric_dtype(df[col]):
          r, p = stats.pearsonr(df[label], df[col])
          output_df.loc[col] = ["r", np.sign(r), abs(round(r, 3)),  round(p,6)]
          scatter(df[col], df[label])
        else:
          F, p = anova(df[[col, label]], col, label)
          output_df.loc[col] = ["F", np.sign(F), round(F, 3),  round(p,6)]
          bar_chart(df, col, label)

      else:
          output_df.loc[col] = [np.nan, np.nan, np.nan, np.nan]

  return output_df.sort_values(by=["Stat", "Effect size"], ascending=[False, False])


def heteroscedasticity(df, feature, label):
  from statsmodels.stats.diagnostic import het_breuschpagan
  from statsmodels.stats.diagnostic import het_white
  import pandas as pd
  import statsmodels.api as sm
  from statsmodels.formula.api import ols

  model = ols(formula=f"{label} ~ Q('{feature}')", data=df).fit()
  output_df = pd.DataFrame(columns=["LM stat", "LM p-value", "F-stat", "F p-value"])

  try: 
    white_test = het_white(model.resid, model.model.exog)
    output_df.loc["White"] = white_test
  except:
    print("Unable to run White test of heteroscedascisity")
  
  bp_test = het_breuschpagan(model.resid, model.model.exog)
  output_df.loc["Br-Pa"] = bp_test
  
  return output_df.round(3)



def scatter(feature, label):
  import seaborn as sns
  from scipy import stats
  import matplotlib.pyplot as plt
  import pandas as pd


  sns.set(color_codes=True)

  m,b,r,p,err = stats.linregress(feature,label)

  textstr = "y = " + str(round(m,2)) + "x + " + str(round(b,2)) + "\n"
  textstr += "r2 = " + str(round(r**2,2)) + "\n"
  textstr += "p = " + str(round(p,2)) + "\n"
  textstr += str(feature.name) + " skew = " + str(round(feature.skew(), 2)) + "\n"
  textstr += str(label.name) + " skew = " + str(round(label.skew(), 2)) + "\n"
  textstr += str(heteroscedasticity(pd.DataFrame(label).join(pd.DataFrame(feature)), feature.name, label.name)) + "\n"


  ax = sns.jointplot(x=feature, y=label, kind="reg")
  ax.fig.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()

def bar_chart(df, feature, label):
  import pandas as pd
  from scipy import stats
  from matplotlib import pyplot as plt
  import seaborn as sns

  oneway = anova(df, feature, label)

  unique_groups = df[feature].unique()
  ttests = []

  for i, group in enumerate(unique_groups):
    for i2, group_2 in enumerate(unique_groups):
      if i2 > i:
        type_1= df[df[feature] == group]
        type_2 = df[df[feature] == group_2]
        t, p = stats.ttest_ind(type_1[label], type_2[label])
        ttests.append([group, group_2, t.round(4), p.round(4) ])

  if len(ttests) > 0:
    p_threshhold = 0.05 / len(ttests)
  else:
    p_threshhold = 0.05


  textstr = f"      ANOVA\n"
  textstr += f"F: {oneway[0].round(2)}\n"
  textstr += f"p-value: {oneway[1].round(2)}\n"
  textstr += f"Sig. comparisions (Bonferrioni)\n"

  for ttest in ttests:
    if ttest[3] <= p_threshhold:
      textstr += ttest[0] + "-" + ttest[1] + ": t=" + str(ttest[2]) + ", p=" + str(ttest[3]) + "\n"
  


  ax = sns.barplot(x= df[feature], y=df[label])


  ax.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()



def mlr_prepare(df):
  import numpy as np 
  import pandas as pd
  from sklearn.preprocessing import MinMaxScaler

  for col in df: 
    if not pd.api.types.is_any_real_numeric_dtype(df[col]):
      df = df.join(pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int))
  
  df = df.select_dtypes(int)
  df_minmax = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

  return df_minmax

def mlr(df, label):
  import statsmodels.api as sm

  y = df[label]
  X = df.drop(columns=[label]).assign(const=1)
  
  results = sm.OLS(y, X).fit()
  return results

def mlr_feature_df(results): 
  import pandas as pd

  df_features = pd.DataFrame({'coef':results.params, "t":abs(results.tvalues),"p":results.pvalues})
  df_features.drop(labels= ['const'], inplace=True)
  df_features = df_features.sort_values(by=['t', 'p'])
  return df_features

def mlr_fit(results, actual, roundto=10):
  import numpy as np

  df_features = mlr_feature_df(results)
  residuals = np.array(actual) - np.array(results.fittedvalues)
  rmse = np.sqrt(sum((residuals**2)/len(actual)))
  mae = np.mean(abs(np.array(actual) - np.array(results.fittedvalues)))
  fit_stats = [round(results.rsquared, roundto), round(results.rsquared_adj, roundto), 
               round(results.rsquared - results.rsquared_adj, roundto), round(rmse, roundto), 
               round(mae, roundto), [df_features.index.values]]
  
  return fit_stats

def mlr_step(df, label, min=2):
  import pandas as pd

  df_models = pd.DataFrame(columns=["R2", "R2a", "diff", "RMSE", "MAE", "features"])
  df = mlr_prepare(df)
  results = mlr(df, label)
  df_models.loc[str(len(results.params))] = mlr_fit(results, df[label], 10)
  df_features = mlr_feature_df(results)

  while len(results.params) >= min:
    df = df.drop(columns=[df_features.index[0]])
    results = mlr(df, label)
    df_features = mlr_feature_df(results)
    df_models.loc[len(results.params)] = mlr_fit(results, df[label], 10)
  
  df_models.to_excel("./" + label + ".xlsx")
  df_models.to_csv("./" + label + ".csv")

  df_models.drop(columns=["features"], inplace=True)

  return df_models
