import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64
import time
import numpy as np

def get_combined_dataframe():
  target_name = ["Meat_consume (raw)"]
  feature_names = ["GDP_per_capita (raw)", "CO2 (raw)", "Urban_population (raw)", "Income_per_capita (raw)"]
  years_str = [str(year) for year in range(2000,2015)]
  years_int = [year for year in range(2000,2015)]
  years_str.extend(["Country Name"])
  years_int.extend(["Country Name"])
  df_target_raw = pd.read_excel("./app/static/allraw.xlsx", sheet_name  = target_name[0])
  df_target_raw = df_target_raw.loc[
                            (df_target_raw["Year"]>=2000) & 
                            (df_target_raw["Year"]<=2014),
                            ["Entity", "Year", "Meat food supply quantity (kg/capita/yr) (FAO, 2020)"]
                            ]
  df_target_raw = df_target_raw.set_index(["Entity","Year"])
  # we set the index to sort properly
  # the index will be resetted later to allow for the column to be inserted proprly
  df_target_raw =df_target_raw.sort_index()
  df_features_raw = pd.DataFrame()
  for indicator in feature_names:
    df_indicator = pd.read_excel("./app/static/allraw.xlsx", sheet_name = indicator, header=4)
    try:
      df_indicator = df_indicator.loc[:,years_str]
    except:
      df_indicator = df_indicator.loc[:,years_int]
    df_indicator.columns = [int(col) if col not in ["Country Name","Indicator Name"] else col for col in df_indicator.columns]
    df_indicator = df_indicator.set_index(["Country Name"])
    # stack to change the df into 1 column
    df_indicator = df_indicator.stack().reset_index()
    df_indicator = df_indicator.rename(columns={"level_1": "Year", 0: indicator[:-6]})
    # here we set again to concat properly
    df_indicator = df_indicator.set_index(["Country Name","Year"])
    if indicator == feature_names[0]:
      df_features_raw = df_indicator
    else:
      df_features_raw = pd.concat([df_features_raw,df_indicator], axis=1)
  df_features_raw = df_features_raw.reset_index()
  df_features_raw = df_features_raw.set_index("Country Name")
  df_target_raw.index.names = ['Country Name', 'Year']
  df_combined = df_features_raw.join(df_target_raw, on = ['Country Name', 'Year'])
  df_combined = df_combined.reset_index()
  df_combined = df_combined.rename(columns={"Meat food supply quantity (kg/capita/yr) (FAO, 2020)":"Meat consumed"})
  return(df_combined)

def get_country(countries):
  df = get_combined_dataframe()
  return df.loc[df["Country Name"].isin(countries), :]

def get_unique_countries():
  return(get_combined_dataframe()["Country Name"].unique())

def draw_plots(countries):
  df = get_country(countries)
  columns = ["GDP_per_capita",	"CO2", "Income_per_capita",	"Urban_population"]
  plots = []
  for variable in columns:
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(variable)
    axis.set_xlabel(variable)
    axis.set_ylabel("Meat Consumed")
    for country in countries:
      df_country = df[df["Country Name"]==country]
      axis.scatter(df_country[variable], df_country["Meat consumed"], label = country)
    axis.legend(loc="lower right")
    # Convert plot to PNG image
    # following code is from https://gitlab.com/snippets/1924163
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    plots.append(pngImageB64String)
  return(plots)

def get_features_targets(df, feature_names, target_names):
    df_feature = df[feature_names]
    df_target = df[target_names]
    return df_feature, df_target

def normalize_z(df, mean=None, std= None):
  if type(mean) == pd.Series:
    z = (df - mean)/std
    return z
  else:
    mean = df.mean(axis = 0)
    std = df.std(axis = 0)
    z = (df - mean)/std
    return z

def prepare_feature(df_feature):
    out = df_feature.to_numpy()
    nrows = np.shape(out)[0]
    out = np.concatenate((np.ones((nrows,1)), out), axis = 1)
    return out

def prepare_target(df_target):
    out = df_target.to_numpy()
    return out

def calc_linear(X, beta):
    return np.matmul(X, beta)

def predict(df_feature, mean=None, std=None):
    beta_init = [[27.61897951], [ 3.69675744], [ 9.06746831], [ 4.36772633], [ 4.15094336]]
    beta_improved = [[27.61897951], [ 5.08183451], [ 6.31857727], [ 5.8035173 ], [ 3.95143081]]
    df_feature = normalize_z(df_feature, mean, std)
    np_feature = prepare_feature(df_feature)
    pred_init = calc_linear(np_feature, beta_init)
    pred_improved = calc_linear(np_feature, beta_improved)
    return (pred_init,pred_improved)

def draw_plots_for_model(countries, mean=None, std=None):
  df = get_country(countries)
  columns = ["GDP_per_capita",	"CO2", "Income_per_capita",	"Urban_population"]
  plots = []
  df_features, df_target = get_features_targets(df ,columns,["Meat consumed"])
  pred_init,pred_improved = predict(df_features, mean, std)
  for variable in columns:
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title(variable)
    axis.set_xlabel(variable)
    axis.set_ylabel("Meat Consumed")
    axis.scatter(df_features[variable], df_target, label = 'Actual')
    axis.scatter(df_features[variable], pred_init, label = 'Predicted-initial model')
    axis.scatter(df_features[variable], pred_improved, label = 'Predicted-improved model')
    axis.legend(loc="lower right")
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    plots.append(pngImageB64String)
  return(plots)

def r2_score(y, ypred):
    y_bar = y.mean()
    ss_tot = np.sum((y-y_bar)**2)
    ss_res = np.sum((y-ypred)**2)
    return 1 - (ss_res/ss_tot)

def adjusted_r2(y, ypred, independent_variables):
    n = y.shape[0]
    return(1-(1-r2_score(y, ypred)) * (n-1) / (n-1-independent_variables))

def r2_both(countries, mean=None, std=None):
  df = get_country(countries)
  columns = ["GDP_per_capita",	"CO2", "Income_per_capita",	"Urban_population"]
  df_features, df_target = get_features_targets(df ,columns,["Meat consumed"])
  pred = predict(df_features, mean, std)
  return([r2_score(prepare_target(df_target),pred[0]),adjusted_r2(prepare_target(df_target),pred[0],4),r2_score(prepare_target(df_target),pred[1]),adjusted_r2(prepare_target(df_target),pred[1],4)])

import math
def split_data(df_feature, df_target, random_state=None, test_size=0.5):
    numrows = len(df_feature)
    trainlen = math.ceil(numrows*(1-test_size))
    testlen = numrows-trainlen
    
    np.random.seed(random_state)
    test_index = np.random.choice(df_feature.index, testlen, replace = False)
    train_index = np.random.choice(df_feature.index, trainlen, replace = False)
    
    df_feature_test = df_feature.loc[test_index, df_feature.columns]
    df_feature_train = df_feature.drop(df_feature_test.index)
    df_target_test = df_target.loc[test_index, df_target.columns]
    df_target_train = df_target.drop(df_target_test.index)
    
    return df_feature_train, df_feature_test, df_target_train, df_target_test

def get_mean_std(df):
    mean = df.mean(axis = 0)
    std = df.std(axis = 0)
    return(mean,std)

def get_df_features_train():
  df = get_combined_dataframe()
  selected_countries = ["Georgia" ,"Kenya" ,"Korea, Rep." ,"Liberia" ,'Madagascar' ,"Maldives" ,"Morocco" ,"Nepal" ,"Senegal" ,"South Africa" ,"Tanzania" ,"Thailand" ,"Uganda" ,"Ukraine" ,"United Kingdom"]
  df_countries = df.loc[df["Country Name"].isin(selected_countries),:]
  columns = ["GDP_per_capita", "CO2", "Income_per_capita", "Urban_population"]
  df_features, df_target = get_features_targets(df_countries ,columns,["Meat consumed"])
  df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target,random_state=100,test_size=0.3)
  return(df_features_train)
