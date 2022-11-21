import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64


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
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    plots.append(pngImageB64String)
  return(plots)

# get_unique_countries()
# (get_country("Georgia"))
# draw_plots(["Georgia"])