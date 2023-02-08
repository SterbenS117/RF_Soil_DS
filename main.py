import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ['CUDA_FORCE_PTX_JIT'] = '1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import xarray as xr
#import tensorflow_datasets as tfds
import tensorflow as tf
import glob
import plotly.express as px
from contextlib import redirect_stdout
import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from datetime import datetime
#import tensorflow_decision_forests as tfdf
import tensorflow_decision_forests as forest
from sklearn.metrics import mean_squared_error
#import shap
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('213213213333333333333333333333777777777777777777777777777777777777777777777')
def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

index1 = (['20121024','20121027','20121030','20130617','20130716','20130719','20130723','20130927','20140416','20140418','20140424','20140708','20140711','20140715','20141014','20141017','20141021','20150416','20150420','20150807','20150811','20150814'])

def scale_datasets(x):
    standard_scaler = StandardScaler().fit(x.values)
    x_scaled = standard_scaler.transform(x.values)
    return x_scaled

task = forest.keras.Task.REGRESSION
print("078914320471928301798423")
#1400m:700, 700m:18000
#t=55000

#tuner = tfdf.tuner.RandomSearch(num_trials=int(3.0148*t))
#tuner = forest.tuner.RandomSearch(num_trials=4 * t)

print("078914320471928301798423")
#0.6 model = tfdf.keras.RandomForestModel(verbose=0, tuner=tuner, num_trees=700,allow_na_conditions=True, task=tfdf.keras.Task.REGRESSION,winner_take_all=True,categorical_algorithm='RANDOM', honest=True,honest_fixed_separation=True, honest_ratio_leaf_examples=0.85)
#!!0.605 model = tfdf.keras.RandomForestModel(verbose=0, tuner=tuner, num_trees=700,allow_na_conditions=True, task=tfdf.keras.Task.REGRESSION,winner_take_all=True,categorical_algorithm='CART', honest=True,honest_fixed_separation=True, honest_ratio_leaf_examples=0.85, bootstrap_size_ratio=1.85, adapt_bootstrap_size_ratio_for_maximum_training_duration=True)
#model = tfdf.keras.RandomForestModel(verbose=0, tuner=tuner, num_trees=700,allow_na_conditions=True, task=task,winner_take_all=True,categorical_algorithm='CART', honest=True,honest_fixed_separation=True, honest_ratio_leaf_examples=0.85, bootstrap_size_ratio=1.795, adapt_bootstrap_size_ratio_for_maximum_training_duration=True)
#1model = tfdf.keras.RandomForestModel(verbose=0, tuner=tuner,allow_na_conditions=True, task=tfdf.keras.Task.REGRESSION,max_num_nodes=-1,max_depth=200, growing_strategy='BEST_FIRST_GLOBAL')
#model = tfdf.keras.RandomForestModel(verbose=0, tuner=tuner,allow_na_conditions=True,num_trees=700, task=tfdf.keras.Task.REGRESSION,split_axis='SPARSE_OBLIQUE',winner_take_all=False,categorical_algorithm='RANDOM',honest=True,honest_fixed_separation=True, honest_ratio_leaf_examples=0.85,keep_non_leaf_label_distribution=False)
#model = tfdf.keras.GradientBoostedTreesModel(verbose=0, tuner=tuner, num_trees=500,allow_na_conditions=True, task=tfdf.keras.Task.REGRESSION, categorical_algorithm='RANDOM', honest=True,honest_fixed_separation=True, honest_ratio_leaf_examples=0.85)

resolution='RF_v1_SSr3_100m'
# train_file ='1000m_reprocessed_.csv'
# test_file ='1000m_reprocessed_.csv'
train_file = 'SS_100m_data_train.csv'
test_file = 'SS_100m_data_test.csv'
var1 = ['Clay','Sand', 'Elevation','Slope','NDVI', 'Lai', 'SMERGE','Albedo'] # training vars
var2 = ['Clay','Sand', 'Elevation','Slope','NDVI', 'Lai', 'Albedo'] # testing vars
var3 = ['Clay','Sand', 'Elevation','Slope','NDVI', 'Lai', 'SMERGE','Albedo'] # vars for export csv
print(00000000000000000000000000000000000000000000000000)
#model.summary()
#model.save("rf_all")
print(00000000000000000000000000000000000000000000000000)
#tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0, max_depth=3)
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
summary_file = resolution+"%smodel_summary.txt" % str(resolution)

#train_data = pd.read_csv(train_file, usecols=['Clay', 'Sand','Silt ','Elevation','Slope', 'Ascept','NDVI', 'Smerge','Air','Date'])
#train_data = pd.read_csv(train_file, usecols=['PageName','Clay','Sand','Silt ','Elevation','Slope','Ascept','NDVI','Smerge','Air','Date','Lai','Albedo','LST_DI','LST_D_','LST_N_'])
train_data = pd.read_csv(train_file, usecols=['SMERGE','Date','PageName','LAI','Albedo','NDVI','Clay','Sand','Silt ','Slope','Elevation','Ascept'])
train_data = train_data[['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'SMERGE','Date','LAI','Albedo']]
train_data.columns = ['Clay', 'Sand','Silt ', 'Elevation','Slope', 'Ascept','NDVI', 'SMERGE','Date','Lai','Albedo']
#train_data = train_data[train_data['Date']==20140416.0]
#train_data = train_data.sample(frac=0.15)
#train_data.to_csv('Y_Air_moss_train_final22%.csv',index=False)
print(train_data)
#train_data = train_data[['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'SMERGE','Date']]
#train_data = train_data[['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'SMERGE','AirMoss','Date']]

#test_data = pd.read_csv(test_file, usecols=['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'Smerge','Air','Date','PageName'])
test_data = pd.read_csv(test_file, usecols=['SMERGE','Date','PageName','LAI','Albedo','NDVI','Clay','Sand','Silt ','Slope','Elevation','Ascept'])
test_page = test_data[['PageName']]
dates = test_data['Date']
test_data.drop(['PageName'],axis=1)
test_data = test_data[['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'SMERGE','Date','LAI','Albedo']]
test_data.columns = ['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'SMERGE','Date','Lai','Albedo']

# test_data = test_data[test_data['Date']==20140416.0]
#test_data = test_data.sample(frac=0.046)
#test_data.to_csv('Y_Air_moss_test_final22%.csv', index=False)
print(test_data)
os.system("pause")
#train_data, test_data = train_test_split(main_data, test_size=0.2)
scale_columns = ['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI','SMERGE']
#sm_train_data=pd.read_csv(train_file, usecols=['SMERGE'])
#train_data['Date'] = pd.to_datetime(train_data['Date'],format="%m/%d/%Y")
#sm_train_data = sm_train_data.dropna(axis=0)
print(train_data)
# train_data=train_data.dropna()
# train_data = train_data.reset_index(drop=True)
t=1100
tuner = forest.tuner.RandomSearch(num_trials=9*t)
#model = forest.keras.RandomForestModel(verbose=1, tuner=tuner,num_trees=t, allow_na_conditions=True, task=task,keep_non_leaf_label_distribution=False)
# model = forest.keras.RandomForestModel(verbose=1, tuner=tuner, num_trees=t,allow_na_conditions=True, task=task,winner_take_all=True,
#                                        categorical_algorithm='CART', honest=True,honest_fixed_separation=True, honest_ratio_leaf_examples=0.85,
#                                        bootstrap_size_ratio=1.8, adapt_bootstrap_size_ratio_for_maximum_training_duration=True, keep_non_leaf_label_distribution=False)
print(train_data)
print("873421390-721-87432-917493-21839-02180-7125794730275809327908759032475093842")
# airmoss = pd.read_csv('/content/drive/MyDrive/Downscaling/1000m_airmoss_mod7.csv', usecols=['AirMoss'])
scale_columns = ['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI']
model_img_file =resolution+"model.png"
n= test_data.shape[0]


print("078914320471928301798423")
index2 = (['2012-10-24','2012-10-27','2012-10-30','2013-06-17','2013-07-16','2013-07-19','2013-07-23',
           '2013-09-27','2014-04-16','2014-04-18','2014-04-24','2014-07-08','2014-07-11','2014-07-15',
           '2014-10-14','2014-10-17','2014-10-21','2015-04-16','2015-04-20','2015-08-07','2015-08-11','2015-08-14'])

index3 =([['2013-06-17'],['2012-10-24','2012-10-27','2012-10-30','2014-10-14','2014-10-17','2014-10-21','2015-04-16',
            '2015-04-20','2014-04-16','2014-04-18','2014-04-24'],['2013-07-16','2013-07-19','2013-07-23','2013-09-27',
          '2014-07-08','2014-07-11','2014-07-15','2015-08-07',
          '2015-08-11','2015-08-14']])
out = np.empty(shape=(0,1))
#train_data['Date'] = (pd.to_datetime(train_data['Date'], format='%d/%m/%Y'))
#test_data['Date'] = (pd.to_datetime(test_data['Date'], format='%d/%m/%Y'))
print(test_data.columns)
#var1 = ['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'SMERGE','Date','Lai','Albedo']
#var2 = ['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'Date','Lai','Albedo']
#var3 = ['Clay', 'Sand','Silt ', 'Elevation', 'Ascept', 'Slope','NDVI', 'SMERGE','AirMoss','Date','Lai','Albedo']
x = pd.DataFrame(columns=var3)
for gh in range(0,1):
    model = forest.keras.RandomForestModel(verbose=2, tuner=tuner, num_trees=t, allow_na_conditions=True, task=task,
                                           winner_take_all=True,
                                           categorical_algorithm='CART', honest=True,
                                           honest_fixed_separation=True, honest_ratio_leaf_examples=0.75,
                                           bootstrap_size_ratio=1.05,
                                           adapt_bootstrap_size_ratio_for_maximum_training_duration=True,
                                           keep_non_leaf_label_distribution=False,num_threads=8, max_depth=9)
    model.predefined_hyperparameters()
    print(index2[gh])
    train_data['Date'] = pd.to_datetime(train_data['Date'], format="%m/%d/%Y").astype(int)
    #train_data['Date'] = train_data['Date'].astype("string")
    #train_data1 = train_data[train_data['Date'].isin(index3[gh])]
    #train_data1 = train_data[train_data['Date'] == index2[gh]]
    train_data1 = train_data
    train_ds = forest.keras.pd_dataframe_to_tf_dataset(
        train_data1[var1], label='SMERGE',
        task=task)
    # train_ds = forest.keras.pd_dataframe_to_tf_dataset(train_data[['Clay', 'Sand','Silt ','NDVI', 'SMERGE','Date']], label='SMERGE', task=task)
    print(train_data1[var1])
    print('dashjkhlfjlkdashfjlkdsahfdlkjsahfjlkdsa')
    model.compile(metrics=["mae"])
    model.fit(train_ds)
    print(
        "0987654============================================================================================================================================================================================================")

    df_out = pd.DataFrame(columns=['ML_'])
    test_data['Date'] = pd.to_datetime(test_data['Date'], format="%m/%d/%Y").astype(int)
    #test_data['Date'] = test_data['Date'].astype("string")
    #test_data1 = test_data[test_data['Date'].isin(index3[gh])]
    #test_data1 = test_data[test_data['Date'] == index2[gh]]
    test_data1 = test_data
    test_ds = forest.keras.pd_dataframe_to_tf_dataset(
        test_data1[var2], task=task)
    print(test_data1[var2])
    out_data = test_data1[var3]
    hope = model.predict(test_ds, verbose=1)
    h = hope
    print(h.shape)
    out = np.vstack([out, h])
    x = pd.concat([x,out_data])
    print(out.shape)
    with open(resolution+"sum_model.txt","w") as txt_file:
        with redirect_stdout(txt_file):
            model.summary()
    with open(resolution+"model.html", "w") as html_file:
        html_file.write(forest.model_plotter.plot_model(model, tree_idx=0, max_depth=10))
    model.reset_states()
print(x.shape)
x['ML_'] = out
df_out = pd.DataFrame(h, columns=['ML_'])
print(h.shape)
print(test_data)
print(test_data.shape)
x['Date'] = dates
x['PageName'] = test_page
# out_ex = out_ex.iloc[:m, :]
x.to_csv(resolution + "7030.csv", index=False)


