import os
import pickle
from sklearn.preprocessing import StandardScaler

from joblib import dump, load
import pandas as pd
import numpy as np

train_original: pd.DataFrame = pd.read_pickle("../training_data/task_3_training_e8da4715deef7d56_f8b7378_pandas.pkl").reset_index()
test_original: pd.DataFrame = pd.read_pickle("../test_data/task_4_test_dd4bd32b08b776e6_daf99ad_pandas.pkl").reset_index()

# get only the low and mid level features + segment_id
train = train_original.loc[:, "essentia_dissonance_mean":"mirtoolbox_roughness_pct_90"]
test = test_original.loc[:, "essentia_dissonance_mean":"mirtoolbox_roughness_pct_90"]

# normalize datasets
train = StandardScaler().fit_transform(train)
test = StandardScaler().fit_transform(test)

hl_feat = ["gems_wonder_binary", "gems_transcendence_binary", "gems_tenderness_binary", "gems_nostalgia_binary", "gems_peacefulness_binary",
           "gems_power_binary", "gems_joyful_activation_binary", "gems_tension_binary", "gems_sadness_binary", "gemmes_movement_binary",
           "gemmes_force_binary", "gemmes_interior_binary", "gemmes_wandering_binary", 'gemmes_flow_binary']

training_data = train_original.copy()
test_data = test_original.copy()

for feature in hl_feat:
    model = load(os.path.join('models', feature))

    pred = model.predict(train)
    training_data = training_data.drop(feature, axis=1)
    training_data.insert(training_data.shape[1], feature, pred)

    pred = model.predict(test)
    test_data.insert(test_data.shape[1], feature, pred)

training_data.to_pickle("high_level_data/train_data_pandas.pkl")
test_data.to_pickle("high_level_data/test_data_pandas.pkl")


train = pd.read_pickle('high_level_data/train_data_pandas.pkl')
test = pd.read_pickle('high_level_data/test_data_pandas.pkl')
print('Finished execution.')

