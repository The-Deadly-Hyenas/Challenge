import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.combine import SMOTEENN
from sklearn.metrics import make_scorer
from score_function import revenue_gain

# load the data and reset index of dataframe
df: pd.DataFrame = pd.read_pickle("../training_data/task_3_training_e8da4715deef7d56_f8b7378_pandas.pkl").reset_index()

# get only the low and mid level features + segment_id
X = df.loc[:, "essentia_dissonance_mean":"mirtoolbox_roughness_pct_90"]
# target value
y = df["quadrant"]

# preprocess dataset
X_std = StandardScaler().fit_transform(X)
X = pd.DataFrame(X_std, columns=X.columns)

# add segment_id to data for filtering segments
X["segment_id"] = df["segment_id"]

# remove segment_id 26 and keep as test/ eval data for later
seg_26_indices = (X["segment_id"] == 26)
X_test = X[seg_26_indices].drop(["segment_id"], axis=1)
y_test = y[seg_26_indices]

X_train = X.drop(X[seg_26_indices].index, axis=0).reset_index(drop=True)
y_train = y.drop(X[seg_26_indices].index, axis=0)

# Combination of over- and under-sampling
# https://imbalanced-learn.org/stable/combine.html
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
# X_resampled, y_resampled = X_train, y_train


# split the data according to segment_id
# store the splits as tuple (train indices, test_indices)
# 2 segments for test, the rest for training (not including segment 26)
cv = []

for i in range(24):
    train_indices = X_resampled[~X_resampled["segment_id"].isin([i, i + 1])].index.to_list()
    test_indices = X_resampled[X_resampled["segment_id"].isin([i, i + 1])].index.to_list()
    cv.append((train_indices, test_indices))

# remove the segment_id as we don't want it in the training data
X_resampled = X_resampled.drop(["segment_id"], axis=1)

# select k best features according to ANOVA F-value between label/feature (for classification tasks)
best_features = SelectKBest(score_func=f_classif, k=15).fit(X_resampled, y_resampled).get_feature_names_out()
X_select = X_resampled[best_features]

# parameters for grid search
params = {
    "n_neighbors": [77],
    "weights": ["uniform"],  # {‘uniform’, ‘distance’}
    "algorithm": ["auto"],  # {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
}

gs_cv = GridSearchCV(KNeighborsClassifier(),
                     params,
                     cv=cv,
                     return_train_score=True,
                     n_jobs=-1,
                     scoring=make_scorer(revenue_gain, greater_is_better=True))

gs_cv.fit(X_select, y_resampled)
print(gs_cv.best_score_, gs_cv.best_params_)

############################################################################################################
# load test dataset and transform
test_data = pd.read_pickle("../test_data/task_4_test_dd4bd32b08b776e6_daf99ad_pandas.pkl").reset_index()
test_features = test_data.loc[:, "essentia_dissonance_mean":]
test_features_std = StandardScaler().fit_transform(test_features)
test_features = pd.DataFrame(test_features_std, columns=test_features.columns)


# apply data transform to all the data
X_all, y_all = smote_enn.fit_resample(X, y)
best_features = SelectKBest(score_func=f_classif, k=15).fit(X_all, y_all).get_feature_names_out()
X_all = X_all.drop(["segment_id"], axis=1)
X_all = X_all[best_features]

# predict labels
knn = KNeighborsClassifier(**gs_cv.best_params_).fit(X_all, y_all)
predicted = knn.predict(test_features[best_features])

# create dataframe with results and save to csv
results = test_data.loc[:, "pianist_id":"snippet_id"]
results["quadrant"] = predicted.astype(int)
results.to_csv("results_knn.csv", index=False)
