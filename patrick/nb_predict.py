import pandas as pd
import numpy as np
from tqdm import tqdm
from simple_colors import *

from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.metrics import make_scorer


df: pd.DataFrame = pd.read_pickle(
    "high_level_data/train_data_pandas.pkl").reset_index()


def revenue_gain(y_true, y_pred):
    """
    Return the score according to the Estimated revenues/ fitmotion gain matrix
    from the slides. Faster than the revenue_challenge.py function.
    labels: 1="happy", 2="angry", 3="sad", 4="relaxed"
    for sklearn use this:
        from sklearn.metrics import make_scorer
        score = make_scorer(revenue_gain, greater_is_better=True)
        score(classifier, X, y) # same as revenue_gain(y, classifier.predict(X))

    :param y_true: true labels as numpy array of int
    :param y_pred: predicted labels as numpy array of int
    :return: accumulates score over all predictions
    """
    # convert to int and remove 1 to use as index for gain_matrix
    y_true = y_true.astype(int) - 1
    y_pred = y_pred.astype(int) - 1

    gain_matrix = np.array([[5, -5, -5, 2],
                            [-5, 10, 2, -5],
                            [-5, 2, 10, -5],
                            [2, -5, -2, 5]])

    revenue = gain_matrix[y_true, y_pred].sum()

    return revenue


class NB:

    def __init__(self, params):
        """
        Default Parameters
        :param params:
        """
        self.params = params

        self.X = None
        self.y = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.model = None
        self.cv = None

    def preprocessing(self):
        """
        Performe Dataset Pre-Processing and Cross-Validation
        :return:
        """
        # Load Training Dataset (Low-Level Features, Mid-Level Features and High-Level Features)
        # X = df.loc[:, "essentia_dissonance_mean":"mirtoolbox_roughness_pct_90"]  # 1

        X = df.copy()  # In order to take information about the ID also into account.  # 2
        X = X.drop(['quadrant', 'valence', 'arousal', 'gems_wonder', 'gems_transcendence', 'gems_tenderness',
                    'gems_nostalgia', 'gems_peacefulness', 'gems_power', 'gems_joyful_activation', 'gems_tension',
                    'gems_sadness', 'gemmes_movement', 'gemmes_force', 'gemmes_interior', 'gemmes_wandering',
                    'gemmes_flow'], axis=1)  # 2

        y = df["quadrant"]

        # Dataset Pre-Processing
        X_std = StandardScaler().fit_transform(X)
        X = pd.DataFrame(X_std, columns=X.columns)

        # add segment_id to training data for doing the cross validation splits
        X["segment_id"] = df["segment_id"]   # 1

        # remove segment_id 26
        seg_26_indices = (X["segment_id"] == 26)
        X_test = X[seg_26_indices].drop(["segment_id"], axis=1)
        y_test = y[seg_26_indices]

        X_train = X.drop(X[seg_26_indices].index, axis=0).reset_index(drop=True)
        y_train = y.drop(X[seg_26_indices].index, axis=0)

        # SMOTE-ENN
        # https://imbalanced-learn.org/stable/combine.html
        smote_enn = SMOTEENN(random_state=0)
        X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

        # CROSS-VALIDATION
        cv = []
        for i in tqdm(range(24), total=len(range(24)), desc=":: Cross-Validation"):
            train_indices = X_resampled[~X_resampled["segment_id"].isin([i, i + 1])].index.to_list()
            test_indices = X_resampled[X_resampled["segment_id"].isin([i, i + 1])].index.to_list()
            cv.append((train_indices, test_indices))

        # Drop Segment-ID
        X_resampled = X_resampled.drop(["segment_id"], axis=1)

        # Select Best Features
        best_features = SelectKBest(score_func=f_classif, k=8).fit(X_resampled, y_resampled).get_feature_names_out()

        self.cv = cv

        self.X = X
        self.y = y

        self.X_train = X_resampled[best_features]
        self.y_train = y_resampled

        self.X_test = X_test[best_features]
        self.y_test = y_test

    def train_model(self):
        """
        Select model from model class
        :return:
        """

        print(green(":: Starting Preprocessing"))
        # Dataset Preprocessing
        self.preprocessing()
        print(green(":: Finished Preprocessing"))

        gridSearchModel = GridSearchCV(GaussianProcessClassifier(), self.params, cv=self.cv, n_jobs=1)
        # gridSearchModel = GridSearchCV(GaussianNB(), self.params, cv=self.cv, n_jobs=1)
        gridSearchModel.fit(self.X_train, self.y_train)

        # Set Model
        self.model = gridSearchModel

        print(green(f"\n:: Best Parameters") + f"\t\t\t\t\t\t{self.model.best_params_}")
        print(green(f":: Trainingset (Score)") + f"\t\t\t\t\t{self.model.score(self.X_train, self.y_train)}")
        print(green(f":: Testset Score (Segment-ID: 26)") + f"\t\t{self.model.score(self.X_test, self.y_test)}")

        means = self.model.cv_results_["mean_test_score"]
        stds = self.model.cv_results_["std_test_score"]

        print(green("\n:: GridSearch Results: "))
        for mean, std, params in zip(means, stds, self.model.cv_results_["params"]):
            print("   %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        print(green(f"\n:: Classification Report: "))
        y_true, y_pred = self.y_test, self.model.predict(self.X_test)
        print(classification_report(y_true, y_pred, zero_division=1))

        print(green(f"\n:: Achieved Score and revenue: "))
        print(f'Revenue gain: {revenue_gain(y_true, y_pred)} out of maximum revenue 465')

    def test_model(self):
        """
        Train model on best model classifiers and test on test dataset
        :return:
        """
        # load test dataset and transform
        test_data = pd.read_pickle("high_level_data/test_data_pandas.pkl").reset_index()

        # test_features = test_data.loc[:, "essentia_dissonance_mean":]  # 1
        test_features = test_data.copy()  # 2
        test_features_std = StandardScaler().fit_transform(test_features)
        test_features = pd.DataFrame(test_features_std, columns=test_features.columns)

        # apply data transform to all the data
        smote_enn = SMOTEENN(random_state=0)
        X_all, y_all = smote_enn.fit_resample(self.X, self.y)
        best_features = SelectKBest(score_func=f_classif, k=17).fit(X_all, y_all).get_feature_names_out()
        X_all = X_all.drop(["segment_id"], axis=1)
        X_all = X_all[best_features]

        # predict labels
        classifier = GaussianProcessClassifier(**self.model.best_params_).fit(X_all, y_all)
        # classifier = GaussianNB(**self.model.best_params_).fit(X_all, y_all)
        predicted = classifier.predict(test_features[best_features])

        # create dataframe with results and save to csv
        results = test_data.loc[:, "pianist_id":"snippet_id"]
        results["quadrant"] = predicted.astype(int)

        results.to_csv("results_svm.csv", index=False)


if __name__ == "__main__":

    params = {"kernel": [None]}

    NB = NB(params=params)
    NB.train_model()
    NB.test_model()
