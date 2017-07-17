import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnClassificationAlgorithm

from autosklearn.pipeline.constants import *


class LGBMachineClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self, learning_rate, feature_fraction,
                 num_leaves, min_data_in_leaf, max_depth,
                 max_bin, min_gain_to_split, min_sum_hessian_in_leaf,
                 bagging_fraction, lambda_l1, lambda_l2,
                 min_data_in_bin, random_state=None, num_threads=1):

        
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_bin = max_bin
        self.feature_fraction = feature_fraction
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.max_depth = max_depth
        self.min_gain_to_split = min_gain_to_split
        if random_state is None:
            self.random_state = np.random.randint(1, 10000, size=1)[0]
        else:
            self.random_state = random_state.randint(1, 10000, size=1)[0]
        self.num_threads = num_threads
        self.bagging_fraction = bagging_fraction
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.min_data_in_bin = min_data_in_bin
        
    def fit(self, X, y):
        import lightgbm as lgbm

        self.learning_rate = float(self.learning_rate)
        self.num_leaves = int(self.num_leaves)
        self.max_bin = int(self.max_bin)
        self.feature_fraction = float(self.feature_fraction)
        self.min_data_in_leaf = int(self.min_data_in_leaf)
        self.min_sum_hessian_in_leaf = float(self.min_sum_hessian_in_leaf)
        self.max_depth = int(self.max_depth)
        self.min_gain_to_split = float(self.min_gain_to_split)
        self.num_threads = int(self.num_threads)
        self.bagging_fraction = float(self.bagging_fraction)
        self.lambda_l1 = float(self.lambda_l1)
        self.lambda_l2 = float(self.lambda_l2)
        self.min_data_in_bin = int(self.min_data_in_bin)

        if len(np.unique(y))==2:
            self.objective = 'binary'
        else:
            self.objective = 'multiclass'
                   
        self.estimator = lgbm.LGBMClassifier(
            learning_rate = self.learning_rate,
            num_leaves = self.num_leaves,
            max_bin = self.max_bin,
            feature_fraction = self.feature_fraction,
            min_data_in_leaf = self.min_data_in_leaf,
            min_sum_hessian_in_leaf = self.min_sum_hessian_in_leaf,
            max_depth = self.max_depth,
            min_gain_to_split = self.min_gain_to_split,
            seed = self.random_state,
            num_threads = self.num_threads,
            bagging_fraction = self.bagging_fraction,
            lambda_l1 = self.lambda_l1,
            lambda_l2 = self.lambda_l2,
            min_data_in_bin = self.min_data_in_bin,
            objective = self.objective
        )

        self.estimator.fit(X,y)
        return self
    
    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(self, X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError
        probas = self.estimator.predict_proba(X)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LGBM',
                'name': 'Light Gradient Boosting Machine Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input':(DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        #Parametrized
        learning_rate = UniformFloatHyperparameter(
            name='learning_rate', lower=0.01, upper=1, default=0.1)
        feature_fraction = UniformFloatHyperparameter(
            name='feature_fraction', lower=0.5, upper=1.0, default=0.9)
        num_leaves = UniformIntegerHyperparameter(
            name='num_leaves', lower=5, upper=500, default=31)
        min_data_in_leaf = UniformIntegerHyperparameter(
            name='min_data_in_leaf', lower=1, upper=30, default=20)
        lambda_l1 = UniformFloatHyperparameter(
            name='lambda_l1', lower=0.0, upper=10.0, default=0.0)
        lambda_l2 = UniformFloatHyperparameter(
            name='lambda_l2', lower=0.0, upper=10.0, default=0.0)
        max_bin = UniformIntegerHyperparameter(
            name='max_bin', lower=50, upper=500, default=255)
        

        #UnParametrized
        bagging_fraction = UnParametrizedHyperparameter(
            name='bagging_fraction', value=1.0)
        bagging_freq = UnParametrizedHyperparameter(
            name='bagging_freq', value=5)
        max_depth = UnParametrizedHyperparameter(
            name='max_depth', value=-1)
        min_data_in_bin = UnParametrizedHyperparameter(
            name='min_data_in_bin', value=5)
        min_gain_to_split = UnParametrizedHyperparameter(
            name='min_gain_to_split', value=0)
        min_sum_hessian_in_leaf = UnParametrizedHyperparameter(
            name='min_sum_hessian_in_leaf', value=1e-3)
        

        cs.add_hyperparameters([learning_rate, feature_fraction, bagging_fraction,
                                num_leaves, min_data_in_leaf, lambda_l1,
                                lambda_l2, min_data_in_bin, max_depth, max_bin,
                                min_sum_hessian_in_leaf, min_gain_to_split]) 
        
        return cs
