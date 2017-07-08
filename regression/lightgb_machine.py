import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter

from autosklearn.pipeline.components.base import \
    AutoSklearnRegressionAlgorithm

from autosklearn.pipeline.constants import *


class LGBMachineRegressor(AutoSklearnRegressionAlgorithm):
    def __init__(self, max_bin, learning_rate, feature_fraction,
                 num_leaves, min_data_in_leaf, min_sum_hessian_in_leaf,
                 max_depth, min_gain_to_split, random_state=None):# num_threads):
        #set instance vars to args passed to __init__

        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_bin = max_bin
        self.feature_fraction = feature_fraction
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.max_depth = max_depth
        self.min_gain_to_split = min_gain_to_split
        if random_state is None:
            self.random_state = numpy.random.randint(1, 10000, size=1)[0]
        else:
            self.random_state = random_state.randint(1, 10000, size=1)[0]

        
    def fit(self, X, y):
        import LightGBM as lgbm
        #run checks, convert data to relevant types

        self.learning_rate = float(self.learning_rate)
        self.num_leaves = int(self.num_leaves)
        self.max_bin = int(self.max_bin)
        self.feature_fraction = float(self.feature_fraction)
        self.min_data_in_leaf = int(self.min_data_in_leaf)
        self.min_sum_hessian_in_leaf = float(self.min_sum_hessian_in_leaf)
        self.max_depth = int(self.max_depth)
        self.min_gain_to_split = float(self.min_gain_to_split)
        

        self.estimator = lgbm.LGBMRegressor(
            learning_rate = self.learning_rate,
            num_leaves = self.num_leaves,
            max_bin = self.max_bin,
            feature_fraction = self.feature_fraction,
            min_data_in_leaf = self.min_data_in_leaf,
            min_sum_hessian_in_leaf = self.min_sum_hessian_in_leaf,
            max_depth = self.max_depth,
            min_gain_to_split = self.min_gain_to_split,
            random_state = self.random_state
        )

        self.estimator.fit(X,y)
        return self
    
    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(self, X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LGBM',
                'name': 'Light Gradient Boosting Machine Regressor',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'input':(DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        max_bin = UnParametrizedHyperparameter(
            name='max_bin', value=255)
        learning_rate = UniformFloatHyperparameter(
            name='learning_rate', lower=0.01, upper=1, default=0.1)
        feature_fraction = UniformFloatHyperparameter(
            name='feature_fraction', lower=0.5, upper=1.0, default=1.0)
        num_leaves = UniformIntegerHyperparameter(
            name='num_leaves', lower=5, upper=500, default=31)
        min_data_in_leaf = UniformIntegerHyperparameter(
            name='min_data_in_leaf', lower=1, upper=30, default=20)
        min_sum_hessian_in_leaf = UnParametrizedHyperparameter(
            name='min_sum_hessian_in_leaf', value=1e-3)
        max_depth = UniformIntegerHyperparameter(
            name='max_depth', lower=-1, upper=10, default=-1)
        min_gain_to_split = UnParametrizedHyperparameter(
            name='min_gain_to_split', value=0)
        # num_threads = Uniform

        cs.add_hyperparameters([max_bin, learning_rate, feature_fraction,
                                num_leaves, min_data_in_leaf,
                                min_sum_hessian_in_leaf, max_depth,
                                min_gain_to_split])
        
        return cs
