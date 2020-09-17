import pandas as pd
import numpy as np
from utils import logger
from sklearn.base import BaseEstimator, TransformerMixin


class HierarchicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, sample_weight_name=None, C=30, disambiguate=True, verbose=False):
        assert C > 1
        self.C = C
        self.cols = cols
        self.sample_weight_name = sample_weight_name
        self.disambiguate=disambiguate
        self.verbose = verbose
        self.gbs = dict()
        
        
    def _disambiguate(self, X, sep='__'):
        """Disambiguate hierarchical categorical columns, f.e. distinguish Paris, US from Paris, France, 
        by concatenating the parent categories values with child categories values.
        Order of cols matters: the feature at the beginning of the cols list is considered to be the parent feature.
        F.e.: [country, city, street].
        __ is used as a value separator in the concatenated values by default.
        """
        for i, c in enumerate(self.cols):
            if i > 0:
                X[c] = X[c].astype('str') + sep + X[self.cols[i-1]].astype('str')
        return X
      
    
    def fit(self, X, y):
        if self.verbose:
            logger.info('Fitting...')
        if self.sample_weight_name is None:
            sample_weight = pd.Series(1, index=X.index)
        else:
            sample_weight = X[self.sample_weight_name]
        if self.cols is None:
            self.cols = [c for c in list(X.columns) if not c==self.sample_weight_name]
        X = X[self.cols].copy()
        if self.disambiguate:
            X = self._disambiguate(X)

        X['target_denominator'] = sample_weight
        X['target_numerator'] = y * sample_weight
               
        self.total_ratio = np.average(y, weights=sample_weight)
        min_sample_std = np.sqrt(self.total_ratio * (1-self.total_ratio) / self.C)
        self.std_mean_ratio = round(min_sample_std / self.total_ratio, 3)
        if self.verbose:
            logger.info(f'STD/AVG for min sample: {self.std_mean_ratio}')
        
        for i in range(len(self.cols)):
            self.gbs[i] = X.groupby(self.cols[:i+1])[['target_numerator', 'target_denominator']].sum().reset_index()
            if i > 0:
                self.gbs[i]['parent_ratio'] = self.gbs[i][self.cols[i-1]].map(self.gbs[i-1].set_index(self.cols[i-1])['ratio'])
                
            elif i==0:
                self.gbs[i]['parent_ratio'] = self.total_ratio
            self.gbs[i]['ratio'] = (self.gbs[i]['target_numerator'] + self.gbs[i]['parent_ratio']*self.C ) /\
                                       (self.gbs[i]['target_denominator'] + self.C)
        return self
    
    
    def transform(self, X, matrix_output=False):
        if self.verbose:
            logger.info('Transforming...')
        X = X[self.cols].copy()
        if self.disambiguate:
            X = self._disambiguate(X)
        expected_len = X.shape[0]
        self.result = pd.Series(index=X.index)
        for i in reversed(range(len(self.cols))):
            if self.verbose:
                logger.info(f'Mapping {self.cols[i]}...')
            bavg = self.gbs[i].reset_index()[[self.cols[i], 'ratio']].set_index(self.cols[i]).to_dict()['ratio']
            mapping = X[self.cols[i]].map(bavg)
            self.result = self.result.fillna(mapping)
            n_na = self.result.isna().sum()
            if self.verbose:
                logger.info(f'Mapping completed, missing values to fill: {n_na} out of {expected_len}')
            if n_na == 0:
                break
        n_na = self.result.isna().sum()
        if n_na > 0:
            logger.info(f'Imputing {n_na} unknown values with global average...')
        self.result = self.result.fillna(self.total_ratio)
        assert ~self.result.isna().any()
        if self.verbose:
            logger.info('Completed.')
        return pd.DataFrame(self.result)
