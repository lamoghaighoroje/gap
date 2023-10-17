import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

class LabelSanitizer(BaseEstimator, TransformerMixin):
    def __init__(self, sanitize_labels):
        self.sanitize_labels = sanitize_labels

    def transform(self, X, corrections):
        X = X.copy(deep=True)

        if not self.sanitize_labels:
            print('Label sanization will be skipped.')
        else:
            print(corrections)

        if len(corrections) and self.sanitize_labels:
            mask = corrections['label'].str.lower().str.contains('\(a\)')
            ids = corrections[mask]['id'].values
            ids = [id for id in ids if id in X.index]
            X.loc[ids, ['a_coref', 'b_coref']] = [True, False]
            # X.loc[ids, ['a_coref']] = [True]

            mask = corrections['label'].str.lower().str.contains('\(b\)')
            ids = corrections[mask]['id'].values
            ids = [id for id in ids if id in X.index]
            # X.loc[ids, ['b_coref']] = [True]
            X.loc[ids, ['a_coref', 'b_coref']] = [False, True]

            mask = corrections['label'].str.lower().str.contains('neither')
            ids = corrections[mask]['id'].values
            ids = [id for id in ids if id in X.index]
            X.loc[ids, ['a_coref', 'b_coref']] = [False, False]

        if 'a_coref' in X.columns or 'b_coref' in X.columns:
            if 'a_coref' in X.columns:
                y = pd.DataFrame(X[['a_coref']].values, columns=['A'])
                y['NEITHER'] = ~y['A']
            # if 'b_coref' in X.columns:
            #     y = pd.DataFrame(X[['b_coref']].values, columns=['B'])
            #     y['NEITHER'] = ~y['B'] 
        else:
            y = pd.DataFrame([[False]]*len(X), columns=['A'])
            y['NEITHER'] = ~y['A']

        return {'X': X, 'y': y}