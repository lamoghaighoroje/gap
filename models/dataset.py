import pandas as pd
import numpy as np
from attrdict import AttrDict
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from IPython.core.display import display, HTML

class Dataset(BaseEstimator, TransformerMixin):
    def __init__(self, n_samples=None):
        self.n_samples = n_samples
        
    def transform(self, 
                    X, 
                    pretrained=None, 
                    label_corrections=None,
                    shift_by_one=True,
                    verbose=0):

        if label_corrections is not None:
            label_corrections = pd.read_csv(label_corrections, 
                                            sep='-', 
                                            header=None, 
                                            comment='#', 
                                            names=['id', 'label'])

            if shift_by_one:
                label_corrections['id'] = label_corrections['id'] -1
        else:
            label_corrections = pd.DataFrame(columns=['id', 'label'])
        
        X1 = pd.read_csv(X, sep='\t')
        X2 = X1.copy()

        X = pd.concat([X1, X2])
        
        if pretrained is not None:
            pretrained = pd.read_csv(pretrained)
        else:
            # pretrained = pd.DataFrame(np.ones((len(X), 3))*0.33)
            pretrained = pd.DataFrame(np.ones((len(X), 2))*0.33)

        if self.n_samples:
            X = X.head(self.n_samples)
            pretrained = pretrained.head(self.n_samples)
            
        # normalizing column names
        X.columns = map(lambda x: x.lower().replace('-', '_'), X.columns)
        X2.columns = map(lambda x: x.lower().replace('-', '_'), X2.columns)
        if verbose:
            with pd.option_context('display.max_rows', 10, 'display.max_colwidth', 15):
                # display(X)
                display(X2)
     
        if 'a_coref' in X2.columns or 'b_coref' in X2.columns:
            if 'a_coref' in X2.columns:
                y_a = pd.DataFrame(X2[['a_coref']].values, columns=['A'])
                # y_a['NEITHER'] = ~y_a['A']
            if 'b_coref' in X2.columns:
                y_b = pd.DataFrame(X2[['b_coref']].values, columns=['A'])
                # y_b['NEITHER'] = ~y_b['B'] 
            y = pd.concat([y_a, y_b]) 
            y['NEITHER'] = ~y['A'] 
        else:
            y = pd.DataFrame([[False]]*len(X), columns=['A'])
            y['NEITHER'] = ~y['A']

        return AttrDict(locals())