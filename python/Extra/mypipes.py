
from sklearn.base import BaseEstimator, TransformerMixin


class VarSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self,var_names):
        self.vars=var_names
    
    def fit(self,x,y=None):
        return self
    
    def transform(self,X):
        return X[self.vars]

class get_dummies_Pipe(BaseEstimator, TransformerMixin):
    
    def __init__(self,freq_cutoff=0):
        self.freq_cutoff=freq_cutoff
        self.var_cat_dict={}
        
    def fit(self,x,y=None):
        data_cols=x.columns
        for col in data_cols:
            k=x[col].value_counts()
            cats=k.index[k>self.freq_cutoff][:-1]
            self.var_cat_dict[col]=cats
        return self
            
    def transform(self,x,y=None):
        dummy_data=x.copy()
        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                name=col+'_'+cat
                dummy_data[name]=(dummy_data[col]==cat).astype(int)
            del dummy_data[col]
        return dummy_data