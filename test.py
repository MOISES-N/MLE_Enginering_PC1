from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from pycaret.regression import load_model
import pandas as pd
import numpy as np

class TestModel:

    def __init__( self ) -> None:
        pass

    def KFolds_Evaluation( self ) :

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        evaluation_data = pd.read_csv( 'data/evaluation_dataset.csv' , sep = ';' )
        print( evaluation_data )
        model = load_model( 'Modelo/final_model')
        scores = cross_val_score(model, evaluation_data.drop('Rings', axis = 1), evaluation_data['Rings'], scoring=scorer, cv=kf)
        print("MSE scores per fold: ", -scores)
        print("Mean MSE: {:.2f}".format(-np.mean(scores)))

def TestModelo():
    TestModel().KFolds_Evaluation()

if __name__ == '__main__' :
    TestModelo()
    
