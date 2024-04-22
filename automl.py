import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.regression import setup, compare_models, create_model, tune_model, finalize_model , save_model , load_model , predict_model 
from dotenv import load_dotenv


class MLSystem:

    def __init__(self) -> None:
        pass

    def read_datasets( self ) :

        self.data_train = pd.read_csv( 'data/train.csv' )
        self.data_test = pd.read_csv( 'data/test.csv' )

    def split_dataset( self ) :

        train_data, test_data = train_test_split(
                                                    self.data_train,                    
                                                    test_size=0.25,          
                                                    random_state=42,         
                                                    stratify=self.data_train['Rings']  
                                                )
        
        self.data_train = train_data
        test_data.to_csv( 'data/evaluation_dataset.csv', index = False )

    def Get_Best_Model_automl( self ) :

        setup(data = self.data_train, 
            train_size       = 0.8,
            numeric_features = ['Length','Diameter','Height','Whole weight','Whole weight.1','Whole weight.2','Shell weight'],  # Example numeric features
            ignore_features = ['Id'],  
            target = 'Rings', 
            session_id=123) 
        
        best = compare_models()
        self.best_model = create_model( best )

    def Tune_Best_Model( self ):

        tuned_model = tune_model( self.best_model, optimize='RMSE')
        self.final_dt = finalize_model( tuned_model )

    def Save_Final_Model( self ):

        save_model( self.final_dt , 'Modelo/final_model')

    def Load_Final_Model( self ):

        self.final_dt = load_model( 'Modelo/final_model')

    def Final_Prediccion( self ):

        prediccion = predict_model( self.final_dt , data=self.data_test)[ [ 'id' , 'prediction_label' ]]
        prediccion = prediccion.rename( columns={'prediction_label': 'Rings'} )
        self.prediccion_path = 'Predicion/Predicion_final.csv'
        prediccion.to_csv( self.prediccion_path , index = False )

    def Submit_Competition( self ) :
        try:
            load_dotenv()
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print( 'Credenciales ok' )
            self.api.competition_submit(file_name = self.prediccion_path,
                                message="First submission",
                                competition="playground-series-s4e4" )
        except AttributeError:
            pass

def AutoML_PyCaret():
    ML_System = MLSystem()
    ML_System.read_datasets()
    ML_System.split_dataset()
    ML_System.Get_Best_Model_automl()
    ML_System.Tune_Best_Model()
    ML_System.Save_Final_Model()
    ML_System.Load_Final_Model()
    ML_System.Final_Prediccion()

def SubmitKaggle():
    try:
            load_dotenv()
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print( 'Credenciales ok' )
            api.competition_submit(file_name = 'Predicion/Predicion_final.csv',
                                message="First submission",
                                competition="playground-series-s4e4" )
    except AttributeError:
        pass

if __name__ == '__main__' :

    AutoML_PyCaret()
    SubmitKaggle()


    