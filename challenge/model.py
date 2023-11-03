import pandas as pd
from typing import Tuple, Union, List
from dataops import DataOps
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from xgboost import plot_importance

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame):
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        data_ops = DataOps(data=data)
        data_ops.add_feature_list()
        pp_data = data_ops.dataset
        if target_column in pp_data.columns:
            return pp_data[target_column]
        else :
            return pp_data
        

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        features = pd.concat([
            pd.get_dummies(features['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(features['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(features['MES'], prefix = 'MES')], 
            axis = 1
        )
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)
        print(f"train shape: {x_train.shape} | test shape: {x_test.shape}")
        y_train.value_counts('%')*100
        y_test.value_counts('%')*100
        xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        xgb_model.fit(x_train, y_train)
        self._model = xgb_model

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Argumentation about selection of xgboost as model. It has better accurancy 
        keeping in mind the possibility of scale the features of the model,
        since for now the target it is delay maybe in the future there is other multidimensional
        approaches that can be handled better in performance by xgboost.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        x_train, x_test, y_train, y_test = train_test_split(features, features['delay'], test_size = 0.33, random_state = 42)
        xgboost_y_preds = self._model.predict(x_test)
        xgboost_y_preds = [1 if y_pred > 0.5 else 0 for y_pred in xgboost_y_preds]

        return xgboost_y_preds

            