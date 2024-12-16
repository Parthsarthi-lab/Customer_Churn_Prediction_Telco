
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

from training.custom_logging import info_logger, error_logger
from training.exception import FeatureEngineeringError, handle_exception

from training.entity.config_entity import FeatureEngineeringConfig
from training.configuration_manager.configuration import ConfigurationManager

class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def load_saved_data(self):
        try:
            info_logger.info("Loading Final Training Saved Data for Transformation")

            final_train_data_path = os.path.join(self.config.final_train_data_path, 'Train.npz')
            final_test_data_path = os.path.join(self.config.final_test_data_path,"Test.npz")

            final_train_data = np.load(final_train_data_path, allow_pickle=True)
            final_test_data = np.load(final_test_data_path, allow_pickle=True)

            xtrain = final_train_data["xtrain"]
            xtest = final_test_data["xtest"]
            ytrain = final_train_data["ytrain"]
            ytest = final_test_data["ytest"]

            info_logger.info("Loaded Final Training Saved Data for Transformation")

            return xtrain, xtest, ytrain, ytest
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)

    def transform_data(self, xtrain, xtest, ytrain, ytest):
        try:
            info_logger.info("Transforming Final Training Data")

            # Load the original dataset to extract column names
            df = pd.read_csv(self.config.data_dir).drop(columns=['customerID', 'Churn'])

            # Define categorical and numerical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

            # Ensure xtrain and xtest are DataFrames with proper column names
            if not isinstance(xtrain, pd.DataFrame):
                xtrain = pd.DataFrame(xtrain, columns=df.columns)
            if not isinstance(xtest, pd.DataFrame):
                xtest = pd.DataFrame(xtest, columns=df.columns)

            # Define the ColumnTransformer
            transform_pipeline = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ]
            )

            info_logger.info("Pipeline defined successfully")

            # Fit the pipeline on xtrain
            transform_pipeline.fit(xtrain)

            # Save the pipeline as an artifact
            pipeline_path = os.path.join(self.config.root_dir, "pipeline.joblib")
            joblib.dump(transform_pipeline, pipeline_path)

            # Transform xtrain and xtest
            xtrain = transform_pipeline.transform(xtrain)
            xtest = transform_pipeline.transform(xtest)

            #info_logger.info(xtrain[0])
            info_logger.info("Transformed Final Training Data")

            return xtrain, xtest, ytrain, ytest
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)


    def save_transformed_data(self, xtrain, xtest, ytrain, ytest):
        try:
            info_logger.info("Saving Final Training Transformed Data")

            final_transform_data_path = self.config.root_dir
            
            #info_logger.info(xtrain[0])
            # Save xtrain and ytrain  to Train.npz
            # Save xtest and ytest to Test.npz
            np.savez(os.path.join(final_transform_data_path, 'Train.npz'), xtrain=xtrain, ytrain=ytrain)
            np.savez(os.path.join(final_transform_data_path, 'Test.npz'),  xtest=xtest, ytest=ytest)

            info_logger.info("Saved Final Training Transformed Data")

            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Feature Engineering status: True")

        except Exception as e:
            handle_exception(e, FeatureEngineeringError)
            

if __name__ == "__main__":
    config = ConfigurationManager()
    feature_engineering_config = config.get_feature_engineering_config()

    feature_engineering = FeatureEngineering(config = feature_engineering_config)
    xtrain, xtest, ytrain, ytest = feature_engineering.load_saved_data()
    xtrain, xtest, ytrain, ytest = feature_engineering.transform_data(xtrain, xtest, ytrain, ytest)
    feature_engineering.save_transformed_data(xtrain, xtest, ytrain, ytest)
