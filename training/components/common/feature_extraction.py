
import os
import sys
import pandas as pd
import numpy as np

from training.exception import FeatureExtractionError, handle_exception
from training.custom_logging import info_logger, error_logger

from training.entity.config_entity import FeatureExtractionConfig
from training.configuration_manager.configuration import ConfigurationManager


class FeatureExtraction:
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config

    def extract_features(self):
        try:
            info_logger.info("Feature Extraction component started")

            status = False
            df = pd.read_csv(self.config.data_dir)
            
            # Convert 'TotalCharges' to numeric, coercing errors to identify anomalies
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

            # Replace missing TotalCharges with 0
            df.fillna({'TotalCharges': 0}, inplace=True)
            
            save_path = os.path.join(self.config.root_dir, "features_extracted.csv")
            df.to_csv(save_path, index=False)

            status = True
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Feature Extraction status: {status}")

            info_logger.info("Feature Extraction Component completed")
        except Exception as e:
            status = False
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Feature Extraction status: {status}")
            handle_exception(e, FeautreExtractionError)

if __name__ == "__main__":
    config = ConfigurationManager()
    feature_extraction_config = config.get_feature_extraction_config()

    feature_extraction =  FeatureExtraction(config = feature_extraction_config)
    feature_extraction.extract_features()
