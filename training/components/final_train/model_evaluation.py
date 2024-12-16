import sys
import os
import joblib
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

from training.exception import ModelEvaluationError, handle_exception
from training.custom_logging import info_logger, error_logger

from training.entity.config_entity import ModelEvaluationConfig
from training.configuration_manager.configuration import ConfigurationManager


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_model(self):
        try:
            info_logger.info("Loading final model from artifacts folder")

            model_path = os.path.join(self.config.model_path, "final_model.joblib")
            final_model = joblib.load(model_path)

            info_logger.info("Final model loaded from artifacts folder")

            return final_model
        except Exception as e:
            handle_exception(e, ModelEvaluationError)

    def load_test_data(self):
        try:
            info_logger.info("Loading test data for final model evaluation")

            test_data_path = os.path.join(self.config.final_test_data_path, "Test.npz")
            test_data = np.load(test_data_path, allow_pickle=True)

            xtest = test_data["xtest"]
            ytest = test_data["ytest"]

            info_logger.info("Loaded test data for final model evaluation")

            return xtest, ytest
        except Exception as e:
            handle_exception(e, ModelEvaluationError)

    def evaluate_model(self, final_model, xtest, ytest):
        try:
            info_logger.info("Model evaluation started")

            # Make predictions
            y_pred = final_model.predict(xtest)
            y_pred_proba = final_model.predict_proba(xtest)[:, 1]  # Probabilities for ROC-AUC

            # Compute evaluation metrics
            accuracy = accuracy_score(ytest, y_pred)
            roc_auc = roc_auc_score(ytest, y_pred_proba)
            report = classification_report(ytest, y_pred)
            confusion = confusion_matrix(ytest, y_pred)

            # Log metrics
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Model Evaluation status: True \n")
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"ROC-AUC: {roc_auc}\n")
                f.write(f"Classification Report: \n{report}\n")
                f.write(f"Confusion Matrix: \n{confusion}\n")

            info_logger.info("Model evaluation completed")
            info_logger.info(f"Accuracy: {accuracy}, ROC-AUC: {roc_auc}")
        except Exception as e:
            handle_exception(e, ModelEvaluationError)


if __name__ == "__main__":
    config = ConfigurationManager()
    model_evaluation_config = config.get_model_evaluation_config()

    model_evaluation = ModelEvaluation(config=model_evaluation_config)
    final_model = model_evaluation.load_model()
    xtest, ytest = model_evaluation.load_test_data()

    model_evaluation.evaluate_model(final_model, xtest, ytest)
