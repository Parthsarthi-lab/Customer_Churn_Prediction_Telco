import pytest
import pandas as pd
import os
from unittest.mock import patch, mock_open
from training.components.common.data_validation import DataValidation
from training.entity.config_entity import DataValidationConfig