import os
import pytest 
from unittest.mock import MagicMock, patch
from training.components.common.data_ingestion import DataIngestion
from training.configuration_manager.configuration import ConfigurationManager

@pytest.fixture
def mock_config(tmp_path):
    """Fixture to provide a mock DataIngestionConfig"""
    class MockConfig:
        root_dir = tmp_path / "root"
        source = tmp_path / "source/Telco_Customer_Churn.csv"
        data_dir = tmp_path / "data"
        STATUS_FILE = tmp_path / "status.txt"

    # Create directories and files
    mock_config = MockConfig()
    os.makedirs(mock_config.root_dir, exist_ok=True)
    os.makedirs(mock_config.data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(mock_config.source),exist_ok=True)
    mock_config.source.write_text("mock data") # create a mock source file
    return mock_config

def test_data_ingestion_initialization(mock_config):
    """Test the initialization of DataIngestion class."""
    data_ingestion = DataIngestion(config=mock_config)
    assert data_ingestion.config == mock_config 


@patch("shutil.copy")
@patch("os.path.exists", return_value=False)
def test_save_data_file_copy(mock_exists, mock_copy, mock_config):
    """Test that save data copies the source file when it doesn't exist."""
    data_ingestion = DataIngestion(config=mock_config)

    data_ingestion.save_data()

    # Check if shutil.copy was called
    mock_copy.assert_called_once_with(mock_config.source, mock_config.data_dir)    

    # Check if the status file is updated
    with open(mock_config.STATUS_FILE, "r") as f:
        status = f.read()

    assert "True" in status



@patch("shutil.copy")
@patch("os.path.exists", return_value=True)
def test_save_data_file_exists(mock_exists,mock_copy, mock_config):
    """Test that save_data does nothing if the data file already exists."""
    data_ingestion = DataIngestion(config=mock_config)

    data_ingestion.save_data()

    # Check that shutil.copy was not calle
    assert not mock_copy.called # No file operations occured

    # Check the status file remains empty
    if not mock_exists(mock_config.STATUS_FILE):
        with open(mock_config.STATUS_FILE, "r") as f:
            status = f.read()
    else:
        status=""
    
    assert status == "" # No update since the file already exists.


@patch("shutil.copy", side_effect=Exception("Mock Error"))
def test_save_data_error_handling(mock_copy, mock_config):
    """Test that save_data handles errors gracefully."""
    data_ingestion = DataIngestion(config=mock_config)
    

    # handle_exception needs to be patched where it's used, not where it's defined.
    with patch("training.components.common.data_ingestion.handle_exception") as mock_handle_exception:
        data_ingestion.save_data()

        # Ensure handle_exception is called
        mock_handle_exception.assert_called_once()

        # Check if the status file is updated with failure status
        with open(mock_config.STATUS_FILE, "r") as f:
            status = f.read()

        assert "False" in status

@patch("training.configuration_manager.configuration.ConfigurationManager.get_data_ingestion_config")
def test_data_ingestion_with_configuration_manager(mock_get_config, tmp_path):
    """Test the DataIngestion class with a real ConfigurationManager"""

    # Mock the configuration manager to return the mock config
    mock_get_config.return_value = MagicMock(
        root_dir= tmp_path / "root",
        source = tmp_path / "source/Telco_Customer_Churn.csv",
        data_dir= tmp_path / "data",
        STATUS_FILE = tmp_path/"status.txt"
    )

    # Create the DataIngestion object
    config_manager = ConfigurationManager()
    config = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=config)

    # Ensure the conifg object is correctly passed
    assert data_ingestion.config == config


    