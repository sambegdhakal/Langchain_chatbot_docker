import os

def test_env_exists():
    assert os.path.exists(".env"), ".env file is missing"

def test_datasets_exist():
    required_paths = [
        "datasets/commentary_data",
        "datasets/playerStats_data",
        "datasets/base_data",
    ]
    for path in required_paths:
        assert os.path.exists(path), f"{path} is missing"
