import yaml

def load_yaml_file(filepath):
    """
    Utility function to load a YAML file.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict or list: The contents of the YAML file as a Python object.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {filepath}: {e}")
            raise


