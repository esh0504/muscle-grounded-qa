import importlib


def find_trainer_def(file_name, class_name):
    """Import trainers.{file_name} and return `{class_name}` (e.g. TrainerS1)."""
    module = importlib.import_module(f"trainers.{file_name}")
    return getattr(module, class_name)
