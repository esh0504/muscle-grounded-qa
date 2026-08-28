import importlib


def find_model_def(file_name, model_name):
    """Import models.{file_name} and return `{model_name}`.

    Supports nested modules, e.g. ``stage1.model`` → ``models.stage1.model``.
    """
    module = importlib.import_module(f"models.{file_name}")
    return getattr(module, model_name)
