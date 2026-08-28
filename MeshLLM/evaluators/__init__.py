import importlib


def find_evaluator_def(file_name, class_name):
    """Import evaluators.{file_name} and return `{class_name}` (e.g. EvaluatorUnseen)."""
    module = importlib.import_module(f"evaluators.{file_name}")
    return getattr(module, class_name)
