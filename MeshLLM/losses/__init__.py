import importlib


def find_loss_def(file_name, loss_name):
    """Import losses.{file_name} and return `{loss_name}` (e.g. LossS1)."""
    module = importlib.import_module(f"losses.{file_name}")
    return getattr(module, loss_name)
