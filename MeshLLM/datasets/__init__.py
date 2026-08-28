import importlib


def find_dataset_def(dataset_name, class_name=None):
    """Import datasets.{dataset_name} and return the dataset class.

    Examples:
      find_dataset_def('mesh_dataset', 'MeshDataset')
      find_dataset_def('mesh_dataset')  # defaults class_name=MeshDataset
    """
    module = importlib.import_module(f"datasets.{dataset_name}")
    if class_name is None:
        # mesh_dataset -> MeshDataset
        class_name = "".join(part.capitalize() for part in dataset_name.split("_"))
    return getattr(module, class_name)
