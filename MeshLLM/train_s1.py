"""Stage-1 training entrypoint (3D displacement → muscle activations)."""

import hydra
from omegaconf import DictConfig, OmegaConf

from datasets import find_dataset_def
from losses import find_loss_def
from models import find_model_def
from trainers import find_trainer_def


@hydra.main(version_base=None, config_path="configs", config_name="train_s1")
def main(cfg: DictConfig):
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    DatasetClass = find_dataset_def(
        cfg.datasets.name, getattr(cfg.datasets, "class_name", None)
    )
    train_dataset = DatasetClass(cfg.datasets)

    val_cfg = OmegaConf.to_container(cfg.datasets, resolve=True)
    val_cfg["split"] = "val"
    val_dataset = DatasetClass(val_cfg)

    ModelClass = find_model_def(cfg.models.name, cfg.models.class_name)
    model = ModelClass(cfg.models)

    LossClass = find_loss_def(cfg.losses.name, cfg.losses.class_name)
    loss_fn = LossClass(cfg.losses)

    TrainerClass = find_trainer_def(cfg.trainers.name, cfg.trainers.class_name)
    trainer = TrainerClass(cfg.trainers, experiment_cfg=cfg)

    print(
        f"[Stage-1] dataset train/val = {len(train_dataset)}/{len(val_dataset)}  "
        f"model={type(model).__name__}  loss={type(loss_fn).__name__}  "
        f"trainer={type(trainer).__name__}"
    )

    if cfg.run.do_train:
        trainer.fit(model, loss_fn, train_dataset, val_dataset=val_dataset)

    if cfg.run.do_eval and not cfg.run.do_train:
        val_loss = trainer.evaluate(model, loss_fn, val_dataset)
        print(f"val_loss={val_loss:.6f}")

    return 0


if __name__ == "__main__":
    main()
