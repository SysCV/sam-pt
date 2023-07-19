import hydra
import pandas as pd
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .train_net_video import *


def main_inner(cfg: DictConfig) -> None:
    # Setup config
    detectron2_config = cfg.DETECTRON2_CONFIG
    default_setup(detectron2_config, {"eval_only": True})

    # Setup logging
    setup_logger(name="point_tracking_vis_eval")
    setup_logger(output=cfg.DETECTRON2_CONFIG.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="point_tracking_video")
    if comm.is_main_process():
        wandb.init(
            entity=cfg.logging.wandb.entity,
            project=cfg.logging.wandb.project,
            name=cfg.logging.exp_id,
            group=cfg.logging.exp_id,
            config={
                "cfg": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                "work_dir": os.getcwd(),
                "hydra_cfg": HydraConfig.get() if HydraConfig.instance().cfg is not None else None,
            },
        )
        wandb.run.log_code(cfg.logging.wandb.log_code_path)
        wandb.run.summary["work_dir"] = os.path.abspath(os.getcwd())

    # Load model
    model = instantiate(cfg.model)
    model = model.to(cfg.device)
    model = model.eval()

    # Evaluate model
    results = Trainer.test(detectron2_config, model)
    print(f"Process {comm.get_rank()} has finished evaluation. Results: {results}")
    if detectron2_config.TEST.AUG.ENABLED:
        raise NotImplementedError
    if comm.is_main_process():
        print("Results verification by the main process has started")
        verify_results(detectron2_config, results)
        print("Results verification has finished")

        df_global = pd.DataFrame.from_dict(results["segm"], orient="index").T
        wandb.log({"df_global": wandb.Table(dataframe=df_global)})
        wandb.run.summary["score"] = df_global["AR100"].item()


@hydra.main(config_path="../../configs", config_name="vis_eval_sam_pt", version_base="1.1")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    launch(
        main_inner,
        num_gpus_per_machine=cfg.num_gpus_per_machine,
        num_machines=cfg.num_machines,
        machine_rank=cfg.machine_rank,
        dist_url=cfg.dist_url,
        args=(cfg,),
    )


if __name__ == "__main__":
    main()
