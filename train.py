from ultralytics import YOLO, settings
from pathlib import Path
from typing import Union, List
import argparse

# for logging
# import wandb


def train(
    model: str = None,
    data: str = None,
    epochs: int = 100,
    patience: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    save: bool = False,
    save_period: int = -1,
    cache: bool = False,
    device=None,
    workers: int = 8,
    project: str = "runs",
    name: str = "exp",
    exist_ok: bool = False,
    pretrained: bool = True,
    optimizer: str = "auto",
    verbose: bool = True,
    seed: int = 0,
    deterministic: bool = True,
    single_cls: bool = True,
    rect: bool = True,
    cos_lr: bool = False,
    close_mosaic: int = 10,
    resume: bool = False,
    amp: bool = True,
    fraction: float = 1.0,
    profile: bool = False,
    freeze: Union[int, List, None] = None,
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3,
    warmup_momentum: float = 0.8,
    warmup_bias_lr: float = 0.1,
    box: float = 7.5,
    cls: float = 0.5,
    dfl: float = 1.5,
    label_smoothing: float = 0.0,
    nbs: int = 64,
    val: bool = True,
    plots: bool = False,
):
    """Train a model.

    Args:
        model (str, optional): path to model file, i.e. yolov8n.pt, yolov8n.yaml
        data (str, optional): path to data file, i.e. coco128.yaml
        epochs (int, optional): number of epochs to train for. Defaults to 100.
        patience (int, optional): epochs to wait for no observable improvement for early stopping of training. Defaults to 50.
        batch (int, optional): number of images per batch (-1 for AutoBatch). Defaults to 16.
        imgsz (int, optional): size of input images as integer. Defaults to 640.
        save (bool, optional): save train checkpoints and predict results. Defaults to False.
        save_period (int, optional): Save checkpoint every x epochs (disabled if < 1). Defaults to -1.
        cache (bool, optional): True/ram, disk or False. Use cache for data loading. Defaults to False.
        device (_type_, optional): device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu. Defaults to None.
        workers (int, optional): number of worker threads for data loading (per RANK if DDP). Defaults to 8.
        project (str, optional): project name. Defaults to "runs".
        name (str, optional): experiment name. Defaults to "exp".
        exist_ok (bool, optional): whether to overwrite existing experiment. Defaults to False.
        pretrained (bool, optional): (bool or str) whether to use a pretrained model (bool) or a model to load weights from (str). Defaults to True.
        optimizer (str, optional): optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]. Defaults to "auto".
        verbose (bool, optional): whether to print verbose output. Defaults to True.
        seed (int, optional): random seed for reproducibility. Defaults to 0.
        deterministic (bool, optional): whether to enable deterministic mode. Defaults to True.
        single_cls (bool, optional): train multi-class data as single-class. Defaults to True.
        rect (bool, optional): rectangular training with each batch collated for minimum padding. Defaults to True.
        cos_lr (bool, optional): use cosine learning rate scheduler. Defaults to False.
        close_mosaic (int, optional): disable mosaic augmentation for final epochs (0 to disable). Defaults to 10.
        resume (bool, optional): resume training from last checkpoint. Defaults to False.
        amp (bool, optional): Automatic Mixed Precision (AMP) training, choices=[True, False]. Defaults to True.
        fraction (float, optional): dataset fraction to train on (default is 1.0, all images in train set). Defaults to 1.0.
        profile (bool, optional): profile ONNX and TensorRT speeds during training for loggers. Defaults to False.
        freeze (Union[int, List, None], optional): (int or list, optional) freeze first n layers, or freeze list of layer indices during training. Defaults to None.
        lr0 (float, optional): initial learning rate (i.e. SGD=1E-2, Adam=1E-3). Defaults to 0.01.
        lrf (float, optional): final learning rate (lr0 * lrf). Defaults to 0.01.
        momentum (float, optional): SGD momentum/Adam beta1. Defaults to 0.937.
        weight_decay (float, optional): optimizer weight decay 5e-4. Defaults to 0.0005.
        warmup_epochs (float, optional): warmup epochs (fractions ok). Defaults to 3.
        warmup_momentum (float, optional): warmup initial momentum. Defaults to 0.8.
        warmup_bias_lr (float, optional): warmup initial bias lr. Defaults to 0.1.
        box (float, optional): box loss gain. Defaults to 7.5.
        cls (float, optional): cls loss gain (scale with pixels). Defaults to 0.5.
        dfl (float, optional): dfl loss gain. Defaults to 1.5.
        label_smoothing (float, optional): label smoothing (fraction). Defaults to 0.0.
        nbs (int, optional): nominal batch size. Defaults to 64.
        val (bool, optional): validate/test during training. Defaults to True.
        plots (bool, optional): save plots and images during train/val. Defaults to False.
    """

    model = YOLO(model)
    results = model.train(
        data=data,
        epochs=epochs,
        patience=patience,
        batch_size=batch,
        imgsz=imgsz,
        save=save,
        save_period=save_period,
        cache_images=cache,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=exist_ok,
        pretrained=pretrained,
        opt=optimizer,
        verbose=verbose,
        seed=seed,
        single_cls=single_cls,
        rect=rect,
        resume=resume,
        amp=amp,
        fraction=fraction,
        profile=profile,
        freeze=freeze,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        box_loss_gain=box,
        cls_loss_gain=cls,
        label_smoothing=label_smoothing,
        dfl_loss_gain=dfl,
        nbs=nbs,
        val=val,
        plots=plots,
        deterministic=deterministic,
        cos_lr=cos_lr,
        close_mosaic=close_mosaic,
    )


def parse_device_arg(device_arg):
    if device_arg is not None:
        try:
            # Try to convert to int
            device_arg = int(device_arg)
        except ValueError:
            # If it fails, try to convert to list of ints
            device_arg = [int(device) for device in device_arg.split(",")]
    return device_arg


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    settings.update(
        {
            "comet": False,
            "mlflow": False,
            "neptune": False,
            "tensorboard": False,
            "dvc": False,
            "wandb": True,  # if you want to log to wandb
        }
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        required=True,
        help="path to model file, i.e. yolov8n.pt, yolov8n.yaml",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default="data.yaml",
        required=True,
        help="path to YAML file. if current dir is data/, else absolute path to YAML file",
    )
    parser.add_argument(
        "--epochs", type=int, required=True, default=100, help="number of epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="patience for early stopping"
    )
    parser.add_argument(
        "--batch", type=int, required=True, default=42, help="batch size"
    )
    parser.add_argument(
        "--imgsz", type=int, required=True, default=1088, help="image size"
    )
    parser.add_argument(
        "--save",
        type=str2bool,
        default=True,
        help="whether to save checkpoint after each epoch",
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=-1,
        help="save checkpoint every x epochs (disabled if < 1)",
    )
    parser.add_argument(
        "--cache",
        type=str2bool,
        default=False,
        help="use cache for data loading",
    )
    parser.add_argument(
        "--device",
        type=parse_device_arg,
        default=None,
        help="device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of worker threads for data loading",
    )
    parser.add_argument(
        "--project", type=str, default="runs", help="project name for logging"
    )
    parser.add_argument(
        "--name", type=str, default="exp", help="experiment name for logging"
    )
    parser.add_argument(
        "--exist_ok",
        type=str2bool,
        default=False,
        help="whether to overwrite existing experiment",
    )
    parser.add_argument(
        "--pretrained",
        type=str2bool,
        default=True,
        help="(bool or str) whether to use a pretrained model (bool) or a model to load weights from (str)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"],
        help="optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]",
    )
    parser.add_argument(
        "--verbose", type=str2bool, default=True, help="whether to print verbose output"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--deterministic",
        type=str2bool,
        default=True,
        help="whether to enable deterministic mode",
    )
    parser.add_argument(
        "--single_cls",
        type=str2bool,
        default=True,
        help="train multi-class data as single-class",
    )
    parser.add_argument(
        "--rect",
        type=str2bool,
        default=True,
        help="rectangular training with each batch collated for minimum padding",
    )
    parser.add_argument(
        "--cos_lr",
        type=str2bool,
        default=False,
        help="use cosine learning rate scheduler",
    )
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=10,
        help="disable mosaic augmentation for final epochs (0 to disable)",
    )
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="resume training from the last checkpoint",
    )
    parser.add_argument(
        "--amp",
        type=str2bool,
        default=True,
        help="Automatic Mixed Precision (AMP) training",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="dataset fraction to train on (default is 1.0, all images in train set)",
    )
    parser.add_argument(
        "--profile",
        type=str2bool,
        default=False,
        help="profile ONNX and TensorRT speeds during training for loggers",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="(int or list, optional) freeze first n layers, or freeze list of layer indices during training",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="initial learning rate (i.e. SGD=1E-2, Adam=1E-3)",
    )
    parser.add_argument(
        "--lrf", type=float, default=0.01, help="final learning rate (lr0 * lrf)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.937, help="SGD momentum/Adam beta1"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="optimizer weight decay"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=3,
        help="warmup epochs (fractions ok)",
    )
    parser.add_argument(
        "--warmup_momentum", type=float, default=0.8, help="warmup initial momentum"
    )
    parser.add_argument(
        "--warmup_bias_lr", type=float, default=0.1, help="warmup initial bias lr"
    )
    parser.add_argument("--box", type=float, default=7.5, help="box loss gain")
    parser.add_argument(
        "--cls", type=float, default=0.5, help="cls loss gain (scale with pixels)"
    )
    parser.add_argument("--dfl", type=float, default=1.5, help="dfl loss gain")
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0, help="label smoothing (fraction)"
    )
    parser.add_argument("--nbs", type=int, default=64, help="nominal batch size")
    parser.add_argument(
        "--val", type=str2bool, default=True, help="validate/test during training"
    )
    parser.add_argument(
        "--plots", type=str2bool, default=True, help="plot training results as png"
    )
    parser.add_argument(
        "--logging", type=str2bool, default=True, help="whether to log to wandb"
    )

    args = parser.parse_args()
    if args.logging:
        import wandb

        # start logging
        wandb.login(relogin=True)
        wandb.init(project=args.project, name=args.name)

    train(
        args.model,
        args.data,
        args.epochs,
        args.patience,
        args.batch,
        args.imgsz,
        args.save,
        args.save_period,
        args.cache,
        args.device,
        args.workers,
        args.project,
        args.name,
        args.exist_ok,
        args.pretrained,
        args.optimizer,
        args.verbose,
        args.seed,
        args.deterministic,
        args.single_cls,
        args.rect,
        args.cos_lr,
        args.close_mosaic,
        args.resume,
        args.amp,
        args.fraction,
        args.profile,
        args.freeze,
        args.lr0,
        args.lrf,
        args.momentum,
        args.weight_decay,
        args.warmup_epochs,
        args.warmup_momentum,
        args.warmup_bias_lr,
        args.box,
        args.cls,
        args.dfl,
        args.label_smoothing,
        args.nbs,
        args.val,
        args.plots,
    )
    
    # end logging
    if args.logging:
        wandb.finish()
