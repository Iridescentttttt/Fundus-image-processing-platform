import torch
import numpy as np
import os
import shutil
from enum import Enum
from typing import Any, Tuple, Union, Optional

import torch
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union, Optional, Any, Tuple

def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def load_state_dict(
        model: nn.Module,
        model_weights_path: str,
        ema_model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        load_mode: Optional[str] = None,
) -> Union[Tuple[nn.Module, Optional[nn.Module], Any, Any, Any, Optional[Optimizer], Optional[_LRScheduler]],
           Tuple[nn.Module, Any, Any, Any, Optional[Optimizer], Optional[_LRScheduler]],
           nn.Module]:
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

    if load_mode == "resume":
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the model weights to the current model (base model)
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        scheduler.load_state_dict(checkpoint["scheduler"])

        if ema_model is not None:
            # Load ema model state dict. Extract the fitted model weights
            ema_model_state_dict = ema_model.state_dict()
            ema_state_dict = {k: v for k, v in checkpoint["ema_state_dict"].items() if k in ema_model_state_dict.keys()}
            # Overwrite the model weights to the current model (ema model)
            ema_model_state_dict.update(ema_state_dict)
            ema_model.load_state_dict(ema_model_state_dict)
            return model, ema_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler

        return model, start_epoch, best_psnr, best_ssim, optimizer, scheduler
    else:
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        # Overwrite the model weights to the current model
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)

        return model


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_checkpoint(
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        is_best: bool = False,
        is_last: bool = False,
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name)
    torch.save(state_dict, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, "LSRGAN_x2.pth.tar"))
    if is_last:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, "last.pth.tar"))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
