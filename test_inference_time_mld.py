import logging
import os
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm

from mld.config import parse_args
# from mld.datasets.get_dataset import get_datasets
from mld.data.get_data import get_datasets
from mld.data.sampling import subsample, upsample
from mld.models.get_model import get_model
from mld.utils.logger import create_logger


def main():
    """
    get input text
    ToDo skip if user input text in command
    current tasks:
         1 text 2 mtion
         2 motion transfer
         3 random sampling
         4 reconstruction

    ToDo
    1 use one funtion for all expoert
    2 fitting smpl and export fbx in this file
    3

    """
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")

    if cfg.DEMO.EXAMPLE:
        # Check txt file input
        # load txt
        from mld.utils.demo_utils import load_example_input

        text, length = load_example_input(cfg.DEMO.EXAMPLE)
        task = "Example"
    elif cfg.DEMO.TASK:
        task = cfg.DEMO.TASK
        text = None
    else:
        # keyborad input
        task = "Keyborad_input"
        text = input("Please enter texts, none for random latent sampling:")
        length = input(
            "Please enter length, range 16~196, e.g. 50, none for random latent sampling:"
        )
        if text:
            motion_path = input(
                "Please enter npy_path for motion transfer, none for skip:")
        # text 2 motion
        if text and not motion_path:
            cfg.DEMO.MOTION_TRANSFER = False
        # motion transfer
        elif text and motion_path:
            # load referred motion
            joints = np.load(motion_path)
            frames = subsample(
                len(joints),
                last_framerate=cfg.DEMO.FRAME_RATE,
                new_framerate=cfg.DATASET.KIT.FRAME_RATE,
            )
            joints_sample = torch.from_numpy(joints[frames]).float()

            features = model.transforms.joints2jfeats(joints_sample[None])
            motion = xx
            # datastruct = model.transforms.Datastruct(features=features).to(model.device)
            cfg.DEMO.MOTION_TRANSFER = True

        # default lengths
        length = 200 if not length else length
        length = [int(length)]
        text = [text]

    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    # cuda options
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")

    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]

    # create mld model
    total_time = time.time()
    model = get_model(cfg, dataset)

    # ToDo
    # 1 choose task, input motion reference, text, lengths
    # 2 print task, input, output path
    #
    # logger.info(f"Input Text: {text}\nInput Length: {length}\nReferred Motion: {motion_path}")
    # random samlping
    if not text:
        logger.info(f"Begin specific task{task}")

    # debugging
    # vae
    # ToDo Remove this
    # temp loading
    # if cfg.TRAIN.PRETRAINED_VAE:
    #     logger.info("Loading pretrain vae from {}".format(cfg.TRAIN.PRETRAINED_VAE))
    #     ckpt = torch.load(cfg.TRAIN.PRETRAINED_VAE, map_location="cpu")
    #     model.load_state_dict(ckpt["state_dict"], strict=False)

    # /apdcephfs/share_1227775/shingxchen/AIMotion/TMOSTData/exps/actor/ACTOR_1010_vae_feats_kl/checkpoints/epoch=1599.ckpt

    # loading checkpoints
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]
    # # remove mismatched and unused params
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     old, new = "denoiser.decoder.0.", "denoiser.decoder."
    #     # old1, new1 = "text_encoder.text_model.text_model", "text_encoder.text_model.vision_model"
    #     old1 = "text_encoder.text_model.vision_model"
    #     if k[: len(old)] == old:
    #         name = k.replace(old, new)
    #     # elif k[: len(old)] == old:
    #     #     name = k.replace(old, new)
    #     else:
    #         name = k

    #     new_state_dict[name] = v
    #     # if k.split(".")[0] not in ["text_encoder", "denoiser"]:
    #     #     new_state_dict[k] = v
    # model.load_state_dict(new_state_dict, strict=False)

    model.load_state_dict(state_dict, strict=True)

    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    inference_times = []

    mld_time = time.time()

    # sample
    with torch.no_grad():
        rep_lst = []
        rep_ref_lst = []
        texts_lst = []
        # task: input or Example
        assert (text, "only test text generation time!")
        text_list = text
        length_list = length
        # prepare batch data
        num_samples = len(text_list)
        # number of iteration required
        assert num_samples % cfg.DEMO.BATCH_SIZE == 0, "Error: num_samples must not be a multiple of batch_size."
        iteration_required = int(num_samples / cfg.DEMO.BATCH_SIZE)

        for i in range(iteration_required):
            text_i_start = i * cfg.DEMO.BATCH_SIZE
            text_i_end = (i + 1) * cfg.DEMO.BATCH_SIZE
            start_time = time.perf_counter()
            batch = {"length": length_list[text_i_start:text_i_end], "text": text_list[text_i_start:text_i_end]}
            # text motion transfer
            if cfg.DEMO.MOTION_TRANSFER:
                joints = model.forward_motion_style_transfer(batch)
            # text to motion synthesis
            else:
                joints = model(batch)

            # cal inference time
            inference_times.append(time.perf_counter() - start_time)

    out_path = output_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    inference_times_mean = np.mean(inference_times)
    inference_times_std = np.std(inference_times)
    with open(os.path.join(out_path, 'inference_times.txt'), 'w') as fw:
        fw.write("batch size: " + str(cfg.DEMO.BATCH_SIZE) + "\n")
        fw.write("mean: " + str(inference_times_mean) + "\n")
        fw.write("std: " + str(inference_times_std) + "\n")

    with open(os.path.join(out_path, 'inference_times.txt'), 'a') as fw:
        fw.write(str(inference_times))

    return


if __name__ == "__main__":
    main()
