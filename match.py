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


from torch.nn.functional import cosine_similarity


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
    test_data = dataset.test_dataloader()

    model = get_model(cfg, dataset)

    # loading checkpoints
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]

    model.load_state_dict(state_dict, strict=True)

    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    # sample
    with torch.no_grad():
        length_lst = []
        motion_rep_lst = []
        texts_rep_lst = []
        
        for i, data in enumerate(test_data):
            texts = data['text']
            lengths = data['length']
            motions = data['motion'].to(model.device)
        
            # text
            uncond_tokens = [""] * len(texts)
            uncond_tokens.extend(texts)
            texts = uncond_tokens
            text_emb = model.text_encoder(texts)
            
            texts_rep_lst += [model._diffusion_reverse(text_emb, lengths)[0]]
            
            # motion
            z, _ = model.vae.encode(motions, lengths)
            motion_rep_lst += [z[0]]
            
            #length
            length_lst.append(lengths)

            id = 0
            npypath = str(output_dir /
                    f"{i}.npy")
            with open(npypath.replace(".npy", ".txt"), "w") as text_file:
                text_file.write(data["text"][0])
            np.save(npypath, model.feats2joints(data['motion'].detach().cpu())[0])
            logger.info(f"Motions are generated here:\n{npypath}")
            
            if i == 1000:
                break

        num_top_matches = 5

        # 计算所有可能的 text-motion 对的余弦相似度
        similarities = []
        for i, text_rep in enumerate(texts_rep_lst):
            for j, motion_rep in enumerate(motion_rep_lst):
                similarity = cosine_similarity(text_rep, motion_rep).item()
                similarities.append((i, j, similarity))

        # 对于每个 text，找出最匹配的三个 motion 并写入文件
        with open("text_to_motion_matches.txt", "w") as f:
            for i in range(len(texts_rep_lst)):
                top_matches = sorted([sim for sim in similarities if sim[0] == i], key=lambda x: x[2], reverse=True)[:num_top_matches]
                match_indices = [match[1] for match in top_matches]
                f.write(f"Text {i} matches with motions {match_indices}\n")

        # 对于每个 motion，找出最匹配的三个 text 并写入文件
        with open("motion_to_text_matches.txt", "w") as f:
            for j in range(len(motion_rep_lst)):
                top_matches = sorted([sim for sim in similarities if sim[1] == j], key=lambda x: x[2], reverse=True)[:num_top_matches]
                match_indices = [match[0] for match in top_matches]
                f.write(f"Motion {j} matches with texts {match_indices}\n")


if __name__ == "__main__":
    main()
