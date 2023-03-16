# the loop for generating new events starting from gen-level information in the files
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import os

sys.path.insert(0, os.path.join("..", "..", "models"))

from modded_basic_nflow import load_model
import nbd_func

if __name__ == "__main__":

    # specify old/new root
    root = "/gpfs/ddn/srm/cms//store/mc/RunIIAutumn18NanoAODv6/TTJets_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/Nano25Oct2019_102X_upgrade2018_realistic_v20_ext1-v1/"
    new_root = "/gpfs/ddn/cms/user/cattafe/TTJets/"
    ttbar_training_files = [
        "250000/047F4368-97D4-1A4E-B896-23C6C72DD2BE.root",
        "240000/B38E7351-C9E4-5642-90A2-F075E2411B00.root",
        "230000/DA422D8F-6198-EE47-8B00-1D94D97950B6.root",
        "230000/393066F3-390A-EC4A-9873-BF4D4D7FBE4F.root",
        "230000/12C9A5BF-1608-DA48-82E9-36F18051CE31.root",
        "230000/12C8AFA5-B554-9540-8603-2DF948304880.root",
        "250000/02B1F58F-7798-FB44-BF80-56C3DC1B6E52.root",
        "230000/78137863-DAD0-E740-B357-D88AF92BE59F.root",
        "230000/91456D0B-2FDE-2B4F-8C7A-8E60260480CD.root",
    ]
    files_paths = [
        os.path.join(d, f)
        for d in os.listdir(root)
        for f in os.listdir(os.path.join(root, d))
    ]  # = [x for x in pathlib.Path(root).glob('**/*')]

    # optionally remove training files if we are generating ttbar dataset
    files_paths = [path for path in files_paths if path not in ttbar_training_files]
    # take remaining files if loop crashes
    files_paths = files_paths[:2]

    # # print(files_paths)

    # root = "/gpfs/ddn/srm/cms//store/mc/RunIIAutumn18NanoAODv6/DY2JetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/Nano25Oct2019_102X_upgrade2018_realistic_v20-v1/"
    # new_root = "/gpfs/ddn/cms/user/cattafe/DYJets/"
    # files_paths = [
    #     os.path.join(d, f)
    #     for d in os.listdir(root)
    #     for f in os.listdir(os.path.join(root, d))
    # ]

    print(f"We will process a total of {len(files_paths)} files")
    # specify device and load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    ele_flow, _, _, _, trh, tsh = load_model(
        device=device,
        model_dir=os.path.dirname(__file__),
        filename="EM1/checkpoint-latest.pt",
    )  # to be changed

    ele_flow = ele_flow.to(device)

    # generation loop
    for path in tqdm(files_paths):
        # torch.cuda.empty_cache()
        path_str = str(path)  # shouldn't be needed
        nbd_func.nbd(ele_flow, root, path_str, new_root)
