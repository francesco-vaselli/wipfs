import basic_nflow

import sys
import os

sys.path.insert(0, os.path.join("..", "utils"))
from dataset import MyDataset


if __name__ == "__main__":

    # define hyperparams
    lr = 1e-5
    total_epochs = 600
    batch_size = 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the train and validations datasets
    train_ds = MyDataset(["./datasets/Ajets_and_muons1+.hdf5"], limit=5000000)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=9
    )
    test_ds = MyDataset(["./datasets/Ajets_and_muons7+.hdf5"], limit=400000)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=9
    )

    # define additional model parameters
    base_param_dict = {
        "num_transform_blocks": 10,
        "activation": "relu",
        "batch_norm": True,
        "num_bins": 128,
        "hidden_dim": 298,
        "block_size": 3,
    }
    
    input_dim = 17
    context_dim = 14
    num_flow_steps = 23
    transform_type = "block-permutation"

    # create model
    flow = create_NDE_model(input_dim, context_dim, num_flow_steps, base_param_dict, transform_type)

    # print total params number and stuff
    total_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print(device)
    print(total_params)
    print(len(train_ds))
    print(len(train_loader.dataset))

    # set optimizer, send to device and train
    # remember that the model is being saved every 10 epochs
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    flow.to(device)

    trh, tsh = train(flow, train_loader, test_loader)
