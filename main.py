"""
Entry point training and testing FDT
"""
import argparse
import numpy as np
import json
import logging
from utils import utils
import time
from models.fdt.FDT import FDT
from models.fdt.FDTCriterion import FDTCriterion
from os.path import join
from datasets.data_loader_lmdb_augmenter import LMDBDataLoaderAugmenter
from datasets.data_loader_lmdb import LMDBDataLoader
from utils import plotutils
import torch

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("backbone_path", help="path to backbone .pth - e.g. efficientnet")
    arg_parser.add_argument("dataset_name", help="name of dataset")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--plot", action="store_true",
                            help="plot results")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution detailsZ
    logging.info("Start {} FDT".format(args.mode))
    logging.info("Using dataset: {}".format(args.dataset_name))

    # Read configuration
    with open('config.json', "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.dataset_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))
    config["backbone"] = args.backbone_path

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model
    model = FDT(config).to(device)
    model.put_heads_on_device(device)

    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(config.checkpoint_path))

    # Loads npys:
    npys = config.get("npys")
    for key, path in npys.items():
        config[key] = np.load(path)

    if args.mode == 'train':
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        model.train()
        criterion = FDTCriterion(config)
        max_norm = config.get("clip_max_norm")

        params = list(model.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                 lr=config.get('lr'),
                                 eps=config.get('eps'),
                                 weight_decay=config.get('weight_decay'))

        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        train_loader = LMDBDataLoaderAugmenter(config, config.get("train_source"))
        logging.info("Training with {} images.".format(len(train_loader.dataset)))
        plot = False
        for epoch in range(config.get("num_epochs")):

            idx = 0
            for idx, data in enumerate(train_loader):
                imgs, targets = data
                if args.plot:
                    with torch.no_grad():
                        gt_boxes_for_plotting = targets[0]['boxes']
                        img_for_plotting = imgs[0].permute(1, 2, 0).cpu().numpy()
                        plotutils.plot_bboxes(img_for_plotting, gt_boxes_for_plotting)

                imgs = [image.unsqueeze(0).to(device) for image in imgs]
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]

                outputs = model.forward(imgs)
                loss_dict = criterion(outputs, targets)
                #weight_dict = criterion.weight_dict
                #losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                losses = sum(loss_dict[k] for k in loss_dict.keys())

                optim.zero_grad()
                losses.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optim.step()

                # Record loss and performance on train set
                if idx % n_freq_print == 0:
                    logging.info("FDT loss: {:.2f}".format(losses.item()))

                # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), config.checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

                # Scheduler update
            scheduler.step()

            logging.info('Training completed')
            torch.save(model.state_dict(), config.checkpoint_prefix + '_final.pth'.format(epoch))

    else:  # eval
        # loads validation dataset generator if a validation dataset is given
        assert config.get("val_source") is not None
        val_loader = LMDBDataLoader(config, config.get("val_source"), False)


