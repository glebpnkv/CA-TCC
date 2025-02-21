import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

from config_files.config_factory import config_factory
from dataloader.dataloader import data_generator
from models.TC import TC
from models.model import BaseModel
from trainer.trainer import run_model_training, model_evaluate, gen_pseudo_labels
from utils import _calc_metrics, copy_files, _logger, set_requires_grad


def main(args):
    start_time = datetime.now()

    device = torch.device(args.device)
    experiment_description = args.experiment_description
    data_type = args.selected_dataset
    training_mode = args.training_mode
    run_description = args.run_description

    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok=True)

    config = config_factory.create(data_type)

    # ##### fix random seeds for reproducibility ########
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    #####################################################

    root_model_save_dir = str(os.path.join(
        logs_save_dir,
        experiment_description,
        run_description
    ))

    experiment_log_dir = os.path.join(root_model_save_dir, training_mode + f"_seed_{SEED}")

    os.makedirs(experiment_log_dir, exist_ok=True)

    # Logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug("=" * 45)

    # Load datasets
    data_path = os.path.join(args.data_path, data_type)
    train_dl, valid_dl, test_dl = data_generator(data_path, config, training_mode)
    logger.debug("Data loaded ...")

    # Load Model
    model = BaseModel(config).to(device)
    temporal_contr_model = TC(config, device).to(device)

    if "fine_tune" in training_mode or "ft_" in training_mode:
        # load saved model of this experiment
        if 'SupCon' not in training_mode:
            load_from = os.path.join(root_model_save_dir, f"self_supervised_seed_{SEED}", "saved_models")
        else:
            load_from = os.path.join(root_model_save_dir, f"SupCon_seed_{SEED}", "saved_models")

        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        model_dict = model.state_dict()
        del_list = ['logits']
        pretrained_dict_copy = pretrained_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del pretrained_dict[i]
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if training_mode == "gen_pseudo_labels":
        ft_perc = "1p"
        load_from = os.path.join(root_model_save_dir, f"ft_{ft_perc}_seed_{SEED}", "saved_models")
        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        model.load_state_dict(pretrained_dict)
        gen_pseudo_labels(model, train_dl, device, data_path)
        sys.exit(0)

    if "train_linear" in training_mode or "tl" in training_mode:
        if 'SupCon' not in training_mode:
            load_from = os.path.join(root_model_save_dir, f"self_supervised_seed_{SEED}", "saved_models")
        else:
            load_from = os.path.join(root_model_save_dir, f"SupCon_seed_{SEED}", "saved_models")
        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # delete these parameters (Ex: the linear layer at the end)
        del_list = ['logits']
        pretrained_dict_copy = pretrained_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del pretrained_dict[i]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

    if training_mode == "random_init":
        model_dict = model.state_dict()

        # delete all the parameters except for logits
        del_list = ['logits']
        pretrained_dict_copy = model_dict.copy()
        for i in pretrained_dict_copy.keys():
            for j in del_list:
                if j in i:
                    del model_dict[i]
        set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.

    if training_mode == "SupCon":
        data_perc = "1p"
        load_from = os.path.join(root_model_save_dir, f"ft_{data_perc}_seed_{SEED}", "saved_models")
        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        model.load_state_dict(pretrained_dict)


    model_optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2),
                                       weight_decay=3e-4)

    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=config.lr,
                                                betas=(config.beta1, config.beta2), weight_decay=3e-4)

    if training_mode == "self_supervised" or training_mode == "SupCon":  # to do it only once
        copy_files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

    # Trainer
    run_model_training(
        model=model,
        temporal_contr_model=temporal_contr_model,
        model_optimizer=model_optimizer,
        temp_cont_optimizer=temporal_contr_optimizer,
        train_dl=train_dl,
        valid_dl=valid_dl,
        test_dl=test_dl,
        device=device,
        logger=logger,
        config=config,
        experiment_log_dir=experiment_log_dir,
        training_mode=training_mode
    )

    if training_mode != "self_supervised" and training_mode != "SupCon" and training_mode != "SupCon_pseudo":
        # Testing
        outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        total_loss, total_acc, pred_labels, true_labels = outs
        _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

    logger.debug(f"Training time is : {datetime.now() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ######################## Model parameters ########################
    home_dir = os.getcwd()
    parser.add_argument(
        '--experiment_description',
        default='HAR_experiments',
        type=str,
        help='Experiment Description'
    )
    parser.add_argument('--run_description', default='test1', type=str, help='Experiment Description')
    parser.add_argument('--seed', default=0, type=int, help='seed value')
    parser.add_argument('--training_mode', default='self_supervised', type=str,
                        help='Modes of choice: random_init, supervised, self_supervised, SupCon, ft_1p, gen_pseudo_labels')

    parser.add_argument('--selected_dataset', default='HAR', type=str,
                        help='Dataset of choice: EEG, HAR, Epilepsy, pFD')
    parser.add_argument('--data_path', default=r'data/', type=str, help='Path containing dataset')

    parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
    parser.add_argument('--device', default='cuda:0', type=str, help='cpu or cuda')
    parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
    args = parser.parse_args()

    main(args)
