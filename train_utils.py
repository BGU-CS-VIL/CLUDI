


import torch
import numpy as np
from vit_new import *
from denoising_diffusion_pytorch import GaussianDiffusion
import math
from ema_pytorch import EMA
import os
import csv

def read_imagenet_classes(file_path):
    """Reads the ImageNet classes file and returns a dictionary mapping IDs to labels."""
    class_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ', 2)  # Splitting only on the first two spaces
            if len(parts) == 3:
                wordnet_id, label_number, label = parts
                class_dict[wordnet_id] = label_number
    return class_dict

def filter_labels(your_list_path, class_dict):
    """Filters and returns the labels for the given list of WordNet IDs."""
    filtered_labels = []
    with open(your_list_path, 'r') as file:
        for line in file:
            wordnet_id = line.strip().split(' ')[0]
            if wordnet_id in class_dict:
                filtered_labels.append(int(class_dict[wordnet_id]))

    return filtered_labels


def save_results_to_csv(results, directory, filename):
    """
    Appends a list of dictionary results to a CSV file, creating the file if it does not exist.

    Args:
    results (list of dicts): The results to be saved, where each dictionary represents a row.
    directory (str): The directory where the CSV file will be saved.
    filename (str): The name of the CSV file to save the results.

    Example of results:
    [{'iteration': 0, 'NMI': 0.85, 'ARI': 0.65, 'ACC': 0.95},
     {'iteration': 1, 'NMI': 0.86, 'ARI': 0.66, 'ACC': 0.96}]
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Path to save the file
    file_path = os.path.join(directory, filename)

    # Determine if the file exists and has content
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

    # Check if results is non-empty and a list of dicts
    if results and isinstance(results, list) and all(isinstance(item, dict) for item in results):
        # Get the fieldnames from the first item
        fieldnames = results[0].keys()

        # Open the file in append mode or write mode as appropriate
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header only if the file did not exist or was empty
            if not file_exists:
                writer.writeheader()

            for result in results:
                writer.writerow(result)
    else:
        raise ValueError("Results must be a non-empty list of dictionaries.")
def save_args_to_csv(args, directory):
    args_dict = vars(args)
    csv_path = os.path.join(directory, 'args.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in args_dict.items():
            writer.writerow([key, value])

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms



def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, rescaling_factor=1.0):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        min_beta = 0.1
        max_beta = 20
        alpha_bar = lambda t: math.exp(-(max_beta - min_beta) / 2 * t ** 2 - min_beta * t)

    elif schedule_name == "cosine":
        shift = 0.008
        alpha_bar = lambda t: math.cos((t + shift) / (1 + shift) * math.pi / 2) ** 2

    elif schedule_name == 'sqrt':
        shift = 0.0001
        alpha_bar = lambda t: 1 - math.sqrt(t + shift)

    elif schedule_name == 'edm':
        rho = 7
        min_sigma = 0.002 ** (1 / rho)
        max_sigma = 80 ** (1 / rho)
        alpha_bar = lambda t: 1 / ((max_sigma + (1 - t) * (min_sigma - max_sigma)) ** (rho * 2) + 1)

    elif schedule_name == 'cdcd':
        rho = 7
        min_sigma = 1 ** (1 / rho)
        max_sigma = 300 ** (1 / rho)
        alpha_bar = lambda t: 1 / ((max_sigma + (1 - t) * (min_sigma - max_sigma)) ** (rho * 2) + 1)

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    f2 = rescaling_factor ** 2
    rescaled_alpha_bar = lambda t: alpha_bar(t) / (f2 - (f2 - 1) * alpha_bar(t))
    return betas_for_alpha_bar(num_diffusion_timesteps, rescaled_alpha_bar)



def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def initialize_models(args):
    student_diff_model = UViTLin(num_clusters=args.num_clusters,embedding_dim = args.embedding_dim,non_linear=args.non_linear)
    teacher_diff_model = EMA(student_diff_model, beta = args.ema_val,update_every = 1)
    student, teacher = student_diff_model.cuda(), teacher_diff_model.cuda()
    teacher.eval()
    student.train()
    return student, teacher

def initialize_diffusion(args):
    if args.pred_v:
        objective = 'pred_v'
    else:
        objective = 'pred_x0'
    student_diff = GaussianDiffusion(objective=objective,vp_rf=args.vp_rf,rescaling_factor=args.rescaling_factor,timesteps=args.diffusion_steps).cuda()
    return student_diff, student_diff , student_diff

def get_optimizer_and_schedulers(student,dino_model, args, data_loader):
    params_groups = get_params_groups(student)
    if dino_model is not None and args.finetune:
        params_groups_dino = get_params_groups(dino_model)
        params_groups = merge_data_structures(params_groups,params_groups_dino)
    optimizer = torch.optim.AdamW(params_groups,betas=(0.9,0.98))

    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None
    lr_schedule = cosine_scheduler(
        args.lr ,
        args.min_lr, args.epochs, len(data_loader), warmup_epochs=args.warmup_epochs
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader)
    )
    momentum_schedule = cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    return optimizer, fp16_scaler, lr_schedule, wd_schedule, momentum_schedule

def load_checkpoint(args):
    to_restore = {"epoch": 0}
    return to_restore["epoch"]


def save_checkpoint(args, student, teacher, optimizer, fp16_scaler, epoch):
    save_dict = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'args': args
    }
    if fp16_scaler is not None:
        save_dict['fp16_scaler'] = fp16_scaler.state_dict()
    torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth.tar"))

def merge_data_structures(structure1, structure2):
    merged = [
        {'params': structure1[0]['params'] + structure2[0]['params']},
        {'params': structure1[1]['params'] + structure2[1]['params'], 'weight_decay': 0.0}
    ]
    return merged


def get_params_groups(model):
    regularized = []
    not_regularized = []
    cluster_params = []  # New list for cluster center parameters

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        elif "clusters_centers" in name:  # Add condition for cluster centers
            cluster_params.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.},
            {'params': cluster_params, 'weight_decay': 0.}]
def feature_dropout(features, min_dropout_rate=0.1, max_dropout_rate=0.25):
    """
    Applies dropout to the features with a random dropout rate for each feature map.

    Args:
    - features (torch.Tensor): The input feature tensor.
    - min_dropout_rate (float): Minimum dropout rate.
    - max_dropout_rate (float): Maximum dropout rate.

    Returns:
    - torch.Tensor: The augmented feature tensor with dropout applied.
    """
    # Generate random dropout rates for each feature map
    batch_size = features.shape[0]
    dropout_rates = torch.rand(batch_size).cuda() * (max_dropout_rate - min_dropout_rate) + min_dropout_rate
    dropout_rates = dropout_rates.view(batch_size, 1, 1).to(features.device)

    # Create and apply a dropout mask for each feature map
    masks = torch.rand_like(features) > dropout_rates
    return features * masks.float()



def add_random_noise(features, min_scale, max_scale):
    """
    Adds random noise to the features within a specified scale range.

    Args:
    - features (torch.Tensor): The input feature tensor.
    - min_scale (float): The minimum scale of noise to be added.
    - max_scale (float): The maximum scale of noise to be added.

    Returns:
    - torch.Tensor: The feature tensor with added noise.
    """
    batch_size = features.shape[0]

    noise_scale = torch.rand(batch_size).cuda() * (max_scale - min_scale) + min_scale
    noise = torch.randn(features.shape).cuda()

    return features + torch.nn.functional.normalize(noise, p=2, dim=-1) * noise_scale.view(batch_size, 1, 1)




