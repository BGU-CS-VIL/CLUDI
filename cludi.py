
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from tqdm import tqdm
from eval_utils import *
from log_utils import *
from train_utils import *
from loss_utils import *
from diff_utils import *
from einops import rearrange, reduce, repeat
from dino_utils import *

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    # Model parameters
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="Base EMA parameter for teacher update.")
    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=bool, default=True, help="Whether or not to use half precision for training.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Initial value of the weight decay.")
    parser.add_argument('--weight_decay_end', type=float, default=0.01, help="Final value of the weight decay.")
    parser.add_argument('--clip_grad', type=float, default=4.0, help="Maximal parameter gradient norm if using gradient clipping.")
    parser.add_argument('--batch_size', default=256, type=int, help='Per-GPU batch-size.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate at the end of linear warmup.")
    parser.add_argument("--warmup_epochs", default=1, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Target LR at the end of optimization.")
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'], help="Type of optimizer.")
    parser.add_argument('--data_path', default='/gmiha/data001/RnD/SSV/Datasets/imagenet/train/', type=str, help='Path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--num_clusters", default=50, type=int, help="Number of clusters for clustering.")
    parser.add_argument("--embedding_dim", default=64, type=int, help="Embedding size.")
    parser.add_argument("--batch_diffusion", default=8, type=int, help="Batch size for diffusion process.")
    parser.add_argument("--ce_lambda", default=50, type=int, help="cross entropy lambda")
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="Number of diffusion steps.")
    parser.add_argument("--ema_val", type=float, default=0.999, help="EMA value.")
    parser.add_argument("--early_stopping", type=int, default=0, help="early stopping")

    parser.add_argument("--noise_schedule", type=str, default="cosine", help="Type of noise schedule.")
    parser.add_argument("--rescaling_factor", type=float, default=49.0, help="Factor for rescaling.")
    parser.add_argument("--decoding_rescaling_factor", type=float, default=49.0, help="Factor for rescaling.")
    parser.add_argument("--vp_rf", action='store_true', help="Boolean flag for vp_rf.")
    parser.add_argument("--attention", action='store_true', help="Attention")
    parser.add_argument("--multiple_k", action='store_true', help="multiple k")
    parser.add_argument("--loop", action='store_true', help="Use loop")
    parser.add_argument("--non_linear", action='store_true', help="None linear output")
    parser.add_argument("--images", action='store_true', help="Train on images")
    parser.add_argument("--finetune", action='store_true', help="finetune dino")
    parser.add_argument("--pred_v", action='store_true', help="pred_v")
    parser.add_argument("--post_hoc", action='store_true', help="pred_v")
    parser.add_argument("--load_model", action='store_true', help="load from checkpoint ")
    parser.add_argument("--supervised", action='store_true', help="supervised ")
    parser.add_argument("--self_classifier", action='store_true', help="self_classifier")
    parser.add_argument("--lr_decay", action='store_true', help="self_classifier")
    parser.add_argument("--superclass", type=int, default=99, help="superclass")
    parser.add_argument("--acc_steps", type=int, default=1, help="acc_steps")


    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    return parser




def train_dino(args):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.results_dir = os.path.join(args.results_dir, current_time)
    os.makedirs(args.results_dir, exist_ok=True)
    save_args_to_csv(args, args.results_dir)
    args.output_dir = args.results_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    samples = torch.load('train.pth')
    samples_test = torch.load('test.pth')
    labels_gt = torch.load('labels_train.pth')
    labels_test_gt = torch.load('labels_test.pth')
    dataset = TensorDataset(samples, labels_gt)

    test_dataset = TensorDataset(samples_test, labels_test_gt)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,drop_last=False, num_workers= 8)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,drop_last=False, num_workers= 2)

    args.results_dir = './results/'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.results_dir = os.path.join(args.results_dir, current_time)
    os.makedirs(args.results_dir,exist_ok=True)
    save_args_to_csv(args, args.results_dir)
    student, teacher = initialize_models(args)
    student_diff, student_gen, teacher_gen = initialize_diffusion(args)
    optimizer, fp16_scaler, lr_schedule, wd_schedule, momentum_schedule = get_optimizer_and_schedulers(student,None, args, data_loader)
    start_epoch = load_checkpoint(args)

    print("Len train: ", len(dataset))
    print("Len test: ", len(test_dataset))
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        val_epoch(student, student_gen, test_data_loader, epoch, args)
        train_stats = train_one_epoch(student, teacher,student_diff, teacher_gen, data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args)
        save_checkpoint(args, student, teacher, optimizer, fp16_scaler, epoch)
        log_stats(args, train_stats, epoch)
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, student_diff, teacher_diff, data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args):
    metric_logger =MetricLogger(delimiter="  ")
    acc_steps = args.acc_steps
    for it, (images, labels_gt) in enumerate(metric_logger.log_every(data_loader, 10, 'Epoch: [{}/{}]'.format(epoch, args.epochs))):
        labels_gt = labels_gt.reshape(1, -1)[0]
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]

        images = images.cuda().unsqueeze(0).repeat(args.batch_diffusion, 1, 1)
        images = torch.nn.functional.normalize(images, p=2, dim=-1)

        teacher_feat = images


        student_feat = feature_dropout(add_random_noise(images, 0.2, 1.0), 0.05, 0.1)
        student.eval()
        z_0, logits,z_0_all = gen_loop(student, teacher_diff, teacher_feat, args, 25, args.decoding_rescaling_factor)
        student.train()

        labels_softmax = [F.softmax(logits[0] / (0.1),dim=-1).half()]


        student_z_0, student_logits,loss_weights, time_steps,student_z_0_loop,target_v,pred_v =  train_iter(student, student_diff, labels_softmax, student_feat ,args)
        loss_diff = F.mse_loss(student_z_0_loop, z_0, reduction = 'none')
        loss_diff = (reduce(loss_diff, 'b ... -> b', 'mean') * loss_weights).mean()

        loss_logits = 0
        for iter_log in range(len(logits)):
            loss_logits += loss_logits+ extended_new_loss_newnew(logits[iter_log], student_logits[iter_log], loss_weights)
        loss_logits = loss_logits/len(logits)

        loss = loss_diff + loss_logits*args.ce_lambda

        cluster_lr = optimizer.param_groups[0]['lr'] / 10
        optimizer.param_groups[2]['lr'] = cluster_lr
        loss_cont = loss_diff


        loss = loss / acc_steps

        if fp16_scaler is None:
            loss.backward()
        else:
            fp16_scaler.scale(loss).backward()

        if ((it + 1) % acc_steps == 0) or ((it + 1) == len(data_loader) * (epoch + 1)):
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                clip_gradients(student, args.clip_grad)
            if fp16_scaler is None:
                optimizer.step()
            else:
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            optimizer.zero_grad()
            teacher.update()

        teacher.update()
        nmi = normalized_mutual_info_score(labels_gt.cpu().numpy(), labels_softmax[0].detach().cpu().numpy()[0].argmax(1))
        ari = adjusted_rand_score(labels_gt.cpu().numpy(), labels_softmax[0].detach().cpu().numpy()[0].argmax(1))
        nmi_s = normalized_mutual_info_score(labels_gt.cpu().numpy(), student_logits[0].detach().cpu().numpy()[0].argmax(1))
        ari_s = adjusted_rand_score(labels_gt.cpu().numpy(), student_logits[0].detach().cpu().numpy()[0].argmax(1))
        metric_logger.update(ari=ari)
        metric_logger.update(ari_s=ari_s)
        metric_logger.update(nmi=nmi)
        metric_logger.update(nmi_s=nmi_s)
        metric_logger.update(loss_diff=loss_diff.item())
        metric_logger.update(loss_logits=loss_logits.item())
        metric_logger.update(loss_cont=loss_cont.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def val_epoch(student, student_diff, data_loader, epoch, args):
    labels_gt_save = []
    labels_prediction = []
    labels_mean = []
    student.eval()
    batch_sample = 64
    with torch.no_grad():
        for it, (images, labels_gt) in tqdm(enumerate(data_loader)):
            images = images.cuda().unsqueeze(0).repeat(batch_sample, 1, 1)
            if not args.self_classifier:
                z_0, logits,_ = gen_loop(student, student_diff, images , args, 50,args.decoding_rescaling_factor)
                logits = logits[0]
            else:
                logits = student(images)
            labels_softmax_avg = logits
            labels_softmax_student_mean = labels_softmax_avg.mean(0).argmax(-1)
            labels_softmax_avg = labels_softmax_avg.argmax(-1).unsqueeze(0)
            labels_gt_save.append(labels_gt)
            labels_prediction.append(labels_softmax_avg)
            labels_mean.append(labels_softmax_student_mean)

    labels_prediction = torch.cat(labels_prediction,-1).cpu().numpy()[0]
    labels_gt_save = torch.cat(labels_gt_save).reshape(-1)
    labels_mean = torch.cat(labels_mean).reshape(-1).cpu().numpy()
    nmi = normalized_mutual_info_score(labels_gt_save, labels_mean.reshape(-1))
    ari = adjusted_rand_score(labels_gt_save, labels_mean.reshape(-1))
    acc = clustering_accuracy(labels_gt_save, labels_mean.reshape(-1))
    print(f"Mean: NMI = {nmi}, ARI = {ari}, ACC = {acc}")
    student.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DIffusionClustering', parents=[get_args_parser()])
    args = parser.parse_args()
    args.results_dir = './results/'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.results_dir = os.path.join(args.results_dir, current_time)
    os.makedirs(args.results_dir, exist_ok=True)
    save_args_to_csv(args, args.results_dir)
    args.output_dir = args.results_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)




