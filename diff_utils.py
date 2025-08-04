
import torch
import torch.nn.functional as F
from random import random
from tqdm import tqdm

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    # Sample from Gumbel(0, 1)
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # Gumbel(0,tau)
    y_soft = gumbels.softmax(dim=-1)

    if hard:
        # Straight through trick: select the max in the forward pass, use the soft gradient in the backward pass
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret


def noise_iter(model,model_diff,args,t):
    z_0 = model.return_clusters_centers()
    noise = torch.randn_like(z_0) #* args.rescaling_factor
    z_t = model_diff.q_sample(z_0, t, noise).type_as(z_0)
    return z_t
def train_iter(model, model_diff, c_0, x, args):

    z_0 = model.return_embedding(c_0[0])
    if args.multiple_k:
        z_2 = model.return_embedding2(c_0[1])
        z_3 = model.return_embedding3(c_0[2])
        z_0 = model.weighted_emb(z_0,z_2,z_3)
    t = torch.randint(0, args.diffusion_steps, [len(z_0)], device=z_0.device)
    model_t = t 
    noise = torch.randn_like(z_0) * args.rescaling_factor
    z_t = model_diff.q_sample(z_0, t, noise).type_as(z_0)
    z_self_cond = torch.zeros_like(z_0)

    if random() < 0.5:
        with torch.no_grad():
            if args.pred_v:
                model_output = model(z_t, x, model_t, z_self_cond)
                z_self_cond = model_diff.predict_start_from_v(z_t, model_t, model_output)
                z_self_cond = torch.clamp(z_self_cond, min=-1., max=1.)
                z_self_cond = z_self_cond.detach()
            else:
                z_self_cond = model(z_t, x, model_t, z_self_cond)
                z_self_cond = z_self_cond.detach()
    if args.pred_v:
        target_v = model_diff.predict_v(z_0,model_t,noise)
        pred_v = model(z_t,x, model_t, z_self_cond)
        z_0_hat = model_diff.predict_start_from_v(z_t, t, pred_v)
    else:
        target_v = 0
        pred_v = 0
        z_0_hat = model(z_t,x, model_t, z_self_cond)
    logits = model.last_layer(z_0_hat)
    if args.multiple_k:
        logits_2 = model.last_layer2(z_0_hat)
        logits_3 = model.last_layer3(z_0_hat)
        z_o_loop = model.return_embedding(F.softmax(logits/0.1,dim=-1))
        z_o_loop2 = model.return_embedding2(F.softmax(logits_2/0.1,dim=-1))
        z_o_loop3 = model.return_embedding3(F.softmax(logits_3/0.1,dim=-1))
        z_0_loop = model.weighted_emb(z_o_loop,z_o_loop2,z_o_loop3)
        return z_0_hat, [logits,logits_2,logits_3], model_diff.loss_weight[t],model_t,z_0_loop,target_v,pred_v
    else:
        z_0_loop = model.return_embedding(F.softmax(logits / 0.1, dim=-1))
        return z_0_hat, [logits], model_diff.loss_weight[t]/args.rescaling_factor, model_t, z_0_loop,target_v,pred_v



def gen_loop(model, model_diff, x , args, steps, rescaling_factor=1,disable=True,debug=False):
    batch, device, total_timesteps, sampling_timesteps, eta = x.shape[0], x.device, model_diff.num_timesteps, model_diff.sampling_timesteps, model_diff.ddim_sampling_eta
    eta = 1
    times = torch.linspace(-1, total_timesteps - 1, steps = steps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    z_t = torch.randn(x.shape[0],x.shape[1],args.embedding_dim, device = x.device)*rescaling_factor
    self_cond = torch.zeros_like(z_t,device= x.device)
    imgs = []

    ind = 1
    for time, time_next in tqdm(time_pairs, desc='sampling loop time step', disable=disable):

        if (len(times)-ind)<args.early_stopping:
            break
        ind = ind+1
        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
        if args.pred_v:
            model_output = model(z_t, x, time_cond , self_cond)
            z_start = model_diff.predict_start_from_v(z_t, time_cond, model_output)
            z_start = torch.clamp(z_start,min = -1., max = 1.)
        else:
            z_start = model(z_t, x, time_cond , self_cond)
            z_start = torch.clamp(z_start,min = -1., max = 1.)
        pred_noise = model_diff.predict_noise_from_start(z_t,time_cond,z_start)
        if time_next < 0:
            z_t = z_start
            imgs.append(model.last_layer(z_t))
            continue

        alpha = model_diff.alphas_cumprod[time]
        alpha_next = model_diff.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(z_t) * rescaling_factor

        z_t = z_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise
        self_cond = z_start
        imgs.append(model.last_layer(z_start))
    logits = model.last_layer(z_start)
    z_start = model.return_embedding(F.softmax(logits / 0.1, dim=-1))
    if args.multiple_k:
        logits_2 = model.last_layer2(z_start)
        logits_3 = model.last_layer3(z_start)


        z_o_loop = model.return_embedding(F.softmax(logits / 0.1, dim=-1))
        z_o_loop2 = model.return_embedding2(F.softmax(logits_2 / 0.1, dim=-1))
        z_o_loop3 = model.return_embedding3(F.softmax(logits_3 / 0.1, dim=-1))

        z_0_loop = model.weighted_emb(z_o_loop, z_o_loop2, z_o_loop3)
        if debug:
            return z_start,[logits,logits_2,logits_3]
        else:
            return z_0_loop, [logits,logits_2,logits_3], torch.stack(imgs)
    else:

        return z_start, [logits], torch.stack(imgs)

def gen_loop_v(model, model_diff, x , args, steps, rescaling_factor=1,disable=True):
    batch, device, total_timesteps, sampling_timesteps, eta = x.shape[0], x.device, model_diff.num_timesteps, model_diff.sampling_timesteps, model_diff.ddim_sampling_eta
    eta = 1
    times = torch.linspace(-1, total_timesteps - 1, steps = steps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    z_t = torch.randn(x.shape[0],x.shape[1],args.embedding_dim, device = x.device)*rescaling_factor
    self_cond = torch.zeros_like(z_t,device= x.device)
    imgs = []


    for time, time_next in tqdm(time_pairs, desc='sampling loop time step', disable=disable):
        time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
        v_hat = model(z_t, x, time_cond , self_cond)
        z_start = model_diff.predict_start_from_v(z_t,time_cond,v_hat)
        pred_noise = model_diff.predict_noise_from_start(z_t,time_cond,z_start)
        if time_next < 0:
            z_t = z_start
            imgs.append(model.last_layer(z_t))
            continue

        alpha = model_diff.alphas_cumprod[time]
        alpha_next = model_diff.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(z_t) * rescaling_factor

        z_t = z_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise
        self_cond = z_start
        imgs.append(model.last_layer(z_start))
    logits = model.last_layer(z_t)

    return z_t, logits, torch.stack(imgs)

