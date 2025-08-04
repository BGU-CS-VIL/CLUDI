

import sys
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
def extended_new_loss_newnew(pred_softmax_1, pred_softmax_2, loss_weights = None, row_tau=0.1 * np.log(10) / np.log(10),
                      col_tau=0.05 * np.log(10) / np.log(10)):
    # Calculate log probabilities with normalization across seq_length
    logs_1 = torch.log(
        pred_softmax_1.shape[1] / pred_softmax_1.shape[-1] * F.normalize(F.softmax(pred_softmax_1 / row_tau, -1), p=1,
                                                                         dim=1, eps=1e-8) + 1e-8).unsqueeze(1)
    logs_2 = torch.log(
        pred_softmax_1.shape[1] / pred_softmax_1.shape[-1] * F.normalize(F.softmax(pred_softmax_2 / row_tau, -1), p=1,
                                                                         dim=1, eps=1e-8) + 1e-8).unsqueeze(0)

    # Normalize across num_classes
    norm_1 = F.normalize(F.softmax(pred_softmax_1 / col_tau, 1), p=1, dim=2, eps=1e-8).unsqueeze(1)
    norm_2 = F.normalize(F.softmax(pred_softmax_2 / col_tau, 1), p=1, dim=2, eps=1e-8).unsqueeze(0)

    # Calculate the loss for all pairings
    l_1 = -torch.mean(torch.sum(norm_1 * logs_2, dim=(3)), dim=-1)

    l_2 = -torch.mean(torch.sum(norm_2 * logs_1, dim=(3)), dim=-1)


    if loss_weights is not None:
        # Adjust weights for all pairings: for each combination, average the weights of the two involved samples
        loss_weights_expanded_1 = loss_weights.unsqueeze(1)  # .expand(-1, loss_weights.size(0)).clone()
        loss_weights_expanded_2 = loss_weights.unsqueeze(0)  # .expand(loss_weights.size(0), -1).clone()

        weighted_loss_1 = (l_1 * loss_weights_expanded_1).mean()
        weighted_loss_2 = (l_2 * loss_weights_expanded_2).mean()

    else:
        weighted_loss_1  = l_1.mean()
        weighted_loss_2 = l_2.mean()

    total_loss = (weighted_loss_1 + weighted_loss_2) / 2

    return total_loss

