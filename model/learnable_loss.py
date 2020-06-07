import torch
from torch import nn


MSE_LOSS = nn.MSELoss()


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)  # b x ch x ch
    return gram


def get_content_loss(y, target):
    return MSE_LOSS(y, target)


def get_style_loss(features_y, gram_style, n_batch):
    loss = 0.
    for ft_y, gm_s in zip(features_y, gram_style):
        gm_y = gram_matrix(ft_y)
        loss += MSE_LOSS(gm_y, gm_s[:n_batch, :, :])
    return loss * 1E4


def get_total_variation_loss(y):
    return (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +\
        torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))) * 1E-4


class LearnableLoss(nn.Module):
    def __init__(self, model, loss_names, device):
        super().__init__()
        self.model = model
        self.loss_names = loss_names
        self.eta = nn.Parameter(torch.zeros(len(loss_names), device=device))
        # self.sigmoid = nn.Sigmoid()

    def get_model(self):
        return self.model

    def forward(self, x):
        return self.model(x)

    def get_total_loss(self, losses):
        loss_tensor = torch.stack(losses, dim=0)
        total_loss = loss_tensor * torch.exp(-self.eta) + self.eta
        total_loss = total_loss.sum()
        # print(f'   loss_tensor: {loss_tensor}, total_loss: {total_loss}')

        meta = {'loss': {}, 'eta': {}}
        for l, n, e in zip(losses, self.loss_names, self.eta.detach().cpu().numpy()):
            meta['loss'][n] = l.item()
            meta['eta'][n] = e
        meta['loss']['total'] = total_loss
        return total_loss, meta
