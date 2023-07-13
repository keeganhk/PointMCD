from .pkgs import *
from .tools import *



def visualize_image_tensor(image_tensor):
    image_tensor = min_max_normalization(image_tensor.cpu())
    unloader = transforms.ToPILImage()
    image_pil = unloader(image_tensor)
    return image_pil


def visualize_one_channel_tensor(one_channel_tensor):
    label_numpy = one_channel_tensor.squeeze(0).cpu().data.numpy() # (H, W)
    label_numpy = min_max_normalization(label_numpy) * 255
    label_pil = Image.fromarray(label_numpy).convert('L')
    return label_pil


def cdw_dist_loss(mv_proj_s, mv_proj_t):
    assert mv_proj_s.size(0) == mv_proj_t.size(0)
    assert mv_proj_s.size(1) == mv_proj_t.size(1)
    loss = F.kl_div(F.softmax(mv_proj_s, dim=-1).log(), F.softmax(mv_proj_t, dim=-1))
    return loss


def mv_cdw_dist_loss(mv_proj_s, mv_proj_t):
    assert mv_proj_s.size(0) == mv_proj_t.size(0)
    assert mv_proj_s.size(1) == mv_proj_t.size(1)
    assert mv_proj_s.size(2) == mv_proj_t.size(2)
    batch_size, num_views, num_channels = mv_proj_s.size()
    mv_proj_s_rs = mv_proj_s.view(batch_size*num_views, num_channels)
    mv_proj_t_rs = mv_proj_t.view(batch_size*num_views, num_channels)
    loss = F.kl_div(F.softmax(mv_proj_s_rs, dim=-1).log(), F.softmax(mv_proj_t_rs, dim=-1))
    return loss


def smooth_cross_entropy(pred, label, eps=0.2):
    num_classes = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, label.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
    log_prb = F.log_softmax(pred, dim=1)
    sce_loss = -(one_hot * log_prb).sum(dim=1).mean()
    return sce_loss


