import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..registry import LOSSES
from .utils import weighted_loss
import numpy as np

BBOX_XFORM_CLIP = np.log(1333. / 16.)#最大尺寸除以stride

def bbox_transform(deltas):
    # wx, wy, ww, wh = weights
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    dw = torch.clamp(dw, max=BBOX_XFORM_CLIP)
    dh = torch.clamp(dh, max=BBOX_XFORM_CLIP)

    pred_ctr_x = dx
    pred_ctr_y = dy
    pred_w = torch.exp(dw)
    pred_h = torch.exp(dh)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    return x1.view(-1), y1.view(-1), x2.view(-1), y2.view(-1)



def giou_loss(pred, target, eps=1e-6,reduction=None,avg_factor=None):
    x1,y1,x2,y2=bbox_transform(pred)
    x1g,y1g,x2g,y2g=bbox_transform(target)
    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    intsctk = torch.zeros(x1.size()).to(pred)#把一个张量放到另一个张量所在的设备上
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)#返回一个bool mask True就是该对bbox有交集
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])#这里mask的用法就是索引
    #>>> import numpy as np
    # >>> a=np.array([-0.0642,0.1014,-0.3007])
    # >>> b=np.array([1.159,1.1792,0.7867])
    # >>> import torch as t
    # >>> a=t.from_numpy(a)
    # >>> b=t.from_numpy(b)
    # >>> mask=np.bool([True,False,True])
    # >>> mask
    # True
    # >>> mask=(a>b)
    # >>> mask
    # tensor([False, False, False])
    # >>> mask=(a<b)
    # >>> mask
    # tensor([True, True, True])
    # >>> a[mask]
    # tensor([-0.0642,  0.1014, -0.3007], dtype=torch.float64)
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk
    iouk=iouk.clamp(min=eps)
    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    giouk = iouk - ((area_c - unionk) / area_c)

    # iouk = 1-iouk
    giouk = 1-giouk
    if reduction=='mean':
        loss=giouk.sum()/avg_factor
    else:
        loss=giouk.sum()
    return loss




@LOSSES.register_module
class GIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            )
        return loss



