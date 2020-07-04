from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import functional as nnfun
import torch.nn.functional as F

# similarity params
SMOOTH_DISTANCE_INIT = 0.3
PRIOR_DISTANCE_INIT = 0.3

# mean-field iterations
NUM_ITER = 4
# connectivity
FILTER_SIZE = [5, 5, 5]
# prior potential dilation
DILATION = 2.0


class CAM(nn.Module):
    def __init__(self, nclasses, img_shape):
        super(CAM, self).__init__()

        self.nclasses = nclasses

        # appearance kernel parameters
        self.smooth_norm = Parameter(torch.Tensor([1 / SMOOTH_DISTANCE_INIT]))
        self.smooth_weight = Parameter(torch.ones(nclasses))

        # prior kernel parameters
        self.prior_norm = Parameter(torch.Tensor([1 / PRIOR_DISTANCE_INIT]))
        self.prior_weight = Parameter(torch.ones(img_shape[0], img_shape[1], img_shape[2]))

        # comparability
        self.compat = nn.Conv3d(
            in_channels=nclasses,
            out_channels=nclasses,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.compat.weight.data=torch.from_numpy((np.eye(nclasses, nclasses, dtype=np.float32)).reshape([nclasses, nclasses, 1, 1, 1]))

        self.filter_size = FILTER_SIZE
        self.dilation = DILATION

    def forward(self, unary, img, atlas, atlas_label, patch_start):
        # unary: [B C W H D], before softmax
        # img: [B 1 W H D]
        # atlas: [B 1 W H D]
        # img_feats: [B C W H D]
        # atlas_feats: [B C W H D]
        # atlas_label: [B W H D C]

        bs, c, x, y, z = unary.shape
        npixels = (x, y, z)

        # features
        smooth_feats = self._create_feats(feats=img, norm=self.smooth_norm)
        prior_feats = (
            self._create_feats(feats=img, norm=self.prior_norm),
            self._create_feats(feats=atlas, norm=self.prior_norm),
        )

        _gaus_list = []
        _gaus_types = []
        _gaus_weight = []

        # smoothness potential
        gaussian = self._compute_smooth_similarity(smooth_feats, npixels)
        _gaus_list.append(gaussian)
        _gaus_weight.append(self.smooth_weight)
        _gaus_types.append('smooth')

        # prior potential
        gaussian = self._compute_prior_similarity(prior_feats[0], prior_feats[1], npixels)
        _gaus_list.append(gaussian)
        _gaus_weight.append(self.prior_weight)
        _gaus_types.append('prior')

        # mean-field inference
        q_values = unary
        atlas_label = atlas_label.permute(0, 4, 1, 2, 3)
        for i in range(NUM_ITER):
            softmax_out = nnfun.softmax(q_values, dim=1)
            message = self._compute(input=softmax_out, atlas_label=atlas_label, patch_start=patch_start, gaus_list=_gaus_list, gaus_weight=_gaus_weight, gaus_types=_gaus_types, npixels=npixels)
            message = self.compat(message)

            q_values = unary + message

        return q_values

    def _create_feats(self, feats, norm):
        norm_feats = feats * norm

        return norm_feats

    def _compute_prior_similarity(self, feat_1, feat_2, npixels):

        span = [x // 2 for x in self.filter_size]
        bs = feat_1.shape[0]

        if self.dilation > 1:
            feat_1 = torch.nn.functional.interpolate(
                input=feat_1,
                scale_factor=(1 / self.dilation),
                mode='trilinear',
                align_corners=True,
            )
            feat_2 = torch.nn.functional.interpolate(
                input=feat_2,
                scale_factor=(1 / self.dilation),
                mode='trilinear',
                align_corners=True,
            )
            updated_npixels = [math.ceil(npixels[0] / self.dilation),
                       math.ceil(npixels[1] / self.dilation),
                       math.ceil(npixels[2] / self.dilation)]
        else:
            updated_npixels = npixels

        gaussian = feat_1.data.new(
            bs, self.filter_size[0], self.filter_size[1], self.filter_size[2], updated_npixels[0], updated_npixels[1], updated_npixels[2]
        ).fill_(0)

        for dx in range(-span[0], span[0] + 1):
            for dy in range(-span[1], span[1] + 1):
                for dz in range(-span[2], span[2] + 1):
                    dx1, dx2 = _get_ind(dx)
                    dy1, dy2 = _get_ind(dy)
                    dz1, dz2 = _get_ind(dz)

                    feat_t = feat_1[:, :, dx2:_negative(dx1), dy2:_negative(dy1), dz2:_negative(dz1)]
                    feat_t2 = feat_2[:, :, dx1:_negative(dx2), dy1:_negative(dy2), dz1:_negative(dz2)]

                    diff = feat_t - feat_t2
                    exp_diff = torch.exp(torch.sum(-0.5 * diff * diff, dim=1))

                    gaussian[:, dx + span[0], dy + span[1], dz + span[2], dx2:_negative(dx1), dy2:_negative(dy1), dz2:_negative(dz1)] = exp_diff
        return gaussian.view(
            bs, 1, self.filter_size[0], self.filter_size[1], self.filter_size[2], updated_npixels[0], updated_npixels[1], updated_npixels[2]
        )

    def _compute_smooth_similarity(self, features, npixels):

        span = [x // 2 for x in self.filter_size]
        bs = features.shape[0]

        gaussian = features.data.new(
            bs, self.filter_size[0], self.filter_size[1], self.filter_size[2], npixels[0], npixels[1], npixels[2],
        ).fill_(0)

        for dx in range(-span[0], span[0] + 1):
            for dy in range(-span[1], span[1] + 1):
                for dz in range(-span[2], span[2] + 1):
                    dx1, dx2 = _get_ind(dx)
                    dy1, dy2 = _get_ind(dy)
                    dz1, dz2 = _get_ind(dz)

                    feat_t = features[:, :, dx1:_negative(dx2), dy1:_negative(dy2),  dz1:_negative(dz2)]
                    feat_t2 = features[:, :, dx2:_negative(dx1), dy2:_negative(dy1), dz2:_negative(dz1)]

                    diff = feat_t - feat_t2
                    exp_diff = torch.exp(torch.sum(-0.5 * diff * diff, dim=1))

                    gaussian[:, dx + span[0], dy + span[1], dz + span[2], dx2:_negative(dx1), dy2:_negative(dy1), dz2:_negative(dz1)] = exp_diff

        return gaussian.view(
            bs, 1, self.filter_size[0], self.filter_size[1], self.filter_size[2], npixels[0], npixels[1], npixels[2]
        )

    def _compute_smooth_message(self, input, gaussian, npixels):
        # input (unary): [bs, c, h, w, d]
        # guassian: [bs, 1, f, f, f, h, w, d]

        span = [x // 2 for x in self.filter_size]

        product_tensor = input.data.new(
            input.shape[0], self.nclasses, npixels[0], npixels[1], npixels[2]
        ).fill_(0)
        product = Variable(product_tensor)

        for dx in range(-span[0], span[0] + 1):
            for dy in range(-span[1], span[1] + 1):
                for dz in range(-span[2], span[2] + 1):
                    dx1, dx2 = _get_ind(dx)
                    dy1, dy2 = _get_ind(dy)
                    dz1, dz2 = _get_ind(dz)

                    feat_t = input[:, :, dx1:_negative(dx2), dy1:_negative(dy2),  dz1:_negative(dz2)]
                    feat_t2 = gaussian[:, :, dx + span[0], dy + span[1], dz + span[2], dx2:_negative(dx1), dy2:_negative(dy1), dz2:_negative(dz1)]

                    product[:, :, dx2:_negative(dx1), dy2:_negative(dy1), dz2:_negative(dz1)] += feat_t * feat_t2

        message = product.view([input.shape[0], self.nclasses, npixels[0], npixels[1], npixels[2]])

        return message

    def _compute_prior_message(self, atlas_label, gaussian, npixels):
        # atlas_label: [bs, c, h, w, d]
        # guassian: [bs, 1, f, f, f, h, w, d]

        span = [x // 2 for x in self.filter_size]

        if self.dilation > 1:
            atlas_label = torch.nn.functional.interpolate(
                input=atlas_label,
                scale_factor=(1.0 / self.dilation),
                mode='trilinear',
                align_corners=True,
            )
            updated_npixels = [math.ceil(npixels[0] / self.dilation), math.ceil(npixels[1] / self.dilation), math.ceil(npixels[2] / self.dilation)]
        else:
            updated_npixels = npixels

        product_tensor = atlas_label.data.new(
            atlas_label.shape[0], self.nclasses, updated_npixels[0], updated_npixels[1], updated_npixels[2]
        ).fill_(0)
        product = Variable(product_tensor)

        for dx in range(-span[0], span[0] + 1):
            for dy in range(-span[1], span[1] + 1):
                for dz in range(-span[2], span[2] + 1):
                    dx1, dx2 = _get_ind(dx)
                    dy1, dy2 = _get_ind(dy)
                    dz1, dz2 = _get_ind(dz)

                    feat_t = atlas_label[:, :, dx1:_negative(dx2), dy1:_negative(dy2),  dz1:_negative(dz2)]
                    feat_t2 = gaussian[:, :, dx + span[0], dy + span[1], dz + span[2], dx2:_negative(dx1), dy2:_negative(dy1), dz2:_negative(dz1)]

                    product[:, :, dx2:_negative(dx1), dy2:_negative(dy1), dz2:_negative(dz1)] += feat_t * feat_t2

        message = product.view([atlas_label.shape[0], self.nclasses, updated_npixels[0], updated_npixels[1], updated_npixels[2]])

        if self.dilation > 1:
            message = torch.nn.functional.interpolate(
                input=message,
                scale_factor=float(self.dilation),
                mode='trilinear',
                align_corners=True,
            )
            message = message.contiguous()

        return message

    def _compute(self, input, atlas_label, patch_start, gaus_list, gaus_weight, gaus_types, npixels):
        assert(len(gaus_list) == len(gaus_types) == len(gaus_weight))

        pred = 0
        for gaus, weight, type in zip(gaus_list, gaus_weight, gaus_types):
            if type == 'smooth':
                message = self._compute_smooth_message(input=input, gaussian=gaus, npixels=npixels)
                pred = pred + weight[None, :, None, None, None] * message
            elif type == 'prior':
                message = self._compute_prior_message(atlas_label=atlas_label, gaussian=gaus, npixels=npixels)

                pred = pred + weight[None, None, patch_start[0][0]:patch_start[0][0] + npixels[0], patch_start[0][1]:patch_start[0][1] + npixels[1], patch_start[0][2]:patch_start[0][2] + npixels[2]] * message

        return pred


def _get_ind(dz):
    if dz == 0:
        return 0, 0
    if dz < 0:
        return 0, -dz
    if dz > 0:
        return dz, 0


def _negative(dz):
    if dz == 0:
        return None
    else:
        return -dz
