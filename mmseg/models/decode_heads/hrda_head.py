# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add return_logits flag
# - Update debug_output
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy

import torch
from torch.nn import functional as F
import torch.nn as nn
from ...core import add_prefix
from ...ops import resize as _resize
from .. import builder
from ..builder import HEADS
from ..segmentors.hrda_encoder_decoder import crop
from .decode_head import BaseDecodeHead
def extract_high_freq_components(x, sigma=1):
    # Determine kernel size: 3 standard deviations rule
    kernel_size = int(2 * (3 * sigma) + 1)  # Kernel size: 3*sigma on both sides
    kernel_size = max(3, kernel_size)  # Ensure kernel size is at least 3
    
    # Create 1D Gaussian kernel
    gauss_kernel_1d = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    gauss_kernel_1d = torch.exp(-gauss_kernel_1d**2 / (2 * sigma**2))
    gauss_kernel_1d = gauss_kernel_1d / gauss_kernel_1d.sum()  # Normalize

    # Create 2D Gaussian kernel by outer product
    gauss_kernel_2d = gauss_kernel_1d[:, None] * gauss_kernel_1d[None, :]  # Outer product
    gauss_kernel_2d = gauss_kernel_2d.view(1, 1, kernel_size, kernel_size)  # [1, 1, K, K]

    # Repeat kernel for each channel
    gauss_kernel_2d = gauss_kernel_2d.repeat(x.size(1), 1, 1, 1)  # [C, 1, K, K]

    # Apply Gaussian blur (low-pass filter)
    blurred = F.conv2d(x, gauss_kernel_2d, padding=kernel_size // 2, groups=x.size(1))

    # Subtract low-pass (blurred) from original to get high-frequency components
    high_freq_components = x - blurred

    return high_freq_components.abs()  # Return absolute values to represent magnitudes


class DualA(nn.Module):
    def __init__(self, kernel_size=3, initial_alpha=0.5):
        super(DualA, self).__init__()
        
        # Initialize alpha as a trainable parameter
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, requires_grad=True))
        
        # Spatial attention for HR features
        self.hr_feature_attn = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        
        # Reduce multi-channel attention map to single channel
        self.reduce_attention_map = nn.Conv2d(
            in_channels=19,  # Number of input channels in attention_map_1
            out_channels=1,  # Desired output channels
            kernel_size=1,   # 1x1 convolution
            bias=False       # No bias needed for weighted sum
        )

    def forward(self, hr_features, hr_un, lr_logits, lr_uncertainty):
        """
        Args:
            hr_features: High-resolution feature map [B, C_hr, H, W].
            hr_uncertainty: Uncertainty map derived from HR features [B, 1, H, W].
            lr_logits: Low-resolution logits from the LR decoder [B, C_lr, H, W].
            lr_uncertainty: Uncertainty map derived from LR logits [B, 1, H, W].
        Returns:
            Refined HR features with enhanced attention to small and large classes.
        """

        # Step 1: Extract High-Frequency Components
        #hr_high_freq = extract_high_freq_components(hr_features)
        #lr_high_freq = extract_high_freq_components(lr_logits)

        # Step 2: Multi-Scale Feature Norm
        feature_norm_global = hr_features.mean(dim=1, keepdim=True)  # Global average pooling
       
        hr_high_freq = extract_high_freq_components(feature_norm_global, sigma=2)  # Apply Gaussian smoothing
        
        #feature_norm_local = F.max_pool2d(hr_features, kernel_size=3, stride=1, padding=1)  # Local max pooling
        #print( feature_norm_local.shape)
        feature_norm = feature_norm_global +  hr_high_freq  # Combine high-frequency details
       
        # Step 3: Attention Map 1 (LR Logits + HR Confidence + High Frequencies)
        lr_logits=self.reduce_attention_map(lr_logits)
        lr_high_freq = extract_high_freq_components(lr_logits, sigma=2) 
        lr_attention_map = torch.sigmoid(lr_logits)  # Normalize LR logits
        attention_map_1 = lr_attention_map * torch.sigmoid(hr_un)  # Combine with HR confidence
        attention_map_1 = torch.sigmoid((attention_map_1 + lr_high_freq))  # Add LR high-frequencies and reduce

        # Step 4: Attention Map 2 (HR Features + LR Uncertainty)
        hr_feature_attention = self.hr_feature_attn(feature_norm)  # Spatial attention
        lr_uncertainty_attention = torch.exp(-lr_uncertainty)  # High uncertainty suppresses attention
        attention_map_2 = torch.sigmoid(hr_feature_attention * lr_uncertainty_attention)

        # Step 5: Combine Attention Maps
        alpha = torch.sigmoid(self.alpha)  # Ensure alpha is in [0, 1]
        final_attention_map = alpha * attention_map_1 + (1 - alpha) * attention_map_2  # Combine maps

        # Step 6: Refine HR Features
        refined_hr_features = hr_features * final_attention_map  # Element-wise refinement
        refined_hr_features = refined_hr_features + hr_features  # Residual connection

        return refined_hr_features

def scale_box(box, scale):
    y1, y2, x1, x2 = box
    # assert y1 % scale == 0
    # assert y2 % scale == 0
    # assert x1 % scale == 0
    # assert x2 % scale == 0
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


@HEADS.register_module()
class HRDAHead(BaseDecodeHead):

    def __init__(self,
                 single_scale_head,
                 lr_loss_weight=0,
                 hr_loss_weight=0,
                 scales=[1],
                 attention_embed_dim=256,
                 attention_classwise=True,
                 enable_hr_crop=False,
                 hr_slide_inference=True,
                 fixed_attention=None,
                 debug_output_attention=False,
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)
        if single_scale_head == 'DAFormerHead':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        elif single_scale_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        else:
            raise NotImplementedError(single_scale_head)
        super(HRDAHead, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        head_cfg['type'] = single_scale_head
        self.head = builder.build_head(head_cfg)

        attn_cfg['type'] = single_scale_head
        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.scale_attention = builder.build_head(attn_cfg)
        else:
            self.scale_attention = None
            self.fixed_attention = fixed_attention
        self.lr_loss_weight = lr_loss_weight
        self.hr_loss_weight = hr_loss_weight
        self.scales = scales
        self.enable_hr_crop = enable_hr_crop
        self.hr_crop_box = None
        self.hr_slide_inference = hr_slide_inference
        self.debug_output_attention = debug_output_attention
        self.DualA=DualA(kernel_size=3, initial_alpha=0.5)

    def set_hr_crop_box(self, boxes):
        self.hr_crop_box = boxes

    def hr_crop_slice(self, scale):
        crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.align_corners)

    def decode_hr(self, inp, bs):
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            features = inp['features']  # level, crop * bs, c, h, w
            boxes = inp['boxes']
            dev = features[0][0].device
            h_img, w_img = 0, 0
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.num_classes, h_img, w_img),
                                device=dev)
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)

            crop_seg_logits = self.head(features)
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            return preds
        else:
            return self.head(inp)

    def get_scale_attention(self, inp):
        if self.scale_attention is not None:
            att = torch.sigmoid(self.scale_attention(inp))
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs):
        assert len(inputs) == 2
        hr_inp = inputs[1]
        hr_scale = self.scales[1]
        lr_inp = inputs[0]
        lr_sc_att_inp = inputs[0]  # separate var necessary for stack hr_fusion
        lr_scale = self.scales[0]
        batch_size = lr_inp[0].shape[0]
        assert lr_scale <= hr_scale

        has_crop = self.hr_crop_box is not None
        if has_crop:
            crop_y1, crop_y2, crop_x1, crop_x2 = self.hr_crop_box

        # print_log(f'lr_inp {[f.shape for f in lr_inp]}', 'mmseg')
        lr_seg = self.head(lr_inp)
        # print_log(f'lr_seg {lr_seg.shape}', 'mmseg')
        lr_un = 1-(torch.softmax(lr_seg.detach(), dim=1))
        lr_un = torch.max(lr_un, dim=1, keepdim=True)[0]  # Reduce to [B, 1, H, W]

        #lr_un = lr_un > 0.6
       
        
        # print_log(f'lr_seg {lr_seg.shape}', 'mmseg')
        #sparse_hr_features = None 
        if isinstance(hr_inp, dict) and 'boxes' in hr_inp.keys():
            sparse_hr_features = []
            fe = hr_inp['features']  # List of high-resolution features
              # Should print 4

            for i, feature in enumerate(fe):
                # Determine group size
                group_size = feature.shape[0] // lr_un.shape[0]  # E.g., 18 // 2 = 9
                feature_grouped = feature.view(lr_un.shape[0], group_size, *feature.shape[1:])  # Shape: [2, 9, C, H, W]

                # Placeholder for the sparsified tensor for this feature
                sparse_feature = []

                for j in range(lr_un.shape[0]):  # Loop over the number of `hr_un` batches
                    lr_un_single = lr_un[j:j+1]  # Extract a single batch of hr_un with shape [1, 1, H, W]
                    feature_single_group = feature_grouped[j]  # Shape: [9, C, H, W]
                    


                    # Interpolate `hr_un` to match feature resolution (broadcasting will handle alignment)
                    lr_un_single = F.interpolate(
                        lr_un_single.float(),
                        size=(feature_single_group.shape[2], feature_single_group.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )  # Shape remains [1, 1, H, W]
                    lr_seg2=lr_seg[j:j+1]
                    lr_seg2 = F.interpolate(
                        lr_seg2,
                        size=(feature_single_group.shape[2], feature_single_group.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )  # Shape remains [1, 1, H, W]

                    hr_un = 1-(torch.softmax(feature_single_group.detach(), dim=1))
                    hr_un = torch.max(hr_un, dim=1, keepdim=True)[0]

                    # Convert to binary after interpolation
                    #hr_un_single = (hr_un_single > 0.5).bool()
                    #hr_c_single=1-hr_un_single.float()
                    # Element-wise multiplication (broadcasting applies hr_un_single to all 9 features)
                    sparse_group=self.DualA(feature_single_group,hr_un,lr_seg2, lr_un_single )
                    
                    #sparse_group = feature_single_group * hr_un_single.float()  # Broadcasting: [9, C, H, W] * [1, 1, H, W]
                    #sparse_group2=feature_single_group * hr_c_single.float()
                    #sparse_group=sparse_group+feature_single_group
                    sparse_feature.append(sparse_group)  # Append the entire group to sparse_feature

                # Combine all groups back into a single tensor for this feature
                sparse_feature = torch.cat(sparse_feature, dim=0)  # Shape: [18, C, H, W] (if 2 groups with 9 features each)

                # Append the sparsified tensor for this feature to sparse_hr_features
                sparse_hr_features.append(sparse_feature)

           

            hr_inp['features'] = sparse_hr_features

        hr_seg = self.decode_hr(hr_inp, batch_size)
        
        att = self.get_scale_attention(lr_sc_att_inp)
        if has_crop:
            mask = lr_seg.new_zeros([lr_seg.shape[0], 1, *lr_seg.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os)
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask
        # print_log(f'att {att.shape}', 'mmseg')
        lr_seg = (1 - att) * lr_seg
        # print_log(f'scaled lr_seg {lr_seg.shape}', 'mmseg')
        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)
        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)

        if has_crop:
            hr_seg_inserted = torch.zeros_like(up_lr_seg)
            slc = self.hr_crop_slice(self.os)
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg
        else:
            hr_seg_inserted = hr_seg

        fused_seg = att * hr_seg_inserted + up_lr_seg

        if self.debug_output_attention:
            att = torch.sum(
                att * torch.softmax(fused_seg, dim=1), dim=1, keepdim=True)
            return att, None, None

        if self.debug:
            self.debug_output.update({
                'High Res':
                torch.max(hr_seg, dim=1)[1].detach().cpu().numpy(),
                'High Res Inserted':
                torch.max(hr_seg_inserted, dim=1)[1].detach().cpu().numpy(),
                'Low Res':
                torch.max(lr_seg, dim=1)[1].detach().cpu().numpy(),
                'Fused':
                torch.max(fused_seg, dim=1)[1].detach().cpu().numpy(),
            })
            if torch.is_tensor(att):
                self.debug_output['Attention'] = torch.sum(
                    att * torch.softmax(fused_seg, dim=1), dim=1,
                    keepdim=True).detach().cpu().numpy()

        return fused_seg, lr_seg, hr_seg

    def reset_crop(self):
        del self.hr_crop_box
        self.hr_crop_box = None

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      return_logits=False):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        if return_logits:
            losses['logits'] = seg_logits
        self.reset_crop()
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute losses."""
        fused_seg, lr_seg, hr_seg = seg_logit
        loss = super(HRDAHead, self).losses(fused_seg, seg_label, seg_weight)
        if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
            return loss

        if self.lr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(lr_seg, seg_label,
                                                 seg_weight), 'lr'))
        if self.hr_loss_weight > 0 and self.enable_hr_crop:
            cropped_seg_label = crop(seg_label, self.hr_crop_box)
            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
            else:
                cropped_seg_weight = seg_weight
            if self.debug:
                self.debug_output['Cropped GT'] = \
                    cropped_seg_label.squeeze(1).detach().cpu().numpy()
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, cropped_seg_label,
                                                 cropped_seg_weight), 'hr'))
        elif self.hr_loss_weight > 0:
            loss.update(
                add_prefix(
                    super(HRDAHead, self).losses(hr_seg, seg_label,
                                                 seg_weight), 'hr'))
        loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
        if self.lr_loss_weight > 0:
            loss['lr.loss_seg'] *= self.lr_loss_weight
        if self.hr_loss_weight > 0:
            loss['hr.loss_seg'] *= self.hr_loss_weight

        if self.debug:
            self.debug_output['GT'] = \
                seg_label.squeeze(1).detach().cpu().numpy()
            # Remove debug output from cross entropy loss
            self.debug_output.pop('Seg. Pred.', None)
            self.debug_output.pop('Seg. GT', None)

        return loss