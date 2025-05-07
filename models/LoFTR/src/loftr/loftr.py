import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class LoFTR(nn.Module):
    def __init__(self, config, return_attn=False):
        super().__init__()
        # Misc
        self.config = config
        #print("LoFTR CONFIG:", self.config)
        self.return_attn = return_attn

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'], config['onnx'], return_attn=return_attn)
        self.coarse_matching = CoarseMatching(config['match_coarse'], config['onnx'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"], config['onnx'], return_attn=False)
        self.fine_matching = FineMatching(config['onnx'])

    #def forward(self, data):
    #def forward(self, img0, img1):
    def forward(self, data, data1=None):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        
        #print("ONNX:", self.config['onnx'])
        if self.config['onnx']: # inputs are two images
            data = {'image0': data, 'image1': data1}

        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })
        #print("Data forward sizes:", data['hw0_i'], data['hw1_i'], data['bs'], "\n")

        #print(data)
        #print(data['hw0_i'][0].item())

        if self.config['onnx']: # onnx complains that tensor should not be converted to boolean
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])
        else:
            if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
                feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
                (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
            else:  # handle different input shapes
                (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        if self.config['onnx']:
            ## Alternative implementation of rearrange for ONNX
            feat_c0 = self.pos_encoding(feat_c0)
            feat_c0 = torch.permute(feat_c0, (0, 2, 3, 1))
            feat_c0 = torch.reshape(feat_c0, (feat_c0.shape[0], feat_c0.shape[1]*feat_c0.shape[2], feat_c0.shape[3]))
            feat_c1 = self.pos_encoding(feat_c1)
            feat_c1 = torch.permute(feat_c1, (0, 2, 3, 1))
            feat_c1 = torch.reshape(feat_c1, (feat_c1.shape[0], feat_c1.shape[1]*feat_c1.shape[2], feat_c1.shape[3]))
        else:
            feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
            feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        if self.return_attn:
            feat_c0, feat_c1, self_attn0, cross_attn0 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
            data['self_attn0'] = self_attn0
            data['cross_attn0'] = cross_attn0
        else:
            feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)

        if self.config['onnx']: # onnx complains that tensor should not be converted to boolean
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)
        else:
            if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
                feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        if self.config['onnx']:
            return data['mkpts0_f'], data['mkpts1_f'], data['mconf'] 


    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
