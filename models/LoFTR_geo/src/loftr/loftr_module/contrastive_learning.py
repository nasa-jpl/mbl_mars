
import torch
import torch.nn as nn
import numpy as np


'''
https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py
https://github.com/openai/CLIP/tree/main/clip
'''

class ContrastiveLearning(nn.Module):
    def __init__(self, config):
        super(ContrastiveLearning, self).__init__()

        self.config = config
        self.patch_size = 128 #64 # ** for now assume 640 x 640 img size. This should be a function of img_size
        self.coarse_scale = self.config['resolution'][0] # scale of coarse 1/8
        self.coarse_patch_size = int(self.patch_size / self.coarse_scale)

        self.enc_dim = int(self.coarse_patch_size*self.coarse_patch_size / 2) # 128

        ## Img encoder layers
        self.img_conv_reduce = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0), # 256 is the default number of features at coarse scale
            nn.ReLU(inplace=True),
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(self.enc_dim*2, self.enc_dim*2, bias=False),
            nn.ReLU(True),
            nn.Linear(self.enc_dim*2, self.enc_dim, bias=False),
        )
        self.img_norm = nn.LayerNorm(self.enc_dim)

        ## Depth encoder layers
        self.depth_conv_reduce = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0), # 256 is the default number of features at coarse scale
            nn.ReLU(inplace=True),
        )
        self.depth_encoder = nn.Sequential(
            nn.Linear(self.enc_dim*2, self.enc_dim*2, bias=False),
            nn.ReLU(True),
            nn.Linear(self.enc_dim*2, self.enc_dim, bias=False),
        )
        self.depth_norm = nn.LayerNorm(self.enc_dim)

        # Learned temperature parameter for scaling the similarities
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))



    def get_patches(self, mask_c1):

        # Avoid sampling patches from the padded areas
        inds_x, inds_y = torch.where(mask_c1.squeeze(0))
        feat_height = max(inds_x).item() + 1
        feat_width = max(inds_y).item() + 1

        nr_vertical_patches = int(feat_width / self.coarse_patch_size + 0.5)
        nr_horizontal_patches = int(feat_height / self.coarse_patch_size + 0.5)
        #print(nr_vertical_patches)
        #print(nr_horizontal_patches)

        patches = {}
        count=0
        for r in range(0, nr_vertical_patches):
            r_start = r * self.coarse_patch_size
            for c in range(0, nr_horizontal_patches):
                c_start = c * self.coarse_patch_size
                #map_gray_crop = map_window_img[c_start:min(feat_height, c_start + self.coarse_patch_size), r_start:min(feat_width, r_start + self.coarse_patch_size)]
                #print(map_gray_crop.shape)
                #patches[count] = (map_gray_crop, r_start, c_start) # keep the start coordinate for each image
                patches[count] = (r_start, c_start) # keep the start coordinate for each image
                #print(count, r_start, c_start)
                count+=1
        return patches


    def forward(self, feat_c1, feat_c2, mask_c1):
        #print(feat_c1.shape)
        #feat_c1 = feat_c1.repeat(2,1,1,1)
        #feat_c2 = feat_c2.repeat(2,1,1,1)
        #print(feat_c1.shape)
        
        B, C, cH, cW = feat_c1.shape # we have an extra batch size dimension (compared to the typical CLIP)

        # Divide the feature volumes in feature patches and convert them to patch embeddings
        # feat_c1 and feat_c2 should have the same dimensions in the batch
        patches = self.get_patches(mask_c1)
        N = len(patches)

        batch_patches_c1, batch_patches_c2 = [], []
        for b in range(B):
            patches_c1, patches_c2 = [], []
            for i in range(N):
                p = patches[i]
                patch_feat_c1 = feat_c1[b,:, p[0]:p[0]+self.coarse_patch_size, p[1]:p[1]+self.coarse_patch_size].unsqueeze(0)
                patch_feat_c2 = feat_c2[b,:, p[0]:p[0]+self.coarse_patch_size, p[1]:p[1]+self.coarse_patch_size].unsqueeze(0)
                patches_c1.append(patch_feat_c1)
                patches_c2.append(patch_feat_c2)

            # Patches are already ordered, i.e. img patch 0 corresponds to depth patch 0 etc
            patches_c1 = torch.cat(patches_c1, dim=0) # N x 256 x coarse_patch_size x coarse_patch_size
            patches_c2 = torch.cat(patches_c2, dim=0)
            batch_patches_c1.append(patches_c1)
            batch_patches_c2.append(patches_c2)
        
        batch_patches_c1 = torch.stack(batch_patches_c1, dim=0) # B x N x 256 x coarse_patch_size x coarse_patch_size
        batch_patches_c2 = torch.stack(batch_patches_c2, dim=0)
        #print(batch_patches_c1.shape)

        patches_c1 = batch_patches_c1.view(B*N, C, self.coarse_patch_size, self.coarse_patch_size)
        patches_c2 = batch_patches_c2.view(B*N, C, self.coarse_patch_size, self.coarse_patch_size)
        #print(patches_c1.shape)

        # Encoders for converting the patches to img and depth embeddings
        patches_feat_c1 = self.img_conv_reduce(patches_c1)
        patches_feat_c1_flat = patches_feat_c1.view(B*N, -1)
        embedd_c1 = self.img_encoder(patches_feat_c1_flat)
        embedd_c1 = self.img_norm(embedd_c1) # B*N x enc_dim
        embedd_c1 = embedd_c1.view(B, N, self.enc_dim)
        #print(embedd_c1.shape)

        patches_feat_c2 = self.depth_conv_reduce(patches_c2)
        patches_feat_c2_flat = patches_feat_c2.view(B*N, -1)
        embedd_c2 = self.depth_encoder(patches_feat_c2_flat)
        embedd_c2 = self.depth_norm(embedd_c2)
        embedd_c2 = embedd_c2.view(B, N, self.enc_dim)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        #print(logit_scale)
        batch_logits_per_img, batch_logits_per_depth = [], []
        for b in range(B):
            logits_per_img = logit_scale * embedd_c1[b,:,:] @ embedd_c2[b,:,:].t() # dot(c1, c2) * exp(t) # N x N
            logits_per_depth = logits_per_img.t()
            #print(logits_per_img[0,:]) # Similarity scores of img patch 0 to all depth patches
            #print(logits_per_img.shape)
            batch_logits_per_img.append(logits_per_img)
            batch_logits_per_depth.append(logits_per_depth)

        batch_logits_per_img = torch.stack(batch_logits_per_img, dim=0) # B x N x N
        batch_logits_per_depth = torch.stack(batch_logits_per_depth, dim=0)
        #print(batch_logits_per_img.shape)

        #log_softmax = nn.LogSoftmax(dim=-1)
        #softmax = nn.Softmax(dim=-1)
        #print("Softmax:", softmax(logits_per_img[0,:]))
        #print(log_softmax(logits_per_img[0,:]))

        #self.symmetric_ce_loss(batch_logits_per_img, batch_logits_per_depth)

        return batch_logits_per_img, batch_logits_per_depth
        


        