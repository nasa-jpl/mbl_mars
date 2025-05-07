
import numpy as np
import os
import cv2
import torch
from einops import repeat



def warp_kpts_mars_sim_to_query(points2D, batch, device):
    
    # Projects points2D (map window image) on query
    #TODO: check if the matched keypoints in output by LoFTR fine matching are in the original input size og the image (before loftr resizing to 480x640)
    # in case they're not, we need to resize them before procesisng the matches

    query_points_2D = torch.zeros((points2D.shape[0], points2D.shape[1], 2), dtype=torch.float32).to(device=device) # B x N x 2
    points2D = points2D.round().long() # B x N x 2

    for k in range(len(points2D)): # batch (k is the batch index)
        K0 = batch['K0'][k]
        fx, fy, cx, cy = K0[0,0], K0[1,1], K0[0,2], K0[1,2]
        map_depth = batch['image1_depth'][k,:,:] # map window depth
        # get depth points using the window coordinates
        #map_points2D_depth = map_depth[points2D[k,:,1], points2D[k,:,0]]
        R_inv = torch.linalg.inv(batch['R0'][k]).float()    # R0 = R_query_wc (from world to camera); R_inv = inv(R_query_wc) = R_query_cw
        T_inv = torch.matmul(-R_inv, batch['t0'][k]).float() #np.dot(-R_inv, batch['t0'][k]) - t0 = t_query_wc ; T_inv =  np.dot(-R_inv, t0) = np.dot(-R_query_cw, t_query_wc) = t_query_cw
        px_resolution = batch['px_resolution'][k]
        cx_map = batch['cx_map'][k]
        cy_map = batch['cy_map'][k]
        map_height, map_width = cy_map*2, cx_map*2
        #TODO: Resume here
        R_map_wc = batch['R_map_wc'][k]
        t_map_wc = batch['t_map_wc'][k]
        t_map_wc = t_map_wc.reshape((3, 1)) # 3 x 1

        [c_start, r_start, c_end, r_end] = batch['image1_map_box'][k]

        x, y = points2D[k,:,0], points2D[k,:,1]
        z = map_depth[y, x]
        # convert window coordinates back to the map
        x += r_start.long()
        y += c_start.long()

        ## Unproject from map and transform to query coordinates    
        local3D = torch.zeros((x.shape[0],3), dtype=torch.float32).to(device=device) # N x 3
        # print(f"local3D shape: {local3D.shape}") # N x 3 (confirmed)
        # print(f"t_map_wc shape: {t_map_wc.shape}") # 3 x 1 (confirmed)
        local3D[:,0] = (x-map_width/2)*px_resolution
        local3D[:,1] = (y-map_height/2)*px_resolution
        local3D[:,2] = z
    
        # Transform local3D to world coordinates
        local3D = local3D.t()  # Transpose to match (3, N)
        w3D = torch.matmul(R_map_wc, local3D) + t_map_wc  # (3, 3) @ (3, N) + (3, 1)
        # print(f"w3D shape: {w3D.shape}")
        # # Transpose back to (N, 3)
        # points3D = w3D.t()

        # Transform world coordinates to local 3D coordinates
        local_query_3D = torch.matmul(R_inv, w3D) + T_inv  # (3, 3) @ (3, N) + (3, 1)
        # print(f"local_query_3D shape: {local_query_3D.shape}") # 3 x N

        ## Project on query
        x_proj = fx * local_query_3D[0,:] / local_query_3D[2,:] + cx # X*f_x/Z + cx
        y_proj = fy * local_query_3D[1,:] / local_query_3D[2,:] + cy # Y*f_y/Z + cy
        query_points_2D[k,:,0] = x_proj#.round().long()
        query_points_2D[k,:,1] = y_proj#.round().long()

        # Apply in-plane rotation on the points, if rotation was applied on the query image in the dataloader
        query_points_2D[k,:,:] = apply_affine_transf(M=batch['M'][k,:,:], points=query_points_2D[k,:,:])

        assert points2D.shape[1] == query_points_2D.shape[1], "Corresponding keypoint numbers are not the same!"

    return query_points_2D


def warp_kpts_mars_sim_to_map(points2D, batch, device):
    # Projects points2D (query) on map window image

    # do back projection and transformation in one function
    map_img_points_2D = torch.zeros((points2D.shape[0], points2D.shape[1], 2), dtype=torch.float32).to(device=device) # B x N x 2
    points2D = points2D.round().long()
    
    for k in range(len(points2D)): # batch
        K0 = batch['K0'][k]
        fx, fy, cx, cy = K0[0,0], K0[1,1], K0[0,2], K0[1,2]
        img_depth = batch['image0_depth'][k,:,:]
        R = batch['R0'][k]  # R_query_ wc
        T = batch['t0'][k]  # t_query_ wc
        px_resolution = batch['px_resolution'][k]
        cx_map = batch['cx_map'][k]
        cy_map = batch['cy_map'][k]
        R_map_wc = batch['R_map_wc'][k]
        t_map_wc = batch['t_map_wc'][k]
        R_map_cw = torch.linalg.inv(R_map_wc).float() 
        t_map_cw = torch.matmul(-R_map_cw, t_map_wc).float()
        [c_start, r_start, c_end, r_end] = batch['image1_map_box'][k]

        # Apply in-plane rotation on the points, if rotation was applied on the query image in the dataloader
        points2D[k,:,:] = apply_affine_transf(M=batch['Minv'][k,:,:], points=points2D[k,:,:])

        x, y = points2D[k,:,0], points2D[k,:,1]        
        z = img_depth[y, x]
        local3D = torch.zeros((x.shape[0],3), dtype=torch.float32).to(device=device)
        
        ## Unproject from query and transform to map coordinates
        a = x - cx
        b = y - cy
        q1 = a[:,None]*z[:,None] / fx
        q2 = b[:,None]*z[:,None] / fy
        local3D[:,0] = q1.reshape(q1.shape[0])
        local3D[:,1] = q2.reshape(q2.shape[0])
        local3D[:,2] = z

        # Transform local3D to world coordinates
        local3D = local3D.t()  # Transpose to match (3, N)
        w3D = torch.matmul(R, local3D) + T  # (3, 3) @ (3, N) + (3, 1)
        
        # Transform world coordinates to local map 3D coordinates
        local_map_3D = torch.matmul(R_map_cw, w3D) + t_map_cw  # (3, 3) @ (3, N) + (3, 1)

        # Project on map 
        x_proj = local_map_3D[0,:]/px_resolution + cx_map
        y_proj = local_map_3D[1,:]/px_resolution + cy_map
        map_img_points_2D[k,:,0] = x_proj#.round().long()
        map_img_points_2D[k,:,1] = y_proj#.round().long()

        # Shift points inside the window
        map_img_points_2D[k,:,0] -= r_start
        map_img_points_2D[k,:,1] -= c_start
         
        assert points2D.shape[1] == map_img_points_2D.shape[1], "Corresponding keypoint numbers are not the same!"

    return map_img_points_2D


def apply_affine_transf(M, points):
    # M: 2 x 3 affine transformation
    # points: N x 2 
    points = torch.cat((points, torch.ones((points.shape[0], 1)).cuda()), dim=1)
    points = torch.transpose(points, dim0=0, dim1=1) # 3 x N
    #print(_points.shape)
    rot = torch.matmul(M, points)
    #print(rot.shape)
    rot = torch.transpose(rot, dim0=0, dim1=1)
    return rot[:,:2]


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def resize_img_gray(img, new_size=None, df=None, padding=False, valid_mask=None):
    if new_size is None:
        #scale = 1.0
        scale = torch.tensor([1.0, 1.0], dtype=torch.float32)
    else:
        h, w = img.shape
        ratio = new_size / max(h, w) # resize the longer edge to keep aspect ratio
        w_new, h_new = int(round(w*ratio)), int(round(h*ratio))
        w_new, h_new = get_divisible_wh(w_new, h_new, df=df)
        #print("New dims:", w_new, h_new)
        #print("Scale:", w/w_new, h/h_new)
        img = cv2.resize(img, (w_new, h_new))

        if valid_mask is not None:
            valid_mask = cv2.resize(valid_mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

        #scale = w/w_new
        scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float32)

    h_new, w_new = img.shape
    if padding:
        pad_to = max(h_new, w_new)
        img, mask = pad_bottom_right(img, pad_to, ret_mask=True)
        if valid_mask is not None: # Merge the valid mask with the padded mask
            valid_mask = valid_mask.astype(bool)
            mask[:valid_mask.shape[0], :valid_mask.shape[1]] = valid_mask
            valid_mask = mask.copy()
        else:
            valid_mask = mask.copy()

    return img, valid_mask, scale


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt
