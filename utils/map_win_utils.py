import os
import numpy as np
from utils.projections_utils import backproject_ortho_points, backproject_persp_points, project_on_ortho, project_on_persp
import utils.plot_utils as plot_utils

########### Map windows utilities #####################################

def computePointCovarianceFromImageToMapMars(input_var, px, py, pz, yaw, pitch, roll, R_wc, intr, R_mw, px_resolution):
    '''
    Compute uncertainty of point projected from the query image to the map, for the Mars image dataset
    input_var: dict containing input uncertainties
    px, py, pz: point (px,py) on image, pz is depth
    tx, ty, tz: noisy image pose translation
    yaw, pitch, roll: noisy image pose rotation angles 
    R_wc: noisy image pose rotation as SO(3), from world (w) to camera (c) frame
    intr: camera intrinsic params
    R_mw: map camera pose rotation as SO(3), from map camera (m) to world (w) frame

                                           [(px-cx)*pz/fx,]                             [cx_map,]
    function: (1 / px_map_res) ( R_mw R_wc |(py-cx)*pz/fy,| +  R_mw t_wc + t_map_mw ) + |cy_map,|
                                           [      pz      ]                             [   1   ]
    We need to compute Jacobian wrt
    px, py, pz, tx, ty, yaw, pitch, roll
    Note that we do not compute derivate wrt tz because Z is not used when projecting on the map

    Rotation matrix according to rotation order 312: R(yaw)*R(pitch)*R(roll) = R(rot_z)*R(rot_x)*R(rot_y)
    rot_z, rot_x, rot_y = yaw_rad, pitch_rad, roll_rad
    cosx, cosy, cosz = np.cos(rot_x), np.cos(rot_y), np.cos(rot_z)
    sinx, siny, sinz = np.sin(rot_x), np.sin(rot_y), np.sin(rot_z)
    R_wc =  numpy.array([[cosz*cosy - sinz*sinx*siny,  sinz*cosx, -cosz*siny -sinz*sinx*cosy],
                         [sinz*cosy + cosz*sinx*siny, -cosz*cosx, -sinz*siny +cosz*sinx*cosy],
                         [                -cosx*siny,      -sinx,                 -cosx*cosy]], dtype=numpy.float64)
    '''
    
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]

    # back_p 1D-array
    back_px = (px-cx)*pz / fx
    back_py = (py-cy)*pz / fy
    back_pz = pz
    back_p = np.array([[back_px, back_py, back_pz]])
    back_p = back_p.reshape((3, 1))

    # Derivatives of back_p 1-D array w.r.t. px, py, pz
    dback_px_dpx = pz / fx
    dback_px_dpy = 0
    dback_px_dpz = (px-cx)/fx

    dback_py_dpx = 0
    dback_py_dpy = pz / fy
    dback_py_dpz = (py-cy)/fy

    dback_pz_dpx = 0
    dback_pz_dpy = 0
    dback_pz_dpz = 1

    dback_p_dpx = np.array([[dback_px_dpx, dback_py_dpx, dback_pz_dpx]])
    dback_p_dpy = np.array([[dback_px_dpy, dback_py_dpy, dback_pz_dpy]])
    dback_p_dpz = np.array([[dback_px_dpz, dback_py_dpz, dback_pz_dpz]])
    dback_p_dpx = dback_p_dpx.reshape((3,1))
    dback_p_dpy = dback_p_dpy.reshape((3,1))
    dback_p_dpz = dback_p_dpz.reshape((3,1))

    # Derivatives of t_wc w.r.t t_wc_x, t_wc_y, t_wc_z
    dt_wc_dtx = np.array([[1, 0, 0]])
    dt_wc_dty = np.array([[0, 1, 0]])
    dt_wc_dtz = np.array([[0, 0, 1]])
    dt_wc_dtx = dt_wc_dtx.reshape((3,1))
    dt_wc_dty = dt_wc_dty.reshape((3,1))
    dt_wc_dtz = dt_wc_dtz.reshape((3,1))  

    # Derivatives of R_wc elements w.r.t. yaw, pitch, roll
    ### Query rotation w.r.t. world ###
    rot_z, rot_x, rot_y = yaw, pitch, roll
    cosx, cosy, cosz = np.cos(rot_x), np.cos(rot_y), np.cos(rot_z)
    sinx, siny, sinz = np.sin(rot_x), np.sin(rot_y), np.sin(rot_z)
    ### Row 0 ###
    dR00_drotx = -sinz*cosx*siny
    dR00_droty = -cosz*siny - sinz*sinx*cosy
    dR00_drotz = -sinz*cosy - cosz*sinx*siny

    dR01_drotx = -sinz*sinx
    dR01_droty =  0
    dR01_drotz =  cosz*cosx

    dR02_drotx = -sinz*cosx*cosy
    dR02_droty = -cosz*cosy +sinz*sinx*siny
    dR02_drotz =  sinz*siny -cosz*sinx*cosy

    ### Row 1 ####
    dR10_drotx =  cosz*cosx*siny
    dR10_droty = -sinz*siny + cosz*sinx*cosy
    dR10_drotz =  cosz*cosy - sinz*sinx*siny

    dR11_drotx =  cosz*sinx
    dR11_droty =  0
    dR11_drotz =  sinz*cosx

    dR12_drotx =  cosz*cosx*cosy
    dR12_droty = -sinz*cosy -cosz*sinx*siny
    dR12_drotz = -cosz*siny -sinz*sinx*cosy

    ### Row 2 ###
    dR20_drotx =  sinx*siny
    dR20_droty = -cosx*cosy
    dR20_drotz =  0

    dR21_drotx =  -cosx
    dR21_droty =   0
    dR21_drotz =   0

    dR22_drotx =   sinx*cosy
    dR22_droty =   cosx*siny
    dR22_drotz =   0


    dRwc_drotx = np.array([[dR00_drotx, dR01_drotx, dR02_drotx],
                        [dR10_drotx, dR11_drotx, dR12_drotx],
                        [dR20_drotx, dR21_drotx, dR22_drotx]])

    dRwc_droty = np.array([[dR00_droty, dR01_droty, dR02_droty],
                        [dR10_droty, dR11_droty, dR12_droty],
                        [dR20_droty, dR21_droty, dR22_droty]])

    dRwc_drotz = np.array([[dR00_drotz, dR01_drotz, dR02_drotz],
                        [dR10_drotz, dR11_drotz, dR12_drotz],
                        [dR20_drotz, dR21_drotz, dR22_drotz]])
    

    # Derivatives of f = [u, v, ...]
    df_dpx = (1/px_resolution)*np.dot(np.matmul(R_mw, R_wc), dback_p_dpx)
    df_dpy = (1/px_resolution)*np.dot(np.matmul(R_mw, R_wc), dback_p_dpy)
    df_dpz = (1/px_resolution)*np.dot(np.matmul(R_mw, R_wc), dback_p_dpz)
    df_dtx = (1/px_resolution)*np.dot(R_mw, dt_wc_dtx)
    df_dty = (1/px_resolution)*np.dot(R_mw, dt_wc_dty)
    df_dx  = (1/px_resolution)*np.dot(np.matmul(R_mw, dRwc_drotx), back_p)
    df_dy  = (1/px_resolution)*np.dot(np.matmul(R_mw, dRwc_droty), back_p)
    df_dz  = (1/px_resolution)*np.dot(np.matmul(R_mw, dRwc_drotz), back_p)
    # print(f"df-dpx shape: {df_dpx.shape}")

    # We need to compute Jacobian wrt
    # px, py, pz, tx, ty, yaw, pitch, roll
    # Note that we do not compute derivate wrt tz because Z is not used when projecting on the map
    J = np.zeros((2,8))
    # print(f"J shape: {J.shape}")
    # print(f"J[:,0] shape: {J[:,0].shape}")
    J[:,0] = df_dpx[:2].squeeze()  # ... / dpx
    J[:,1] = df_dpy[:2].squeeze()  # ... / dpy
    J[:,2] = df_dpz[:2].squeeze()  # ... / dpz
    J[:,3] = df_dtx[:2].squeeze()  # ... / dtx
    J[:,4] = df_dty[:2].squeeze()  # ... / dty
    J[:,5] = df_dz[:2].squeeze()   # ... / dz = ... / dyaw
    J[:,6] = df_dx[:2].squeeze()   # ... / dx = ... / dpitch
    J[:,7] = df_dy[:2].squeeze()   # ... / dy = ... / droll
    
    # Compute Covariance
    cov_x = np.diag([input_var['var_px'], input_var['var_py'], input_var['var_z'], # var_z is either var_altimeter or var_pz
                     input_var['var_tx'], input_var['var_ty'], input_var['var_yaw'], 
                     input_var['var_pitch'], input_var['var_roll']])

    cov_p = np.dot(np.dot(J, cov_x), J.transpose())

    return cov_p

def get_map_search_window(map_height, map_width, pose_uncertainty, camera_intr, px_resolution, points2D, map_points2D_noisy, points2D_depth_noisy, only_altitude, noisy_pose, R_ortho2world):
    #### Estimate window over the map to be used for matching based on the pose uncertainty
    # This is done by projecting multiple corner image pixels to find an initial window size before expansion

    yaw_noisy, roll_noisy, pitch_noisy, R_wc_noisy = noisy_pose

    # Add map keypoint uncertainty based on map resolution
    # Variance of uniform distribution = step^2/12
    pose_uncertainty['var_px'] = px_resolution ** 2 / 12
    pose_uncertainty['var_py'] = px_resolution ** 2 / 12
    pose_uncertainty['var_z'] = pose_uncertainty['var_altimeter']  # this changes later based on the args.only_altitude

    # When using the corner pixels, we estimate the cov_uv for all and use range from the pixel with the highest uncertainty
    # The base size of the window is found from the projected points
    ranges = []
    
    for i in range(len(points2D)):
        px, py = points2D[i,0], points2D[i,1]
        pz = points2D_depth_noisy[i] #img_depth[py,px]

        # If we have only altimeter (not depth) then we use the larger variance for all points except the center (point 0)
        if (i>0) and only_altitude:
            pose_uncertainty['var_z'] = pose_uncertainty['var_pz']
        cov_uv = computePointCovarianceFromImageToMapMars(pose_uncertainty, px, py, pz, yaw_noisy, pitch_noisy, roll_noisy, R_wc_noisy, camera_intr, R_ortho2world, px_resolution)
        x_range = int(3 * np.sqrt(cov_uv[0, 0]) + 0.5)
        y_range = int(3 * np.sqrt(cov_uv[1, 1]) + 0.5)
        ranges.append(x_range)
        ranges.append(y_range)
    #print(ranges)
    max_range = np.amax(np.asarray(ranges))
    #print(max_range)

    min_u = np.amin(map_points2D_noisy[:,0])
    max_u = np.amax(map_points2D_noisy[:,0])
    min_v = np.amin(map_points2D_noisy[:,1])
    max_v = np.amax(map_points2D_noisy[:,1])
    #max_height = max_v - min_v
    #max_width = max_u - min_u
    #print(max_height, max_width)
    c_start = max(0, min_v - max_range) # left
    c_end = min(map_height-1, max_v + max_range) # right
    r_start = max(0, min_u - max_range) # top
    r_end = min(map_width-1, max_u + max_range) # bottom

    # box = [r_start, c_start, r_end, c_end]
    box = [int(r_start), int(c_start), int(r_end), int(c_end)]
    return box

def get_map_window_images(map_img, map_search_box, win_width=1024, win_height=1024, win_overlap=128, black_thresh=0.25):
    
    #TODO: there may be map_windows with height, width not exactley equal to the desired ones.
    # Option 1 (this code): leave the map windows with possible smaller size
    # Option 2: discard map windows with size not corresponding to the one give as input
    # Option 3: pad the map winwo to reach the desired size


    print(f"\n... getting map windows with (width, height)={(win_width, win_height)} pixels and overlap={win_overlap} pixels")

    [search_r_start, search_c_start, search_r_end, search_c_end] = map_search_box
    map_crop_img   =   map_img[search_c_start:search_c_end, search_r_start:search_r_end]
    height, width = np.shape(map_crop_img)

    nr_vertical_windows = int(width / (win_width - win_overlap) + 0.5) # 5
    nr_horizontal_windows = int(height / (win_height - win_overlap) + 0.5) # 5

    map_windows = {}
    count=0
    for r in range(0, nr_vertical_windows):
        r_start = r * (win_width - win_overlap)
        r_end = min(width, r_start + win_width)

        for c in range(0, nr_horizontal_windows):
            c_start = c * (win_height - win_overlap)
            c_end = min(height, c_start + win_height)

            map_window_img = map_crop_img[c_start:c_end, r_start:r_end]

            # # Pad the image to ensure it's the input size (Option 3)
            # map_gray_crop = np.pad(map_gray_crop, 
            #         ((0, win_width - map_gray_crop.shape[0]), 
            #             (0, win_height - map_gray_crop.shape[1])),
            #         mode='constant', constant_values=0)


            # Save map window with percentage of black pixels below a threshold
            #   Calculate the percentage of black pixels
            black_pixel_count = np.sum( map_window_img == 0)
            total_pixel_count =  map_window_img.size
            black_percentage = black_pixel_count / total_pixel_count
            if black_percentage <= black_thresh:
                map_windows[count] = ( map_window_img, r_start + search_r_start, c_start + search_c_start) # keep the start coordinate for each image
                #print(count, r_start, c_start)
                count+=1

    print(f"... complete.")
    print(f"Number of map windows: {count}")
 
    

    return map_windows

def get_map_window_images_and_depth_w_box(map_img, map_depth, map_search_box, cam_clip_start, win_width=1024, win_height=1024, win_overlap=128, depth_thresh=0.25, shadow_thresh=0.6):
    
    print(f"\n... getting map windows with (width, height)={(win_width, win_height)} pixels and overlap={win_overlap} pixels")
    [search_r_start, search_c_start, search_r_end, search_c_end] = map_search_box
    map_crop_img   =   map_img[search_c_start:search_c_end, search_r_start:search_r_end]
    map_crop_depth = map_depth[search_c_start:search_c_end, search_r_start:search_r_end]
    height, width = np.shape(map_crop_img)


    nr_vertical_windows = int(width / (win_width - win_overlap) + 0.5) # 5
    nr_horizontal_windows = int(height / (win_height - win_overlap) + 0.5) # 5

    map_windows = {}
    count=0
    for r in range(0, nr_vertical_windows):
        r_start = r * (win_width - win_overlap)
        r_end = min(width, r_start + win_width)

        for c in range(0, nr_horizontal_windows):
            c_start = c * (win_height - win_overlap)
            c_end = min(height, c_start + win_height)

            map_win_img   = map_crop_img[c_start:c_end, r_start:r_end]
            map_win_depth = map_crop_depth[c_start:c_end, r_start:r_end]
            #print(map_gray_crop.shape)

            # Compute depth mask
            # print(f"...map win # {count}")
            # print(f"...compute depth mask")
            depth_valid, depth_mask = check_for_depth(map_win_depth, cam_clip_start, thresh=depth_thresh)

            # Compute shadow mask
            # print(f"...compute shadow mask")
            shadow_valid, shadow_mask = check_for_shadows(map_win_img, depth_mask, thresh=shadow_thresh)

            
            # Keep map wins with per of non-valid depth pixels < depth_thresh and with shadow per. < shadow_thresh)
            # print(f"...check for valid depth and shadowed pixels")
            if depth_valid and shadow_valid:
                win_r_start = r_start + search_r_start
                win_c_start = c_start + search_c_start
                win_r_end   = r_end   + search_r_start
                win_c_end   = c_end   + search_c_start 
                map_windows[count] = (map_win_img, map_win_depth, win_r_start, win_c_start, win_r_end, win_c_end, depth_mask, shadow_mask) # keep the start coordinate for each image
                #print(count, r_start, c_start)
                count+=1
    # print(f"... complete.")
    print(f"Number of map windows: {count}")
    return map_windows

def select_map_window(query_depth,
                      query_clip_start,
                      query_intr,
                      R_query_wc, 
                      t_query_wc,
                      map_windows,
                      map_depth,
                      map_px_resolution,
                      map_cx,
                      map_cy,
                      R_map_wc,
                      t_map_wc,
                      query_gray,
                      map_gray,
                      overlap_thresh = 0.25,
                      point_3D_err_thresh = 1,
                      corresp_save_dir=False
                      ):

    # Finds a corresponding image pair from the map
    # Returns the window box

    inds = np.where(query_depth> query_clip_start)
    query_points_2D = np.zeros((len(inds[0]), 2), dtype=int)
    query_points_2D[:,0] = inds[1] # inds[1] is x (width coordinate)
    query_points_2D[:,1] = inds[0]
    #points2D = points2D[::100, :]

    print(query_points_2D.shape)
    # points2D_depth = img_depth[points2D[:,1], points2D[:,0]]
    #print(points2D_depth.shape)
    
    # Project points from the image on the map
    R_map_cw = R_map_wc.T
    t_map_cw = np.dot(R_map_cw, -t_map_wc)
    import time
    start_t = time.time()
    query_points_3D = backproject_persp_points(query_points_2D, query_depth, query_intr, R=R_query_wc, T=t_query_wc)
    map_points_2D = project_on_ortho(query_points_3D, map_px_resolution, cx=map_cx, cy=map_cy, R=R_map_cw, T=t_map_cw)
    print("Time elapsed:", time.time()-start_t)
    # viz_utils.plot_correspondences(query_gray, map_gray, query_points_2D, map_points_2D)
    # plt.show()

    # Selct only the points that are co-visible from both queries and map
    # TODO: test with point_3D_err > 1 and < 10
    valid_map_points_2D = []
    # valid_map_points_3D = []
    valid_query_points_2D = []
    # valid_query_points_3D = []
    for i in range(map_points_2D.shape[0]):
        x,y = map_points_2D[i,:]
        map_point_3D = backproject_ortho_points(np.expand_dims(map_points_2D[i,:], axis=0), map_depth, map_px_resolution, R=R_map_wc, T=t_map_wc)
        point_3D_err = np.linalg.norm(query_points_3D[i,:]- map_point_3D, axis=1) # m
        if point_3D_err < point_3D_err_thresh:
            # query_point_2D_reproj = project_on_persp(map_point_3D, query_intr, R=R_query_cw, T=t_query_cw_c)
            valid_query_points_2D.append(query_points_2D[i,:])
            # valid_query_points_3D.append(query_points_3D[i,:])
            # valid_query_points_2D_reproj.append(query_point_2D_reproj[0])

            valid_map_points_2D.append(map_points_2D[i,:])
            # valid_map_points_3D.append(map_point_3D[0])
    query_points_2D = np.asarray(valid_query_points_2D)
    # query_points_3D = np.asarray(valid_query_points_3D)
    map_points_2D = np.asarray(valid_map_points_2D)
    # map_points_3D = np.asarray(valid_map_points_3D)

    # Search through map windows for ovelap
    win_keys = [] # store win ids with sufficient overlap
    for i in map_windows.keys():
        win_r_start = map_windows[i][1]
        win_c_start = map_windows[i][2]
        win_r_end = map_windows[i][3]
        win_c_end = map_windows[i][4]

        #print(map_points2D.shape)
        #print(win_r_start, win_r_end, win_c_start, win_c_end)

        inds = np.where( (map_points_2D[:,0]>=win_r_start) & (map_points_2D[:,0]<win_r_end) & (map_points_2D[:,1]>=win_c_start) & (map_points_2D[:,1]<win_c_end) )[0]
        # TODO: visualize ccorrespondences
        # if len(inds)>0:
        #     print(f"win key: {i}", len(inds), len(inds)/query_points_2D.shape[0])
        #     qpoints = query_points_2D[inds,:]
        #     mpoints = map_points_2D[inds,:]
        #     viz_utils.plot_correspondences(query_img_gray, map_gray, qpoints, mpoints)
        
        if len(inds)/query_points_2D.shape[0] > overlap_thresh:
            win_keys.append(i)
        
            if corresp_save_dir:
                # Check the folder exists
                if not os.path.exists(corresp_save_dir):
                    os.makedirs(corresp_save_dir)

                overlap = len(inds)/query_points_2D.shape[0]
                print(f"win key: {i}", f"len(inds): {len(inds)}", f"overlap: {overlap*100}%")
                qpoints = query_points_2D[inds,:]
                mpoints = map_points_2D[inds,:]
                wmpoints =   mpoints - np.array([win_r_start, win_c_start])
                filepath = os.path.join(corresp_save_dir,"pair_%06i" % i + f"_overlap_{int(overlap*100)}.png")
                plot_utils.plot_correspondences(query_gray, map_gray[win_c_start:win_c_end, win_r_start:win_r_end], qpoints, wmpoints, points_thresh=True, text=f"Overlap: {int(overlap*100)}%", save_filepath=filepath)

    return win_keys


    # # Select the window over the map
    # min_u = np.amin(map_points2D[:,0])
    # max_u = np.amax(map_points2D[:,0])
    # min_v = np.amin(map_points2D[:,1])
    # max_v = np.amax(map_points2D[:,1])
    # #max_height = max_v - min_v
    # #max_width = max_u - min_u
    # #print(max_height, max_width)
    # c_start = max(0, min_v) # left
    # c_end = min(map_height-1, max_v) # right
    # r_start = max(0, min_u) # top
    # r_end = min(map_width-1, max_u) # bottom

    # # ** Here select whether to crop / enlarge window (for data augmentation based on expected image overlap) and
    # # ** filter the remaining points

    # # Choose to crop such that the image size is divisible by 8
    # #print(c_start, r_start, c_end, r_end, c_end-c_start, r_end-r_start)
    # w_win, h_win = utils.get_divisible_wh(c_end-c_start, r_end-r_start, df=df)
    # c_end = c_start + w_win
    # r_end = r_start + h_win
    # #print(c_start, r_start, c_end, r_end, c_end-c_start, r_end-r_start)

    # #map_window_img = self.map_gray[c_start:c_end, r_start:r_end]
    # window = [c_start, r_start, c_end, r_end]
    # return window

###### Mask functions #########################################################
def check_for_depth(depth, clip_start, thresh):
    
    # Find indices with invalid depth values (i.e. falling outside the terrain)
    inds_x, inds_y = np.where(depth < clip_start)

    # Calculate the total number of depth image pixels
    n_pixels = depth.shape[0]*depth.shape[1]

    # Calculate the percentage of invalid depth pixels
    per = inds_x.shape[0] / n_pixels
    # print(f"per invalid depth: {per}")

    # Determine whether the percentage exceeds the threshold
    valid = True
    if per >= thresh:
        valid = False

    # Create the binary mask, marking invalid depth areas as 0  
    mask = np.ones(depth.shape)   
    mask[inds_x, inds_y] = 0   

    return valid, mask

def check_for_shadows(img, depth_mask, thresh):
    '''
        Check the amount of shadow in an image
        Simply estimate percentage of black pixels among the ones that have valid depth values 
        Return binary mask indicating valid areas for sampling correspondences
    '''


    # Find indices where the image meets the shadow condition (img <= 5)
    inds_x, inds_y = np.where(img <= 5)
    n_total_pixels = img.shape[0]*img.shape[1]
    # print(f"per black areas: {inds_x.shape[0]/n_total_pixels}")

    # Check only for valid depth pixels (depth_mask == 1) at the same indices
    valid_inds = depth_mask[inds_x, inds_y] == 1  # Create a boolean array of valid pixels

    # Filter indices to only include valid depth pixels
    valid_shadow_inds_x = inds_x[valid_inds]
    valid_shadow_inds_y = inds_y[valid_inds]

    # Calculate the total number of valid depth pixels
    n_valid_depth_pixels = np.sum(depth_mask == 1)
    
    # Calculate the percentage of valid shadow pixels (those with img <= 5 and depth_mask == 1)
    per = valid_shadow_inds_x.shape[0] / n_valid_depth_pixels
    # print(f"per valid shadows: {per}")

    # Determine whether the percentage exceeds the threshold
    valid = True
    if per >= thresh:
        valid = False
    #print(per)
        
    # Create the binary mask, marking valid shadow areas as 0   
    mask = np.ones(img.shape)
    mask[inds_x, inds_y] = 0
    
    return valid, mask
