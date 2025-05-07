import numpy as np
import cv2
import random

###### Matches utilities ############################################################################################

def get_top_k(mkpts0, mkpts1, mconf, top_k):
    # keep only the top k matches for visual clarity
    if mconf.shape[0] < top_k:
        top_k = mconf.shape[0]
    inds = np.argpartition(mconf, -top_k)[-top_k:]
    return mkpts0[inds], mkpts1[inds], mconf[inds], inds

def filter_with_conf(conf_kpts, query_kpts, map_kpts, conf_thresh):
    # If no matches are found, keep reducing the confidence thresh (adaptive).
    # This is not common, usually some matches are found with >=0.95 confidence
    count = 0
    match_inds = conf_kpts > conf_thresh
    while (len(np.where(match_inds==True)[0]) < 100):
        conf_thresh -= 0.05
        match_inds = conf_kpts > conf_thresh
        #print(conf_thresh)
        #print(len(np.where(match_inds==True)[0]))
        if count > 20:
            break
        count += 1
        
    match_query_kpts = query_kpts[match_inds]
    match_map_kpts = map_kpts[match_inds]
    match_conf = conf_kpts[match_inds]

    # # without adaptive conf thresh
    # match_inds = conf_kpts > conf_thresh
    # match_query_kpts = query_kpts[match_inds]
    # match_map_kpts = map_kpts[match_inds]
    # match_conf = conf_kpts[match_inds]

    return match_query_kpts, match_map_kpts, match_conf

def check_bounds(kpts0, kpts1, mconf, img0_size, img1_size):
    # remove kpts that are out-of-bounds
    valid_inds = np.where( (kpts0[:,0]>0) & (kpts0[:,1]>0) & (kpts0[:,0]<img0_size[1]-1) & (kpts0[:,1]<img0_size[0]-1) & 
                              (kpts1[:,0]>0) & (kpts1[:,1]>0) & (kpts1[:,0]<img1_size[1]-1) & (kpts1[:,1]<img1_size[0]-1) )[0]

    kpts0 = kpts0[valid_inds, :]
    kpts1 = kpts1[valid_inds, :]
    mconf = mconf[valid_inds]
    return kpts0, kpts1, mconf

def extract_map_features(map_img, config, feature_type):

    win_size = config['WIN_SIZE']
    win_overlap = config['WIN_OVERLAP']
    nr_bins = config['NR_BINS']
    max_nr_features_per_bin = config['MAX_NR_FEATURES_PER_BIN']

    # Feature extraction
    if feature_type == 'SIFT':
        #Feature = cv2.xfeatures2d.SIFT_create(nfeatures=config['MAX_FEATS_PER_WINDOW'])
        Feature = cv2.SIFT_create(nfeatures=config['MAX_FEATS_PER_WINDOW'])
    elif feature_type == 'ORB':
        Feature = cv2.ORB_create(nfeatures=config['MAX_FEATS_PER_WINDOW'])
    elif feature_type == 'SURF':
        #Feature = cv2.xfeatures2d.SURF_create(400)
        Feature = cv2.SURF_create(400) # ** SURF is currently not working
    elif feature_type == 'BRISK':
        Feature = cv2.BRISK_create() # ** Look more carefully in BRISK parameters

    height, width = np.shape(map_img)

    nr_vertical_windows = int(width / (win_size - win_overlap) + 0.5)
    nr_horizontal_windows = int(height / (win_size - win_overlap) + 0.5)
    kpts_map = []
    descriptors_map = []
    for r in range(0, nr_vertical_windows):
        r_start = r * (win_size - win_overlap)
        for c in range(0, nr_horizontal_windows):
            c_start = c * (win_size - win_overlap)
            map_gray_crop = map_img[c_start:min(height, c_start + win_size), r_start:min(width, r_start + win_size)]
            kpts_crop_map, descriptors_crop_map = Feature.detectAndCompute(map_gray_crop, None)

            if len(kpts_crop_map)==0:
                continue

            for kpt_id in range(len(kpts_crop_map)):
                kpts_crop_map[kpt_id].pt = (kpts_crop_map[kpt_id].pt[0] + r_start,
                                            kpts_crop_map[kpt_id].pt[1] + c_start)
            kpts_map.extend(kpts_crop_map)
            descriptors_map.extend(descriptors_crop_map)

    print('Number of features extracted: ', len(kpts_map))

    # Apply grid filtering
    hist_counter = np.zeros((nr_bins, nr_bins))
    mask = np.zeros((height, width), dtype='bool')
    kpts_map_filtered = []
    descriptors_map_filtered = []

    # Shuffle order so that first keypoints are not favored
    kpts_shuffle_ids = np.arange(len(kpts_map))
    random.shuffle(kpts_shuffle_ids)

    for kpt_id in kpts_shuffle_ids:
        x = round(kpts_map[kpt_id].pt[0])
        y = round(kpts_map[kpt_id].pt[1])

        if not mask[y, x]:  # Skips duplicates
            bin_x_id = int((x / width) * nr_bins)
            bin_y_id = int((y / height) * nr_bins)

            if hist_counter[bin_y_id, bin_x_id] < max_nr_features_per_bin:  # Is it full?
                hist_counter[bin_y_id, bin_x_id] += 1
                mask[y, x] = True
                kpts_map_filtered.append(kpts_map[kpt_id])
                descriptors_map_filtered.append(descriptors_map[kpt_id])

    print('Percentage of features retained after grid filter: ', len(kpts_map_filtered) / len(kpts_map))

    #print(len(kpts_map_filtered))
    #print(len(descriptors_map_filtered))

    return kpts_map_filtered, descriptors_map_filtered


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

