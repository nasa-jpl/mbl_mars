import cv2
import os
import platform
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import yaml
from scipy.stats import gaussian_kde
import time
import torch
torch.cuda.empty_cache()


# MbL utils
from utils.data_load_utils import load_map_data, load_observation_data
from utils.loftr_utils import resize_img_gray, pad_bottom_right, get_divisible_wh
from utils.map_win_utils import get_map_search_window, get_map_window_images_and_depth_w_box
from utils.match_utils import extract_map_features, check_bounds, get_top_k, filter_with_conf,check_for_depth, check_for_shadows
from utils.math_utils import XtoSO3, Eul312toSO3
import utils.misc as misc
from utils.pose_utils import perturb_pose, project_and_print, generate_noisy_depths
from utils.plot_utils import show_matches_on_map, show_matches, plot_correspondences
from utils.projections_utils import backproject_persp_points, project_on_ortho, backproject_ortho_points, project_on_persp

# Default LoFTR model
from models.LoFTR.src.loftr import LoFTR, default_cfg
from models.LoFTR.src.config.default import get_cfg_defaults
from models.LoFTR.src.utils.plotting import make_matching_figure

# Geo-LoFTR model with input map depth
from models.LoFTR_geo.src.loftr import LoFTR_geo
from models.LoFTR_geo.src.config.default import get_cfg_defaults as get_cfg_defaults_geo

##### Models loading ####################################################################################################
def load_loftr(model_type, weights_path):

    print(f"\nLoading LoFTR model:")
    print(f"MODEL TYPE: {model_type}")
    if model_type == 'pretrained':
        matcher = LoFTR(config=default_cfg)
        model_path = os.path.join(weights_path, 'pretrained/outdoor_ds.ckpt')

        # Check if the machine is a Mac
        if platform.system() == 'Darwin':  # 'Darwin' is the system name for macOS
            print("Loading model on CPU because CUDA is not available on macOS.")
            model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            # Load on GPU if available
            model = torch.load(model_path)

        matcher.load_state_dict(model['state_dict'])

    elif model_type == 'finetuned':

        # Load config
        config = get_cfg_defaults()
        main_cfg_path = 'models/LoFTR/configs/loftr/outdoor/loftr_ds_dense.py' #options.main_config_file
        config.merge_from_file(main_cfg_path)
        _config = misc.lower_config(config)

        # Load matcher
        matcher = LoFTR(config=_config['loftr'])
        model_path = os.path.join(weights_path,'V1/finetuned/2024_08_23-05_52_53.pt')
        #   Check if the machine is a Mac
        if platform.system() == 'Darwin':  # 'Darwin' is the system name for macOS
            print("Loading model on CPU because CUDA is not available on macOS.")
            model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            # Load on GPU if available
            model = torch.load(model_path)
        matcher.load_state_dict(model['models']['matcher'])

    elif model_type == 'geo' or model_type == 'geo_ctx':
        
        # Load config files
        config = get_cfg_defaults_geo()
        main_cfg_path = 'models/LoFTR_geo/configs/loftr/outdoor/loftr_ds_dense.py'
        config.merge_from_file(main_cfg_path)
        config.LOFTR.CONTRASTIVE = False
        _config = misc.lower_config(config)
        
        if model_type == 'geo':
            model_path = os.path.join(weights_path,'V1/geo/2024_08_29-17_28_02.pt')
        elif model_type == 'geo_ctx':
            model_path = os.path.join(weights_path,'geo_ctx/2025_03_25-00_23_02.pt')       

        # Load the matcher
        matcher = LoFTR_geo(config=_config['loftr'])
        
        #   Check if the machine is a Mac
        if platform.system() == 'Darwin':  # 'Darwin' is the system name for macOS
            print("Loading model on CPU because CUDA is not available on macOS.")
            model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            # Load on GPU if available
            model = torch.load(model_path)
        matcher.load_state_dict(model['models']['matcher'])

    else:
        raise(f"Model type not supported. Choose between 'pretrained', 'finetuned', 'geo' and 'geo_ctx'.")        
    

    print(f"MODEL PATH: {model_path}")
    matcher = matcher.eval()
    
    # Move to GPU if available and not on Mac
    if platform.system() != 'Darwin' and torch.cuda.is_available():
        matcher = matcher.cuda()

    print(f"Loading complete")
    return matcher



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--method', type=str, default='loftr', choices=['sift', 'loftr'], required=False)  
    parser.add_argument('--map_data_path', type=str, default='/Users/dariopisanti/Documents/PhD_Space/NASA_JPL/LORNA/dataset/jezero_crater/maps/HiRISE_res', required=False)
    parser.add_argument("--test_data_path", default='/Users/dariopisanti/Documents/PhD_Space/NASA_JPL/LORNA/dataset/jezero_crater/test_queries/test_queries_alt64-200m_0angles', help="Dataset with query images", required=False)
    parser.add_argument("--dest_dir", default='./results/jezero_HiRISEtest', help="Destination directory", required=False)
    parser.add_argument("--n", type=int, default=0, help="Number of queries to use for testing. If 0 then use all of them", required=False)
    parser.add_argument('--elev_maps', nargs='+', type=int, default=[30, 60, 90], help='List of Sun elevation angles for map [deg]')
    parser.add_argument('--azim_maps', nargs='+', type=int, default=[0, 90, 180, 270], help='List of Sun azimuth angles for map [deg]')
    parser.add_argument('--elev_tests', nargs='+', type=int, default=[40,], help='List of Sun elevation angles for test observations [deg]')
    parser.add_argument('--azim_tests', nargs='+', type=int, default=[180,], help='List of Sun azimuth angles for tets observations [deg]')
    parser.add_argument('--depth_thresh', type=float, default=0.10, help="Threshold for the depth masks.")
    parser.add_argument('--shadow_thresh', type=float, default=0.6, help="Threshold for shadow masks.")
    
    # SIFT argument group
    sift_group = parser.add_argument_group('sift', 'Arguments related to sift method')
    sift_group.add_argument("--sift_config", default='./config/config_sift_matching.json', help="Config params to extract features", required=False)
    # LoFTR argument group
    loftr_group = parser.add_argument_group('loftr', 'Arguments related to loftr method')
    loftr_group.add_argument('--loftr_model_type', type=str, default='pretrained', choices=['pretrained', # pre-trained model
                                                                                'finetuned', # fine-tuned model
                                                                                'geo', # Geo-LoFTR trained on HiRISE data
                                                                                'geo_ctx'], # Geo-LoFTR trained on CTX data 
                                                                                required=False, 
                                help='pretrained: LoFTR off-the-shelf, finetuned: LoFTR fine-tuned on Mars data with HiRISE-like maps generate with MARTIAN'
                                     'geo: Geo-LoFTR trained on Mars data with HiRISE-like maps generated with MARTIAN, geo_ctx: Geo-LoFTR trained on Mars data with CTX-like maps generated with MARTIAN')
    loftr_group.add_argument('--loftr_weight_path', type=str, default='weights/loftr', help='Path to the LoFTR model', required=False)
    loftr_group.add_argument("--loftr_config", default='./config/config_loftr_matching.json', help="Config params to extract features", required=False)
    loftr_group.add_argument('--resize', type=bool, default=True, help="If enabled, it resizes the images (and depths) before being passed to LoFTR (LoFTR + geo). Provide the new size with --input_img_size.")
    loftr_group.add_argument('--input_img_size', dest='input_img_size', type=int, default=640)
    loftr_group.add_argument('--df', type=int, default=8, help='Resize to divisible dimensions')
    loftr_group.add_argument('--img_padding', default=True, action='store_false')

    # Pose prior argument group
    pose_prior_group = parser.add_argument_group('pose_prior', 'Arguments related to pose prior')
    pose_prior_group.add_argument('--pose_prior', default=False, action='store_true', help='Enable pose prior options')
    pose_prior_group.add_argument("--pose_uncertainty", default='./config/pose_uncertainty_low.json', help="Config params to extract features", required=False)
    pose_prior_group.add_argument("--only_altitude", default=False, action='store_true', help='Assumes query has no depth, only altimeter is available') 
    args = parser.parse_args()
      
    # Create destination directory
    os.makedirs(args.dest_dir, exist_ok=True)
    
    # Load data for pose prior
    if args.pose_prior:
        # Check the right args are provided
        if args.pose_uncertainty is None:
            parser.error("--pose_prior requires --pose_uncertainty to be specified.")
        # Load pose prior uncertainty
        with open(args.pose_uncertainty) as f:
            pose_uncertainty = json.load(f)
        # Variables to save in file names
        uncertainty_mode = args.pose_uncertainty.split('.')[-2].split('_')[-1]
        var = str(int(pose_uncertainty['var_tx'])) # to include in the save filenames

    # Set devce
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    #     DEVICE = 'mps'
    else:
        DEVICE = 'cpu'

    # Load matcher model
    if args.method=="sift":
        # Load feature extraction config
        with open(args.sift_config) as f:
            config = json.load(f)
        # Matcher
        # sift_feature = cv2.SIFT_create(nfeatures=args.max_feats)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        method_name = args.method   
    elif args.method=="loftr":
        # Load feature extraction config
        with open(args.loftr_config) as f:
            config = json.load(f)
        input_img_dim = (config['INPUT_IMG_H'], config['INPUT_IMG_W']) # 480 x 640
        # Resizing variable
        new_size = args.input_img_size if args.resize else None
        # Global variables:
        LOFTR_RESOLUTION = (8, 2)
        coarse_scale = 1 / LOFTR_RESOLUTION[0] # needed to resize the masks 
        # Matcher
        matcher = load_loftr(model_type=args.loftr_model_type, weights_path=args.loftr_weight_path)
        matcher = matcher.eval().to(DEVICE)
        method_name = args.method
    else:
        raise Exception('Invalid method!')
    


    # Print config and args
    print("\nARGS:")
    print(args)
    print(f"\nMETHOD: {method_name}")
    print("\nCONFIG:")
    print(config)


    # File name strigs
    if args.pose_prior:
        var_str = f"_var_{var}"
    else:
        var_str = ''
    if args.method == "loftr":
        match_params_str = f"{config['CONF_THRESH']}conf_thr"
        if config['TOP_K'] > 0:
            match_params_str+=f"_top{config['TOP_K']}"
    elif args.method == "sift":
        match_params_str = f"{config['RATIO_THRESH']}ratio_thr"

    # Choose sun angle ranges to use for maps and observations
    elev_list_maps = args.elev_maps #[60,]
    azim_list_maps = args.azim_maps #[0]
    elev_list_test = args.elev_tests
    azim_list_test = args.azim_tests 

    # Load map and test data
    map_data = load_map_data(args.map_data_path, elev_list_maps, azim_list_maps)
    test_data = load_observation_data(args.test_data_path, elev_list_test, azim_list_test)


    # print(map_data.keys())
    for map_name in map_data.keys():
            
        # Get the map data dictionary
        map_dict = map_data[map_name]

        # Load map image and depth
        map_img = map_dict['gray']
        map_depth = map_dict['depth']

        # Get map camera params
        map_cx, map_cy = map_dict['cx'], map_dict['cy']
        map_img_height, map_img_width = map_dict['img_height'], map_dict['img_width']
        map_clip_start = map_dict['clip_start']
        map_px_resolution = map_dict['px_resolution']

        # Get map pose
        R_map_wc = map_dict['R_wc']
        t_map_wc = map_dict['t_wc']
        R_map_cw = R_map_wc.T
        t_map_cw = np.dot(R_map_cw, -t_map_wc)

        if args.method == "sift":
            # Extract features
            print(f"\nExtracting SIFT features from map...")
            start_map_t = time.time()
            kpts_map, descriptors_map = extract_map_features(map_img, config, 'SIFT')
            elapsed_time_map = time.time()-start_map_t
            print(f"...complete.")
            print(f"Elapsed time: {elapsed_time_map} s")

            # # Get feature 3D coordinates    
            # kpts_map_3d = backproject_ortho_points(kpts_map, map_depth, map_px_resolution, R=R_map_wc, T=t_map_wc) # N x 3

            # keep copies
            kpts_map_all = kpts_map.copy()
            descriptors_map_all = descriptors_map.copy()
            # kpts_map_3d_all = kpts_map_3d.copy()
        
        # verify map windows visually
        #for i in map_windows.keys():
        #    cv2.imwrite(str(i)+'.png', map_windows[i][0])

        for obsv_name in test_data.keys():
                       
            print("\nMAP:", map_name)
            print("OBSV:", obsv_name)
            # Get the observations data dictionary
            obsv_dict = test_data[obsv_name]
                    
            # Get save directory filename:
            sun_comb_name = f"map_{map_dict['elev']}_{map_dict['azim']}_obsv_{obsv_dict['elev']}_{obsv_dict['azim']}"
            if args.pose_prior:
                pose_prior_str = f"pose_uncertainty_{uncertainty_mode}"
            else:
                pose_prior_str = "wo_pose_prior"
            if args.method == "loftr":
                method_str = f"{method_name}/{args.loftr_model_type}"
            elif args.method == "sift":
                method_str = f"{method_name}"
            results_dir = os.path.join(args.dest_dir,
                                os.path.join(sun_comb_name, f"{pose_prior_str}/{method_str}" ) )
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
             

            # comb_save_dir = os.path.join(args.dest_dir, f"map_{map_dict['elev']}_{map_dict['azim']}_obsv_{obsv_dict['elev']}_{obsv_dict['azim']}_{args.pairs_dir_id}")
            # if not os.path.exists(comb_save_dir):
            #     os.makedirs(comb_save_dir)


            # Get query camera params
            query_cx, query_cy = obsv_dict['cx'], obsv_dict['cy']
            query_fx, query_fy = obsv_dict['fx'], obsv_dict['fy']
            query_img_height, query_img_width = obsv_dict['img_height'], obsv_dict['img_width']
            print(f"query img height, width: {(query_img_height, query_img_width)}")
            query_clip_start = obsv_dict['clip_start']
            query_intr = (query_fx, query_fy, query_cx, query_cy)
            camera_matrix = np.array([[obsv_dict['fx'], 0, obsv_dict['cx']],
                                  [0, obsv_dict['fy'], obsv_dict['cy']],
                                  [0, 0, 1]],np.float32)

            #### Generate matches of map to queries
            per_query_res = []
            location_err_acc = []
            nadir_acc = []
            altitude_acc = []
            nr_pnp_fails = 0
            elapsed_time = []
            # Match accuracy
            err_1to0_list = []
            err_0to1_list = []
            err_1to0_inliers_list = []
            err_0to1_inliers_list = []
            conf_match_list = []
            conf_inliers_list = []
            n_matches_list = []
            correct_matches_acc = []
            correct_matches_acc_inliers = []
            n_inliers_list = []
            #query_inds = np.asarray([2], dtype=int)
            #rgb_image_path_list = np.asarray(rgb_image_path_list)[query_inds]

            if (args.n==0) or (args.n > len(obsv_dict['rgb_image_path_list'])):
                n_queries = len(obsv_dict['rgb_image_path_list'])
            else:
                n_queries = args.n


            for query_id in range(n_queries): #query_inds: #range(len(rgb_image_path_list)): #range(n_queries):
                
                altitude = obsv_dict['altitude_queries'][query_id]
                
                print("\nTesting query", query_id)
                
                # Load queries  
                query_depth = np.load(obsv_dict['depth_path_list'][query_id])
                img_bgr = cv2.imread(obsv_dict['rgb_image_path_list'][query_id], cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                query_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


                # Get gt pose of the query image
                yaw, pitch, roll = obsv_dict['ypr_queries'][query_id] # radians
                rot_z, rot_x, rot_y = yaw, pitch, roll
                R_query_wc = np.matmul(Eul312toSO3(rot_x, rot_y, rot_z), XtoSO3(np.pi)) # the convention in wc is the other way round
                R_query_cw = R_query_wc.T
                t_query_wc = np.transpose(np.array(obsv_dict['locs_queries'][query_id])) # Column vector
                t_query_wc = t_query_wc.reshape((3, 1))
                t_query_cw = np.dot(R_query_cw, -t_query_wc)                


                # Get nadir angle
                nadir_acc.append(np.arccos(np.dot(R_query_wc, np.array([[0],[0],[-1]]))[-1,0])*180/np.pi)
                altitude_acc.append(altitude)
                
                
                if args.pose_prior: # Search map windows based on pose prior
                    # Get noisy pose: Perturb pose
                    t_query_noisy_wc, R_query_noisy_wc, yaw_noisy, roll_noisy, pitch_noisy = perturb_pose(t_query_wc, yaw, roll, pitch, pose_uncertainty)

                    # Select edge pixels from the test image. Point 0 is the center
                    points2D = np.asarray([[int(query_cx),int(query_cy)], [1,1], [query_img_width-2, 1], [1,query_img_height-2], [query_img_width-2,query_img_height-2]])
                    
                    center_depth = query_depth[int(query_cy), int(query_cx)] if args.only_altitude else None # + query_clip_start # Simulated altimeter reading (this should be the same as "altitude")
                    points2D_depth_noisy = generate_noisy_depths(points2D, query_depth, pose_uncertainty, args.only_altitude, center_depth)                   

                    # Project these points from the image on the map
                    # ...using the gt pose
                    # Get non-noisy depth values
                    points2D_depth = [query_depth[points2D[k, 1], points2D[k, 0]] for k in range(points2D.shape[0])]
                    map_points2D = project_and_print(points2D, points2D_depth, query_intr, R_query_wc, t_query_wc, map_px_resolution, map_cx, map_cy, R_map_cw, t_map_cw)            
                    # ...using the noisy pose
                    map_points2D_noisy = project_and_print(points2D, points2D_depth_noisy, query_intr, R_query_noisy_wc, t_query_noisy_wc, map_px_resolution, map_cx, map_cy, R_map_cw, t_map_cw)       

                    # # Uncomment to show gt and noisy correspondences
                    # corresp_save_dir = os.path.join(results_dir, "projections")
                    # plot_correspondences(im1=query_img_gray, 
                    #                         im2=map_gray, 
                    #                         points1=points2D,
                    #                         points2=map_points2D,
                    #                         points_noisy=map_points2D_noisy,
                    #                         query_id=query_id,
                    #                         save_dir=corresp_save_dir)
                    
                    map_search_box = get_map_search_window(map_img_height, map_img_width, pose_uncertainty, query_intr, map_px_resolution, points2D, map_points2D_noisy, 
                                                    points2D_depth_noisy, args.only_altitude, noisy_pose=(yaw_noisy, roll_noisy, pitch_noisy, R_query_noisy_wc), R_ortho2world=R_map_cw)
                    search_r_start, search_c_start, search_r_end, search_c_end = map_search_box
                    
                else: # Select a map crop box of 1 km x 1km (without pose prior) centered on the map-projection of the observation image center pixel
                    crop_size = 1000 # m
                    img_center2D = np.asarray([[int(query_cx),int(query_cy)]])
                    img_center2D_depth = np.array([query_depth[int(query_cy), int(query_cx)]])
                    img_center3D = backproject_persp_points(img_center2D, img_center2D_depth , query_intr, R=R_query_wc, T=t_query_wc)
                    map_point2D  = project_on_ortho(img_center3D , map_px_resolution, cx=map_cx, cy=map_cy, R=R_map_cw, T=t_map_cw)
                    U_map, V_map = map_point2D[0,0], map_point2D[0,1] # float
                    print("Observation image center projected on map:\n", U_map, V_map)
                    search_c_start = max(0, int(V_map - 0.5*crop_size/map_px_resolution)) # left
                    search_c_end = min(map_img_height-1, int(V_map + 0.5*crop_size/map_px_resolution)) # right
                    search_r_start = max(0, int(U_map - 0.5*crop_size/map_px_resolution)) # top
                    search_r_end = min(map_img_width-1, int(U_map + 0.5*crop_size/map_px_resolution)) # bottom
                    map_search_box = [search_r_start, search_c_start, search_r_end, search_c_end]

                # # Uncomment to show query and map with selected window
                # save_dir = os.path.join(args.dest_dir, "loftr_mbl_pose_prior")
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)
                # plot_map_box(query_img_gray, map_gray, box=map_search_box, query_id=query_id, save_dir=save_dir)


                # query_ortho_size = (query_img_width/query_fx)*altitude
                # print(f"query altitude: {altitude}")
                # print(f"query ortho size: {query_ortho_size} m")
                # print(f"query resolution: {query_ortho_size/query_img_width} m / pixel")
                # print(f"map window size: (width, height):{(map_win_width, map_win_height)}")
                # print(f"map window overlap: {map_win_overlap}")
                # print(f"map resolution: {map_dict['px_resolution']} m / pixel")
                print(f"map search box: {map_search_box}")
                
                if args.method == 'loftr':

                    # Sample map windows from map search area
                    if args.loftr_model_type == 'geo_ctx':
                        map_win_width  = int(0.75*(search_c_end - search_c_start))
                        map_win_height = int(0.75*(search_r_end - search_r_start))
                        print(f"map window size: (width, height):{(map_win_width, map_win_height)}")
                    else:
                        map_win_width  = config['WIN_SIZE'] # 1024
                        map_win_height = int((query_img_height / query_img_width)*map_win_width)
                    map_win_overlap = int(0.12*map_win_width)
                    map_windows = get_map_window_images_and_depth_w_box(map_img, map_depth, map_search_box, map_clip_start, win_width=map_win_width, win_height=map_win_height, win_overlap=map_win_overlap, depth_thresh=args.depth_thresh, shadow_thresh=args.shadow_thresh)
                     # Query img and depth resizing
                    query_img_in, _, query_scale = resize_img_gray(query_img.copy(), new_size, df=args.df, padding=args.img_padding)
                    query_depth, _ = pad_bottom_right(query_depth, config['WIN_SIZE'])
                    query_depth    = torch.from_numpy(query_depth).float()

                    # Defining matched keypoints lists
                    query_kpts = []
                    map_kpts = []
                    conf_kpts = []
                    win_kpts = []

                    start_t = time.time()
                    
                    # Run query through map windows
                    for i in range(len(map_windows.keys())):
                        print(f"\nWindow id {i}")
                        map_win_img   = map_windows[i][0]
                        map_win_depth = map_windows[i][1]
                        # valid_depth_mask = map_windows[i][2]
                        r_start, c_start, r_end, c_end = map_windows[i][2:6]
                        map_win_box = [c_start, r_start, c_end, r_end]
                        print(f"map window shape: img={map_win_img.shape}, depth={map_win_depth.shape}")                                 
                        # print(f"depth mask shape: {valid_depth_mask.shape}")          
                        print(f"map window left corner coords.: r_start={r_start}, c_start={c_start}")
                        print("Query id", query_id, "shape:", query_img.shape)
                        
                        # Prepare map window depth as input to LoFTR geo 
                        if args.loftr_model_type == "geo_shadow" or args.loftr_model_type == "geo" or args.loftr_model_type == "geo_contr_shadow" or args.loftr_model_type == "geo_ctx":
                            map_win_depth_in, _, _ = resize_img_gray(np.copy(map_win_depth), new_size, df=args.df, padding=args.img_padding) # resize based on width=640
                            image1_depth_in = torch.from_numpy(map_win_depth_in).float()[None][None] / np.amax(map_win_depth_in) 
                            image1_depth_in = image1_depth_in.to(DEVICE)

                        # Pad map window depth:
                        # you can't scale map_win_depth and use it for back projection.
                        # But if query and map windows are padded, you need to pad the query and map depth as well                     
                        map_win_depth, _ = pad_bottom_right(map_win_depth, config['WIN_SIZE'])
                        map_win_depth    = torch.from_numpy(map_win_depth).float()

                        # Resize map window
                        map_win_img_in, _, map_scale = resize_img_gray(map_win_img.copy(), new_size, df=args.df, padding=args.img_padding)

                        # Pre-process LoFTR inputs
                        img0 = torch.from_numpy(query_img_in).float()[None][None] / 255. # (h, w) -> (1, h, w) and normalized
                        img1 = torch.from_numpy(map_win_img_in).float()[None][None] / 255. # (h, w) -> (1, h, w) and normalized
                        img0 = img0.to(DEVICE)
                        img1 = img1.to(DEVICE)
                        scale0 = query_scale.unsqueeze(0).to(DEVICE)
                        scale1 = map_scale.unsqueeze(0).to(DEVICE)

                        # Create input batch dictionary
                        batch = {
                                'image0': img0,           # (torch.Tensor): (1, H, W)
                                'image1': img1,         # (torch.Tensor): (1, H, W) 
                                'scale0': scale0,
                                'scale1': scale1,
                                }

                        # Process the depth
                        if args.loftr_model_type == "geo" or args.loftr_model_type == "geo_shadow" or args.loftr_model_type == "geo_contr_shadow" or args.loftr_model_type == "geo_ctx":
                            batch['image1_depth_in'] = image1_depth_in

                        # Run LoFTR
                        with torch.no_grad():
                            matcher(batch)
                            mkpts0 = batch['mkpts0_f'].cpu().numpy()
                            mkpts1 = batch['mkpts1_f'].cpu().numpy()
                            mconf = batch['mconf'].cpu().numpy()

                        print(f"mkpts0 shape: {mkpts0.shape}")
                        print(f"mkpts1 shape: {mkpts1.shape}")

                        # Get top k matches based on confidence, for each map window
                        if config['TOP_K'] > 0:
                            mkpts0, mkpts1, mconf, _ = get_top_k(mkpts0, mkpts1, mconf, top_k=config['TOP_K'])


                        # Remove any matches at the edges of the images
                        mkpts0, mkpts1, mconf = check_bounds(mkpts0, mkpts1, mconf, img0_size=query_img.shape, img1_size=map_win_img.shape)

                        # # Ucomment to visualize matches with each window
                        # save_dir = os.path.join(results_dir, f"matches")
                        # if not os.path.exists(save_dir):
                        #     os.makedirs(save_dir)
                        # filename = str(query_id)+'_matches_to_map_win_'+str(i)+f"_{match_params_str}{var_str}.png"
                        # show_matches(batch, top_k=config['TOP_K'] if config['TOP_K'] > 0 else 10, path=os.path.join(save_dir, filename))

                        ## When scale_0 and scale_1 are provided, LoFTR retruns the matches in the original scales of the input images
                        ## so the code below is not needed
                        # # Transform matches back to original map and query coordinates
                        # # By default, LoFTR resizes images to 640x480 before matching
                        # mkpts0[:,0] = (mkpts0[:,0] / input_img_dim[1]) * query_img_gray.shape[1]
                        # mkpts0[:,1] = (mkpts0[:,1] / input_img_dim[0]) * query_img_gray.shape[0]
                        # mkpts1[:,0] = (mkpts1[:,0] / input_img_dim[1]) * map_win.shape[1]
                        # mkpts1[:,1] = (mkpts1[:,1] / input_img_dim[0]) * map_win.shape[0]                    
                        # # map_win_kpts.extend(mkpts1) #  Keypoints in map window coordinates

                    
                        mkpts1_win = mkpts1.copy() # save matches in window coordinates for later

                        # translate the window coordinate to the map coordinate
                        # print(f"before correction - mkpts1: {mkpts1[:4,:]}")
                        # print(f"(r_start, c_start) = {(r_start, c_start)}")
                        mkpts1[:,0] += r_start
                        mkpts1[:,1] += c_start
                        

                        # # Ucomment to visualize matches with each window over the map
                        # color = cm.jet(mconf, alpha=0.7)
                        # #text = ['LoFTR','Matches: {}'.format(len(mkpts0))]
                        # text = ['','']
                        # save_dir = os.path.join(results_dir, f"matches")
                        # if not os.path.exists(save_dir):
                        #     os.makedirs(save_dir)
                        # filename = str(query_id)+'_matches_to_map_win_'+str(i)+f"_map_{match_params_str}{var_str}.png"
                        # fig = make_matching_figure(query_img, map_img, mkpts0, mkpts1, color, mkpts0, mkpts1, text, dpi=150, path=os.path.join(save_dir, filename))

                        query_kpts.extend(mkpts0)
                        map_kpts.extend(mkpts1)
                        conf_kpts.extend(mconf)
                        win_kpts.extend(mkpts1_win)

                    elapsed_time.append(time.time() - start_t)


                    query_kpts = np.asarray(query_kpts)
                    map_kpts = np.asarray(map_kpts)
                    conf_kpts = np.asarray(conf_kpts)
                    win_kpts = np.asarray(win_kpts)
                
                    # Get matches based on confidence thresh
                    match_query_kpts, match_map_kpts, match_conf = filter_with_conf(conf_kpts, query_kpts, map_kpts, conf_thresh=config['CONF_THRESH'])
                    _, match_win_kpts, _ = filter_with_conf(conf_kpts, query_kpts, win_kpts, conf_thresh=config['CONF_THRESH'])

                elif args.method == 'sift':
                    ### Keep kpts_map_3d, descriptors_map where kpts_map fall within the map_window_box
                    [left, top, right, bottom] = map_search_box
                    kpts_map = []
                    descriptors_map = []
                    # kpts_map_3d = []

                    for kpt_id in range(len(kpts_map_all)):
                        kpt_x = kpts_map_all[kpt_id].pt[0]
                        kpt_y = kpts_map_all[kpt_id].pt[1] 
                        if (kpt_x > left) and (kpt_x < right) and (kpt_y > top) and (kpt_y < bottom):
                            kpts_map.append(kpts_map_all[kpt_id])
                            descriptors_map.append(descriptors_map_all[kpt_id])
                            # kpts_map_3d.append(kpts_map_3d_all[kpt_id])
                    print("Kept", len(kpts_map), "out of", len(kpts_map_all), " map keypoints!")


                    start_t = time.time()
                    # Feature extraction
                    # if args.feature_type=='SIFT':
                        #Feature = cv2.xfeatures2d.SIFT_create(nfeatures=config['MAX_FEATS_PER_QUERY_IMAGE'])
                    Feature = cv2.SIFT_create(nfeatures=config['MAX_FEATS_PER_QUERY_IMAGE'])

                    kpts_test, descriptors_test = Feature.detectAndCompute(query_img, None)
                    #print(len(kpts_test))
                    #print(kpts_test[0].pt)
                    
                    print('Number of image query keypoints:',len(kpts_test))
                    # print("Data type of descriptors_test:", descriptors_test.dtype)
                    # print("Data type of descriptors_map:", np.array(descriptors_map).dtype)
                    if (descriptors_test is not None) and (descriptors_map is not None):
                        if descriptors_test.dtype == np.array(descriptors_map).dtype and len(descriptors_test)>1 and len(descriptors_map)>1:

                            # Exhaustive Matching
                            matches_knn = matcher.knnMatch(descriptors_test, np.array(descriptors_map), 2)
                            ratio_tests = [m.distance / n.distance for m, n in matches_knn]
                            # Sort matches by ratio test
                            ratio_test_ids_sorted = np.argsort(ratio_tests)
                            ratio_tests_sorted = [ratio_tests[id] for id in ratio_test_ids_sorted]
                            matches = [matches_knn[id] for id in ratio_test_ids_sorted]

                            nr_matches = len(matches)

                            elapsed_time.append(time.time() - start_t)

                            # Get matching results
                            kpts_map_matched = []
                            # kpts_map_3d_matched = []
                            kpts_query_matched = []
                            mconf_matched = []

                            for match_id in range(nr_matches):
                                m, n = matches[match_id]

                                if m.distance < config['RATIO_THRESH']*n.distance:
                                    # Get match indices
                                    kpts_map_idx = m.trainIdx
                                    kpts_query_idx = m.queryIdx

                                    # Get image coords
                                    kpt_map = kpts_map[kpts_map_idx].pt
                                    kpt_query = kpts_test[kpts_query_idx].pt
                                    # kpt_map_3d = np.array([kpts_map_3d[kpts_map_idx]]).transpose()

                                    # Used for PNP
                                    if match_id <= config['PNP_NR_MATCHES']:
                                        kpts_map_matched.append(kpt_map)
                                        kpts_query_matched.append(kpt_query)
                                        mconf_matched.append((1/m.distance)*100)
                                        # invert the ratio and normalize from 0 to 1
                                        # kpts_map_3d_matched.append(kpts_map_3d[kpts_map_idx])

                            match_map_kpts   = np.array(kpts_map_matched)
                            match_query_kpts = np.array(kpts_query_matched)
                            match_conf = np.array(mconf_matched)
                        else:
                            # Get matching results
                            kpts_map_matched = []
                            # kpts_map_3d_matched = []
                            kpts_query_matched = []
                            mconf_matched = []
                            match_map_kpts   = np.array(kpts_map_matched)
                            match_query_kpts = np.array(kpts_query_matched)
                            match_conf = np.array(mconf_matched)
                    else:
                        # Get matching results
                        kpts_map_matched = []
                        # kpts_map_3d_matched = []
                        kpts_query_matched = []
                        mconf_matched = []
                        match_map_kpts   = np.array(kpts_map_matched)
                        match_query_kpts = np.array(kpts_query_matched)
                        match_conf = np.array(mconf_matched)
                


                
                # Print matches info per query
                print(f"\nQuery id: {query_id} - All matches")
                print(f"match_query_kpts: {match_query_kpts.shape}")
                print(f"match_map_kpts: {match_map_kpts.shape}")

                if match_query_kpts.size > 2 and match_map_kpts.size > 2: # Check if both match_query_kpts and match_map_kpts have at least 6 points

                    if query_id < 100: # Show up to 100 matches figures
                        # Ucomment to visualize matches with the map crop
                        save_dir = os.path.join(results_dir, f"matches")
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        filename = str(query_id)+f"_matches_to_map_box_{match_params_str}{var_str}.png"
                        show_matches_on_map(query_img, map_img, match_query_kpts, match_map_kpts, match_conf, top_k=config['TOP_K'], map_box=map_search_box,  path=os.path.join(save_dir, filename))
                    
                        # # Ucomment to visualize matches with the map
                        # save_dir = os.path.join(results_dir, f"matches")
                        # if not os.path.exists(save_dir):
                        #     os.makedirs(save_dir)
                        # filename = str(query_id)+f"_matches_to_map_{match_params_str}{var_str}.png"
                        # show_matches_on_map(query_img, map_img, match_query_kpts, match_map_kpts, match_conf, top_k=config['TOP_K'], map_box=None,  path=os.path.join(save_dir, filename))
                    
                    
                    #   Unproject map points into map camera coordinate frame
                    match_map_kpts_3d = backproject_ortho_points(match_map_kpts, map_depth, map_px_resolution, R=R_map_wc, T=t_map_wc) # N x 3
                    
                    # Reprojection from map to query
                    #  Transform map points to camera coordinate frame (local3D) of test image, and prohect on the image frame
                    match_query_kpts_reproj = project_on_persp(match_map_kpts_3d, query_intr, R=R_query_cw, T=t_query_cw)

                    # Reprojection from query to map  
                    query_kpts_depth = []
                    for k in range(match_query_kpts.shape[0]):
                        # print((match_query_kpts.round().astype(np.int64)[k,1], match_query_kpts.round().astype(np.int64)[k,0] ))
                        query_kpts_depth.append(float(query_depth[match_query_kpts.round().astype(np.int64)[k,1], match_query_kpts.round().astype(np.int64)[k,0] ] ))
                    
                    #   Unproject query points into query camera coordinate frame
                    query_kpts_3d = backproject_persp_points(match_query_kpts, query_kpts_depth, query_intr, R=R_query_wc, T=t_query_wc)
                    #   Transform query points to camera coordinate frame (local3D) of map image, and project on the map image frame
                    match_map_kpts_reproj = project_on_ortho(query_kpts_3d, map_px_resolution, cx=map_cx, cy=map_cy, R=R_map_cw, T=t_map_cw)
                    
                    # Compute  and store reprojection errors
                    n_matches = match_query_kpts.shape[0]
                    err_1to0 = np.linalg.norm(match_query_kpts - match_query_kpts_reproj, axis=1)
                    err_0to1 = np.linalg.norm(match_map_kpts   - match_map_kpts_reproj,   axis=1)
                    err_1to0_list.extend(err_1to0)
                    err_0to1_list.extend(err_0to1)
                    conf_match_list.extend(match_conf)
                    n_matches_list.append(n_matches)

                    # Compute quantities per query
                    #   Reproj. errors
                    err = (err_1to0 + err_0to1)/2 
                    mErr_1to0 = np.mean(err_1to0)
                    mErr_0to1 = np.mean(err_0to1)
                    mErr = np.mean(err)              
                    #   Percentage of matches under reprojection error of 1,2,5 
                    err_1to0_per_1 = np.sum(np.where(err_1to0 < 1.0, 1.0, 0.0)) / err_1to0.shape[0]
                    err_0to1_per_1 = np.sum(np.where(err_0to1 < 1.0, 1.0, 0.0)) / err_1to0.shape[0]
                    err_per_1 = np.sum(np.where(err < 1.0, 1.0, 0.0)) / err.shape[0]

                    err_1to0_per_2 = np.sum(np.where(err_1to0 < 2.0, 1.0, 0.0)) / err_1to0.shape[0]
                    err_0to1_per_2 = np.sum(np.where(err_0to1 < 2.0, 1.0, 0.0)) / err_1to0.shape[0]
                    err_per_2 = np.sum(np.where(err < 2.0, 1.0, 0.0)) / err.shape[0]

                    err_1to0_per_5 = np.sum(np.where(err_1to0 < 5.0, 1.0, 0.0)) / err_1to0.shape[0]
                    err_0to1_per_5 = np.sum(np.where(err_0to1 < 5.0, 1.0, 0.0)) / err_1to0.shape[0]
                    err_per_5 = np.sum(np.where(err < 5.0, 1.0, 0.0)) / err.shape[0]

                    correct_matches_acc.append(100.0*err_per_1)
                    print('Number of correct matches: ',np.sum(np.array(err)<1),'/',len(err))

                    # Use top matches for PnP-RANSAC
                    # top_inds = np.argpartition(match_conf, -config['PNP_NR_MATCHES'])[-config['PNP_NR_MATCHES']:]
                    # match_map_kpts_3d = match_map_kpts_3d[top_inds,:]
                    # match_query_kpts = match_query_kpts[top_inds,:]
                    if match_query_kpts.shape[0] < 4: # it is possible although highly improbable that we will get 0 matches
                        success = False
                    else:
                        # Non-linear solution by default. Transformation is T_cw
                        success, r, t, inliers =\
                            cv2.solvePnPRansac(match_map_kpts_3d, match_query_kpts, camera_matrix, np.zeros((4,1),np.float32), 
                                                            iterationsCount = config['RANSAC_ITERS'], reprojectionError=config['PNP_REPROJ_ERR'])

                    if success:
                        R_query_camera_from_world, _ = cv2.Rodrigues(r)
                        R_hat = R_query_camera_from_world.transpose() # camera to world
                        t_hat = - np.dot(R_hat,t).transpose()
                        print('\nRANSAC inliers, ',len(inliers),'/', match_query_kpts.shape[0])
                        print('Location error:', np.linalg.norm(t_hat - t_query_wc.transpose()))
                        location_err_acc.append(np.linalg.norm(t_hat - t_query_wc.transpose()))
                    else:
                        location_err_acc.append(10000000)
                        nr_pnp_fails += 1
                        inliers = []
                        print("\nRANSAC PNP FAILED")

                    
                    # Compute reprojection errors on inliers
                    
                    if success:

                        inliers = inliers.squeeze()
                        n_inliers_list.extend(inliers)
                        inlier_query_kpts = match_query_kpts[inliers]
                        inlier_map_kpts   = match_map_kpts[inliers]
                        inlier_query_kpts_reproj = match_query_kpts_reproj[inliers]
                        inlier_map_kpts_reproj = match_map_kpts_reproj[inliers]
                        inlier_conf = match_conf[inliers]

                        print(f"\nQuery id: {query_id} - Inliers")
                        print(f"inlier_query_kpts: {inlier_query_kpts.shape}")
                        print(f"inlier_map_kpts: {inlier_map_kpts.shape}")

                        if query_id < 100: # Show up to 100 matches figures
                            # Ucomment to visualize matches with the map
                            save_dir = os.path.join(results_dir, f"matches")
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            filename = str(query_id)+f"_inlier_matches_to_map_{match_params_str}{var_str}.png"
                            show_matches_on_map(query_img, map_img, inlier_query_kpts, inlier_map_kpts, inlier_conf, top_k=config['TOP_K'], map_box=map_search_box, path=os.path.join(save_dir, filename))
                            
                            # # Ucomment to visualize matches with the map
                            # save_dir = os.path.join(results_dir, f"matches")
                            # if not os.path.exists(save_dir):
                            #     os.makedirs(save_dir)
                            # filename = str(query_id)+f"_inlier_matches_to_map_{match_params_str}{var_str}.png"
                            # show_matches_on_map(query_img, map_img, inlier_query_kpts, inlier_map_kpts, inlier_conf, top_k=config['TOP_K'], map_box=None, path=os.path.join(save_dir, filename))
                        

                        # Compute  and store reprojection errors
                        n_inliers_matches = len(inliers)
                        err_1to0_inliers = np.linalg.norm(inlier_query_kpts - inlier_query_kpts_reproj, axis=1)
                        err_0to1_inliers = np.linalg.norm(inlier_map_kpts   - inlier_map_kpts_reproj,   axis=1)
                        err_1to0_inliers_list.extend(err_1to0_inliers)
                        err_0to1_inliers_list.extend(err_0to1_inliers)
                        conf_inliers_list.extend(inlier_conf)
                        print(f"err_0to1_inliers.shape: {err_0to1_inliers.shape}")
                        print(f"err_1to0_inliers.shape: {err_1to0_inliers.shape}")
                        print(f"inlier_conf.shape: {inlier_conf.shape}")
                        

                        # Compute quantities per query
                        #   Reproj. errors
                        err_inliers = (err_1to0_inliers + err_0to1_inliers)/2 
                        mErr_1to0_inliers = np.mean(err_1to0_inliers)
                        mErr_0to1_inliers = np.mean(err_0to1_inliers)
                        mErr_inliers = np.mean(err_inliers)              
                        #   Percentage of matches under reprojection error of 1,2,5 
                        err_1to0_inliers_per_1 = np.sum(np.where(err_1to0_inliers < 1.0, 1.0, 0.0)) / err_1to0_inliers.shape[0]
                        err_0to1_inliers_per_1 = np.sum(np.where(err_0to1_inliers < 1.0, 1.0, 0.0)) / err_0to1_inliers.shape[0]
                        err_inliers_per_1 = np.sum(np.where(err_inliers < 1.0, 1.0, 0.0)) / err_inliers.shape[0]

                        err_1to0_inliers_per_2 = np.sum(np.where(err_1to0_inliers < 2.0, 1.0, 0.0)) / err_1to0_inliers.shape[0]
                        err_0to1_inliers_per_2 = np.sum(np.where(err_0to1_inliers < 2.0, 1.0, 0.0)) / err_0to1_inliers.shape[0]
                        err_inliers_per_2 = np.sum(np.where(err_inliers < 2.0, 1.0, 0.0)) / err_inliers.shape[0]

                        err_1to0_inliers_per_5 = np.sum(np.where(err_1to0_inliers < 5.0, 1.0, 0.0)) / err_1to0_inliers.shape[0]
                        err_0to1_inliers_per_5 = np.sum(np.where(err_0to1_inliers < 5.0, 1.0, 0.0)) / err_0to1_inliers.shape[0]
                        err_inliers_per_5 = np.sum(np.where(err_inliers < 5.0, 1.0, 0.0)) / err_inliers.shape[0]

                        correct_matches_acc_inliers.append(100.0*err_inliers_per_1)
                        print('Number of correct inlier matches : ',np.sum(np.array(err_inliers)<1),'/',len(err_inliers))

                        # Store stats per query image
                        per_query_res.append( {"Total matches": n_matches,
                                                "altitude": float(altitude),
                                                "t_query_wc":t_query_wc.astype(np.float64),
                                                "R_query_wc":R_query_wc.astype(np.float64),
                                                "err_1to0": err_1to0.astype(np.float64),
                                                "err_0to1": err_0to1.astype(np.float64),
                                                "match_conf": match_conf.astype(np.float64),
                                                "mean reprojection error": float(mErr), 
                                                "Correct matches [<1px]": int(np.sum(np.array(err)<1)),
                                                "Correct matches [<1px] per": float(100*err_per_1),
                                                "Correct matches [<2px]": int(np.sum(np.array(err)<2)),
                                                "Correct matches [<2px] per": float(100*err_per_2),
                                                "Correct matches [<5px]": int(np.sum(np.array(err)<5)),
                                                "Correct matches [<5px] per": float(100*err_per_5),
                                                "RANSAC Inliers": len(inliers),
                                                "Location error": float(location_err_acc[query_id]),
                                                "err_1to0_inliers": err_1to0_inliers.astype(np.float64),
                                                "err_0to1_inliers": err_0to1_inliers.astype(np.float64),
                                                "inlier_conf": inlier_conf.astype(np.float64),
                                                "Correct inlier matches [<1px]": int(np.sum(np.array(err_inliers)<1)),
                                                "Correct inlier matches [<1px] per": float(100*err_inliers_per_1),
                                                "Correct inlier matches [<2px]": int(np.sum(np.array(err_inliers)<2)),
                                                "Correct inlier matches [<2px] per": float(100*err_inliers_per_2),
                                                "Correct inlier matches [<5px]": int(np.sum(np.array(err_inliers)<5)),
                                                "Correct inlier matches [<5px] per": float(100*err_inliers_per_5),
                                                "mean reprojection error (inliers)": float(mErr_inliers), 
                                                })
                    else:
                        correct_matches_acc_inliers.append(10000000)
                        print(f"\nQuery id: {query_id} - Fail")
                else:
                    success = False
                    correct_matches_acc.append(10000000)
                    correct_matches_acc_inliers.append(10000000)
                    location_err_acc.append(10000000)
                    nr_pnp_fails += 1
                    per_query_res.append( {"Total matches": 0,
                                        "altitude": float(altitude),
                                        "t_query_wc":t_query_wc.astype(np.float64),
                                        "R_query_wc":R_query_wc.astype(np.float64)})
                    print(f"\nNo matches are found for query id: {query_id} - Fail")

                

            ##############################################
            ########## Result outputs and plots ##########
            ##############################################

            print("Altitude:", altitude_acc)
            elevation_diff = np.max(altitude_acc)-np.min(altitude_acc)
            


            # Cumulative distribution
            bins = np.arange(0.1, 10, 0.1)
            # All matches accuracies
            rep_err_mean = (np.array(err_1to0_list) + np.array(err_0to1_list))/2
            rep_err_1to0_cum = [100*np.sum(np.array(err_1to0_list) < Th) / len(err_1to0_list) for Th in bins]
            rep_err_0to1_cum = [100*np.sum(np.array(err_0to1_list) < Th) / len(err_0to1_list) for Th in bins]
            rep_err_mean_cum = [100*np.sum(rep_err_mean < Th) / len(rep_err_mean) for Th in bins]
            # Inlier matches accuracies
            rep_err_inliers_mean = (np.array(err_1to0_inliers_list) + np.array(err_0to1_inliers_list))/2
            rep_err_1to0_inliers_cum = [100*np.sum(np.array(err_1to0_inliers_list) < Th) / len(err_1to0_inliers_list) for Th in bins]
            rep_err_0to1_inliers_cum = [100*np.sum(np.array(err_0to1_inliers_list) < Th) / len(err_0to1_inliers_list) for Th in bins]
            rep_err_mean_inliers_cum = [100*np.sum(rep_err_inliers_mean < Th) / len(rep_err_inliers_mean) for Th in bins]
            # Location accuracies
            loc_err_cum = [100*np.sum(np.array(location_err_acc) < Th)/ len(location_err_acc) for Th in bins]
            
            # Print results summary
            print("\nRESULTS")
            # print('----------- matches----------------------')
            # for i in range(100):
            #     print(f"\nmatch #{i}")
            #     print(f"map to query reproj. error: {err_1to0_list[i]}")
            #     print(f"query to map reproj. error: {err_0to1_list[i]}")
            #     print(f"mean reproj. error: {rep_err_mean[i]}")
            
            print('--------------Location Accuracy--------------------------')
            print('Queries w/ pose error < 2 m: {:.3f}%'.format(100.0*np.sum(np.array(location_err_acc)<2)/len(location_err_acc)))
            print('Queries w/ pose error < 1 m: {:.3f}%'.format(100.0*np.sum(np.array(location_err_acc)<1)/len(location_err_acc)))
            print('Queries w/ pose error < 0.5 m: {:.3f}%'.format(100.0*np.sum(np.array(location_err_acc)<0.5)/len(location_err_acc)))
            print('Nr of PNP fails:',nr_pnp_fails)
            print('--------------All matches accuracy--------------------------')
            print(f"N.o. matches: {np.sum(np.array(n_matches_list))}")
            print('Matches [%] w/ map-to-query reproj. err. <1 px:  {:.1f}% '.format(100.0*float(np.sum(np.array(err_1to0_list)<1))/len(err_1to0_list)))
            print('Matches [%] w/ query-to-map reproj. err. <1 px:  {:.1f}% '.format(100.0*float(np.sum(np.array(err_0to1_list)<1))/len(err_0to1_list)))
            print('Matches [%] w/ mean reproj. err. <1 px:  {:.1f}% '.format(100.0*float(np.sum(rep_err_mean<1))/len(rep_err_mean)))
            print('--------------Inlier matches accuracy--------------------------')
            print(f"N.o. matches: {np.sum(np.array(n_inliers_list))}")
            print(f"np.sum(np.array(err_1to0_inliers_list)<1): {np.sum(np.array(err_1to0_inliers_list)<1)}")
            print(f"len(err_1to0_inliers_list): {len(err_1to0_inliers_list)}")
            print('Matches [%] w/ map-to-query reproj. err. <1 px:  {:.1f}% '.format(100.0*float(np.sum(np.array(err_1to0_inliers_list)<1))/len(err_1to0_inliers_list)))
            print('Matches [%] w/ query-to-map reproj. err. <1 px:  {:.1f}% '.format(100.0*float(np.sum(np.array(err_0to1_inliers_list)<1))/len(err_0to1_inliers_list)))
            print('Matches [%] w/ mean reproj. err. <1 px:  {:.1f}% '.format(100.0*float(np.sum(rep_err_inliers_mean<1))/len(rep_err_inliers_mean)))
            #print('Repeatability [<1 px]:  {:.1f}% '.format(np.mean(repeatability_acc)))
            
            #print('Reprojection error bias:', reprojection_bias.transpose()/(np.sum(heat_map_reprojection)))
            print('----------------------------------------')
            
            path_2_results = os.path.join(results_dir, f"accuracy" )
            #path_2_results = './results/' + os.path.basename(os.path.normpath(args.testset_path))
            if not os.path.isdir(path_2_results):
                #os.mkdir(path_2_results)
                os.makedirs(path_2_results)

            # Dump results summary
            print(f"\nWriting results summary: ...")
            results_file_name = os.path.basename(os.path.normpath(args.test_data_path)) + f"_{match_params_str}_n{n_queries}{var_str}.json"
            results_dict = {'Matches [%] w/ map-to-query reproj. err. <1 px': 100.0*float(np.sum(np.array(err_1to0_list)<1))/len(err_1to0_list),
                            'Matches [%] w/ query-to-map reproj. err. <1 px': 100.0*float(np.sum(np.array(err_0to1_list)<1))/len(err_0to1_list),
                            'Matches [%] w/ mean reproj. err. <1 px': 100.0*float(np.sum(rep_err_mean<1))/len(rep_err_mean),
                            'Inlier matches [%] w/ map-to-query reproj. err. <1 px': 100.0*float(np.sum(np.array(err_1to0_inliers_list)<1))/len(err_1to0_inliers_list),
                            'Inlier matches [%] w/ query-to-map reproj. err. <1 px': 100.0*float(np.sum(np.array(err_0to1_inliers_list)<1))/len(err_0to1_inliers_list),
                            'Inlier matches [%] w/ mean reproj. err. <1 px': 100.0*float(np.sum(rep_err_inliers_mean<1))/len(rep_err_inliers_mean),
                            #'Repeatability[%]': np.mean(repeatability_acc),
                            'Poses[%] w/ error less than 1 m': 100.0*np.sum(np.array(location_err_acc)<1)/len(location_err_acc),
                            'bins': bins.tolist(),
                            'Cumulative map-to-query match accuracy': rep_err_1to0_cum,
                            'Cumulative query-to-map accuracy': rep_err_0to1_cum,
                            'Cumulative match accuracy': rep_err_mean_cum,
                            'Cumulative inliers map-to-query match accuracy': rep_err_1to0_inliers_cum,
                            'Cumulative inliers query-to-map accuracy': rep_err_0to1_inliers_cum,
                            'Cumulative inliers match accuracy': rep_err_mean_inliers_cum,
                            'Cumulative pose accuracy': loc_err_cum
                            }

            # Writing to sample.json
            with open(os.path.join(path_2_results, results_file_name), "w") as outfile:
                json.dump(results_dict, outfile, indent=4)
            print("...complete.")

            # Write per query results
            print(f"\nWriting per query results: ...")
            total_elapsed_time = np.sum(np.asarray(elapsed_time)) #+ elapsed_time_map
            per_query_res.append({
                'Mean Location Error': float(np.mean(np.asarray(location_err_acc))),
                'Nr of PNP Fails': nr_pnp_fails,
                'Mean query time': float(total_elapsed_time / n_queries)
            })
            #per_query_res_filename = os.path.basename(os.path.normpath(args.testset_path)) + '_loftr_per_query' + '.json'
            per_query_res_filename = os.path.basename(os.path.normpath(args.test_data_path)) + f"_per_query_{match_params_str}_n{n_queries}{var_str}.npz"
            np.savez_compressed(os.path.join(path_2_results, per_query_res_filename), per_query_res=per_query_res)
            print("...complete.")


            # Write result arrays for offline visualization
            print(f"\nWriting result arrays for offline visualization: ...")
            filename = os.path.basename(os.path.normpath(args.test_data_path)) + f"_results_{match_params_str}_n{n_queries}{var_str}.npz"
            np.savez_compressed(os.path.join(path_2_results, filename),
                                altitude_acc=altitude_acc,
                                location_err_acc=location_err_acc,
                                nr_pnp_fails=nr_pnp_fails,
                                err_1to0_list=err_1to0_list,
                                err_0to1_list=err_0to1_list,
                                conf_match_list=conf_match_list,
                                rep_err_mean=rep_err_mean,
                                n_matches_list = n_matches_list,
                                n_inliers_list = n_inliers_list,
                                err_1to0_inliers_list=err_1to0_inliers_list,
                                err_0to1_inliers_list=err_0to1_inliers_list,
                                conf_inliers_list=conf_inliers_list,
                                rep_err_inliers_mean=rep_err_inliers_mean,
                                elapsed_time=elapsed_time,
                                #elapsed_time_map=elapsed_time_map,
                                n_queries=n_queries
                                #repeatability_acc=repeatability_acc,
                                )
            print("...complete.")


            f, axs = plt.subplots(2, 2, figsize=(6, 5))
            # Plot match reprojection error
            axs[0,0].plot(bins, rep_err_1to0_cum, label="map-to-query")
            axs[0,0].plot(bins, rep_err_0to1_cum, label="query-to-map")
            axs[0,0].plot(bins, rep_err_mean_cum, label="mean")
            axs[0,0].grid(axis='x', alpha=0.75)
            axs[0,0].grid(axis='y', alpha=0.75)
            axs[0,0].set_xlabel('Reprojection error [px]')
            axs[0,0].set_ylabel('Matches [%]')
            axs[0,0].set_title('Cumulative matches')
            axs[0,0].set_ylim(0, 100)
            axs[0,0].legend()

            # Plot loc error
            axs[1,0].plot(bins, loc_err_cum)
            axs[1,0].grid(axis='x', alpha=0.75)
            axs[1,0].grid(axis='y', alpha=0.75)
            axs[1,0].set_xlabel('Location error [m]')
            axs[1,0].set_ylabel('Queries [%]')
            axs[1,0].set_title('Cumulative location precision')
            axs[1,0].set_ylim(0, 100)

            # Plot matches vs altitude
            try:
                if len(set(altitude_acc)) == 1 or len(set(correct_matches_acc)) == 1:
                    # If all values are the same, use a simpler scatter plot (no KDE)
                    axs[0,1].scatter(altitude_acc, correct_matches_acc, s=50, edgecolor='face', c='r')  # red color for instance
                    axs[0,1].set_title('Correct matches vs altitude (no variance)')
                else:
                    xy = np.vstack([altitude_acc, correct_matches_acc])
                    z = gaussian_kde(xy)(xy)
                    idx = z.argsort()
                    x, y, z = np.array(altitude_acc)[idx], np.array(correct_matches_acc)[idx], z[idx]
                    axs[0,1].scatter(x, y, c=z, s=50, edgecolor='face')
                axs[0,1].grid(axis='x', alpha=0.75)
                axs[0,1].grid(axis='y', alpha=0.75)
                axs[0,1].set_xlabel('Altitude [m]')
                axs[0,1].set_ylabel('Correct matches [%]')
                axs[0,1].set_title('Correct matches vs altitude')
                axs[0,1].set_ylim(0, 100)
            except np.linalg.LinAlgError as e:
                # Catch the specific error and print a message (or log it)
                print(f"Skipping plot due to error: {e}")

            # Plot matches vs nadir
            try:
                # Attempt to perform KDE and plotting
                if len(set(nadir_acc)) == 1 or len(set(correct_matches_acc)) == 1:
                    # If all values are the same, use a simpler scatter plot (no KDE)
                    axs[1,1].scatter(nadir_acc, correct_matches_acc, s=50, edgecolor='face', c='r')  # red color for instance
                    axs[1,1].set_title('Correct matches vs nadir angle (no variance)')
                else:
                    xy = np.vstack([nadir_acc, correct_matches_acc])
                    z = gaussian_kde(xy)(xy)
                    idx = z.argsort()
                    x, y, z = np.array(nadir_acc)[idx], np.array(correct_matches_acc)[idx], z[idx]
                    axs[1,1].scatter(x, y, c=z, s=50, edgecolor='face')
                axs[1,1].grid(axis='x', alpha=0.75)
                axs[1,1].grid(axis='y', alpha=0.75)
                axs[1,1].set_xlabel('Nadir angle [deg]')
                axs[1,1].set_ylabel('Correct matches [%]')
                axs[1,1].set_title('Correct matches vs nadir angle')
                axs[1,1].set_ylim(0, 100)
            except np.linalg.LinAlgError as e:
                # Catch the specific error and print a message (or log it)
                print(f"Skipping plot due to error: {e}")
            plt.tight_layout()

            # Save figure
            fig_file_name = os.path.basename(os.path.normpath(args.test_data_path))+ f"_{match_params_str}_n{n_queries}{var_str}.png"
            f.savefig(os.path.join(path_2_results,fig_file_name))
            plt.close(f) # Close figure


            # Plot for inliers
            f, axs = plt.subplots(2, 2, figsize=(6, 5))
            # Plot match reprojection error
            axs[0,0].plot(bins, rep_err_1to0_inliers_cum, label="map-to-query")
            axs[0,0].plot(bins, rep_err_0to1_inliers_cum, label="query-to-map")
            axs[0,0].plot(bins, rep_err_mean_inliers_cum, label="mean")
            axs[0,0].grid(axis='x', alpha=0.75)
            axs[0,0].grid(axis='y', alpha=0.75)
            axs[0,0].set_xlabel('Reprojection error [px]')
            axs[0,0].set_ylabel('Matches [%]')
            axs[0,0].set_title('Cumulative matches')
            axs[0,0].set_ylim(0, 100)
            axs[0,0].legend()

            # Plot loc error
            axs[1,0].plot(bins, loc_err_cum)
            axs[1,0].grid(axis='x', alpha=0.75)
            axs[1,0].grid(axis='y', alpha=0.75)
            axs[1,0].set_xlabel('Location error [m]')
            axs[1,0].set_ylabel('Queries [%]')
            axs[1,0].set_title('Cumulative location precision')
            axs[1,0].set_ylim(0, 100)

            # Plot matches vs altitude
            try:
                # Attempt to perform KDE and plotting
                if len(set(altitude_acc)) == 1 or len(set(correct_matches_acc_inliers)) == 1:
                    # If all values are the same, use a simpler scatter plot (no KDE)
                    axs[0,1].scatter(altitude_acc, correct_matches_acc_inliers, s=50, edgecolor='face', c='r')  # red color for instance
                    axs[0,1].set_title('Correct matches vs altitude (no variance)')
                else:
                    xy = np.vstack([altitude_acc, correct_matches_acc_inliers])
                    z = gaussian_kde(xy)(xy)
                    idx = z.argsort()
                    x, y, z = np.array(altitude_acc)[idx], np.array(correct_matches_acc_inliers)[idx], z[idx]
                    axs[0,1].scatter(x, y, c=z, s=50, edgecolor='face')
                axs[0,1].grid(axis='x', alpha=0.75)
                axs[0,1].grid(axis='y', alpha=0.75)
                axs[0,1].set_xlabel('Altitude [m]')
                axs[0,1].set_ylabel('Correct matches [%]')
                axs[0,1].set_title('Correct matches vs altitude')
                axs[0,1].set_ylim(0, 100)
            except np.linalg.LinAlgError as e:
                # Catch the specific error and print a message (or log it)
                print(f"Skipping plot due to error: {e}")
            
            # Plot matches vs nadir angle
            try:
               # Attempt to perform KDE and plotting
                if len(set(nadir_acc)) == 1 or len(set(correct_matches_acc_inliers)) == 1:
                    # If all values are the same, use a simpler scatter plot (no KDE)
                    axs[1,1].scatter(nadir_acc, correct_matches_acc_inliers, s=50, edgecolor='face', c='r')  # red color for instance
                    axs[1,1].set_title('Correct matches vs nadir angle (no variance)')
                else:
                    xy = np.vstack([nadir_acc, correct_matches_acc_inliers])
                    z = gaussian_kde(xy)(xy)
                    idx = z.argsort()
                    x, y, z = np.array(nadir_acc)[idx], np.array(correct_matches_acc_inliers)[idx], z[idx]
                    axs[1,1].scatter(x, y, c=z, s=50, edgecolor='face')
                axs[1,1].grid(axis='x', alpha=0.75)
                axs[1,1].grid(axis='y', alpha=0.75)
                axs[1,1].set_xlabel('Nadir angle [deg]')
                axs[1,1].set_ylabel('Correct matches [%]')
                axs[1,1].set_title('Correct matches vs nadir angle')
                axs[1,1].set_ylim(0, 100)
            except np.linalg.LinAlgError as e:
                # Catch the specific error and print a message (or log it)
                print(f"Skipping plot due to error: {e}")

            plt.tight_layout()

            # Save figure
            fig_file_name = os.path.basename(os.path.normpath(args.test_data_path))+ f"_inliers_{match_params_str}_n{n_queries}{var_str}.png"
            f.savefig(os.path.join(path_2_results,fig_file_name))
            plt.close(f) # Close figure


            # Plot match reprojection errors - All vs Inliers
            f, axs = plt.subplots(1, 2, figsize=(6, 5))
            # Plot match reprojection error
            # all matches
            axs[0].plot(bins, rep_err_1to0_cum, label="map-to-query")
            axs[0].plot(bins, rep_err_0to1_cum, label="query-to-map")
            axs[0].plot(bins, rep_err_mean_cum, label="mean")
            axs[0].grid(axis='x', alpha=0.75)
            axs[0].grid(axis='y', alpha=0.75)
            axs[0].set_xlabel('Reprojection error [px]')
            axs[0].set_ylabel('Matches [%]')
            axs[0].set_title('Cumulative matches')
            axs[0].set_ylim(0, 100)
            axs[0].legend()
            # inliers
            axs[1].plot(bins, rep_err_1to0_inliers_cum, label="map-to-query")
            axs[1].plot(bins, rep_err_0to1_inliers_cum, label="query-to-map")
            axs[1].plot(bins, rep_err_mean_inliers_cum, label="mean")
            axs[1].grid(axis='x', alpha=0.75)
            axs[1].grid(axis='y', alpha=0.75)
            axs[1].set_xlabel('Reprojection error [px]')
            axs[1].set_ylabel('Matches [%]')
            axs[1].set_title('Cumulative inlier matches')
            axs[1].set_ylim(0, 100)
            axs[1].legend()

            # Save figure
            fig_file_name = os.path.basename(os.path.normpath(args.test_data_path))+ f"_match_acc_w_inliers_{match_params_str}_n{n_queries}{var_str}.png"
            f.savefig(os.path.join(path_2_results,fig_file_name))
            plt.close(f) # Close figure

            print(f"Complete.")


