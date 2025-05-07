import os
import numpy as np
import json
from PIL import Image
from utils.math_utils import XtoSO3, Eul312toSO3


########### Data loading utilities ################################
def load_mars_sim_query_set(path):
    image_count = 0
    rgb_image_path_list = []
    depth_path_list = []
    locs_queries = []
    ypr_queries = []
    altitude_queries = []

    while True:
        rgb_image_path = os.path.join(path, "images/PerspCam0_%04i.png" % image_count)
        depth_path = os.path.join(path, "depth/PerspCam0_%04i_depth.npy" % image_count)
        # pose_path = os.path.join(path, 'cam_poses/PerspCam0_%04i_pose.json' % image_count)

        if not os.path.isfile(rgb_image_path):
            break

        # Append image and depth path to lists
        rgb_image_path_list.append(rgb_image_path)
        depth_path_list.append(depth_path)

        # Append poses to lists
        with open(os.path.join(path, 'cam_poses/PerspCam0_%04i_pose.json' % image_count)) as f:
            cam_pose = json.load(f)

        locs_queries.append(cam_pose['t_wc_w']) # Location in world reference frame (Blender frame) [m]
        ypr_queries.append(cam_pose['ypr']) # Yaw, pitch, roll [radians]
        altitude_queries.append(cam_pose['altitude']) # Altitude [m]

        image_count += 1

    # # Convert lists to numpy arrays
    # locs_queries = np.array(locs_queries)
    # ypr_queries = np.array(ypr_queries)
    # altitude_queries = np.array(altitude_queries)
   
    print('Number of query images found:',len(rgb_image_path_list))
    return rgb_image_path_list, depth_path_list, locs_queries, ypr_queries, altitude_queries

def paste_map_tiles(tiles_dir, overall_width, overall_height, n_tiles_x, n_tiles_y):
    final_image_gray = Image.new('L', (overall_width, overall_height))
    
    img_id = 0
    cam_name = "OrthoCam"
    img_path = tiles_dir + cam_name + "_%04i" % img_id

    # Paste each tile into the final image
    tile_width = overall_width // n_tiles_x
    tile_height = overall_height // n_tiles_y
    for y in range(n_tiles_y):
        for x in range(n_tiles_x):
            tile_path = img_path + f"_tile_{x}_{y}.png"
            tile_rgb = Image.open(tile_path) # load the tile RGB image
            tile_gray = tile_rgb.convert('L')   # Convert to gray
            final_image_gray.paste(tile_gray, (x * tile_width, (n_tiles_y - y - 1) * tile_height))

    # Convert the final image to a NumPy array
    final_image_gray_np = np.array(final_image_gray)

    return final_image_gray_np

def load_map_data(maps_path, elev_list, azim_list):
    
    print(f"\n... loading map data")
    # Organize all map data in a dict
    print(f"maps path: {maps_path}")
    map_dirs = os.listdir(maps_path)
    map_dirs = [ x for x in map_dirs if os.path.isdir(os.path.join(maps_path,x)) and x != "configs"]
    # print(f"map dirs: {map_dirs}")
    map_data = {}

    # Load map cam metadata
    with open(os.path.join(maps_path, 'cam_data.json')) as f:
        cam_data = json.load(f)
    map_ortho_width = cam_data['ortho_scale']
    map_width, map_height = cam_data['img_width'], cam_data['img_height']
    map_aspect_x, map_aspect_y = cam_data['aspect_x'], cam_data['aspect_y']
    map_aspect_ratio = map_aspect_x / map_aspect_y
    map_cx, map_cy = map_width / 2., map_height / 2.
    map_clip_start = cam_data['clip_start']
    px_resolution = map_ortho_width / map_width

    # Load map data params
    with open(os.path.join(maps_path, 'params.json')) as f:
        data_params = json.load(f)
    tiles_x = data_params["tiles_x"]
    tiles_y = data_params["tiles_y"]

    for map_name in map_dirs:
        azim = int(map_name.split('_')[-1])
        elev = int(map_name.split('_')[-3])
        # load maps with specified elevation and azimuth
        if (elev in elev_list) and (azim in azim_list):
            
            # Load map
            map_path = os.path.join(maps_path, map_name)
            print(f"\n        map path: {map_path}")
            print(f"    ...loading map depth")
            map_depth = np.load(os.path.join(map_path, 'depth/OrthoCam_0000_depth.npy'))
            #   Load map image tiles by tiles
            print(f"    ...loading map image tile by tile")
            map_gray = paste_map_tiles(tiles_dir=map_path+"/images/", overall_width=map_width, overall_height=map_height, n_tiles_x=tiles_x, n_tiles_y=tiles_y)
            print(f"        map depth shape: {map_depth.shape}")
            print(f"        map gray shape: {map_gray.shape}")

            # Load pose
            with open(os.path.join(map_path, 'cam_poses/OrthoCam_0000_pose.json')) as f:
                map_pose = json.load(f)
            # R_wc = quat2SO3_right_handed(map_pose['q_wc'])
            # t_map = R_wc.transpose() * np.array([map_pose['translation']]).transpose()
            yaw, pitch, roll = map_pose["ypr"] # radians
            rot_z, rot_x, rot_y = yaw, pitch, roll
            R_wc = np.matmul(Eul312toSO3(rot_x, rot_y, rot_z), XtoSO3(np.pi)) # the convention in wc is the other way round
            # R_cw = R_wc.T
            t_wc_w = np.transpose(np.array(map_pose["t_wc_w"])) # Column vector
            t_wc_w = t_wc_w.reshape((3, 1))
            # t_cw_w = -t_wc_w
            # t_cw_c = np.dot(R_cw, t_cw_w)


            map_data[map_name] = {
                'gray':map_gray,
                'depth': map_depth,
                'R_wc':R_wc,
                't_wc':t_wc_w,
                'px_resolution':px_resolution,
                'cx': map_cx,
                'cy': map_cy,
                'img_height': map_height,
                'img_width': map_width,
                'clip_start':map_clip_start,
                'elev': elev,
                'azim': azim
            }
    print(f"... complete.")
    return map_data


def load_observation_data(obsvs_path, elev_list, azim_list):

    print(f"\n... loading observation data")
    print(f"observations path: {obsvs_path}")
    obsv_dirs = os.listdir(obsvs_path)
    obsv_dirs = [ x for x in obsv_dirs if os.path.isdir(os.path.join(obsvs_path,x)) and x != "configs"]

    obsv_data = {}

    # Load query cam metadata
    with open(os.path.join(obsvs_path, 'cam_data.json')) as f:
        cam_data = json.load(f)
    clip_start = cam_data['clip_start']
    img_width, img_height = cam_data['img_width'], cam_data['img_height']
    fx, fy = cam_data['fx'], cam_data['fy']
    cx, cy = cam_data['cx'], cam_data['cy']

    for obsv_name in obsv_dirs:
        azim = int(obsv_name.split('_')[-1])
        elev = int(obsv_name.split('_')[-3])
        # load observations with specified elevation and azimuth
        if (elev in elev_list) and (azim in azim_list):
            obsv_path = os.path.join(obsvs_path,obsv_name)           

            rgb_image_path_list, depth_path_list, locs_queries, ypr_queries, altitude_queries = load_mars_sim_query_set(obsv_path)

            obsv_data[obsv_name] = {
                'img_width':img_width, 'img_height':img_height,
                'clip_start':clip_start,
                'fx':fx, 'fy':fy, 'cx':cx, 'cy':cy,   
                'rgb_image_path_list':rgb_image_path_list,
                'depth_path_list':depth_path_list,
                'locs_queries':locs_queries, # location in world frame (Blender frame) [m]
                'ypr_queries':ypr_queries, # pitch, roll, yaw [radians]
                'altitude_queries':altitude_queries, # altitude [m]
                'elev': elev, # [deg]
                'azim': azim, # [deg]
            }

    print(f"... complete.")
    return obsv_data

########### Data loading utilities ################################
def load_mars_sim_query_set(path):
    image_count = 0 #TODO: modify repos starting idx
    rgb_image_path_list = []
    depth_path_list = []
    locs_queries = []
    ypr_queries = []
    altitude_queries = []

    while True:
        rgb_image_path = os.path.join(path, "images/PerspCam0_%04i.png" % image_count)
        depth_path = os.path.join(path, "depth/PerspCam0_%04i_depth.npy" % image_count)
        # pose_path = os.path.join(path, 'cam_poses/PerspCam0_%04i_pose.json' % image_count)

        if not os.path.isfile(rgb_image_path):
            break

        # Append image and depth path to lists
        rgb_image_path_list.append(rgb_image_path)
        depth_path_list.append(depth_path)

        # Append poses to lists
        with open(os.path.join(path, 'cam_poses/PerspCam0_%04i_pose.json' % image_count)) as f:
            cam_pose = json.load(f)

        locs_queries.append(cam_pose['t_wc_w']) # Location in world reference frame (Blender frame) [m]
        ypr_queries.append(cam_pose['ypr']) # Yaw, pitch, roll [radians]
        altitude_queries.append(cam_pose['altitude']) # Altitude [m]

        image_count += 1

    # # Convert lists to numpy arrays
    # locs_queries = np.array(locs_queries)
    # ypr_queries = np.array(ypr_queries)
    # altitude_queries = np.array(altitude_queries)
   
    print('Number of query images found:',len(rgb_image_path_list))
    return rgb_image_path_list, depth_path_list, locs_queries, ypr_queries, altitude_queries

def paste_map_tiles(tiles_dir, overall_width, overall_height, n_tiles_x, n_tiles_y):
    final_image_gray = Image.new('L', (overall_width, overall_height))
    
    img_id = 0
    cam_name = "OrthoCam"
    img_path = tiles_dir + cam_name + "_%04i" % img_id

    # Paste each tile into the final image
    tile_width = overall_width // n_tiles_x
    tile_height = overall_height // n_tiles_y
    for y in range(n_tiles_y):
        for x in range(n_tiles_x):
            tile_path = img_path + f"_tile_{x}_{y}.png"
            tile_rgb = Image.open(tile_path) # load the tile RGB image
            tile_gray = tile_rgb.convert('L')   # Convert to gray
            final_image_gray.paste(tile_gray, (x * tile_width, (n_tiles_y - y - 1) * tile_height))

    # Convert the final image to a NumPy array
    final_image_gray_np = np.array(final_image_gray)

    return final_image_gray_np

def load_map_data(maps_path, elev_list, azim_list):
    
    print(f"\n... loading map data")
    # Organize all map data in a dict
    print(f"maps path: {maps_path}")
    map_dirs = os.listdir(maps_path)
    map_dirs = [ x for x in map_dirs if os.path.isdir(os.path.join(maps_path,x)) and x != "configs" and x != "map_depth"]
    print(f"map dirs: {map_dirs}")
    map_data = {}

    # Load map cam metadata
    with open(os.path.join(maps_path, 'cam_data.json')) as f:
        cam_data = json.load(f)
    map_ortho_width = cam_data['ortho_scale']
    map_width, map_height = cam_data['img_width'], cam_data['img_height']
    map_aspect_x, map_aspect_y = cam_data['aspect_x'], cam_data['aspect_y']
    map_aspect_ratio = map_aspect_x / map_aspect_y
    map_cx, map_cy = map_width / 2., map_height / 2.
    map_clip_start = cam_data['clip_start']
    px_resolution = map_ortho_width / map_width

    # Load map data params
    with open(os.path.join(maps_path, 'params.json')) as f:
        data_params = json.load(f)
    tiles_x = data_params["tiles_x"]
    tiles_y = data_params["tiles_y"]

    # Load map depth
    print(f"    ...loading map depth")
    map_depth = np.load(os.path.join(maps_path, 'map_depth/depth/OrthoCam_0000_depth.npy'))
    print(f"        map depth shape: {map_depth.shape}")

    # Load 
    for map_name in map_dirs:
        print(f"map name: {map_name}")
        azim = int(map_name.split('_')[-1])
        elev = int(map_name.split('_')[-3])
        # load maps with specified elevation and azimuth
        if (elev in elev_list) and (azim in azim_list):
            
            # Load map
            map_path = os.path.join(maps_path, map_name)
            print(f"\n        map path: {map_path}")
            #   Load map image tiles by tiles
            print(f"    ...loading map image tile by tile")
            map_gray = paste_map_tiles(tiles_dir=map_path+"/images/", overall_width=map_width, overall_height=map_height, n_tiles_x=tiles_x, n_tiles_y=tiles_y)
            print(f"        map gray shape: {map_gray.shape}")

            # Load pose
            with open(os.path.join(map_path, 'cam_poses/OrthoCam_0000_pose.json')) as f:
                map_pose = json.load(f)
            # R_wc = quat2SO3_right_handed(map_pose['q_wc'])
            # t_map = R_wc.transpose() * np.array([map_pose['translation']]).transpose()
            yaw, pitch, roll = map_pose["ypr"] # radians
            rot_z, rot_x, rot_y = yaw, pitch, roll
            R_wc = np.matmul(Eul312toSO3(rot_x, rot_y, rot_z), XtoSO3(np.pi)) # the convention in wc is the other way round
            # R_cw = R_wc.T
            t_wc_w = np.transpose(np.array(map_pose["t_wc_w"])) # Column vector
            t_wc_w = t_wc_w.reshape((3, 1))
            # t_cw_w = -t_wc_w
            # t_cw_c = np.dot(R_cw, t_cw_w)


            map_data[map_name] = {
                'gray':map_gray,
                'depth': map_depth,
                'R_wc':R_wc,
                't_wc':t_wc_w,
                'px_resolution':px_resolution,
                'cx': map_cx,
                'cy': map_cy,
                'img_height': map_height,
                'img_width': map_width,
                'clip_start':map_clip_start,
                'elev': elev,
                'azim': azim
            }
    print(f"... complete.")
    return map_data


def load_observation_data(obsvs_path, elev_list, azim_list):

    print(f"\n... loading observation data")
    print(f"observations path: {obsvs_path}")
    obsv_dirs = os.listdir(obsvs_path)
    obsv_dirs = [ x for x in obsv_dirs if os.path.isdir(os.path.join(obsvs_path,x)) and x != "configs"]

    obsv_data = {}

    # Load query cam metadata
    with open(os.path.join(obsvs_path, 'cam_data.json')) as f:
        cam_data = json.load(f)
    clip_start = cam_data['clip_start']
    img_width, img_height = cam_data['img_width'], cam_data['img_height']
    fx, fy = cam_data['fx'], cam_data['fy']
    cx, cy = cam_data['cx'], cam_data['cy']

    for obsv_name in obsv_dirs:
        azim = int(obsv_name.split('_')[-1])
        elev = int(obsv_name.split('_')[-3])
        # load observations with specified elevation and azimuth
        if (elev in elev_list) and (azim in azim_list):
            obsv_path = os.path.join(obsvs_path,obsv_name)
            print(f"Obsv_path: {obsv_path}")          

            rgb_image_path_list, depth_path_list, locs_queries, ypr_queries, altitude_queries = load_mars_sim_query_set(obsv_path)

            obsv_data[obsv_name] = {
                'img_width':img_width, 'img_height':img_height,
                'clip_start':clip_start,
                'fx':fx, 'fy':fy, 'cx':cx, 'cy':cy,   
                'rgb_image_path_list':rgb_image_path_list,
                'depth_path_list':depth_path_list,
                'locs_queries':locs_queries, # location in world frame (Blender frame) [m]
                'ypr_queries':ypr_queries, # pitch, roll, yaw [radians]
                'altitude_queries':altitude_queries, # altitude [m]
                'elev': elev, # [deg]
                'azim': azim, # [deg]
            }

    print(f"... complete.")
    return obsv_data
