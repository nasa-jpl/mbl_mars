import numpy as np
from utils.math_utils import Eul312toSO3, XtoSO3
from utils.projections_utils import backproject_persp_points, project_on_ortho

######## Pose utilities #############################################################################

def perturb_pose(t_query_wc, yaw, roll, pitch, pose_uncertainty):
    """Perturb the pose using Gaussian noise based on pose uncertainty."""
    tx_noisy = np.random.normal(t_query_wc[0, 0], np.sqrt(pose_uncertainty['var_tx']))
    ty_noisy = np.random.normal(t_query_wc[1, 0], np.sqrt(pose_uncertainty['var_ty']))
    tz_noisy = np.random.normal(t_query_wc[2, 0], np.sqrt(pose_uncertainty['var_tz']))
    t_query_noisy_wc = np.array([tx_noisy, ty_noisy, tz_noisy]).reshape((3, 1))

    yaw_noisy = np.random.normal(yaw, np.sqrt(pose_uncertainty['var_yaw']))
    roll_noisy = np.random.normal(roll, np.sqrt(pose_uncertainty['var_roll']))
    pitch_noisy = np.random.normal(pitch, np.sqrt(pose_uncertainty['var_pitch']))
    rot_noisy_z, rot_noisy_x, rot_noisy_y = yaw_noisy, pitch_noisy, roll_noisy
    R_query_noisy_wc = np.matmul(Eul312toSO3(rot_noisy_x, rot_noisy_y, rot_noisy_z), XtoSO3(np.pi))
    return t_query_noisy_wc, R_query_noisy_wc, yaw_noisy, roll_noisy, pitch_noisy

def generate_noisy_depths(points2D, query_img_depth, pose_uncertainty, use_altimeter, center_depth):
    """Generate noisy depth values for given points."""
    points2D_depth_noisy = []
    if use_altimeter:
        center_depth_noisy = np.random.normal(center_depth, np.sqrt(pose_uncertainty['var_altimeter'])) 
        points2D_depth_noisy.append(center_depth_noisy)
        # Use the same altimeter reading with larger variance for other points
        for _ in range(1, points2D.shape[0]):
            points2D_depth_noisy.append(np.random.normal(center_depth, np.sqrt(pose_uncertainty['var_pz'])))
    else:
        # Depth image is available
        for k in range(points2D.shape[0]):
            px, py = points2D[k, 0], points2D[k, 1]
            points2D_depth_noisy.append(np.random.normal(query_img_depth[py, px], np.sqrt(pose_uncertainty['var_altimeter'])))
    return points2D_depth_noisy   
   
def project_and_print(points2D, points2D_depth, query_intr, R_query_wc, t_query_wc, map_px_resolution, map_cx, map_cy, R_map_cw, t_map_cw):
    """Project points onto the map and print their coordinates."""
    query_points_3D = backproject_persp_points(points2D, points2D_depth, query_intr, R=R_query_wc, T=t_query_wc)
    map_points2D = project_on_ortho(query_points_3D, map_px_resolution, cx=map_cx, cy=map_cy, R=R_map_cw, T=t_map_cw)
    U, V = map_points2D[0, 0], map_points2D[0, 1]
    print("Projected points on map:\n", U, V)
    return map_points2D

def create_joint_result_matrix(azim_diff_list, elev_diff_list, met_list):
    mat_dict = {}
    res_dict = {}
    for met in met_list:
        res_dict[met] = {}
        for i in range(len(elev_diff_list)):
            for j in range(len(azim_diff_list)):
                slot_name = str(elev_diff_list[i])+"_"+str(azim_diff_list[j])
                res_dict[met][slot_name] = []
                mat_dict[slot_name] = (i,j)
    return res_dict, mat_dict
                    