import numpy as np

# # Assuming we are running this from root dir
# sys.path.insert(0, '.')


#######################################################################
### Functions for projecting points from images to the map in the MarsSim datasets

def backproject_ortho_points(points2D, depth, pixel_res, R, T):
    # TODO: account for non-square pixel (when aspect_x != aspect_y)
    height, width = np.shape(depth)
    points2D = points2D.round().astype(np.int64)
    points3D = np.zeros((points2D.shape[0],3)) # N x 3 
    for i in range(len(points2D)):
        # Sample depth. TODO: Bilinear interpolation
        x = points2D[i][0] # need to verify x,y
        y = points2D[i][1]
        z = depth[y, x]
        # z = float(depth[int(y), int(x)])
        # z = bilinear_interpolation(depth, x, y)

        # local3D - Position of the point in the map camera frame in map camera coordinate
        local3D = np.zeros((3,1), dtype=np.float32) # 3 x 1
        local3D[0] = (x - width/2) * pixel_res
        local3D[1] = (y - height/2) * pixel_res
        local3D[2] = z
        # print(f"Backproject from map - Local3D: {local3D}")

        w3D = np.dot(R,local3D) + T
        # print("w3D:", w3D.shape)
        # print("T:", T.shape)
    
        points3D[i,:] = w3D.transpose()

        # data_dic ={
        #     'local3D':local3D, # local 3D map
        #     'z':z # depth from map
        # }
    return points3D

def backproject_persp_points(points2D, points2D_depth, intr, R, T):
    # print(f"\n***Inside backproject_persp_points****")
    # TODO: account for non-square pixel (when aspect_x != aspect_y)
    # TODO: acocunt for shift_x and shift_y of the persp camera
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    points3D = np.zeros((points2D.shape[0], 3), dtype=np.float32)
    for i in range(len(points2D)):
        x = points2D[i][0] # need to verify x,y
        y = points2D[i][1]
        z = points2D_depth[i] # float(depth[int(y), int(x)])
        # z = bilinear_interpolation(depth, x, y)

        local3D = np.zeros((3,1), dtype=np.float32)
        local3D[0] = (x-cx)*z / fx
        local3D[1] = (y-cy)*z / fy
        local3D[2] = z

        w3D =  np.dot(R,local3D) + T # Column vector
        # print(f"\nx: {x}, y:{y}, z:{z}")
        # print(f"local3D: {local3D}")
        # print(f"R: {R}")
        # print(f"T: {T}")
        # print(f"wl3D: {w3D}")
       
        points3D[i,:] = w3D.transpose()

        # data_dic ={
        #     'local3D':local3D, # local 3D query
        #     'z':z # depth from query
        # }
    return points3D

def project_on_persp(points3D, intr, R, T):
    # print(f"\n***Inside project_on_persp****")
    # TODO: account for non-square pixel (when aspect_x != aspect_y)
    # TODO: acocunt for shift_x and shift_y of the persp camera
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    points2D = np.zeros((len(points3D), 2), dtype=np.float32)
    # print(f"points2D: {points2D}")
    point3D = np.zeros((3,1), dtype=np.float32)
    for i in range(len(points3D)):
        point3D[:,0] = points3D[i,:].transpose()
        local3D = np.dot(R,point3D) + T # Column vector
        # print(f"\nPoint3D: {point3D}")
        # print(f"R: {R}")
        # print(f"T: {T}")
        # print(f"local3D: {local3D}")
        X, Y, Z = local3D[0,0], local3D[1,0], local3D[2,0]
        # Project local 3D points from camera frame to image frame
        # print(f"X: {X}, Y: {Y}, Z: {Z}")
        # print(f"cx: {cx}, cy: {cy}")
        # print(f"fx: {fx}, fy: {fy}")
        u = cx + fx*X/Z
        v = cy + fy*Y/Z

        points2D[i,0] = u
        points2D[i,1] = v
        # print(f"x:{u}, y:{v}")
        # data_dic ={
        #     'u':u,
        #     'v':v,
        #     'local_query_3D':local3D # local query 3D
        # }


    return points2D

def project_on_ortho(points3D, pixel_res, cx, cy, R, T):
    # Project 3D world points on map plane
    # points3D Nx3
    # pixe_res_x (y): pixel resolution [m/pixel] along x (y)
    # TODO: account for non-square pixel (when aspect_x != aspect_y)
    points2D = np.zeros((len(points3D), 2), dtype=np.float32)
    point3D = np.zeros((3,1), dtype=np.float32)
    for i in range(len(points3D)):
        point3D[:,0] = points3D[i,:].transpose()
        local3D = np.dot(R, point3D) + T
        # Project local 3D points from camera frame to image frame
        X, Y, Z = local3D[0,0] , local3D[1,0], local3D[2,0]
        u =  X*(1/pixel_res) + cx
        v =  Y*(1/pixel_res) + cy

        points2D[i,0] = u
        points2D[i,1] = v
        # print(f"points2D[{i},0]: {points2D[i,0]}")
        # print(f"points2D[{i},1]: {points2D[i,1]}")

    # data_dic ={
    #         'u':u,
    #         'v':v,
    #         'local_map_3D':local3D # local map 3D
    #     }
    
    return points2D
