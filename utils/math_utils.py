import numpy as np
import math


######### Math utilities ###################

def get_rot_tuple(tuple_deg):
    # Convers tuple from degrees to radians
    return ( math.radians(tuple_deg[0]), 
             math.radians(tuple_deg[1]), 
             math.radians(tuple_deg[2]) )

def XtoSO3(angle):
    '''  Convert rotation angle around X to SO3 rotation matrix
    angle [rad] - rotation angle
    '''
    cos = np.cos(angle)
    sin = np.sin(angle)

    R =  np.array((
        (1,     0,     0),
        (0,  cos,    -sin),
        (0,  sin,     cos)
    ), dtype=np.float64)

    # print(F"X_inv - X_transp: {np.linalg.inv(R) - R.transpose()}")

    return np.asmatrix(R)

def YtoSO3(angle):
    '''  Convert rotation angle around Y to SO3 rotation matrix
    angle [rad] - rotation angle
    '''
    cos = np.cos(angle)
    sin = np.sin(angle)

    R =  np.array((
        (cos,    0,   sin),
        (0,      1,     0),
        (-sin,   0,   cos)
    ), dtype=np.float64)

    # print(F"Y_inv - Y_transp: {np.linalg.inv(R) - R.transpose()}")
    return np.asmatrix(R)

def ZtoSO3(angle):
    '''  Convert rotation angle around Z to SO3 rotation matrix
    angle [rad] - rotation angle
    '''
    cos = np.cos(angle)
    sin = np.sin(angle)

    R =  np.array((
        (cos,  -sin,     0),
        (sin,   cos,     0),
        (0,       0,     1)
    ), dtype=np.float64)

    # print(F"Z_inv - Z_transp: {np.linalg.inv(R) - R.transpose()}")

    return np.asmatrix(R)

def Eul312toSO3(angle_X, angle_Y, angle_Z):
    '''  Convert rot_X, rot_Y, rot_Z angles to SO3 matrix according to rotation order 312: R(rot_Y)*R(rot_X)*R(rot_Z) 
    angle_Z - Around Z axis (pointing upward in Blender object frame) [rad]
    angle_Y - Around Y axis (pointing in the direction of motion in Blender object frame) [rad]
    angle_X - Around X axis (pointing rightward w.r.t direction of motion) [rad]
    
    
    '''
    R =  np.dot(ZtoSO3(angle_Z), np.dot(XtoSO3(angle_X), YtoSO3(angle_Y)) )

    # print(F"R_inv - R_transp: {np.linalg.inv(R) - R.transpose()}")

    return R