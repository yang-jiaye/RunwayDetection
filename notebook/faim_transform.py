'''
Created: 2024-03-18 18:58:42
Author : Zihao Zhang
Email : zh.zhang@sjtu.edu.cn
Last Modified : 2024-03-29 16:30:52
-----
Description: 
'''

import numpy as np
import pymap3d as pm
import scipy.spatial.transform as st

def solve(X, Y):
    """
    Solve the rigid transform problem.
    X: Nx3 numpy array of source points in NED frame
    Y: Nx3 numpy array of target points in Runway frame
    Returns
    R: 3x3 rotation matrix
    t: 3x1 translation vector
    """
    # 1. Find the centroids of X and Y
    X_centroid = np.mean(X, axis=0)
    Y_centroid = np.mean(Y, axis=0)
    # 2. Subtract the centroids from X and Y
    X_prime = X - X_centroid
    Y_prime = Y - Y_centroid
    # 3. Compute the 3x3 matrix H
    H = np.dot(X_prime.T, Y_prime)
    # 4. Compute the SVD of H
    U, S, Vt = np.linalg.svd(H)
    # 5. Compute R
    R = np.dot(Vt.T, U.T)
    # 6. Special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2] *= -1
        R = np.dot(Vt.T, U.T)
    # 7. Compute t
    t = Y_centroid - np.dot(R, X_centroid)
    return R, t

def main():
    O_LLA = np.array([-45.02011667, 168.73495556, 350])
    
    P1_LLA = np.array([-45.01625000, 168.75830556, 350])
    P1_NED = pm.geodetic2ned(P1_LLA[0], P1_LLA[1], P1_LLA[2], O_LLA[0], O_LLA[1], O_LLA[2])
    print("P1_NED:", P1_NED)
    P1_RUN = np.array([1891, -22.5, 0])
    
    P2_LLA = np.array([-45.01663611, 168.75842500, 350])
    P2_NED = pm.geodetic2ned(P2_LLA[0], P2_LLA[1], P2_LLA[2], O_LLA[0], O_LLA[1], O_LLA[2])
    print("P2_NED:", P2_NED)
    P2_RUN = np.array([1891, 22.5, 0])
    
    P3_LLA = np.array([-45.01991944, 168.73489444, 350])
    P3_NED = pm.geodetic2ned(P3_LLA[0], P3_LLA[1], P3_LLA[2], O_LLA[0], O_LLA[1], O_LLA[2])
    print("P3_NED:", P3_NED)
    P3_RUN = np.array([0, -22.5, 0])
    
    P4_LLA = np.array([-45.02031111, 168.73501667, 350])
    P4_NED = pm.geodetic2ned(P4_LLA[0], P4_LLA[1], P4_LLA[2], O_LLA[0], O_LLA[1], O_LLA[2])
    print("P4_NED:", P4_NED)
    P4_RUN = np.array([0, 22.5, 0])

    X = np.array([P1_NED, P2_NED, P3_NED])
    Y = np.array([P1_RUN, P2_RUN, P3_RUN])
    R, t = solve(X, Y)
    R_tmp = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
    R = R @ R_tmp
    print("R:", R)
    print("t:", t)
    # rot = st.Rotation.from_matrix(R)
    # print("\nrot:", rot.as_euler('xyz', degrees=True))

    print("============test===========")
    a = np.array([-50.3129005,	-1.97795E-07,	-49.64810562]).T
    b = np.array([-4.9126205213715100000E+01,	-1.0870414985923800000E+01,	-4.9646474121268200000E+01]).T
    print(R @ a + t)
    print(b)

if __name__ == "__main__":
    main()