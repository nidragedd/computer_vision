"""
This package contains utility and helper functions related to geometry, angles, etc

@author: nidragedd
"""
import numpy as np


def get_angle_degree(keypoints_coords):
    """
    Remember that dot product of vectors A and B == norm(A) x norm(B) x cos(angle)
        ==> cos(angle) = dot product (A, B) / (norm(A) x norm(B))
        ==> pick arccos to get the angle
    :param keypoints_coords: (array) positions (x, y) for each detected keypoint, None if not detected
    """
    keypoint_a = np.array(keypoints_coords[0])
    keypoint_b = np.array(keypoints_coords[1])
    keypoint_c = np.array(keypoints_coords[2])

    # Vector building
    vector_ab = keypoint_a - keypoint_b
    vector_bc = keypoint_b - keypoint_c

    cosine_angle = np.dot(vector_ab, vector_bc) / (np.linalg.norm(vector_ab) * np.linalg.norm(vector_bc))
    return np.arccos(cosine_angle)
