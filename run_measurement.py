import numpy as np


def get_measurements(keypoints):
    real_world_coordinates = transform_points_coord(keypoints)
    sc = scale_coord(real_world_coordinates, 1.5)
    left_shoulder = sc[5]
    right_shoulder = sc[6]
    left_elbow = sc[7]
    # right_elbow = sc[8]
    left_wrist = sc[9]
    # right_wrist = sc[10]
    left_hip = sc[11]
    right_hip = sc[12]
    left_knee = sc[13]
    # right_knee = sc[14]
    left_ankle = sc[15]
    # right_ankle = sc[16]

    calculate_body_measurements(left_shoulder,
                                right_shoulder,
                                left_hip,
                                right_hip,
                                left_knee,
                                left_elbow,
                                left_wrist,
                                left_ankle)

    return


# Load calibration parameters
mtx_loaded = None
dist_loaded = None


def load_cal_param(cal_param_path, param='both'):
    global mtx_loaded, dist_loaded
    if mtx_loaded is None or dist_loaded is None:
        loaded_params = np.load(cal_param_path)
        mtx_loaded, dist_loaded = loaded_params['mtx'], loaded_params['dist']

    if param == 'mtx':
        return mtx_loaded
    elif param == 'dist':
        return dist_loaded
    else:
        return mtx_loaded, dist_loaded


def transform_points_coord(keypoints):
    # Load the camera matrix
    mtx_loaded = load_cal_param(cal_param_path, param='mtx')

    # Squeeze points into a new dim
    keypoints_squeeze = np.squeeze(keypoints)

    # Add a row of ones to keypoints_xy to make it homogeneous coordinates
    homogeneous_coords = np.hstack(
        (keypoints_squeeze, np.ones((keypoints_squeeze.shape[0], 1))))

    # Transpose homogeneous_coords to have shape (3, N)
    homogeneous_coords = homogeneous_coords.T

    # Convert to real-world coordinates
    real_world_coordinates_homogeneous = np.dot(
        np.linalg.inv(mtx_loaded), homogeneous_coords)

    # Transpose back to have shape (N, 3)
    real_world_coordinates_homogeneous = real_world_coordinates_homogeneous.T

    # Extract non-homogeneous coordinates
    real_world_coordinates = real_world_coordinates_homogeneous[:, :2]

    return real_world_coordinates


def scale_coord(real_world_coordinates, known_square_size=4):

    # Measured size from real-world coordinates
    measured_square_size = np.abs(
        real_world_coordinates[1] - real_world_coordinates[0])

    # Scaling factor
    scaling_factor = known_square_size / measured_square_size

    scaled_real_world_coordinates = real_world_coordinates * scaling_factor

    return scaled_real_world_coordinates


# Function to calculate Euclidean distance between two keypoints
def euclidean_distance(keypoint1, keypoint2):
    return np.sqrt(np.sum((keypoint1 - keypoint2) ** 2))


# Function to find midpoints
def find_midpoint(keypoint1, keypoint2):
    # Find midpoint
    midpoint = (keypoint1 + keypoint2) / 2

    return midpoint


# Circumference of keypoints
def calculate_circumference(keypoint1, keypoint2):

    # Find midpoints
    midpoint = find_midpoint(keypoint1, keypoint2)

    # Calculate radius
    radius = np.sqrt(np.sum((midpoint - keypoint2) ** 2))

    # Calculate circumference
    circumference = 2 * np.pi * radius

    return circumference


def calculate_body_measurements(left_shoulder,
                                right_shoulder,
                                left_hip,
                                right_hip,
                                left_knee,
                                left_elbow,
                                left_wrist,
                                left_ankle):

    # Top Measurement
    shoulder_hip_measurement = euclidean_distance(left_shoulder, left_hip)
    hip_knee_measurement = euclidean_distance(left_hip, left_knee) / 2
    top_measurement = shoulder_hip_measurement + hip_knee_measurement

    # Shoulder Measurement
    shoulder_measurement = euclidean_distance(right_shoulder, left_shoulder)

    # Chest Measurement
    chest_measurement = calculate_circumference(
        right_shoulder, left_shoulder) / 2 * 2

    # Hand measurement
    leftShoulder_leftElbow_measurement = euclidean_distance(
        left_shoulder, left_elbow)
    leftElbow_leftWrist_measurement = euclidean_distance(
        left_elbow, left_wrist)
    hand_measurement = leftShoulder_leftElbow_measurement + \
        leftElbow_leftWrist_measurement

    # Short Hand Measurement
    short_measurement = hand_measurement / 2

    # Hip measurement
    hip_measurement = calculate_circumference(left_hip, right_hip) / 2

    # Neck Measurement
    neck_measurement = euclidean_distance(left_hip, right_hip) / 2 * 3

    # Calculate distances
    hip_to_knee_distance = euclidean_distance(left_hip, left_knee)
    knee_to_ankle_distance = euclidean_distance(left_knee, left_ankle)
    thigh_measurement = hip_to_knee_distance + knee_to_ankle_distance

    # Leg Measurment
    leftHip_leftKnee_measurement = euclidean_distance(left_hip, left_knee)
    leftKnee_leftAnkle_measurement = euclidean_distance(left_knee, left_ankle)
    leg_measurement = leftHip_leftKnee_measurement + leftKnee_leftAnkle_measurement

    return top_measurement, shoulder_measurement, chest_measurement,
    hand_measurement, short_measurement, hip_measurement, neck_measurement,
    thigh_measurement, leg_measurement
