import numpy as np

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
    print(f'The top measurement is {top_measurement}')

    # Shoulder Measurement
    shoulder_measurement = euclidean_distance(right_shoulder, left_shoulder)
    print(f'The shoulder measurement is {shoulder_measurement}')

    # Chest Measurement
    chest_measurement = calculate_circumference(
        right_shoulder, left_shoulder) / 2 * 2
    print(f'The chest measurement is {chest_measurement}')

    # Hand measurement
    leftShoulder_leftElbow_measurement = euclidean_distance(
        left_shoulder, left_elbow)
    leftElbow_leftWrist_measurement = euclidean_distance(
        left_elbow, left_wrist)
    hand_measurement = leftShoulder_leftElbow_measurement + \
        leftElbow_leftWrist_measurement
    print(f'The hand measurement is {hand_measurement}')

    # Short Hand Measurement
    short_measurement = hand_measurement / 2
    print(f'The short hand measurement is {short_measurement}')

    # Hip measurement
    hip_measurement = calculate_circumference(left_hip, right_hip) / 2
    print(f'The hip measurement is {hip_measurement}')

    # Neck Measurement
    neck_measurement = euclidean_distance(left_hip, right_hip) / 2 * 3
    print(f'The neck measurement is {neck_measurement}')

    # Calculate distances
    hip_to_knee_distance = euclidean_distance(left_hip, left_knee)
    knee_to_ankle_distance = euclidean_distance(left_knee, left_ankle)
    thigh_measurement = hip_to_knee_distance + knee_to_ankle_distance
    print(f'The thigh measurement is {thigh_measurement}')

    # Leg Measurment
    leftHip_leftKnee_measurement = euclidean_distance(left_hip, left_knee)
    leftKnee_leftAnkle_measurement = euclidean_distance(left_knee, left_ankle)
    leg_measurement = leftHip_leftKnee_measurement + leftKnee_leftAnkle_measurement
    print(f'The leg measurement is {leg_measurement}')
