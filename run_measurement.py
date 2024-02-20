import numpy as np

cal_param_path = 'calibration paramters/calibration_parameters2.npz'

# Load calibration parameters
mtx_loaded = None
dist_loaded = None


def load_cal_param(cal_param_path=cal_param_path, param='both'):
    global mtx_loaded, dist_loaded
    try:
        if mtx_loaded is None or dist_loaded is None:
            loaded_params = np.load(cal_param_path)
            mtx_loaded, dist_loaded = loaded_params['mtx'], loaded_params['dist']

        if param == 'mtx':
            return mtx_loaded
        elif param == 'dist':
            return dist_loaded
        else:
            return mtx_loaded, dist_loaded
    except Exception as e:
        # Internal Server Error
        return {'error': f'loading calibration parameters failed, {str(e)}'}, 500


def transform_points_coord(keypoints):
    try:
        # Load the camera matrix
        mtx_loaded = load_cal_param(param='mtx')

        # Ensure keypoints has the expected shape
        if len(keypoints.shape) == 4:
            # Extract the relevant dimension
            keypoints_squeeze = np.squeeze(keypoints, axis=(0, 1))
        elif len(keypoints.shape) == 3:
            # Squeeze points into a new dimension
            keypoints_squeeze = np.squeeze(keypoints)
        else:
            # Handle the case where keypoints has unexpected shape
            raise ValueError(
                f"Unexpected shape of keypoints array {keypoints}")

        # Add a row of ones to keypoints_xy to make it homogeneous coordinates
        homogeneous_coords = np.hstack(
            (keypoints_squeeze, np.ones((keypoints_squeeze.shape[0], 1))))

        # Transpose homogeneous_coords to have shape (3, N)
        homogeneous_coords = homogeneous_coords.T

        # Ensure mtx_loaded has shape (3, 3)
        if mtx_loaded.shape != (3, 3):
            raise ValueError("Camera matrix must have shape (3, 3)")

        # Convert to real-world coordinates
        real_world_coordinates_homogeneous = np.dot(
            np.linalg.pinv(mtx_loaded), homogeneous_coords)

        # Transpose back to have shape (N, 3)
        real_world_coordinates_homogeneous = real_world_coordinates_homogeneous.T

        # Extract non-homogeneous coordinates
        real_world_coordinates = real_world_coordinates_homogeneous[:, :2]

        return real_world_coordinates
    except Exception as e:
        # Internal Server Error
        return {'error': f'conversion to real-world coordinates error, {str(e)}'}, 500


def scale_coord(real_world_coordinates, known_square_size=4):
    try:
        # Measured size from real-world coordinates
        measured_square_size = np.abs(
            real_world_coordinates[1] - real_world_coordinates[0])

        # Scaling factor
        scaling_factor = known_square_size / measured_square_size

        scaled_real_world_coordinates = real_world_coordinates * scaling_factor

        return scaled_real_world_coordinates
    except Exception as e:
        # Internal Server Error
        return {'error': f'conversion to scaled real world coorinates error, {str(e)}'}, 500


def euclidean_distance(keypoint1, keypoint2):
    try:
        return np.sqrt(np.sum((keypoint1 - keypoint2) ** 2))
    except Exception as e:
        # Internal Server Error
        return {'error': f'euclidean distance not able to compute, {str(e)}'}, 500


def find_midpoint(keypoint1, keypoint2):
    try:
        # Find midpoint
        midpoint = (keypoint1 + keypoint2) / 2

        return midpoint
    except Exception as e:
        # Internal Server Error
        return {'error': f'midpoint computation error, {str(e)}'}, 500


def calculate_circumference(keypoint1, keypoint2):
    try:
        # Find midpoints
        midpoint = find_midpoint(keypoint1, keypoint2)

        # Calculate radius
        radius = np.sqrt(np.sum((midpoint - keypoint2) ** 2))

        # Calculate circumference
        circumference = 2 * np.pi * radius

        return circumference
    except Exception as e:
        # Internal Server Error
        return {'error': f'circumference calculation errror, {str(e)}'}, 500


def calculate_body_measurements(left_shoulder, right_shoulder, left_hip, right_hip, left_knee, left_elbow, left_wrist, left_ankle):
    try:
        # Top Measurement
        shoulder_hip_measurement = euclidean_distance(left_shoulder, left_hip)
        hip_knee_measurement = euclidean_distance(left_hip, left_knee) / 2
        top_measurement = round(
            shoulder_hip_measurement + hip_knee_measurement, 1)

        # Shoulder Measurement
        shoulder_measurement = round(
            euclidean_distance(right_shoulder, left_shoulder), 1)

        # Chest Measurement
        chest_measurement = round(calculate_circumference(
            right_shoulder, left_shoulder) / 2 * 2, 1)

        # Hand measurement
        leftShoulder_leftElbow_measurement = round(
            euclidean_distance(left_shoulder, left_elbow), 1)
        leftElbow_leftWrist_measurement = round(
            euclidean_distance(left_elbow, left_wrist), 1)
        hand_measurement = round(
            leftShoulder_leftElbow_measurement + leftElbow_leftWrist_measurement, 1)

        # Short Hand Measurement
        short_measurement = round(hand_measurement / 2, 1)

        # Hip measurement
        hip_measurement = round(
            calculate_circumference(left_hip, right_hip), 1)

        # Neck Measurement
        neck_measurement = round(euclidean_distance(
            left_shoulder, right_shoulder) / 2 * 3, 1)

        # Calculate distances
        hip_to_knee_distance = round(
            euclidean_distance(left_hip, left_knee), 1)
        knee_to_ankle_distance = round(
            euclidean_distance(left_knee, left_ankle), 1)
        thigh_measurement = round(
            hip_to_knee_distance + knee_to_ankle_distance / 2, 1)

        # Leg Measurement
        leftHip_leftKnee_measurement = round(
            euclidean_distance(left_hip, left_knee), 1)
        leftKnee_leftAnkle_measurement = round(
            euclidean_distance(left_knee, left_ankle), 1)
        leg_measurement = round(leftHip_leftKnee_measurement +
                                leftKnee_leftAnkle_measurement, 1)

        return top_measurement, shoulder_measurement, chest_measurement, hand_measurement, short_measurement, hip_measurement, neck_measurement, thigh_measurement, leg_measurement
    except Exception as e:
        # Internal Server Error
        return {'error': f'measurement calculation error, {str(e)}'}, 500


def get_measurements(keypoints):
    try:
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

        top_measurement, shoulder_measurement, chest_measurement, hand_measurement, short_measurement, hip_measurement, neck_measurement, thigh_measurement, leg_measurement = calculate_body_measurements(left_shoulder,
                                                                                                                                                                                                           right_shoulder,
                                                                                                                                                                                                           left_hip,
                                                                                                                                                                                                           right_hip,
                                                                                                                                                                                                           left_knee,
                                                                                                                                                                                                           left_elbow,
                                                                                                                                                                                                           left_wrist,
                                                                                                                                                                                                           left_ankle)
        results = {'top_measurement': top_measurement,
                   'shoulder_measurement': shoulder_measurement,
                   'chest_measurement': chest_measurement,
                   'hand_measurement': hand_measurement,
                   'short_measurement': short_measurement,
                   'hip_measurement': hip_measurement,
                   'neck_measurement': neck_measurement,
                   'thigh_measurement': thigh_measurement,
                   'leg_measurement': leg_measurement}
        return results
    except Exception as e:
        # Internal Server Error
        return {'error': f'get measurements error, {str(e)}'}, 500
