#!/usr/bin/env python3
"""
Unit tests for IBVSController math logic (ROS2-free).

Covers:
  - quat_to_R        : quaternion → rotation matrix
  - R_b_c            : camera OpenCV → FRD body rotation
  - Eq.(3)           : normalized image error (ex, ey)
  - Eq.(5)           : LOS unit vector  ray_cam → ray_body → ray_ned → n_t
  - Eq.(7)           : LOS angles q_y (elevation), q_z (azimuth)
  - Eq.(13)          : FOV yaw rate controller
"""

import math
import pytest
import numpy as np

# ── Replicate functions/constants from ibvs_controller.py (no ROS2 needed) ────

def quat_to_R(q):
    """[w,x,y,z] → 3x3 rotation matrix  (FRD body → NED world)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)


# Camera (OpenCV) → Body (FRD) rotation matrix
R_b_c = np.array([
    [0, 0, 1],   # body_x ← cam_z  (forward)
    [1, 0, 0],   # body_y ← cam_x  (right)
    [0, 1, 0],   # body_z ← cam_y  (down)
], dtype=float)

# Default camera intrinsics (typhoon_h480 640×360, FOV=2.0 rad)
FX, FY = 205.5, 205.5
CX, CY = 320.0, 180.0

# FOV controller gains
FOV_KP, FOV_KD = 1.5, 0.1


def compute_ibvs(u, v, R_e_b, b_omega_z=0.0):
    """
    Run full IBVS pipeline for a single detection at pixel (u, v).
    Returns (ex, ey, q_y, q_z, fov_yaw_rate).
    """
    # Eq.(3)
    ex = (u - CX) / FX
    ey = (v - CY) / FY

    # Eq.(5)
    ray_cam  = np.array([ex, ey, 1.0])
    ray_body = R_b_c @ ray_cam
    ray_ned  = R_e_b @ ray_body
    norm = np.linalg.norm(ray_ned)
    n_t = ray_ned / norm

    # Eq.(7)
    q_z = math.atan2(n_t[1], n_t[0])
    q_y = math.atan2(-n_t[2], math.sqrt(n_t[0]**2 + n_t[1]**2))

    # Eq.(13)
    ex_dot = -(1.0 + ex**2) * b_omega_z
    fov_yaw_rate = FOV_KP * ex + FOV_KD * ex_dot

    return ex, ey, q_y, q_z, fov_yaw_rate


# ── Helpers ────────────────────────────────────────────────────────────────────

def R_identity():
    """Level drone facing North (heading=0): FRD = NED, R_e_b = I."""
    return np.eye(3)


def R_yaw(yaw_rad):
    """
    Rotation matrix for drone facing yaw_rad (NED azimuth, CW from North).
    FRD body → NED world for a level drone.
    """
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1],
    ])


# ── Tests: quat_to_R ───────────────────────────────────────────────────────────

class TestQuatToR:
    def test_identity_quaternion_gives_identity_matrix(self):
        R = quat_to_R([1, 0, 0, 0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-9)

    def test_90deg_yaw_around_z(self):
        """
        NED: 90° yaw (East-facing drone).
        Quaternion for +90° rotation about NED-z (Down axis):
          w = cos(45°), z = sin(45°)  but NED-z = FRD-z for level drone,
          and FRD z is Down → right-hand rule: w=cos(45°), z=sin(45°).
        Expected R_e_b: body x (forward) → NED East (y), body y (right) → NED South (-x).
        """
        angle = math.pi / 2
        q = [math.cos(angle/2), 0, 0, math.sin(angle/2)]
        R = quat_to_R(q)
        expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-9)

    def test_rotation_matrix_is_orthogonal(self):
        """R @ R^T = I for any valid quaternion."""
        q = [0.6, 0.2, -0.3, 0.7]
        q = np.array(q) / np.linalg.norm(q)
        R = quat_to_R(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_rotation_matrix_det_is_plus_one(self):
        q = [0.6, 0.2, -0.3, 0.7]
        q = np.array(q) / np.linalg.norm(q)
        R = quat_to_R(q)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9


# ── Tests: R_b_c ──────────────────────────────────────────────────────────────

class TestRbc:
    def test_target_straight_ahead_maps_to_body_forward(self):
        """
        ray_cam = [0, 0, 1] (target at image center, no offset)
        Should map to body x-axis [1, 0, 0] (forward).
        """
        ray_cam = np.array([0.0, 0.0, 1.0])
        ray_body = R_b_c @ ray_cam
        np.testing.assert_allclose(ray_body, [1, 0, 0], atol=1e-9)

    def test_target_right_of_center_maps_to_body_right(self):
        """
        ray_cam = [1, 0, 0] (pure right in OpenCV)
        Should map to body y-axis [0, 1, 0] (right in FRD).
        """
        ray_cam = np.array([1.0, 0.0, 0.0])
        ray_body = R_b_c @ ray_cam
        np.testing.assert_allclose(ray_body, [0, 1, 0], atol=1e-9)

    def test_target_below_center_maps_to_body_down(self):
        """
        ray_cam = [0, 1, 0] (pure down in OpenCV)
        Should map to body z-axis [0, 0, 1] (down in FRD).
        """
        ray_cam = np.array([0.0, 1.0, 0.0])
        ray_body = R_b_c @ ray_cam
        np.testing.assert_allclose(ray_body, [0, 0, 1], atol=1e-9)


# ── Tests: Eq.(3) normalized image error ──────────────────────────────────────

class TestEq3:
    def test_target_at_image_center_gives_zero_error(self):
        ex, ey, *_ = compute_ibvs(CX, CY, R_identity())
        assert ex == pytest.approx(0.0)
        assert ey == pytest.approx(0.0)

    def test_target_to_right_gives_positive_ex(self):
        ex, ey, *_ = compute_ibvs(CX + 100, CY, R_identity())
        assert ex > 0

    def test_target_to_left_gives_negative_ex(self):
        ex, ey, *_ = compute_ibvs(CX - 100, CY, R_identity())
        assert ex < 0

    def test_target_below_gives_positive_ey(self):
        ex, ey, *_ = compute_ibvs(CX, CY + 50, R_identity())
        assert ey > 0

    def test_ex_scale(self):
        """ex = (u - cx) / fx  →  100 px right = 100/205.5."""
        ex, *_ = compute_ibvs(CX + 100, CY, R_identity())
        assert ex == pytest.approx(100.0 / FX)


# ── Tests: Eq.(5) + Eq.(7) LOS angles ────────────────────────────────────────

class TestEq5Eq7:
    def test_target_at_center_heading_north_azimuth_is_zero(self):
        """
        Image center (ex=0, ey=0) + level drone facing North (R=I).
        n_t = [1, 0, 0] → q_z = 0 (North), q_y = 0 (level).
        """
        _, _, q_y, q_z, _ = compute_ibvs(CX, CY, R_identity())
        assert q_z == pytest.approx(0.0, abs=1e-9)
        assert q_y == pytest.approx(0.0, abs=1e-9)

    def test_target_right_of_center_gives_positive_azimuth(self):
        """
        Target at right half of image → q_z > 0 (East of North).
        """
        _, _, q_y, q_z, _ = compute_ibvs(CX + 80, CY, R_identity())
        assert q_z > 0

    def test_target_left_of_center_gives_negative_azimuth(self):
        _, _, q_y, q_z, _ = compute_ibvs(CX - 80, CY, R_identity())
        assert q_z < 0

    def test_target_below_center_gives_negative_elevation(self):
        """
        Target below image center → balloon is below horizontal → q_y < 0.
        """
        _, _, q_y, q_z, _ = compute_ibvs(CX, CY + 60, R_identity())
        assert q_y < 0

    def test_target_above_center_gives_positive_elevation(self):
        _, _, q_y, q_z, _ = compute_ibvs(CX, CY - 60, R_identity())
        assert q_y > 0

    def test_azimuth_sign_symmetry(self):
        """q_z for right target == -q_z for left target (symmetric camera)."""
        _, _, _, q_z_r, _ = compute_ibvs(CX + 80, CY, R_identity())
        _, _, _, q_z_l, _ = compute_ibvs(CX - 80, CY, R_identity())
        assert q_z_r == pytest.approx(-q_z_l, abs=1e-9)

    def test_drone_facing_east_target_at_center_azimuth_is_90deg(self):
        """
        Level drone facing East (yaw=90°).
        Target at image center → drone forward = East → q_z ≈ +90°.
        """
        R_e_b = R_yaw(math.pi / 2)
        _, _, q_y, q_z, _ = compute_ibvs(CX, CY, R_e_b)
        assert q_z == pytest.approx(math.pi / 2, abs=1e-6)
        assert q_y == pytest.approx(0.0, abs=1e-6)

    def test_known_geometry(self):
        """
        Drone facing North, target at 45° right and 30° below horizontal.
        Compute expected pixel position and verify LOS angles round-trip.

        In NED: n_t = [cos(-30°)cos(45°), cos(-30°)sin(45°), -sin(-30°)]
                     = [cos30*cos45, cos30*sin45, sin30]
        In FRD (level, North-facing): ray_body = R_e_b^T @ n_t = n_t (identity)
        In camera: ray_cam = R_b_c^T @ ray_body
          R_b_c^T = [[0,1,0],[0,0,1],[1,0,0]]
          ray_cam = [ray_body[1], ray_body[2], ray_body[0]]
                  = [cos30*sin45, sin30, cos30*cos45]
        Normalized: divide by z_c = cos30*cos45
          ex = (cos30*sin45) / (cos30*cos45) = tan45 = 1.0
          ey = sin30 / (cos30*cos45) = tan30/cos45
        """
        az   = math.radians(45.0)   # azimuth right
        el   = math.radians(-30.0)  # elevation below
        ex_exp = math.tan(az)                               # = 1.0
        ey_exp = math.tan(-el) / math.cos(az)              # = tan30/cos45

        u = CX + ex_exp * FX
        v = CY + ey_exp * FY

        _, _, q_y, q_z, _ = compute_ibvs(u, v, R_identity())
        assert q_z == pytest.approx(az,  abs=1e-5)
        assert q_y == pytest.approx(el,  abs=1e-5)


# ── Tests: Eq.(13) FOV yaw rate controller ────────────────────────────────────

class TestEq13:
    def test_target_at_center_no_yaw_zero_rate(self):
        """ex=0, b_omega_z=0 → fov_yaw_rate = 0."""
        *_, fov = compute_ibvs(CX, CY, R_identity(), b_omega_z=0.0)
        assert fov == pytest.approx(0.0, abs=1e-9)

    def test_target_at_center_still_yawing_gives_damping(self):
        """
        ex=0 (target centered) but drone is still yawing (b_omega_z=0.5 rad/s).
        ex_dot = -(1+0²)*0.5 = -0.5 → fov = kp*0 + kd*(-0.5) = -0.05.
        Derivative term counter-acts the residual yaw (correct damping behavior).
        """
        *_, fov = compute_ibvs(CX, CY, R_identity(), b_omega_z=0.5)
        assert fov == pytest.approx(FOV_KD * (-(1.0) * 0.5), abs=1e-9)

    def test_target_right_gives_positive_yaw_rate(self):
        """ex > 0 (target to the right) → yaw right (positive)."""
        *_, fov = compute_ibvs(CX + 100, CY, R_identity(), b_omega_z=0.0)
        assert fov > 0

    def test_target_left_gives_negative_yaw_rate(self):
        """ex < 0 (target to the left) → yaw left (negative)."""
        *_, fov = compute_ibvs(CX - 100, CY, R_identity(), b_omega_z=0.0)
        assert fov < 0

    def test_proportional_term_only(self):
        """b_omega_z=0 → fov_yaw_rate = kp * ex."""
        ex, _, _, _, fov = compute_ibvs(CX + 82, CY, R_identity(), b_omega_z=0.0)
        assert fov == pytest.approx(FOV_KP * ex, abs=1e-9)

    def test_derivative_damping_with_matching_yaw(self):
        """
        When already yawing right at the right rate to track the target,
        the derivative term should reduce the command (damping effect).
        ex > 0, b_omega_z > 0 (yawing right) → ex_dot < 0 → fov < kp*ex.
        """
        ex, _, _, _, fov_nodamp = compute_ibvs(CX + 100, CY, R_identity(), b_omega_z=0.0)
        _,  _, _, _, fov_damp   = compute_ibvs(CX + 100, CY, R_identity(), b_omega_z=1.0)
        assert fov_damp < fov_nodamp


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
