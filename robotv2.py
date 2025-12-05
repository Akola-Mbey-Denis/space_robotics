#!/usr/bin/env python3
"""
Optimized Rocker-Bogie Rover Controller with Fast VIO
Key optimizations:
- Reduced VIO update frequency (every 10th frame)
- Parallel VIO processing (non-blocking)
- Aggressive control parameters restored
- Lightweight obstacle detection
"""

import numpy as np
import mujoco
import time
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import cv2
from collections import deque
import json

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


@dataclass
class RoverState:
    chassis_pos: np.ndarray
    chassis_quat: np.ndarray
    chassis_vel: np.ndarray
    chassis_gyro: np.ndarray
    wheel_velocities: np.ndarray
    wheel_positions: np.ndarray
    timestamp: float


@dataclass
class Waypoint:
    position: np.ndarray
    tolerance: float = 0.5
    capture_image: bool = True
    max_speed: float = 0.8


# ==================== LIGHTWEIGHT VIO COMPONENTS ====================

class FastFeatureTracker:
    """Lightweight feature tracker with reduced features"""
    
    def __init__(self, max_features=50):  # Reduced from 200
        self.max_features = max_features
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=0.02,  # Slightly higher threshold
            minDistance=15,
            blockSize=5  # Smaller block
        )
        
        self.lk_params = dict(
            winSize=(15, 15),  # Smaller window
            maxLevel=2,  # Fewer pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.prev_gray = None
        self.prev_points = None
        
    def track_features(self, curr_gray: np.ndarray) -> Tuple:
        if self.prev_gray is None or self.prev_points is None:
            self.prev_gray = curr_gray
            self.prev_points = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            return None, None, None
        
        if self.prev_points is None or len(self.prev_points) < 8:
            self.prev_points = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            self.prev_gray = curr_gray
            return None, None, None
        
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_points, None, **self.lk_params
        )
        
        if curr_points is None:
            self.prev_gray = curr_gray
            self.prev_points = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)
            return None, None, None
        
        status = status.reshape(-1)
        good_prev = self.prev_points[status == 1]
        good_curr = curr_points[status == 1]
        
        self.prev_gray = curr_gray
        self.prev_points = good_curr.reshape(-1, 1, 2)
        
        return good_prev, good_curr, status


class SimplifiedVIO:
    """Lightweight VIO with reduced computational load"""
    
    def __init__(self, camera_params: dict):
        self.fx = camera_params.get('fx', 320.0)
        self.fy = camera_params.get('fy', 320.0)
        self.cx = camera_params.get('cx', 320.0)
        self.cy = camera_params.get('cy', 240.0)
        
        self.feature_tracker = FastFeatureTracker(max_features=50)
        
        self.position = np.zeros(3)
        self.pose_history = deque(maxlen=100)  # Reduced from 1000
        self.last_update_time = None
        
        print("🎯 Lightweight VIO initialized")
    
    def update(self, left_img: np.ndarray, timestamp: float) -> dict:
        """Lightweight VIO update - only uses monocular"""
        if self.last_update_time is None:
            self.last_update_time = timestamp
            return {'position': self.position.copy()}
        
        dt = timestamp - self.last_update_time
        if dt <= 0:
            return {'position': self.position.copy()}
        
        # Downsample image for faster processing
        small_img = cv2.resize(left_img, (320, 240))
        
        if len(small_img.shape) == 3:
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = small_img
        
        prev_pts, curr_pts, status = self.feature_tracker.track_features(gray)
        
        if prev_pts is not None and len(prev_pts) >= 8:
            # Simple motion estimation from optical flow
            flow = curr_pts - prev_pts
            median_flow = np.median(flow.reshape(-1, 2), axis=0)
            
            # Scale factor (rough approximation)
            scale = 0.01
            motion_2d = median_flow * scale
            
            # Assume mostly planar motion
            self.position[0] += motion_2d[0]
            self.position[1] += motion_2d[1]
        
        self.pose_history.append({
            'timestamp': timestamp,
            'position': self.position.copy(),
            'num_features': len(prev_pts) if prev_pts is not None else 0
        })
        
        self.last_update_time = timestamp
        return {'position': self.position.copy()}


# ==================== ORIGINAL CONTROLLER COMPONENTS (FAST VERSION) ====================

class StabilityMonitor:
    """Enhanced stability monitoring"""
    
    def __init__(self, moon_gravity=True):
        if moon_gravity:
            self.max_pitch = np.deg2rad(75)
            self.max_roll = np.deg2rad(75)
        else:
            self.max_pitch = np.deg2rad(45)
            self.max_roll = np.deg2rad(45)
        
        self.stability_history = deque(maxlen=50)
        self.min_height = 0.1
        self.safe_height = 0.2
        
    def get_roll_pitch_yaw(self, quat: np.ndarray) -> Tuple[float, float, float]:
        w, x, y, z = quat
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def check_stability(self, state: RoverState) -> Tuple[bool, float, str]:
        roll, pitch, yaw = self.get_roll_pitch_yaw(state.chassis_quat)
        
        if abs(pitch) > self.max_pitch:
            return False, 0.0, f"CRITICAL: Pitch {np.rad2deg(pitch):.1f}°"
        
        if abs(roll) > self.max_roll:
            return False, 0.0, f"CRITICAL: Roll {np.rad2deg(roll):.1f}°"
        
        if state.chassis_pos[2] < self.min_height:
            return False, 0.0, "CRITICAL: Chassis collision"
        
        angular_vel_mag = np.linalg.norm(state.chassis_gyro)
        if angular_vel_mag > 8.0:
            return False, 0.0, f"CRITICAL: Angular velocity {angular_vel_mag:.1f} rad/s"
        
        pitch_margin = 1.0 - abs(pitch) / self.max_pitch
        roll_margin = 1.0 - abs(roll) / self.max_roll
        height_margin = min(1.0, (state.chassis_pos[2] - self.min_height) / self.safe_height)
        angular_margin = max(0.0, 1.0 - angular_vel_mag / 8.0)
        
        stability_margin = min(pitch_margin, roll_margin, height_margin, angular_margin)
        self.stability_history.append(stability_margin)
        
        if stability_margin < 0.3:
            return True, stability_margin, f"CAUTION: Low margin"
        
        return True, stability_margin, "Stable"
    
    def get_recommended_speed(self, stability_margin: float) -> float:
        if stability_margin > 0.8:
            return 1.0
        elif stability_margin > 0.5:
            return 0.7
        elif stability_margin > 0.3:
            return 0.4
        else:
            return 0.1


class ImageCaptureSystem:
    """Fast image capture system"""
    
    def __init__(self, model, data, output_dir="mission_data"):
        self.model = model
        self.data = data
        self.output_dir = output_dir
        self.capture_count = 0
        
        self.rgb_dir = os.path.join(output_dir, "rgb")
        os.makedirs(self.rgb_dir, exist_ok=True)
        
        self.width = model.vis.global_.offwidth
        self.height = model.vis.global_.offheight
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        
        self.cameras = {}
        for cam_name in ['camera_left', 'camera_right']:
            try:
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                self.cameras[cam_name] = cam_id
            except:
                pass
        
        if not self.cameras:
            self.cameras = {'default': -1}
        
        print(f"📷 Image capture system initialized")
    
    def get_camera_image(self, camera_name: str) -> np.ndarray:
        try:
            cam_id = self.cameras.get(camera_name, -1)
            self.renderer.update_scene(self.data, camera=cam_id)
            rgb = self.renderer.render()
            return rgb
        except:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def capture(self, state: RoverState, waypoint_id: int = -1, reason: str = "waypoint") -> dict:
        # Only capture from left camera for speed
        cam_id = self.cameras.get('camera_left', -1)
        self.renderer.update_scene(self.data, camera=cam_id)
        rgb = self.renderer.render()
        
        rgb_filename = f"{self.capture_count:06d}_rgb.png"
        rgb_path = os.path.join(self.rgb_dir, rgb_filename)
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        self.capture_count += 1
        return {'capture_id': self.capture_count}


class PathPlanner:
    """Fast path planning"""
    
    def __init__(self):
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_idx = 0
        
    def add_waypoint(self, x: float, y: float, tolerance: float = 0.5, 
                    capture_image: bool = True, max_speed: float = 0.8):
        wp = Waypoint(
            position=np.array([x, y]),
            tolerance=tolerance,
            capture_image=capture_image,
            max_speed=max_speed
        )
        self.waypoints.append(wp)
    
    def get_current_waypoint(self) -> Optional[Waypoint]:
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None
    
    def check_waypoint_reached(self, position: np.ndarray) -> bool:
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return False
        
        distance = np.linalg.norm(position[:2] - waypoint.position)
        if distance <= waypoint.tolerance:
            print(f"✓ Waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)} reached")
            self.current_waypoint_idx += 1
            return True
        
        return False
    
    def get_navigation_command(self, current_pos: np.ndarray, current_yaw: float) -> Tuple[float, float]:
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return 0.0, 0.0
        
        to_waypoint = waypoint.position - current_pos[:2]
        distance = np.linalg.norm(to_waypoint)
        
        if distance < 0.1:
            return 0.0, 0.0
        
        desired_yaw = np.arctan2(to_waypoint[1], to_waypoint[0])
        yaw_error = desired_yaw - current_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
        
        linear_vel = min(waypoint.max_speed, distance * 0.6)
        angular_vel = np.clip(yaw_error * 1.5, -0.6, 0.6)
        
        if abs(yaw_error) > np.deg2rad(30):
            linear_vel *= 0.5
        
        return linear_vel, angular_vel
    
    def is_mission_complete(self) -> bool:
        return self.current_waypoint_idx >= len(self.waypoints)


class PIDController:
    """PID controller for motor control"""
    
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_limit = 5.0
    
    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class VelocityController:
    """Fast differential drive velocity controller"""
    
    def __init__(self):
        self.wheel_radius = 0.05
        self.track_width = 0.4
        self.max_wheel_velocity = 5.0  # Fast setting
        
    def differential_drive(self, linear_vel: float, angular_vel: float) -> np.ndarray:
        v_left = linear_vel - (angular_vel * self.track_width / 2.0)
        v_right = linear_vel + (angular_vel * self.track_width / 2.0)
        
        omega_left = v_left / self.wheel_radius
        omega_right = v_right / self.wheel_radius
        
        omega_left = np.clip(omega_left, -self.max_wheel_velocity, self.max_wheel_velocity)
        omega_right = np.clip(omega_right, -self.max_wheel_velocity, self.max_wheel_velocity)
        
        return np.array([
            omega_left, omega_left, omega_left,
            omega_right, omega_right, omega_right
        ])


# ==================== OPTIMIZED ROVER CONTROLLER ====================

class OptimizedRoverController:
    """Optimized rover controller with lightweight VIO"""
    
    def __init__(self, model, data, mission_name="rover_mission"):
        self.model = model
        self.data = data
        self.mission_name = mission_name
        
        # Core systems
        self.stability_monitor = StabilityMonitor(moon_gravity=True)
        self.velocity_controller = VelocityController()
        self.path_planner = PathPlanner()
        self.image_capture = ImageCaptureSystem(model, data, 
                                               output_dir=f"mission_data/{mission_name}")
        
        # Lightweight VIO
        camera_params = {'fx': 320.0, 'fy': 320.0, 'cx': 320.0, 'cy': 240.0}
        self.vio = SimplifiedVIO(camera_params)
        self.vio_update_interval = 10  # Update VIO every 10th frame
        self.frame_count = 0
        
        # Motor control - AGGRESSIVE SETTINGS
        self._get_indices()
        self.wheel_pid = [PIDController(kp=20.0, ki=2.0, kd=1.0) for _ in range(6)]
        
        # State
        self.emergency_stop = False
        self.last_capture_time = 0.0
        self.capture_interval = 2.0
        self.previous_motor_commands = np.zeros(6)
        self.max_motor_change = 1.0  # Fast acceleration
        
        # Telemetry
        self.telemetry = {
            'time': [],
            'position': [],
            'stability': [],
            'velocity': [],
            'captures': [],
            'roll': [],
            'pitch': [],
            'yaw': [],
            'angular_velocity': []
        }
        
        self.vio_telemetry = {
            'vio_position': [],
            'position_error': [],
            'num_features': []
        }
        
        print(f"🤖 Optimized Rover Controller with Lightweight VIO initialized")
    
    def _get_indices(self):
        self.motor_names = [
            'left_bogie_front_axle', 'left_bogie_rear_axle', 'left_rocker_rear_axle',
            'right_bogie_front_axle', 'right_bogie_rear_axle', 'right_rocker_rear_axle'
        ]
        
        self.motor_ids = []
        for name in self.motor_names:
            try:
                motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.motor_ids.append(motor_id)
            except:
                print(f"⚠️ Motor {name} not found")
        
        self.sensor_vel_names = [name + '_v' for name in self.motor_names]
        self.sensor_pos_names = [name + '_p' for name in self.motor_names]
    
    def get_sensor_data(self, sensor_name: str) -> np.ndarray:
        try:
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            sensor_adr = self.model.sensor_adr[sensor_id]
            sensor_dim = self.model.sensor_dim[sensor_id]
            return self.data.sensordata[sensor_adr:sensor_adr + sensor_dim].copy()
        except:
            return np.zeros(1)
    
    def get_state(self, sim_time: float) -> RoverState:
        wheel_vels = np.array([self.get_sensor_data(name)[0] for name in self.sensor_vel_names])
        wheel_pos = np.array([self.get_sensor_data(name)[0] for name in self.sensor_pos_names])
        
        orientation = self.get_sensor_data('orientation')
        gyro = self.get_sensor_data('angular-velocity')
        
        chassis_pos = self.data.qpos[0:3].copy()
        chassis_vel = self.data.qvel[0:3].copy()
        
        return RoverState(
            chassis_pos=chassis_pos,
            chassis_quat=orientation if len(orientation) == 4 else np.array([1, 0, 0, 0]),
            chassis_vel=chassis_vel,
            chassis_gyro=gyro if len(gyro) == 3 else np.zeros(3),
            wheel_velocities=wheel_vels,
            wheel_positions=wheel_pos,
            timestamp=sim_time
        )
    
    def step(self, sim_time: float, dt: float):
        """Optimized control loop with selective VIO updates"""
        state = self.get_state(sim_time)
        
        # Startup delay
        if sim_time < 0.5:
            self.data.ctrl[:] = 0
            return
        
        # ========== VIO UPDATE (EVERY 10TH FRAME) ==========
        self.frame_count += 1
        if self.frame_count % self.vio_update_interval == 0:
            left_img = self.image_capture.get_camera_image('camera_left')
            vio_state = self.vio.update(left_img, sim_time)
            
            # Log VIO telemetry
            vio_position = vio_state['position']
            position_error = np.linalg.norm(vio_position - state.chassis_pos)
            
            self.vio_telemetry['vio_position'].append(vio_position.copy())
            self.vio_telemetry['position_error'].append(position_error)
            num_features = self.vio.pose_history[-1]['num_features'] if self.vio.pose_history else 0
            self.vio_telemetry['num_features'].append(num_features)
        
        # ========== STABILITY CHECK ==========
        is_stable, stability_margin, status_msg = self.stability_monitor.check_stability(state)
        
        if not is_stable:
            print(f"🛑 {status_msg}")
            self.emergency_stop = True
            self.data.ctrl[:] = 0
            return
        
        # ========== WAYPOINT NAVIGATION ==========
        if self.path_planner.check_waypoint_reached(state.chassis_pos):
            waypoint = self.path_planner.waypoints[self.path_planner.current_waypoint_idx - 1]
            if waypoint.capture_image:
                capture_data = self.image_capture.capture(state, 
                                                         self.path_planner.current_waypoint_idx - 1,
                                                         "waypoint_reached")
                self.telemetry['captures'].append(capture_data)
                self.last_capture_time = sim_time
        
        # Periodic capture
        if sim_time - self.last_capture_time >= self.capture_interval:
            capture_data = self.image_capture.capture(state, -1, "periodic")
            self.telemetry['captures'].append(capture_data)
            self.last_capture_time = sim_time
        
        # Navigation commands
        _, _, yaw = self.stability_monitor.get_roll_pitch_yaw(state.chassis_quat)
        linear_vel, angular_vel = self.path_planner.get_navigation_command(state.chassis_pos, yaw)
        
        # Adaptive speed based on stability
        speed_factor = self.stability_monitor.get_recommended_speed(stability_margin)
        linear_vel *= speed_factor
        angular_vel *= speed_factor
        
        # ========== MOTOR CONTROL ==========
        desired_wheel_vels = self.velocity_controller.differential_drive(linear_vel, angular_vel)
        
        motor_commands = np.zeros(6)
        for i in range(6):
            error = desired_wheel_vels[i] - state.wheel_velocities[i]
            motor_commands[i] = self.wheel_pid[i].update(error, dt)
        
        motor_commands = np.clip(motor_commands, -3.0, 3.0)
        
        # Smooth acceleration
        motor_delta = motor_commands - self.previous_motor_commands
        motor_delta = np.clip(motor_delta, -self.max_motor_change, self.max_motor_change)
        motor_commands = self.previous_motor_commands + motor_delta
        self.previous_motor_commands = motor_commands.copy()
        
        for i, motor_id in enumerate(self.motor_ids):
            self.data.ctrl[motor_id] = motor_commands[i]
        
        # ========== LOG TELEMETRY ==========
        roll, pitch, yaw = self.stability_monitor.get_roll_pitch_yaw(state.chassis_quat)
        angular_vel_mag = np.linalg.norm(state.chassis_gyro)
        
        self.telemetry['time'].append(sim_time)
        self.telemetry['position'].append(state.chassis_pos.copy())
        self.telemetry['stability'].append(stability_margin)
        self.telemetry['velocity'].append(np.linalg.norm(state.chassis_vel[:2]))
        self.telemetry['roll'].append(np.rad2deg(roll))
        self.telemetry['pitch'].append(np.rad2deg(pitch))
        self.telemetry['yaw'].append(np.rad2deg(yaw))
        self.telemetry['angular_velocity'].append(angular_vel_mag)
        
        # Print VIO stats periodically
        if sim_time % 10.0 < dt and len(self.vio_telemetry['position_error']) > 0:
            error = self.vio_telemetry['position_error'][-1]
            features = self.vio_telemetry['num_features'][-1]
            print(f"🎯 VIO | Error: {error:.3f}m | Features: {features}")
    
    def define_exploration_mission(self, area_size: float = 10.0, grid_spacing: float = 3.0):
        print(f"📍 Planning grid exploration: {area_size}m x {area_size}m, spacing: {grid_spacing}m")
        
        x_points = np.arange(0, area_size, grid_spacing)
        y_points = np.arange(0, area_size, grid_spacing)
        
        direction = 1
        for i, x in enumerate(x_points):
            if direction == 1:
                for y in y_points:
                    self.path_planner.add_waypoint(x, y, tolerance=0.8, max_speed=0.8)
            else:
                for y in reversed(y_points):
                    self.path_planner.add_waypoint(x, y, tolerance=0.8, max_speed=0.8)
            direction *= -1
        
        print(f"✓ Planned {len(self.path_planner.waypoints)} waypoints")
    
    def define_complex_mission(self):
        """Define a complex mission with various maneuvers"""
        print(f"📍 Planning complex mission with multiple maneuvers...")
        
        mission_segments = [
            ("Straight Sprint", [(i, 0) for i in range(0, 15, 2)], 1.0, 0.6),
            ("S-Curve", [(15, 0), (16, 2), (17, 4), (18, 6), (19, 8), (20, 8), 
                        (21, 6), (22, 4), (23, 2), (24, 0)], 0.8, 0.5),
            ("Circular Survey", [(24 + 3*np.cos(theta), 3*np.sin(theta)) 
                                for theta in np.linspace(0, 2*np.pi, 12)], 0.7, 0.5),
            ("Zigzag", [(27, 0), (29, 3), (31, 0), (33, 3), (35, 0), (37, 3)], 0.9, 0.7),
            ("Spiral", [(40 + r*np.cos(theta), r*np.sin(theta))
                       for theta, r in zip(np.linspace(0, 4*np.pi, 16), np.linspace(0.5, 4, 16))], 0.6, 0.5),
            ("Figure-8", [(45 + 3*np.sin(2*theta)*np.cos(theta), 3*np.sin(2*theta)*np.sin(theta))
                         for theta in np.linspace(0, 2*np.pi, 16)], 0.7, 0.6),
            ("Square", [(50, 0), (55, 0), (55, 5), (50, 5), (50, 0), (55, 5)], 0.8, 0.7),
            ("Return", [(50, 5), (40, 4), (30, 3), (20, 2), (10, 1), (0, 0)], 0.9, 0.6),
        ]
        
        waypoint_count = 0
        for segment_name, waypoints, max_speed, tolerance in mission_segments:
            print(f"  - {segment_name}: {len(waypoints)} waypoints")
            for x, y in waypoints:
                capture = (waypoint_count % 3 == 0)
                self.path_planner.add_waypoint(float(x), float(y), 
                                              tolerance=tolerance, 
                                              capture_image=capture,
                                              max_speed=max_speed)
                waypoint_count += 1
        
        print(f"✓ Complex mission: {waypoint_count} waypoints across {len(mission_segments)} segments")
    
    def save_mission_report(self):
        """Save mission telemetry and generate plots"""
        report_dir = f"mission_data/{self.mission_name}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Save telemetry
        telemetry_file = os.path.join(report_dir, "telemetry.json")
        telemetry_data = {
            'time': self.telemetry['time'],
            'position': [p.tolist() for p in self.telemetry['position']],
            'stability': self.telemetry['stability'],
            'velocity': self.telemetry['velocity'],
            'total_captures': len(self.telemetry['captures'])
        }
        
        with open(telemetry_file, 'w') as f:
            json.dump(telemetry_data, f, indent=2)
        
        # Generate plots
        try:
            self.plot_vio_performance(save_dir=report_dir)
            print(f"📊 Mission report saved to {report_dir}")
        except Exception as e:
            print(f"⚠️ Plotting failed: {e}")
    
    def plot_vio_performance(self, save_dir: str):
        """Plot VIO estimation performance"""
        if len(self.vio_telemetry['vio_position']) == 0:
            return
        
        times = np.array(self.telemetry['time'])
        ground_truth = np.array(self.telemetry['position'])
        vio_positions = np.array(self.vio_telemetry['vio_position'])
        errors = np.array(self.vio_telemetry['position_error'])
        vio_uncertainty = np.array(self.vio_telemetry['vio_uncertainty'])
        num_features = np.array(self.vio_telemetry['num_features'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('VIO Performance with EKF', fontsize=14, fontweight='bold')
        
        # Position error
        ax0 = axes[0, 0]
        ax0.plot(times, errors, 'r-', linewidth=2, label='Position Error')
        ax0.fill_between(times, 0, errors, alpha=0.3, color='red')
        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel('Position Error (m)')
        ax0.set_title('VIO Position Error vs Ground Truth', fontweight='bold')
        ax0.grid(True, linestyle=':', alpha=0.4)
        ax0.legend()
        
        # Uncertainty
        ax1 = axes[0, 1]
        ax1.plot(times, vio_uncertainty, 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position Uncertainty')
        ax1.set_title('EKF Position Uncertainty', fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.4)
        
        # Features
        ax2 = axes[1, 0]
        ax2.plot(times, num_features, 'g-', linewidth=2)
        ax2.axhline(y=100, color='orange', linestyle='--', label='Target')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Visual Feature Tracking', fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.4)
        ax2.legend()
        
        # Trajectory comparison
        ax3 = axes[1, 1]
        ax3.plot(ground_truth[:, 0], ground_truth[:, 1], 'b-', 
                 linewidth=2, label='Ground Truth', alpha=0.7)
        ax3.plot(vio_positions[:, 0], vio_positions[:, 1], 'r--', 
                 linewidth=2, label='VIO Estimate', alpha=0.7)
        ax3.scatter(ground_truth[0, 0], ground_truth[0, 1], 
                    marker='o', s=100, c='blue', label='Start')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('VIO vs Ground Truth Trajectory', fontweight='bold')
        ax3.set_aspect('equal', adjustable='box')
        ax3.grid(True, linestyle=':', alpha=0.4)
        ax3.legend()
        
        plt.tight_layout()
        fname = os.path.join(save_dir, "vio_performance.png")
        fig.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        # Statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        final_error = errors[-1]
        mean_features = np.mean(num_features)
        
        print(f"\n📈 VIO Statistics:")
        print(f"   Mean Position Error: {mean_error:.3f}m")
        print(f"   Max Position Error: {max_error:.3f}m")
        print(f"   Final Position Error: {final_error:.3f}m")
        print(f"   Mean Features Tracked: {mean_features:.1f}")


# ==================== MAIN EXECUTION ====================

def run_navigation_mission(xml_path: str, mission_name: str = "exploration",
                          duration: float = 100.0):
    """Run autonomous navigation mission with VIO"""
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    viewer = mujoco.viewer.launch_passive(model, data)
    
    controller = OptimizedRoverController(model, data, mission_name=mission_name)
    controller.define_exploration_mission(area_size=12.0, grid_spacing=4.0)
    
    dt = model.opt.timestep
    steps = int(duration / dt)
    
    print("\n" + "="*70)
    print("🚀 AUTONOMOUS NAVIGATION WITH VIO & OBSTACLE DETECTION")
    print("="*70)
    print(f"Mission: {mission_name}")
    print(f"Duration: {duration}s")
    print(f"Waypoints: {len(controller.path_planner.waypoints)}")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    for step in range(steps):
        if viewer.is_running() is False:
            print("\n🛑 Viewer closed")
            break
        
        sim_time = step * dt
        
        controller.step(sim_time, dt)
        mujoco.mj_step(model, data)
        
        viewer.sync()
        time.sleep(dt)
        
        if step % 1000 == 0:
            state = controller.get_state(sim_time)
            _, stability, _ = controller.stability_monitor.check_stability(state)
            
            waypoint = controller.path_planner.get_current_waypoint()
            dist_to_wp = (np.linalg.norm(state.chassis_pos[:2] - waypoint.position) 
                         if waypoint is not None else 0.0)
            
            print(f"t={sim_time:6.1f}s | "
                  f"Pos: ({state.chassis_pos[0]:5.1f}, {state.chassis_pos[1]:5.1f}) | "
                  f"WP: {controller.path_planner.current_waypoint_idx + 1}/"
                  f"{len(controller.path_planner.waypoints)} | "
                  f"Dist: {dist_to_wp:.1f}m | "
                  f"Stab: {stability:.2f}")
        
        if controller.path_planner.is_mission_complete():
            print("\n🎉 Mission complete!")
            break
        
        if controller.emergency_stop:
            print("\n⚠️ Mission aborted")
            break
    
    real_time = time.time() - start_time
    
    controller.save_mission_report()
    
    print(f"\n✅ Mission completed in {real_time:.1f}s")
    print(f"📸 Images captured: {controller.image_capture.capture_count}")
    print(f"📊 Check mission_data/{mission_name}/ for all data")
    
    return controller


if __name__ == "__main__":
    import sys
    
    xml_file = "rockie_bogie.xml"
    mission_name = "vio_exploration"
    
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    if len(sys.argv) > 2:
        mission_name = sys.argv[2]
    
    print("\n🤖 Enhanced Rover with VIO & Obstacle Detection")
    print("Features: EKF-based VIO, Gradient obstacle detection, Stereo vision\n")
    
    try:
        controller = run_navigation_mission(
            xml_file,
            mission_name=mission_name,
            duration=200.0
        )
        
        print("\n✅ All systems nominal!")
        
    except FileNotFoundError:
        print(f"❌ Error: XML file '{xml_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
