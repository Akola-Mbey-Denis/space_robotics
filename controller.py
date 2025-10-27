
"""
Rocker-Bogie Rover Stability Control System with Slip Ratio Simulation
Includes wheel slip detection, traction control, and comprehensive logging
"""

import numpy as np
import mujoco
import time
import os
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class RoverState:
    """Current state of the rover"""
    chassis_pos: np.ndarray
    chassis_quat: np.ndarray
    chassis_vel: np.ndarray
    chassis_gyro: np.ndarray
    wheel_velocities: np.ndarray
    wheel_positions: np.ndarray
    slip_ratios: Optional[np.ndarray] = None
    wheel_forces: Optional[np.ndarray] = None


class SlipRatioEstimator:
    """Estimates wheel slip ratio for traction control"""
    
    def __init__(self, wheel_radius: float = 0.05):
        self.wheel_radius = wheel_radius
        self.slip_history = []
        
    def compute_slip_ratio(self, wheel_angular_vel: np.ndarray, 
                          chassis_linear_vel: np.ndarray) -> np.ndarray:
        """
        Compute longitudinal slip ratio for each wheel
        Slip ratio = (V_wheel - V_chassis) / max(V_wheel, V_chassis, epsilon)
        
        Slip ratio ranges:
        - 0: No slip (perfect rolling)
        - > 0: Wheel spinning faster than chassis (acceleration slip)
        - < 0: Wheel spinning slower than chassis (braking slip)
        """
        # Compute wheel linear velocity from angular velocity
        V_wheel = wheel_angular_vel * self.wheel_radius
        
        # Use forward velocity component of chassis
        V_chassis = np.linalg.norm(chassis_linear_vel[:2])  # XY plane velocity
        
        # Compute slip ratio with numerical stability
        epsilon = 0.01  # Avoid division by zero
        
        slip_ratios = np.zeros_like(wheel_angular_vel)
        for i in range(len(wheel_angular_vel)):
            denominator = max(abs(V_wheel[i]), V_chassis, epsilon)
            slip_ratios[i] = (V_wheel[i] - V_chassis) / denominator
        
        self.slip_history.append(slip_ratios.copy())
        return slip_ratios
    
    def detect_excessive_slip(self, slip_ratios: np.ndarray, 
                             threshold: float = 0.3) -> Tuple[bool, np.ndarray]:
        """Detect wheels with excessive slip"""
        excessive_slip = np.abs(slip_ratios) > threshold
        return np.any(excessive_slip), excessive_slip
    
    def get_traction_coefficient(self, slip_ratio: float) -> float:
        """
        Simplified Pacejka-like tire model
        Returns normalized traction coefficient based on slip
        """
        # Peak at around 15% slip, then decreases
        optimal_slip = 0.15
        peak_traction = 1.0
        
        if abs(slip_ratio) < optimal_slip:
            # Linear region
            return peak_traction * (abs(slip_ratio) / optimal_slip)
        else:
            # Sliding region - reduced traction
            return peak_traction * np.exp(-3.0 * (abs(slip_ratio) - optimal_slip))


class StabilityMonitor:
    """Monitors rover stability"""
    
    def __init__(self):
        self.max_pitch = np.deg2rad(45)
        self.max_roll = np.deg2rad(45)
        
    def get_roll_pitch_yaw(self, quat: np.ndarray) -> Tuple[float, float, float]:
        """Convert quaternion to roll, pitch, yaw"""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def check_stability(self, state: RoverState) -> Tuple[bool, float, str]:
        """Check if rover is stable"""
        roll, pitch, yaw = self.get_roll_pitch_yaw(state.chassis_quat)
        
        if abs(pitch) > self.max_pitch:
            return False, 0.0, f"Excessive pitch: {np.rad2deg(pitch):.1f}°"
        
        if abs(roll) > self.max_roll:
            return False, 0.0, f"Excessive roll: {np.rad2deg(roll):.1f}°"
        
        if state.chassis_pos[2] < 0.1:
            return False, 0.0, "Chassis too low - possible tip-over"
        
        angular_vel_magnitude = np.linalg.norm(state.chassis_gyro)
        if angular_vel_magnitude > 5.0:
            return False, 0.0, "Excessive angular velocity"
        
        pitch_margin = 1.0 - abs(pitch) / self.max_pitch
        roll_margin = 1.0 - abs(roll) / self.max_roll
        height_margin = min(1.0, (state.chassis_pos[2] - 0.1) / 0.2)
        
        stability_margin = min(pitch_margin, roll_margin, height_margin)
        
        return True, stability_margin, "Stable"


class VelocityController:
    """Controls wheel velocities with differential drive and traction control"""
    
    def __init__(self):
        self.wheel_radius = 0.05  # From XML geom
        self.max_wheel_velocity = 20.0
        self.traction_control_enabled = True
        
    def differential_drive(self, linear_vel: float, angular_vel: float) -> np.ndarray:
        """Compute individual wheel velocities for differential drive"""
        track_width = 0.4  # Distance between left and right wheels
        
        v_left = linear_vel - (angular_vel * track_width / 2.0)
        v_right = linear_vel + (angular_vel * track_width / 2.0)
        
        omega_left = v_left / self.wheel_radius
        omega_right = v_right / self.wheel_radius
        
        omega_left = np.clip(omega_left, -self.max_wheel_velocity, self.max_wheel_velocity)
        omega_right = np.clip(omega_right, -self.max_wheel_velocity, self.max_wheel_velocity)
        
        wheel_velocities = np.array([
            omega_left,   # left_bogie_front
            omega_left,   # left_bogie_rear
            omega_left,   # left_rocker_rear
            omega_right,  # right_bogie_front
            omega_right,  # right_bogie_rear
            omega_right   # right_rocker_rear
        ])
        
        return wheel_velocities
    
    def apply_traction_control(self, desired_vels: np.ndarray, 
                               slip_ratios: np.ndarray,
                               slip_threshold: float = 0.25) -> np.ndarray:
        """Reduce wheel velocity commands when excessive slip detected"""
        if not self.traction_control_enabled:
            return desired_vels
        
        controlled_vels = desired_vels.copy()
        
        for i in range(len(desired_vels)):
            if abs(slip_ratios[i]) > slip_threshold:
                # Reduce commanded velocity proportionally to excess slip
                reduction_factor = 1.0 - min(0.5, (abs(slip_ratios[i]) - slip_threshold) / 0.3)
                controlled_vels[i] *= reduction_factor
        
        return controlled_vels
    
    def adaptive_speed(self, stability_margin: float, avg_slip: float = 0.0) -> float:
        """Adjust speed based on stability and slip"""
        base_speed = 1.0
        
        # Stability-based adjustment
        if stability_margin > 0.8:
            speed_factor = 1.3
        elif stability_margin > 0.5:
            speed_factor = 1.0
        else:
            speed_factor = stability_margin
        
        # Slip-based adjustment
        if avg_slip > 0.2:
            slip_factor = max(0.5, 1.0 - (avg_slip - 0.2) / 0.3)
        else:
            slip_factor = 1.0
        
        return base_speed * speed_factor * slip_factor


class StereoCameraRecorder:
    """Records RGB and depth maps from stereo cameras as videos"""
    
    def __init__(self, model, width=640, height=480, save_dir="stereo_output"):
        self.model = model
        self.width = width
        self.height = height
        
        # Camera names from XML
        self.cam_names = ["camera_left", "camera_right"]
        self.cam_ids = []
        
        for name in self.cam_names:
            try:
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                self.cam_ids.append(cam_id)
            except:
                print(f"Warning: Camera {name} not found")
        
        # Create save directories
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create video writers
        import imageio
        self.video_writers = {}
        for name in self.cam_names:
            cam_dir = os.path.join(save_dir, name)
            os.makedirs(cam_dir, exist_ok=True)
            video_path = os.path.join(cam_dir, f"{name}.mp4")
            self.video_writers[name] = imageio.get_writer(video_path, fps=30, codec='libx264')
        
        # Create renderer
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        
        print(f"📷 Camera recorder initialized: {len(self.cam_ids)} cameras found")
    
    def capture(self, data, step: int):
        """Capture RGB and depth for current simulation step and append to video"""
        for name, cam_id in zip(self.cam_names, self.cam_ids):
            # Update scene
            self.renderer.update_scene(data, camera=cam_id)
            
            # Render RGB
            rgb = self.renderer.render()
            
            # Write frame to video
            self.video_writers[name].append_data(rgb)
    
    def close(self):
        """Close video writers"""
        for writer in self.video_writers.values():
            writer.close()
        print(f"✅ Videos saved to {self.save_dir}/")


class PIDController:
    """Simple PID controller"""
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
    
    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        self.integral = np.clip(self.integral, -10.0, 10.0)
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class RockerBogieController:
    """Main controller for rocker-bogie rover with slip ratio monitoring"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.stability_monitor = StabilityMonitor()
        self.velocity_controller = VelocityController()
        self.slip_estimator = SlipRatioEstimator()
        
        self._get_indices()
        
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0
        self.emergency_stop = False
        self.stability_history = []
        self.slip_history = []
        
        # PID controllers for 6 wheels
        self.wheel_pid = [PIDController(kp=50.0, ki=5.0, kd=2.0) for _ in range(6)]
        
    def _get_indices(self):
        """Get indices for actuators and sensors from XML"""
        # Actuator names from XML
        self.motor_names = [
            'left_bogie_front_axle',
            'left_bogie_rear_axle',
            'left_rocker_rear_axle',
            'right_bogie_front_axle',
            'right_bogie_rear_axle',
            'right_rocker_rear_axle'
        ]
        
        self.motor_ids = []
        for name in self.motor_names:
            try:
                motor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.motor_ids.append(motor_id)
            except:
                print(f"Warning: Motor {name} not found")
        
        # Sensor names from XML
        self.sensor_vel_names = [
            'left_bogie_front_axle_v',
            'left_bogie_rear_axle_v',
            'left_rocker_rear_axle_v',
            'right_bogie_front_axle_v',
            'right_bogie_rear_axle_v',
            'right_rocker_rear_axle_v'
        ]
        
        self.sensor_pos_names = [
            'left_bogie_front_axle_p',
            'left_bogie_rear_axle_p',
            'left_rocker_rear_axle_p',
            'right_bogie_front_axle_p',
            'right_bogie_rear_axle_p',
            'right_rocker_rear_axle_p'
        ]
        
        print(f"✓ Found {len(self.motor_ids)} motors")
    
    def get_sensor_data(self, sensor_name: str) -> np.ndarray:
        """Get data from named sensor"""
        try:
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            sensor_adr = self.model.sensor_adr[sensor_id]
            sensor_dim = self.model.sensor_dim[sensor_id]
            return self.data.sensordata[sensor_adr:sensor_adr + sensor_dim].copy()
        except:
            return np.zeros(1)
    
    def get_state(self) -> RoverState:
        """Get current rover state with slip ratio"""
        # Get wheel velocities and positions
        wheel_vels = np.array([self.get_sensor_data(name)[0] for name in self.sensor_vel_names])
        wheel_pos = np.array([self.get_sensor_data(name)[0] for name in self.sensor_pos_names])
        
        # Get orientation and gyro
        orientation = self.get_sensor_data('orientation')
        gyro = self.get_sensor_data('angular-velocity')
        
        # Get position from freejoint (first 3 elements of qpos)
        chassis_pos = self.data.qpos[0:3].copy()
        
        # Get velocity from freejoint
        chassis_vel = self.data.qvel[0:3].copy()
        
        # Compute slip ratios
        slip_ratios = self.slip_estimator.compute_slip_ratio(wheel_vels, chassis_vel)
        
        return RoverState(
            chassis_pos=chassis_pos,
            chassis_quat=orientation if len(orientation) == 4 else np.array([1, 0, 0, 0]),
            chassis_vel=chassis_vel,
            chassis_gyro=gyro if len(gyro) == 3 else np.zeros(3),
            wheel_velocities=wheel_vels,
            wheel_positions=wheel_pos,
            slip_ratios=slip_ratios
        )
    
    def compute_control(self, dt: float) -> np.ndarray:
        """Main control loop with slip-aware traction control"""
        state = self.get_state()
        
        is_stable, stability_margin, reason = self.stability_monitor.check_stability(state)
        self.stability_history.append(stability_margin)
        self.slip_history.append(state.slip_ratios.copy())
        
        # Check for excessive slip
        has_slip, slip_mask = self.slip_estimator.detect_excessive_slip(state.slip_ratios)
        
        if not is_stable:
            print(f"⚠️  STABILITY WARNING: {reason}")
            if stability_margin < 0.2:
                print(" EMERGENCY STOP")
                self.emergency_stop = True
                return np.zeros(6)
        
        if has_slip:
            slipping_wheels = [i for i, slipping in enumerate(slip_mask) if slipping]
            print(f"⚠️  SLIP DETECTED on wheels: {slipping_wheels}")
        
        if self.emergency_stop:
            if np.max(np.abs(state.wheel_velocities)) < 0.1:
                print("✓ Stopped safely")
            return np.zeros(6)
        
        avg_slip = np.mean(np.abs(state.slip_ratios))
        max_speed = self.velocity_controller.adaptive_speed(stability_margin, avg_slip)
        
        if abs(self.target_linear_vel) > max_speed:
            scale = max_speed / abs(self.target_linear_vel)
            linear_vel = self.target_linear_vel * scale
            angular_vel = self.target_angular_vel * scale
        else:
            linear_vel = self.target_linear_vel
            angular_vel = self.target_angular_vel
        
        desired_wheel_vels = self.velocity_controller.differential_drive(linear_vel, angular_vel)
        
        # Apply traction control
        desired_wheel_vels = self.velocity_controller.apply_traction_control(
            desired_wheel_vels, state.slip_ratios
        )
        
        motor_commands = np.zeros(6)
        for i in range(min(6, len(state.wheel_velocities))):
            error = desired_wheel_vels[i] - state.wheel_velocities[i]
            motor_commands[i] = self.wheel_pid[i].update(error, dt)
        
        # Clip to motor limits from XML
        motor_commands = np.clip(motor_commands, -5.12766, 5.12766)
        
        return motor_commands
    
    def set_velocity(self, linear: float, angular: float):
        """Set target velocities"""
        self.target_linear_vel = linear
        self.target_angular_vel = angular
        self.emergency_stop = False
    
    def step(self, dt: float):
        """Execute one control step"""
        motor_commands = self.compute_control(dt)
        
        for i, motor_id in enumerate(self.motor_ids):
            if i < len(motor_commands):
                self.data.ctrl[motor_id] = motor_commands[i]


def plot_results(time_log, stability_log, position_log, slip_log):
    """Plot simulation results including slip ratios"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Stability margin
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_log, stability_log, 'b-', linewidth=2)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Warning threshold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Stability Margin')
    ax1.set_title('Stability Over Time')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim([0, 1.1])
    
    # XY trajectory
    positions = np.array(position_log)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(positions[:, 0], positions[:, 1], 'g-', linewidth=2)
    ax2.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax2.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Trajectory')
    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')
    
    # Height over time
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time_log, positions[:, 2], 'm-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Chassis Height')
    ax3.grid(True)
    
    # Slip ratios for all wheels
    slip_array = np.array(slip_log)
    ax4 = fig.add_subplot(gs[1, :])
    wheel_names = ['LF', 'LR', 'LRR', 'RF', 'RR', 'RRR']
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    
    for i in range(6):
        ax4.plot(time_log, slip_array[:, i], label=f'Wheel {wheel_names[i]}', 
                color=colors[i], linewidth=1.5)
    
    ax4.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Slip threshold')
    ax4.axhline(y=-0.25, color='orange', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.15, color='green', linestyle=':', alpha=0.5, label='Optimal slip')
    ax4.axhline(y=-0.15, color='green', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Slip Ratio')
    ax4.set_title('Wheel Slip Ratios Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', ncol=3)
    ax4.set_ylim([-0.5, 0.5])
    
    # Average slip magnitude
    ax5 = fig.add_subplot(gs[2, 0])
    avg_slip = np.mean(np.abs(slip_array), axis=1)
    ax5.plot(time_log, avg_slip, 'r-', linewidth=2)
    ax5.axhline(y=0.15, color='g', linestyle='--', label='Optimal')
    ax5.axhline(y=0.25, color='orange', linestyle='--', label='High')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Average |Slip|')
    ax5.set_title('Average Slip Magnitude')
    ax5.grid(True)
    ax5.legend()
    
    # Distance traveled
    ax6 = fig.add_subplot(gs[2, 1])
    distances = [0]
    for i in range(1, len(positions)):
        dist = np.linalg.norm(positions[i, :2] - positions[i-1, :2])
        distances.append(distances[-1] + dist)
    
    ax6.plot(time_log, distances, 'c-', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Distance (m)')
    ax6.set_title('Total Distance Traveled')
    ax6.grid(True)
    
    # Slip statistics heatmap
    ax7 = fig.add_subplot(gs[2, 2])
    slip_stats = np.array([
        np.mean(np.abs(slip_array[:, i])) for i in range(6)
    ]).reshape(2, 3)
    
    im = ax7.imshow(slip_stats, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.3)
    ax7.set_xticks([0, 1, 2])
    ax7.set_xticklabels(['Front', 'Rear', 'Rocker'])
    ax7.set_yticks([0, 1])
    ax7.set_yticklabels(['Left', 'Right'])
    ax7.set_title('Average Slip by Wheel')
    
    for i in range(2):
        for j in range(3):
            text = ax7.text(j, i, f'{slip_stats[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax7, label='Avg |Slip|')
    
    plt.savefig('rocker_bogie_slip_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 Results saved to: rocker_bogie_slip_analysis.png")
    plt.show()


def run_simulation(xml_path: str, duration: float = 20000.0, log_cameras: bool = True):
    """Run rocker-bogie simulation with stability control, slip monitoring, and camera logging"""
    
    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Create controller
    controller = RockerBogieController(model, data)
    
    # Create camera recorder
    camera_recorder = None
    if log_cameras:
        try:
            camera_recorder = StereoCameraRecorder(model, width=640, height=480)
        except Exception as e:
            print(f"Camera recorder disabled: {e}")
    
    # Simulation parameters
    dt = model.opt.timestep
    steps = int(duration / dt)
    
    # Data logging
    time_log = []
    stability_log = []
    position_log = []
    slip_log = []
    
    print("\n" + "="*70)
    print("ROCKER-BOGIE STABILITY & SLIP CONTROL SIMULATION")
    print("="*70)
    print(f"Duration: {duration}s")
    print(f"Timestep: {dt}s")
    print(f"Total steps: {steps}")
    print(f"Camera logging: {'ON' if camera_recorder else 'OFF'}")
    print(f"Traction control: {'ON' if controller.velocity_controller.traction_control_enabled else 'OFF'}")
    print("="*70 + "\n")
    
    # Mission profile
    mission_phases = [
        (0.0, 2.0, 0.3, 0.0),    # Move forward slowly
        (2.0, 4.0, 0.6, 0.0),    # Increase speed (may induce slip)
        (4.0, 6.0, 0.5, 0.2),    # Turn right
        (6.0, 8.0, 0.5, -0.2),   # Turn left
        (8.0, 10.0, 0.0, 0.0),   # Stop
    ]
    
    start_time = time.time()
    camera_log_interval = 50  # Log cameras every 50 steps
    
    for step in range(steps):
        sim_time = step * dt
        
        # Update mission phase
        for t_start, t_end, v_linear, v_angular in mission_phases:
            if t_start <= sim_time < t_end:
                controller.set_velocity(v_linear, v_angular)
                break
        
        # Control + physics step
        controller.step(dt)
        mujoco.mj_step(model, data)
        
        # Log camera data
        if camera_recorder and step % camera_log_interval == 0:
            try:
                camera_recorder.capture(data, step // camera_log_interval)
            except Exception as e:
                print(f"Camera capture error: {e}")
        
        # Log telemetry
        if step % 100 == 0:
            state = controller.get_state()
            _, stability_margin, _ = controller.stability_monitor.check_stability(state)
            
            time_log.append(sim_time)
            stability_log.append(stability_margin)
            position_log.append(state.chassis_pos.copy())
            slip_log.append(state.slip_ratios.copy())
            
            roll, pitch, yaw = controller.stability_monitor.get_roll_pitch_yaw(state.chassis_quat)
            avg_slip = np.mean(np.abs(state.slip_ratios))
            max_slip = np.max(np.abs(state.slip_ratios))
            
            print(f"t={sim_time:5.2f}s | "
                  f"Pos: ({state.chassis_pos[0]:5.2f}, {state.chassis_pos[1]:5.2f}, {state.chassis_pos[2]:5.2f}) | "
                  f"Roll: {np.rad2deg(roll):5.1f}° | "
                  f"Pitch: {np.rad2deg(pitch):5.1f}° | "
                  f"Stability: {stability_margin:4.2f} | "
                  f"Slip(avg/max): {avg_slip:4.2f}/{max_slip:4.2f}")
    
    real_time = time.time() - start_time
    print(f"\n✓ Simulation completed in {real_time:.2f}s (real-time factor: {duration/real_time:.2f}x)")
    
    # Close camera recorder
    if camera_recorder:
        camera_recorder.close()
    
    # Plot results
    plot_results(time_log, stability_log, position_log, slip_log)
    
    # Print statistics
    slip_array = np.array(slip_log)
    print("\n" + "="*70)
    print("STABILITY & SLIP STATISTICS")
    print("="*70)
    print(f"Average stability margin: {np.mean(stability_log):.3f}")
    print(f"Minimum stability margin: {np.min(stability_log):.3f}")
    print(f"Stability maintained > 0.5: {100 * np.sum(np.array(stability_log) > 0.5) / len(stability_log):.1f}%")
    print(f"\nAverage slip ratio: {np.mean(np.abs(slip_array)):.3f}")
    print(f"Maximum slip ratio: {np.max(np.abs(slip_array)):.3f}")
    print(f"Time with excessive slip (>25%): {100 * np.sum(np.any(np.abs(slip_array) > 0.25, axis=1)) / len(slip_log):.1f}%")
    print(f"Time in optimal slip range (10-20%): {100 * np.sum(np.all((np.abs(slip_array) > 0.1) & (np.abs(slip_array) < 0.2), axis=1)) / len(slip_log):.1f}%")
    
    print(f"\nPer-wheel slip statistics:")
    wheel_names = ['Left Front', 'Left Rear', 'Left Rocker', 'Right Front', 'Right Rear', 'Right Rocker']
    for i, name in enumerate(wheel_names):
        avg_slip = np.mean(np.abs(slip_array[:, i]))
        max_slip = np.max(np.abs(slip_array[:, i]))
        print(f"  {name:15s}: avg={avg_slip:.3f}, max={max_slip:.3f}")
    
    if camera_recorder:
        print(f"\nCamera frames captured: {step // camera_log_interval}")
        print(f"Camera data saved to: {camera_recorder.save_dir}/")
    print("="*70 + "\n")
    
    return controller, time_log, stability_log, slip_log


if __name__ == "__main__":
    import sys
    
    xml_file = "rockie_bogie.xml"
    
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    
    print("\n🤖 Rocker-Bogie Stability & Slip Control System")
    print("With Stereo Camera Data Logging & Traction Control\n")
    
    try:
        controller, time_log, stability_log, slip_log = run_simulation(
            xml_file, 
            duration=10.0,
            log_cameras=True
        )
        
        # Additional slip analysis
        slip_array = np.array(slip_log)
        print("\n📈 SLIP ANALYSIS SUMMARY")
        print("="*70)
        print("Slip Ratio Interpretation:")
        print("  • 0.00 - 0.10: Minimal slip (underpowered)")
        print("  • 0.10 - 0.20: Optimal slip (good traction)")
        print("  • 0.20 - 0.30: Moderate slip (traction control active)")
        print("  • 0.30+      : Excessive slip (poor traction/spinning)")
        print("\nTraction Control Status:")
        if controller.velocity_controller.traction_control_enabled:
            print("  ✓ Traction control was ACTIVE during simulation")
            print("  ✓ Wheel velocities reduced when slip exceeded 25%")
        else:
            print("  ✗ Traction control was DISABLED")
        print("="*70)
        
    except FileNotFoundError:
        print(f"❌ Error: XML file '{xml_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)