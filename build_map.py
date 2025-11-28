import os
import numpy as np
import imageio
import open3d as o3d
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def load_intrinsics(K_file):
    """Load 3x3 intrinsic matrix from txt"""
    return np.loadtxt(K_file)

def load_pose(pose_file):
    """Load 4x4 camera pose matrix from txt"""
    with open(pose_file, "r") as f:
        lines = f.readlines()
    # Ignore lines starting with #
    data = [list(map(float, line.split())) for line in lines if not line.startswith("#")]
    return np.array(data)


def world2cam_to_cam2world(RT_w2c: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 world-to-camera homogeneous transformation matrix 
    to a camera-to-world matrix.

    Args:
        RT_w2c (np.ndarray): 4x4 world-to-camera matrix.

    Returns:
        np.ndarray: 4x4 camera-to-world matrix.
    """
    if RT_w2c.shape != (4, 4):
        raise ValueError("Input RT_w2c must be a 4x4 matrix")

    R = RT_w2c[:3, :3]
    t = RT_w2c[:3, 3]

    RT_c2w = np.eye(4)
    RT_c2w[:3, :3] = R.T
    RT_c2w[:3, 3] = -R.T @ t

    return RT_c2w


def build_point_cloud_from_depth(rgb_path, depth_path, K, pose):
    """
    Convert a single RGB + depth frame to a point cloud in world coordinates
    rgb_path: path to RGB PNG
    depth_path: path to depth EXR
    K: 3x3 camera intrinsic
    pose: 4x4 camera pose (world) (world2cam)
    """
    # Load images
    rgb = imageio.imread(rgb_path) / 255.0  # normalize to [0,1]
    # depth = imageio.imread(depth_path).astype(np.float32)
    depth  = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # breakpoint()

    H, W = depth.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # Generate pixel coordinates
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u,v)
    
    # Convert depth to camera coordinates
    X = (uu - cx) * depth / fx
    Y = (vv - cy) * depth / fy
    Z = depth

    pts_cam = np.stack([X,Y,Z], axis=-1).reshape(-1,3)

    # Transform points to world coordinates
    pose = world2cam_to_cam2world(RT_w2c = pose)
    R = pose[:3,:3]
    t = pose[:3,3]
    pts_world = (R @ pts_cam.T).T + t

    # Flatten colors
    colors = rgb.reshape(-1,3)

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def merge_logged_data(log_dir):
    """
    Merge all logged frames into one point cloud.
    Assumes directory structure:
        log_dir/
            rgb/camera_left/*.png
            rgb/camera_right/*.png
            depth/camera_left/*.exr
            depth/camera_right/*.exr
            poses/camera_left.txt
            poses/camera_right.txt
            intrinsics/camera_left.txt
            intrinsics/camera_right.txt
    """
    pcd_all = o3d.geometry.PointCloud()

    cameras = [d for d in os.listdir(os.path.join(log_dir, "rgb")) if os.path.isdir(os.path.join(log_dir, "rgb", d))]
    
    for cam in cameras:
        rgb_dir = os.path.join(log_dir, "rgb", cam)
        depth_dir = os.path.join(log_dir, "depth", cam)
        pose_file = os.path.join(log_dir, "poses", f"{cam}.txt")
        K_file = os.path.join(log_dir, "intrinsics", f"{cam}.txt")
        # /Users/denismbeyakola/Desktop/space_robotics/video_output/depth/camera_left/000000.exr

        K = load_intrinsics(K_file)
        poses = []
        with open(pose_file,"r") as f:
            frame_lines = []
            for line in f:
                if line.startswith("#"):
                    continue
                if line.strip() == "":
                    if frame_lines:
                        poses.append(np.array([list(map(float, l.split())) for l in frame_lines]))
                        frame_lines = []
                else:
                    frame_lines.append(line)
            if frame_lines:
                poses.append(np.array([list(map(float,l.split())) for l in frame_lines]))

        rgb_files = sorted(os.listdir(rgb_dir))
        depth_files = sorted(os.listdir(depth_dir))

        for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
            if i%40 ==  0:
                rgb_path = os.path.join(rgb_dir, rgb_file)
                depth_path = os.path.join(depth_dir, depth_file)
                pose = poses[i]
                pcd = build_point_cloud_from_depth(rgb_path, depth_path, K, pose)
                pcd_all += pcd

    return pcd_all

if __name__ == "__main__":
    log_dir = "video_output"  # change to your logged data directory
    pcd = merge_logged_data(log_dir)
    print(f"Total points: {len(pcd.points)}")

    # Visualize
    o3d.visualization.draw_geometries([pcd])

    # Optionally, save
    o3d.io.write_point_cloud(os.path.join(log_dir,"map.ply"), pcd)
    print(f"Saved map.ply in {log_dir}")
