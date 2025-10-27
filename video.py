import os
import imageio.v2 as imageio  # avoids warning

def images_to_video(image_folder, output_path, fps=30):
    images = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    if not images:
        raise ValueError("No images found")

    writer = imageio.get_writer(output_path, fps=fps, codec='mpeg4', quality=8)
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        image = imageio.imread(img_path)
        writer.append_data(image)
    writer.close()
    print(f"Video saved to {output_path}")

# Example:
images_to_video("stereo_output/camera_right/rgb", "camera_right.avi", fps=30)
