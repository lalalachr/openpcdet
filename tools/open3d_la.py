import open3d as o3d
import numpy as np

# 读取 .bin 文件并转换为点云
def read_bin_file(bin_file_path):
    # 加载点云数据
    point_cloud = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud

# 可视化点云数据
def visualize_point_cloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # 只使用 x, y, z 三个维度
    o3d.visualization.draw_geometries([pcd])

def main():
    bin_file_path = "../data/custom/training/mylidar/04088.bin"  # 你的 .bin 文件路径
    point_cloud = read_bin_file(bin_file_path)
    visualize_point_cloud(point_cloud)

if __name__ == "__main__":
    main()
