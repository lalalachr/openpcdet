import numpy as np
import struct
import open3d

def read_bin_velodyne(path):
    '''read bin file and transfer to array data'''
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)

def main():
    pc_path='/media/zyj/T7/mydata/custom/custom_gt_database/00000_ech_0.bin'
    # pc_path = '/media/zyj/T7/mydata/custom/training/mylidar/00000_ech_0.bin'
    # example=read_bin_velodyne(pc_path)
    example = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape(-1, 4)
    example_xyz=example[:,:3]
    example_xyz=example_xyz[example_xyz[:,2]>-3]

    num_points = len(example_xyz)
    pcd = open3d.open3d.geometry.PointCloud()
    pcd.points= open3d.open3d.utility.Vector3dVector(example_xyz)
    red_color = [1,0,0]
    colors = np.array([red_color for _ in range(num_points)])
    pcd.colors = open3d.utility.Vector3dVector(colors)

    # From numpy to Open3D
    vis_ = open3d.visualization.Visualizer()
    vis_.create_window()
    vis_.add_geometry(pcd)
    render_options = vis_.get_render_option()
    render_options.point_size = 0.01
    render_options.background_color = np.array([0, 0, 0])
    vis_.run()
    vis_.destroy_window()
    # pcd.points= open3d.open3d.utility.Vector3dVector(example)
    # open3d.open3d.visualization.draw_geometries([pcd])

if __name__=="__main__":
    main()

