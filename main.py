from MatrixConstruct import *
from scipy.sparse.linalg import cg

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    filename = "./example/bunny2.ply"
    pcd = read_point_cloud(filename=filename)

    # 显示点云及其法向量
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # 处理 pcd
    # 创建体素
    P = np.asarray(pcd.points)
    N = np.asarray(pcd.normals)

    # 接下来的步骤与之前相同，计算边界框和网格参数
    bound_box_size = np.max(P.max(axis=0) - P.min(axis=0))  # 计算 max(xmax-xmin,ymax-ymin,zmax-zmin)
    pad = 8
    h = bound_box_size / 20  # 步长大小，对最大包围盒进行 20 划分，同时也可往外扩大 pad
    corner = P.min(axis=0) - pad * h  # 网格左下角，x,y,z 较小的角

    # 计算每个维度上的格子数
    # 创建一个 nx*ny*nz 的网格，大小为 h
    nx = int(np.ceil((P[:, 0].max() - P[:, 0].min()) / h)) + 2 * pad
    ny = int(np.ceil((P[:, 1].max() - P[:, 1].min()) / h)) + 2 * pad
    nz = int(np.ceil((P[:, 2].max() - P[:, 2].min()) / h)) + 2 * pad

    # 生成体素网格左下点
    voxel_pt = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                center = corner + np.array([i, j, k]) * h
                voxel_pt.append(center)

    voxel_pt = np.array(voxel_pt)

    # 计算每个点所在的体素索引
    voxel_indices = (np.floor((P - corner) / h)).astype(int)

    # 计算向量场在边界附近的向量 V 的系数
    trilinear_weight = construct_trilinear_weight(P, voxel_pt, voxel_indices, nx, ny, nz, h)
    trilinear_weight = trilinear_weight.T
    # 得到了还没做内积计算的 nx*ny*nz*3 的系数 alpha * s.N，可拆成三维的 nx*ny*nz*1 向量
    # b_pre = trilinear_weight.dot(N)

    # 得到积分表
    int_ff_idx, int_fdf_idx, int_fddf_idx = integration_index(h)

    # 计算所有内积系数矩阵
    matrix_x_dir, matrix_y_dir, matrix_z_dir, coef_matrix, value_matrix = construct_dot_matrix_1(nx, ny, nz, h,
                                                                                   int_ff_idx, int_fdf_idx, int_fddf_idx)
    # 得到右侧向量值 b
    b = ((matrix_x_dir * trilinear_weight) * N[:, 0] +
         (matrix_y_dir * trilinear_weight) * N[:, 1] +
         (matrix_z_dir * trilinear_weight) * N[:, 2])

    # 计算 x 值，利用 ATAx=ATb
    coef_matrix = coef_matrix.T * coef_matrix
    b = coef_matrix.T * b
    x, exit_code = cg(coef_matrix, b)

    # 此时计算每个体素点处的实行函数值
    voxel_value = value_matrix * x
    voxel_value = voxel_value.reshape((nx, ny, nz))

    marching_cubes_display(voxel_value)
