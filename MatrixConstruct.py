from ReadPointCloud import *


def construct_trilinear_weight(point_arr: np.ndarray,
                               voxel_pt_arr: np.ndarray, voxel_indices_arr: np.ndarray,
                               nx, ny, nz, pace: float):
    pt_num = point_arr.shape[0]

    triplet = []
    # 计算出点列相对于左下角的相对位置
    for pt_id, pt in enumerate(point_arr):
        # point 所在的 voxel 三维索引
        pt_in_voxel_id = voxel_indices_arr[pt_id, :]
        [i, j, k] = pt_in_voxel_id
        voxel_id = flatten_idx_ijk_1(pt_in_voxel_id, nx, ny, nz)

        # voxel 左下角坐标
        voxel_pt = voxel_pt_arr[voxel_id, :]

        [c_x, c_y, c_z] = (pt - voxel_pt) / pace

        # 计算三线性插值系数
        triplet.append((pt_id, flatten_idx_ijk(i, j, k, nx, ny, nz), (1 - c_x) * (1 - c_y) * (1 - c_z)))
        triplet.append((pt_id, flatten_idx_ijk(i + 1, j, k, nx, ny, nz), c_x * (1 - c_y) * (1 - c_z)))
        triplet.append((pt_id, flatten_idx_ijk(i, j + 1, k, nx, ny, nz), (1 - c_x) * c_y * (1 - c_z)))
        triplet.append((pt_id, flatten_idx_ijk(i, j, k + 1, nx, ny, nz), (1 - c_x) * (1 - c_y) * c_z))
        triplet.append((pt_id, flatten_idx_ijk(i + 1, j, k + 1, nx, ny, nz), c_x * (1 - c_y) * c_z))
        triplet.append((pt_id, flatten_idx_ijk(i, j + 1, k + 1, nx, ny, nz), (1 - c_x) * c_y * c_z))
        triplet.append((pt_id, flatten_idx_ijk(i + 1, j + 1, k, nx, ny, nz), c_x * c_y * (1 - c_z)))
        triplet.append((pt_id, flatten_idx_ijk(i + 1, j + 1, k + 1, nx, ny, nz), c_x * c_y * c_z))

    # # 将三元组列表转换为行索引、列索引和值的三个数组
    # rows, cols, data = zip(*triplet)
    #
    # # 将这些数组转换为NumPy数组
    # rows = np.array(rows)
    # cols = np.array(cols)
    # data = np.array(data)
    #
    # n_rows = pt_num
    # n_cols = nx * ny * nz
    #
    # # 创建COO格式的稀疏矩阵
    # sparse_matrix = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    return change_triplet_to_sparse_matrix(triplet, pt_num, nx * ny * nz)


# 构建 F1, 梯度F2 的内积矩阵，总共返回三个的 nx*ny*nz 矩阵; 同时构建 F1，L(F2) 的内积矩阵
def construct_dot_matrix_1(nx, ny, nz, h, int_ff_idx, int_fdf_idx, int_fddf_idx):
    dim = nx * ny * nz
    # 存放 F1, 梯度F2 的内积矩阵
    matrix_x_dir = []
    matrix_y_dir = []
    matrix_z_dir = []

    # 构建 F1，DD(F2) 的内积矩阵，即最终 AX=b 里的 A
    coef_matrix = []

    # 构建直接计算值的矩阵，最终与 x 乘在一起得到每个体素网格上的取值
    value_matrix = []

    # 每个体素网格的二次 b 样条值实际上都一样
    value_pt = base_function(0.5)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                elem_idx = flatten_idx_ijk(i, j, k, nx, ny, nz)

                # 从中取出积分不为零的几项，不超过 125 个
                for l in range(max(0, i - 2), min(nx, i + 3)):
                    for m in range(max(0, j - 2), min(ny, j + 3)):
                        for n in range(max(0, k - 2), min(nz, k + 3)):
                            # 要处理 elem_idx, target_idx 对应的基函数的内积
                            target_idx = flatten_idx_ijk(l, m, n, nx, ny, nz)

                            # f,f 积分结果
                            int_result_f_f_x = int_ff_idx[l - i + 2]
                            int_result_f_f_y = int_ff_idx[m - j + 2]
                            int_result_f_f_z = int_ff_idx[n - k + 2]

                            # f,df 积分结果
                            int_result_f_df_x = int_fdf_idx[l - i + 2]
                            int_result_f_df_y = int_fdf_idx[m - j + 2]
                            int_result_f_df_z = int_fdf_idx[n - k + 2]

                            # f,ddf 积分结果
                            int_result_f_ddf_x = int_fddf_idx[l - i + 2]
                            int_result_f_ddf_y = int_fddf_idx[m - j + 2]
                            int_result_f_ddf_z = int_fddf_idx[n - k + 2]

                            matrix_x_dir.append((elem_idx, target_idx,
                                                 int_result_f_df_x * int_result_f_f_y * int_result_f_f_z))
                            matrix_y_dir.append((elem_idx, target_idx,
                                                 int_result_f_df_y * int_result_f_f_x * int_result_f_f_z))
                            matrix_z_dir.append((elem_idx, target_idx,
                                                 int_result_f_df_z * int_result_f_f_x * int_result_f_f_y))

                            value = (int_result_f_ddf_x * int_result_f_f_y * int_result_f_f_z +
                                     int_result_f_ddf_y * int_result_f_f_x * int_result_f_f_z +
                                     int_result_f_ddf_z * int_result_f_f_x * int_result_f_f_y)
                            coef_matrix.append((elem_idx, target_idx, value))

                for l in range(max(0, i - 1), i + 1):
                    for m in range(max(0, j - 1), j + 1):
                        for n in range(max(0, k - 1), k + 1):
                            # 要处理 elem_idx, target_idx 对应的基函数的内积
                            target_idx = flatten_idx_ijk(l, m, n, nx, ny, nz)

                            value = (value_pt / h) ** 3
                            value_matrix.append((elem_idx, target_idx, value))


    # 将三元组列表转换为行索引、列索引和值的三个数组
    matrix_x_dir = change_triplet_to_sparse_matrix(matrix_x_dir, dim, dim)
    matrix_y_dir = change_triplet_to_sparse_matrix(matrix_y_dir, dim, dim)
    matrix_z_dir = change_triplet_to_sparse_matrix(matrix_z_dir, dim, dim)
    coef_matrix = change_triplet_to_sparse_matrix(coef_matrix, dim, dim)
    value_matrix = change_triplet_to_sparse_matrix(value_matrix, dim, dim)
    return matrix_x_dir, matrix_y_dir, matrix_z_dir, coef_matrix,value_matrix
