import open3d as o3d
import numpy as np
from skimage import measure
from scipy.integrate import quad
from scipy.sparse import coo_matrix

import networkx as nx


# 读取一个 网格 文件
def read_point_cloud(filename: str):
    # 读取OBJ文件
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()  # 确保法向量被计算

    # 从三角形网格中提取顶点作为点云
    points = mesh.vertices
    normals = mesh.vertex_normals  # 提取法向量
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)  # 设置点云的法向量

    # 如果你的模型包含顶点颜色，并希望一同显示，可以将颜色信息也转移给点云对象
    # 注意：这一步是可选的，仅在OBJ文件中确实有颜色信息时适用
    if mesh.vertex_colors:
        point_cloud.colors = mesh.vertex_colors
    return point_cloud


# 构造基函数
def base_function(input_value: float):
    if -1.5 <= input_value <= -0.5:
        return 9./8. + 1.5 * input_value + 0.5 * input_value ** 2
    elif -0.5 < input_value <= 0.5:
        return 3./4. - input_value ** 2
    elif 0.5 < input_value <= 1.5:
        return 9./8. - 1.5 * input_value + 0.5 * input_value ** 2
    else:
        return 0


# 基函数导数
def base_function_gradient(input_value: float):
    if -1.5 <= input_value <= -0.5:
        return 1.5 + input_value
    elif -0.5 < input_value <= 0.5:
        return -2 * input_value
    elif 0.5 < input_value <= 1.5:
        return -1.5 + input_value
    else:
        return 0


# 基函数二次导
def base_function_hessian(input_value: float):
    if -1.5 <= input_value <= -0.5:
        return 1.
    elif -0.5 < input_value <= 0.5:
        return -2.
    elif 0.5 < input_value <= 1.5:
        return 1.
    else:
        return 0


# 建立积分索引，<fi,fj> 积分，<fi,fj'> 积分，<fi,fj''> 积分，只有五种可能性
# 建立数值索引，fi(x)
def integration_index(h):
    int_ff_idx = []
    int_fdf_idx = []
    int_fddf_idx = []
    value_idx = []
    for d in range(-2, 3):
        f = lambda x: base_function(x) * base_function(x + d)
        integral, _ = quad(func=f, a=-1.5, b=1.5)
        int_ff_idx.append(integral * h)

        f = lambda x: base_function(x) * base_function_gradient(x + d)
        integral, _ = quad(func=f, a=-1.5, b=1.5)
        int_fdf_idx.append(integral)

        f = lambda x: base_function(x) * base_function_hessian(x + d)
        integral, _ = quad(func=f, a=-1.5, b=1.5)
        int_fddf_idx.append(integral / h)

    return int_ff_idx, int_fdf_idx, int_fddf_idx


# 显示面，点对应的三角网格结果
def mesh_display(selected_verts, selected_faces):
    # 将结果转换为Open3D可用的格式
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(selected_verts)
    mesh.triangles = o3d.utility.Vector3iVector(selected_faces)

    # 计算顶点法线，以便更好地可视化
    mesh.compute_vertex_normals()

    # 可视化结果
    o3d.visualization.draw_geometries([mesh])


# 使用 marching cube 将体素值进行表示并显示
def marching_cubes_display(voxel_values: np.ndarray):

    # 假设 voxel_values 是包含体素中心示性函数值的三维数组
    # 例如，voxel_values = np.random.randn(nx, ny, nz)

    # 使用Marching Cubes算法提取零等值面
    vert, faces, _, _ = measure.marching_cubes(voxel_values, level=0)

    mesh_display(vert, faces)

    # 提取出连通集合
    # 构建图来表示三角形之间的连接关系
    G = nx.Graph()
    for face in faces:
        edges = [(face[i], face[(i + 1) % 3]) for i in range(3)]
        G.add_edges_from(edges)

    # 寻找图中的连通分支
    connected_components = list(nx.connected_components(G))
    sorted_components = sorted(connected_components, key=len, reverse=True)

    for connected_component in sorted_components[:2]:
        # 输出所有连通分支
        selected_verts = vert[list(connected_component)]

        # 重建面片索引，这需要将最大连通分支中的原始顶点索引映射到新索引
        vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(connected_component)}
        selected_faces = np.array(
            [[vertex_mapping[idx] for idx in face] for face in faces if set(face).issubset(connected_component)])

        mesh_display(selected_verts, selected_faces)


# 从体素网格内取到 i,j,k 表示的 向量下标
def flatten_idx_ijk(i, j, k, nx, ny, nz):
    idx = k + nz * (j + i * ny)
    return idx


# 从体素网格内取到 i,j,k 表示的 向量下标
def flatten_idx_ijk_1(idx_arr, nx, ny, nz):
    [i, j, k] = idx_arr
    idx = k + nz * (j + i * ny)
    return idx


def change_triplet_to_sparse_matrix(triplet, n_rows, n_cols):
    # 将三元组列表转换为行索引、列索引和值的三个数组
    rows, cols, data = zip(*triplet)

    # 将这些数组转换为NumPy数组
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)

    # 创建COO格式的稀疏矩阵
    sparse_matrix = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    return sparse_matrix


if __name__ == "__main__":
    int_ff_idx, int_fdf_idx, int_fddf_idx = integration_index(0.5)
    print(int_ff_idx, int_fdf_idx, int_fddf_idx)
