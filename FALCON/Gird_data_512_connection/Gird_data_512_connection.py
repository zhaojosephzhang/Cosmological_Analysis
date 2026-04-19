"""Construct a stitched light cone with the parallel rotating-LoS method.

This script is extracted from Gird_data_512_connection.ipynb and retains only
the data-generation path plus HDF5 export. Diagnostic plotting is intentionally
left in the notebook version.
"""

# 基础库
import os
import re
import time
import sys  # 保留 sys 库
import subprocess
import random
import copy
import pickle
import gc
import warnings
import math
import bisect

# 科学计算和数据处理库
import numpy as np
from numpy import array, sqrt, sin, cos, tan, histogram2d

# h5py 库用于处理 HDF5 数据
import h5py

# Matplotlib 用于可视化
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import units
from matplotlib.gridspec import GridSpec

# Astropy 库和常量
import astropy
from astropy.io import ascii
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import G, c, m_e, m_p

# SciPy 库，用于数值计算和数学运算
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.spatial import cKDTree
from scipy.stats import lognorm

# 多进程和并行计算
import multiprocessing
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
from tqdm import tqdm

# 3D 绘图
from mpl_toolkits.mplot3d import Axes3D

# 设置宇宙学常量
H0 = cosmo.H0

# 设置默认值
default_level_input = "8"
default_AGN_name = "f"
default_X_num = 100
default_Y_num = 100
default_LC_bins = 400
default_threshold = 1e-4
default_vectors = "on"
# 过滤无关参数，保留不以 '--' 开头的参数
args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

# 定义函数来获取参数
def get_int_arg(args, index, default):
    try:
        return int(args[index])
    except (IndexError, ValueError):
        return default

def get_float_arg(args, index, default):
    try:
        return float(args[index])
    except (IndexError, ValueError):
        return default
def get_str_arg(args, index, default):
    try:
        return str(args[index])
    except (IndexError, ValueError):
        return default
# 提取参数
level_input = args[0] if len(args) > 0 else default_level_input
AGN_name = args[1] if len(args) > 1 else default_AGN_name
X_num = get_int_arg(args, 2, default_X_num)
Y_num = get_int_arg(args, 3, default_Y_num)
LC_bins = get_int_arg(args, 4, default_LC_bins)
threshold = get_float_arg(args, 5, default_threshold)
threshold_name = get_str_arg(args, 5, default_threshold)
vectors_switch = args[0] if len(args) > 0 else default_vectors
# 打印参数验证
print(f"Level Input: {level_input}")
print(f"AGN Name: {AGN_name}")
print(f"X Num: {X_num}")
print(f"Y Num: {Y_num}")
print(f"LC Bins: {LC_bins}")
print(f"Threshold: {threshold}")
Boxname = "100"
#AGN_info = "NoBH"
Bin_num = 2**int(level_input)
# 解析 level 参数
level = f"lv{level_input}"  # 转换为 "lv9"

# 根据 AGN_name 确定 AGN_info
AGN_info_dict = {
    "f": "Fiducial",
    "n": "NoBH"
    # 其他映射规则可以根据需要添加
}
AGN_info = AGN_info_dict.get(AGN_name, "Unknown")

font_label = {
    'family': 'serif',
    'size': 25
}

tick_size = 25

font_legend= {
    'family': 'serif',
    'size': 20
}
mp_value = m_p.to('g').value
# 宇宙学参数
Omega_0 = 0.3090  # 物质密度参数
Omega_Lambda = 0.6910  # 暗能量密度参数
H_0 = 67.66  # 哈勃常数，单位为 km/s/Mpc
c = 299792.458  # 光速，单位为 km/s

# 计算共动距离和尺度因子的函数
def comoving_distance(z, Omega_0 = 0.3090, Omega_Lambda = 0.6910, H_0 = 67.66):
    integrand = lambda zp: 1.0 / np.sqrt(Omega_0 * (1.0+zp)**3 + Omega_Lambda)
    integral, _ = quad(integrand, 0, z)
    return c/H_0 * integral

def scale_factor(z):
    return 1.0 / (1.0 + z)

# 已知共动距离，求红移和尺度因子的函数
def find_redshift_and_scale_factor(dC):
    # 定义被求解的方程
    equation = lambda z: comoving_distance(z) - dC

    # 使用root_scalar函数求解方程
    sol = root_scalar(equation, bracket=[0, 10])

    # 检查解是否收敛，如果不收敛则返回None
    if not sol.converged:
        return None, None

    # 返回红移和尺度因子
    z = sol.root
    a = scale_factor(z)
    return z, a
def compute_electron_density(gas_density, electron_abundance):
    return gas_density * 6.77e-31 * electron_abundance / mp_value

def find_nearest_grid_points(kdtree, lc_positions):
    # 使用 KDTree 查找最近的 grid cell
    _, indices = kdtree.query(lc_positions, k=1)  # 找到最近的一个网格单元
    return indices
def process_snapshot(file_path):
    with h5py.File(file_path, 'r') as file:
        # 读取 GridPos 和其他数据
        gridpos_data = file['/Cell/GridPos'][:]
        electron_abundance_data = file['/Cell/ElectronAbundance'][:]
        gas_density_data = file['/Cell/GasDensity'][:]
        Boxsize = file['Parameters'].attrs['BoxSize']
        # 读取缩放因子
        a = file['/Header'].attrs['Time']
        h = file['Parameters'].attrs['HubbleParam']
        a_scaling = file['/Cell/GasDensity'].attrs['a_scaling']
        h_scaling = file['/Cell/GasDensity'].attrs['h_scaling']
        # 计算电子数密度
        print("a:", a, a_scaling)
        print("h", h, h_scaling)
        electron_density_data = compute_electron_density(gas_density_data, electron_abundance_data)*h**h_scaling
        #electron_density = gas_density_data*6.77e-31 * electron_abundance_data / mp
        # 输出数据形状和前几行数据来验证
        #print("GridPos ple of ElectronAbundance data:", electron_abundance_data[:5])
        #print("Sadata shape:", gridpos_data.shape)
        #print("ElectronAbundance data shape:", electron_abundance_data.shape)
        #print("GasDensity data shape:", gas_density_data.shape)
        #print("Sample of GridPos data:", gridpos_data[:5])
        #print("Sammple of GasDensity data:", gas_density_data[:5])
        print("Boxsize:", Boxsize)
        sys.stdout.flush()
        # 创建 KD 树
        kdtree = cKDTree(gridpos_data)

        return kdtree, electron_density_data, Boxsize

def process_snapshot_connection(file_path_low, file_path_up):
    with h5py.File(file_path_low, 'r') as file:
        # 读取 GridPos 和其他数据
        gridpos_data_low = file['/Cell/GridPos'][:]
        electron_abundance_data_low = file['/Cell/ElectronAbundance'][:]
        gas_density_data_low = file['/Cell/GasDensity'][:]
        Boxsize_low = file['Parameters'].attrs['BoxSize']
        # 读取缩放因子
        a_low = file['/Header'].attrs['Time']
        h_low = file['Parameters'].attrs['HubbleParam']
        a_scaling = file['/Cell/GasDensity'].attrs['a_scaling']
        h_scaling = file['/Cell/GasDensity'].attrs['h_scaling']
        # 计算电子数密度
        print("a_low:", a_low, a_scaling)
        print("h_low", h_low, h_scaling)
        electron_density_data_low = compute_electron_density(gas_density_data_low, electron_abundance_data_low)*a_low**a_scaling*h_low**h_scaling
        #electron_density = gas_density_data*6.77e-31 * electron_abundance_data / mp
        # 输出数据形状和前几行数据来验证
        #print("GridPos ple of ElectronAbundance data:", electron_abundance_data[:5])
        #print("Sadata shape:", gridpos_data.shape)
        #print("ElectronAbundance data shape:", electron_abundance_data.shape)
        #print("GasDensity data shape:", gas_density_data.shape)
        #print("Sample of GridPos data:", gridpos_data[:5])
        #print("Sammple of GasDensity data:", gas_density_data[:5])
        print("Boxsize:", Boxsize_low)
        print(f"gridpos_data_low top10:{gridpos_data_low[:10]}")
        sys.stdout.flush()
        # 创建 KD 树
    with h5py.File(file_path_up, 'r') as file:
        # 读取 GridPos 和其他数据
        gridpos_data_up = file['/Cell/GridPos'][:]
        electron_abundance_data_up = file['/Cell/ElectronAbundance'][:]
        gas_density_data_up = file['/Cell/GasDensity'][:]
        Boxsize_up = file['Parameters'].attrs['BoxSize']
        # 读取缩放因子
        a_up = file['/Header'].attrs['Time']
        h_up = file['Parameters'].attrs['HubbleParam']
        a_scaling = file['/Cell/GasDensity'].attrs['a_scaling']
        h_scaling = file['/Cell/GasDensity'].attrs['h_scaling']
        # 计算电子数密度
        print("a_up:", a_up, a_scaling)
        print("h_up", h_up, h_scaling)
        electron_density_data_up = compute_electron_density(gas_density_data_up, electron_abundance_data_up)*a_up**a_scaling*h_up**h_scaling
        #electron_density = gas_density_data*6.77e-31 * electron_abundance_data / mp
        # 输出数据形状和前几行数据来验证
        #print("GridPos ple of ElectronAbundance data:", electron_abundance_data[:5])
        #print("Sadata shape:", gridpos_data.shape)
        #print("ElectronAbundance data shape:", electron_abundance_data.shape)
        #print("GasDensity data shape:", gas_density_data.shape)
        #print("Sample of GridPos data:", gridpos_data[:5])
        #print("Sammple of GasDensity data:", gas_density_data[:5])
        print("Boxsize_up:", Boxsize_up)
        print(f"gridpos_data_low top10:{gridpos_data_up[:10]}")
        sys.stdout.flush()

    #kdtree = cKDTree(gridpos_data)
    return kdtree, electron_density_data, Boxsize

def assign_density_to_lc_bins(LC_shifted, kdtree, electron_density_data):
    # 将 LC_shifted 变为二维数组，以便于 KDTree 查询
    LC_shifted_flat = LC_shifted.reshape(-1, 3)
    
    # 找到每个 bin 的最近网格点索引
    indices = find_nearest_grid_points(kdtree, LC_shifted_flat)
    
    # 根据索引赋值电子数密度
    electron_density_in_lc = electron_density_data[indices]
    
    # 将结果还原为 LC_shifted 的形状
    ne_LC_shifted = electron_density_in_lc.reshape(LC_shifted.shape[:2])
    return ne_LC_shifted 

# 并行化 assign_density_to_lc_bins 的计算
def parallel_assign_density_to_lc_bins(LC_shifted, kdtree, electron_density_data, n_jobs=-1):
    LC_shifted_flat = LC_shifted.reshape(-1, 3)
    indices = Parallel(n_jobs=n_jobs)(delayed(kdtree.query)(pos) for pos in LC_shifted_flat)
    indices = np.array([i[1] for i in indices])  # 获取索引
    electron_density_in_lc = electron_density_data[indices]
    return electron_density_in_lc.reshape(LC_shifted.shape[:2])
# 并行化 calculate_DM_IGM 的计算
def parallel_calculate_DM_IGM(ne_LC_shifted, z_list, dz, DM_IGM_LC_tot, h, h_scaling, n_jobs=-1, c_value = 299792.458):
    LC_num = ne_LC_shifted.shape[0]
    
    def calculate_row(i):
        DM_row = np.zeros(ne_LC_shifted.shape[1])
        for j in range(ne_LC_shifted.shape[1] - 1):
            if j == 0:
                DM_row[0] = c_value / H_0 * ne_LC_shifted[i][0] * (1 + z_list[0]) * dz[0] / np.sqrt(Omega_0 * (1 + z_list[0])**3 + Omega_Lambda) * 1e6
            else:
                DM_row[j] = DM_row[j - 1] + c_value / H_0 * ne_LC_shifted[i][j] * (1 + z_list[j]) * dz[j] / np.sqrt(Omega_0 * (1 + z_list[j])**3 + Omega_Lambda) * 1e6
        DM_row /= h  # 如果需要，转化为物理单位
        return DM_row
    
    DM_IGM_LC = Parallel(n_jobs=n_jobs)(delayed(calculate_row)(i) for i in range(LC_num))
    return np.array(DM_IGM_LC)

# 定义计算单条观测线的 DM 函数
def calculate_DM_IGM(ne_LC_shifted, z_list, dz, DM_IGM_LC_tot, h=0.6774, h_scaling=1, c_value=299792.458):
    LC_num, DM_IGM_num = ne_LC_shifted.shape
    DM_IGM_LC = np.zeros((LC_num, DM_IGM_num))
    
    # 检查长度一致性
    min_length = min(DM_IGM_num, len(z_list), len(dz))
    z_list = z_list[:min_length]
    dz = dz[:min_length]
    
    for i in range(LC_num):
        for j in range(min_length - 1):
            if len(DM_IGM_LC_tot) == 0:
                DM_IGM_LC[i][0] = (
                    c_value / H_0 * ne_LC_shifted[i][0] * (1 + z_list[0]) * dz[0] 
                    / np.sqrt(Omega_0 * (1 + z_list[0])**3 + Omega_Lambda) * 1e6
                )
            else:
                DM_IGM_LC[i][0] = (
                    DM_IGM_LC_tot[-1][i][-1] + c_value / H_0 * ne_LC_shifted[i][0] * (1 + z_list[0]) * dz[0] 
                    / np.sqrt(Omega_0 * (1 + z_list[0])**3 + Omega_Lambda) * 1e6
                )
            DM_IGM_LC[i][j+1] = (
                DM_IGM_LC[i][j] + c_value / H_0 * ne_LC_shifted[i][j+1] * (1 + z_list[j+1]) * dz[j+1] 
                / np.sqrt(Omega_0 * (1 + z_list[j+1])**3 + Omega_Lambda) * 1e6
            )
    return DM_IGM_LC

# 打开 HDF5 文件
file_path = f'/home/zhaozhang/local/Grid_data/L100N1024_{AGN_info}/'
storage_path = f"/home/zhaozhang/local/Grid_data/Grid_figure/L100N1024_{AGN_info}/"
if os.path.exists(storage_path) !=1:
    os.mkdir(storage_path)
file_N_min = 12
file_N_max = 20
gridfile = []
grid_data_name = [name for name in os.listdir(file_path) if re.match(f'fullbox_\\d+_{AGN_name}_lv{level_input}.hdf5', name)]
grid_numbers = [int(re.findall(r'\d+', name)[0]) for name in grid_data_name ]
grid_numbers.sort(reverse=True)
redshifts = []
prev_redshift = None 
for num in grid_numbers:
    if (num >= file_N_min) & (num<=file_N_max):
        grid_file = "fullbox_{:03d}_{}_lv{}.hdf5".format(num, AGN_name, level_input)
        grid_file_path = os.path.join(file_path, grid_file)
        with h5py.File(grid_file_path, 'r') as file:
            redshift = file["Header"].attrs["Redshift"]
            if prev_redshift is None or redshift - prev_redshift >= 0.1:
                gridfile.append(grid_file_path)
                redshifts.append(redshift)
                prev_redshift = redshift  # Update the previous redshift
for path, redshift in zip(gridfile, redshifts):
    print(redshift, path)
    sys.stdout.flush()
#z_max = np.max(redshifts)
z_max = 1.0
Snap_num = len(gridfile)
X_Bound_shift = 0
Y_Bound_shift = 0
Z_Bound_shift = 0
#X_num = 100
#Y_num = 100
LC_num = Y_num*X_num
n_bins = LC_bins
z_temp_tot = []
z_list_tot =  []
z_list_center =  []
a_list_tot = []
dz_tot = []
DM_IGM_num = n_bins
ne_LC_shifted_tot = []
ne_LC_Ion_shifted_tot = []
DM_IGM_LC_tot = []
Bin_centers_shifted_tot = []
Distacne = []
start_time = time.time() 
z_temp = 0 
LC_length_total = 0
index_z = 0
overlap_range = 5
mp = m_p.to(u.g).value
rank = 0
if vectors_switch == "on":
    vectors = [
        [1.2329498412532303, -1.6584734880051313, 1.0],
        [-1.1539290392325818, -1.0, -3.040251011741983],
        [1.744995976117328, 13.752138755257217, -1.0],
        [-1.0, 2.9812078515745277, -1.3489175116209866],
        [5.076528778669732, -1.0, -1.1361961103048872],
        [-6.813263281120863, 1.134827238165392, 1.0],
        [-1.0, -1.5290302860876819, -3.2819785904825287],
        [-40.0549738736401, 1.0, -1.3184549847052658],
        [1.0, -3.415390704583581, -2.0608445853097486],
        [1.0, 1.2628610975439722, 1.2395592438012262],
        [-1.0, -2.1188857980761937, 1.8534181650330381],
        [-25.22732855898138, -1.0, 2.2480963353240346],
        [-1.0, -1.0592479438301083, -3.5114161086979037],
        [-2.628661503957581, -5.443052749112832, -1.0],
        [-3.3680050405460173, 2.336377020616031, 1.0],
        [-4.350208614522061, -3.026052442657449, -1.0],
        [-1.644234746160297, 5.240466355192378, -1.0],
        [1.045461547790559, -3.678980348250516, -1.0],
        [1.692172470990982, -1.0, -11.013018646497054]
    ]
    vectors_index = 0
if vectors_switch == "off":
    vectors = []
for ii in range(Snap_num):
    with h5py.File(gridfile[ii], 'r') as file:
        a = file['/Header'].attrs['Time']
        h = file['Parameters'].attrs['HubbleParam']
        a_scaling = file['/Cell/GasDensity'].attrs['a_scaling']
        h_scaling = file['/Cell/GasDensity'].attrs['h_scaling']
        # 计算电子数密度
    kdtree, electron_density_data, Boxsize = process_snapshot(gridfile[ii])
    ne_LC_shifted = np.zeros([LC_num, n_bins])
    ne_LC_Ion_shifted = np.zeros([LC_num, n_bins])
    DM_IGM_LC = np.zeros([LC_num, DM_IGM_num]) 
    Lbox =  Boxsize/h
    L = Lbox
    print("Start the DM calculate part of {}".format(gridfile[ii]))
    z_low = redshifts[ii]
    #z_up = redshifts[ii+1]
    z_up = redshifts[ii + 1] if ii + 1 < Snap_num else z_max
    # 达到高红移处，对两个BOX进行重叠处理
    random_num_0_2 = 0
    m = 0
    n = 0
    o = 0
    while (z_temp  < z_up):
        if z_temp < z_max:
            # 当前 Box 内的 DM 计算
            print(f"Processing within Box: z_temp = {z_temp}, z_up = {z_up}")
            sys.stdout.flush()
            if rank == 0: 
                print(f"rank {rank}: z_temp<z_up")
                sys.stdout.flush()
                print(f"rank {rank}: z_temp = {z_temp}")
                sys.stdout.flush()
                print(f"rank {rank}: z_up = {z_up}")
                sys.stdout.flush()
            if rank == 0:     
                if vectors_switch == "off":
                    d_index = np.zeros(3)
                    d_index[0] = random.uniform(0, 1) if random.random() < 0.5 else random.uniform(-1, 0)
                    d_index[1] = random.uniform(0, 1) if random.random() < 0.5 else random.uniform(-1, 0)
                    d_index[2] = random.uniform(0, 1) if random.random() < 0.5 else random.uniform(-1, 0)
                    random_num_0_2 = random.randint(0, 2)
                    d_index[random_num_0_2] = 1*np.sign(d_index[random_num_0_2])
                    major_axis = d_index[random_num_0_2]
                    m = 1/d_index[0]
                    n = 1/d_index[1]
                    o = 1/d_index[2]
                    vectors.append([m, n, o])
                if vectors_switch == "on":
                    m = vectors[vectors_index][0]
                    n = vectors[vectors_index][1]
                    o = vectors[vectors_index][2]
                    vectors_index += 1
            #random_num_0_2 = comm.bcast(random_num_0_2 if rank == 0 else None, root=0)
            #m = comm.bcast(m if rank == 0 else None, root=0)
            #n = comm.bcast(n if rank == 0 else None, root=0)
            #o = comm.bcast(o if rank == 0 else None, root=0)
            #major_axis = comm.bcast(major_axis if rank == 0 else None, root=0)
            print(f"m ,n, o: {m, n ,o}")
            sys.stdout.flush()
            LOS = np.array([L/m, L/n, L/o])
            nmo = n * m * o
            mo = m*o
            no = n*o
            nm = m*n
            #image = np.array([n*L, m*L, nm*L])
            print("m,n,o, LOS = ", m, n, o, LOS)
            sys.stdout.flush()  
            # Calculate unit vectors in new coordinate system 
            u3 = (np.array([no, mo, nm]) / (no**2 + mo**2 + nm**2)**0.5)
            axis = np.argmin([no, mo, nm])
            u1 = np.zeros(3)
            u1[axis] = 1
            u1 = np.cross(u1, u3)
            u1 /= np.linalg.norm(u1)
            u2 = np.cross(u3, u1)
            u2 /= np.linalg.norm(u2)

            LC_length = sqrt(sum(LOS**2))
            Bin_edges = np.linspace(0, LC_length, n_bins+1)
            Bin_edges_shifted = np.linspace(LC_length_total, LC_length_total+LC_length, n_bins+1)
            Bin_centers = (Bin_edges[:-1] + Bin_edges[1:]) / 2
            Bin_centers_shifted = (Bin_edges_shifted[:-1] + Bin_edges_shifted[1:]) / 2
            Bin_centers_shifted_tot.append(Bin_centers_shifted)
            Bin_centers_pos = np.array([u3*x for x in Bin_centers])
            if rank == 0: 
                print('Bin_edges_bound = ', Bin_edges_shifted[0], Bin_edges_shifted[-1])
                sys.stdout.flush()
                #print("  Bin_centers  =",   Bin_centers_shifted)
                #sys.stdout.flush()
                print("  LC_length  =",   LC_length )
                sys.stdout.flush()
            X_shift = np.linspace(0, Lbox, X_num+1)
            Y_shift = np.linspace(0, Lbox, Y_num+1)
            LC_origin = np.zeros([(X_num)*(Y_num),3])
            for i in range(X_num):
                for j in range(Y_num):
                    LC_origin[i*Y_num+j, 0] = X_shift[i] + X_Bound_shift
                    LC_origin[i*Y_num+j, 1] = Y_shift[j] + Y_Bound_shift
                    LC_origin[i*Y_num+j, 2] = Z_Bound_shift
            LC_shifted = np.array([Bin_centers_pos+x for x in LC_origin])
            # Apply periodic boundary conditions to the x and y coordinates
            LC_shifted[:, :, 0] %= Lbox
            LC_shifted[:, :, 1] %= Lbox
            LC_shifted[:, :, 2] %= Lbox
            if ii == 0:
                LC_1000_1 = LC_shifted
            # Apply wrap-around for x and y coordinates that are less than 0
            LC_shifted[:, :, 0] += Lbox * (LC_shifted[:, :, 0] < 0)
            LC_shifted[:, :, 1] += Lbox * (LC_shifted[:, :, 1] < 0)
            LC_shifted[:, :, 2] += Lbox * (LC_shifted[:, :, 2] < 0)

            #################### 大尺度结构#################
            z_list = np.array([output1 for output1, output2 in map(find_redshift_and_scale_factor, Bin_edges_shifted)]) #redshift
            a_list = np.array([output2 for output1, output2 in map(find_redshift_and_scale_factor, Bin_edges_shifted)]) #scale factor
            dz = np.zeros(len(z_list))
            dz =  z_list[1:] - z_list[:-1]
            z_center =  (z_list[1:] + z_list[:-1])/2
            z_list_tot.append(z_list)
            z_list_center.append(z_center)
            a_list_tot.append(a_list)
            dz_tot.append(z_list[1:] - z_list[:-1])

            ne_LC_shifted = assign_density_to_lc_bins(LC_shifted, kdtree, electron_density_data)
            ########Paralell
            #ne_LC_shifted = parallel_assign_density_to_lc_bins(LC_shifted, kdtree, electron_density_data)

            DM_IGM_LC = calculate_DM_IGM(ne_LC_shifted, z_list, dz, DM_IGM_LC_tot, h, h_scaling)
            ########Paralell
            #DM_IGM_LC = parallel_calculate_DM_IGM(ne_LC_shifted, z_list, dz, DM_IGM_LC_tot, h, h_scaling)

            ne_LC_shifted_tot.append(copy.copy(ne_LC_shifted))
            DM_IGM_LC_tot.append(copy.copy(DM_IGM_LC))
            
                    
            if rank == 0: 
                print("LC_length  =",   LC_length )
                sys.stdout.flush()
                #comm = MPI.COMM_WORLD
                #rank = comm.Get_rank()  # 获取当前进程的编号
                #size = comm.Get_size()  # 获取总进程数
                #print(f"rank {rank}:size: {size}")
                sys.stdout.flush()
            X_Bound_shift += L/(m*(random_num_0_2 != 0) + np.sign(m)*(random_num_0_2 == 0))
            Y_Bound_shift += L/(n*(random_num_0_2 != 1) + np.sign(n)*(random_num_0_2 == 1))
            Z_Bound_shift += L/(o*(random_num_0_2 != 2) + np.sign(o)*(random_num_0_2 == 2))
            LC_length_total +=  LC_length
            D_C_up = comoving_distance(z_up)
            D_C = comoving_distance(z_temp)
            z_temp,_ = find_redshift_and_scale_factor(LC_length_total)
            #z_temp = comm.bcast(z_temp if rank == 0 else None, root=0)
            D_C = comoving_distance(z_temp)
            z_temp_tot.append(z_temp)
            if rank == 0:
                print("Bound = ", X_Bound_shift,Y_Bound_shift,Z_Bound_shift )
                sys.stdout.flush()
                print("z_low: ", z_low, "z_up: ", z_up, "z_temp_update: ", z_temp)
                sys.stdout.flush()
                print("D_C_up: ", D_C_up, "DC_update: ", D_C , "LC_length_total_update: ", LC_length_total )
                sys.stdout.flush()

            index_z += 1
            # 检查当前 z_temp 是否仍在 Box 内
            if z_temp >= z_up:
                print(f"z_temp ({z_temp}) has reached z_up ({z_up}). Transitioning to next Box...")
                sys.stdout.flush()
                break  # 跳出内层 while 循环
        else:
            print("z_temp has reached z_max. Exiting while loop...")
            sys.stdout.flush()
            break
    if (z_temp >= z_up) & (z_temp<z_max):
        if rank == 0: 
            print("z_temp>=z_up")
            print("z_temp",z_temp)
            print("z_up",z_up)
            sys.stdout.flush()
        with h5py.File(gridfile[ii+1], 'r') as file:
            a_up = file['/Header'].attrs['Time']
            h_up = file['Parameters'].attrs['HubbleParam']
            a_scaling = file['/Cell/GasDensity'].attrs['a_scaling']
            h_scaling = file['/Cell/GasDensity'].attrs['h_scaling']
        kdtree, electron_density_data_low, Boxsize = process_snapshot(gridfile[ii])
        kdtree, electron_density_data_up, Boxsize = process_snapshot(gridfile[ii+1])
        ne_LC_shifted_low = np.zeros([LC_num, n_bins])
        ne_LC_Ion_shifted_low = np.zeros([LC_num, n_bins])
        DM_IGM_LC_low = np.zeros([LC_num, DM_IGM_num]) 
        ne_LC_shifted_up = np.zeros([LC_num, n_bins])
        ne_LC_Ion_shifted_up = np.zeros([LC_num, n_bins])
        DM_IGM_LC_up = np.zeros([LC_num, DM_IGM_num]) 
        Lbox =  Boxsize/h
        L = Lbox
        print("Start the DM connection of Box {}, Box {}".format(gridfile[ii], gridfile[ii+1]))
        # 达到高红移处，对两个BOX进行重叠处理
        if rank == 0:
            if vectors_switch == "off":       
                d_index = np.zeros(3)
                d_index[0] = random.uniform(0, 1) if random.random() < 0.5 else random.uniform(-1, 0)
                d_index[1] = random.uniform(0, 1) if random.random() < 0.5 else random.uniform(-1, 0)
                d_index[2] = random.uniform(0, 1) if random.random() < 0.5 else random.uniform(-1, 0)
                random_num_0_2 = random.randint(0, 2)
                d_index[random_num_0_2] = 1*np.sign(d_index[random_num_0_2])
                major_axis = d_index[random_num_0_2]
                m = 1/d_index[0]
                n = 1/d_index[1]
                o = 1/d_index[2]
            if vectors_switch == "on":
                m = vectors[vectors_index][0]
                n = vectors[vectors_index][1]
                o = vectors[vectors_index][2]
                vectors_index += 1
        #random_num_0_2 = comm.bcast(random_num_0_2 if rank == 0 else None, root=0)
        #m = comm.bcast(m if rank == 0 else None, root=0)
        #n = comm.bcast(n if rank == 0 else None, root=0)
        #o = comm.bcast(o if rank == 0 else None, root=0)
        #major_axis = comm.bcast(major_axis if rank == 0 else None, root=0)
        print(f"m ,n, o: {m, n ,o}")
        sys.stdout.flush()
        LOS = np.array([L/m, L/n, L/o])
        # Calculate unit vectors in new coordinate system 
        u3 = (np.array([no, mo, nm]) / (no**2 + mo**2 + nm**2)**0.5)
        axis = np.argmin([no, mo, nm])
        u1 = np.zeros(3)
        u1[axis] = 1
        u1 = np.cross(u1, u3)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(u3, u1)
        u2 /= np.linalg.norm(u2)
        
        LC_length = sqrt(sum(LOS**2))
        Bin_edges = np.linspace(0, LC_length, n_bins+1)
        Bin_edges_shifted = np.linspace(LC_length_total, LC_length_total+LC_length, n_bins+1)
        Bin_centers = (Bin_edges[:-1] + Bin_edges[1:]) / 2
        Bin_centers_shifted = (Bin_edges_shifted[:-1] + Bin_edges_shifted[1:]) / 2
        Bin_centers_shifted_tot.append(Bin_centers_shifted)
        Bin_centers_pos = np.array([u3*x for x in Bin_centers])
        if rank == 0: 
            print('Bin_edges_bound = ', Bin_edges_shifted[0], Bin_edges_shifted[-1])
            sys.stdout.flush()
            #print("  Bin_centers  =",   Bin_centers_shifted)
            #sys.stdout.flush()
            print("  LC_length  =",   LC_length )
            sys.stdout.flush()
        X_shift = np.linspace(0, Lbox, X_num+1)
        Y_shift = np.linspace(0, Lbox, Y_num+1)
        LC_origin = np.zeros([(X_num)*(Y_num),3])
        for i in range(X_num):
            for j in range(Y_num):
                LC_origin[i*Y_num+j, 0] = X_shift[i] + X_Bound_shift
                LC_origin[i*Y_num+j, 1] = Y_shift[j] + Y_Bound_shift
                LC_origin[i*Y_num+j, 2] = Z_Bound_shift
        LC_shifted = np.array([Bin_centers_pos+x for x in LC_origin])
        # Apply periodic boundary conditions to the x and y coordinates
        LC_shifted[:, :, 0] %= Lbox
        LC_shifted[:, :, 1] %= Lbox
        LC_shifted[:, :, 2] %= Lbox
        if ii == 0:
            LC_1000_1 = LC_shifted
        # Apply wrap-around for x and y coordinates that are less than 0
        LC_shifted[:, :, 0] += Lbox * (LC_shifted[:, :, 0] < 0)
        LC_shifted[:, :, 1] += Lbox * (LC_shifted[:, :, 1] < 0)
        LC_shifted[:, :, 2] += Lbox * (LC_shifted[:, :, 2] < 0)

        #################### 大尺度结构#################
        z_list = np.array([output1 for output1, output2 in map(find_redshift_and_scale_factor, Bin_edges_shifted)]) #redshift
        a_list = np.array([output2 for output1, output2 in map(find_redshift_and_scale_factor, Bin_edges_shifted)]) #scale factor
        dz = np.zeros(len(z_list))
        dz =  z_list[1:] - z_list[:-1]
        z_center =  (z_list[1:] + z_list[:-1])/2
        z_list_tot.append(z_list)
        z_list_center.append(z_center)
        a_list_tot.append(a_list)
        dz_tot.append(z_list[1:] - z_list[:-1])
        
        ne_LC_shifted_low = assign_density_to_lc_bins(LC_shifted, kdtree, electron_density_data_low)
        ne_LC_shifted_up = assign_density_to_lc_bins(LC_shifted, kdtree, electron_density_data_up)
        ne_LC_shifted_connection = np.copy(ne_LC_shifted_up)
        bin_length =  LC_length/n_bins
        #overlapping_bin_num = int(np.round(overlap_range/bin_length))
        overlapping_bin_num = int(np.round(overlap_range/Bin_num))
        print(f"overlap bin_num: {overlapping_bin_num}")
        sys.stdout.flush()
        for LC_index in range(LC_num):
            for i in range(overlapping_bin_num):
                weight_low = (overlapping_bin_num - i) / overlapping_bin_num
                weight_up = i / overlapping_bin_num
                ne_LC_shifted_connection[LC_index, i] = (
                    weight_low * ne_LC_shifted_low[LC_index, i]
                    + weight_up * ne_LC_shifted_up[LC_index, i]
                )

        print(f"overlap ne_low: {ne_LC_shifted_low[0, :overlapping_bin_num]}")    
        print(f"overlap ne_up: {ne_LC_shifted_up[0, :overlapping_bin_num]}")   
        print(f"overlap ne_connection: {ne_LC_shifted_connection[0, :overlapping_bin_num]}")
        sys.stdout.flush()
        ########Paralell
        #ne_LC_shifted = parallel_assign_density_to_lc_bins(LC_shifted, kdtree, electron_density_data)

        DM_IGM_LC_connection = calculate_DM_IGM(ne_LC_shifted_connection, z_list, dz, DM_IGM_LC_tot, h, h_scaling)
        ########Paralell
        #DM_IGM_LC = parallel_calculate_DM_IGM(ne_LC_shifted, z_list, dz, DM_IGM_LC_tot, h, h_scaling)

        ne_LC_shifted_tot.append(copy.copy(ne_LC_shifted_connection))
        DM_IGM_LC_tot.append(copy.copy(DM_IGM_LC_connection))
        
                
        if rank == 0: 
            print("LC_length  =",   LC_length )
            sys.stdout.flush()
            #comm = MPI.COMM_WORLD
            #rank = comm.Get_rank()  # 获取当前进程的编号
            #size = comm.Get_size()  # 获取总进程数
            #print(f"rank {rank}:size: {size}")
            sys.stdout.flush()
        X_Bound_shift += L/(m*(random_num_0_2 != 0) + np.sign(m)*(random_num_0_2 == 0))
        Y_Bound_shift += L/(n*(random_num_0_2 != 1) + np.sign(n)*(random_num_0_2 == 1))
        Z_Bound_shift += L/(o*(random_num_0_2 != 2) + np.sign(o)*(random_num_0_2 == 2))
        LC_length_total +=  LC_length
        D_C_up = comoving_distance(z_up)
        D_C = comoving_distance(z_temp)
        z_temp,_ = find_redshift_and_scale_factor(LC_length_total)
        #z_temp = comm.bcast(z_temp if rank == 0 else None, root=0)
        D_C = comoving_distance(z_temp)
        z_temp_tot.append(z_temp)
        if rank == 0:
            print("Bound = ", X_Bound_shift,Y_Bound_shift,Z_Bound_shift)
            sys.stdout.flush()
            print("z_low: ", z_low, "z_up: ", z_up, "z_temp_update: ", z_temp)
            sys.stdout.flush()
            print("D_C_up: ", D_C_up, "DC_update: ", D_C , "LC_length_total_update: ", LC_length_total )
            sys.stdout.flush()

        index_z += 1

    # 如果 z_temp 超过 z_max，终止循环
    if z_temp >= z_max: 
        print("z_temp has reached z_max. Exiting for loop...")
        sys.stdout.flush()
        break
        # 如果 z_temp 超过 z_max，终止循环

end_time = time.time()  # 记录程序结束时间
run_time = end_time - start_time  # 计算程序运行时间，单位为秒
with open(f"{storage_path}/generated_vectors.txt", "w") as f:
    for vector in vectors:
        f.write(f"{vector}\n")
print(f"程序总运行时间为：{run_time/3600:.2f}小时 ")


# ---------------------------
# Export the generated light-cone products
# ---------------------------

def export_light_cone_products(storage_dir, agn_info, dm_igm_lc_tot, ne_lc_shifted_tot,
                               bin_centers_shifted_tot, z_list_center, lc_num, h_value):
    dm_igm = np.stack(dm_igm_lc_tot)
    ne_igm = np.stack(ne_lc_shifted_tot) * h_value ** 2
    distance = np.array(bin_centers_shifted_tot)
    distance_flat = distance.flatten()
    redshift = np.array(z_list_center)
    redshift_flat = redshift.flatten()

    out_path = os.path.join(storage_dir, f"DM_diff_analysis_{agn_info}.hdf5")
    with h5py.File(out_path, 'w') as handle:
        handle.create_dataset('Redshift', data=redshift_flat)
        handle.create_dataset('Distance', data=distance_flat)
        handle['Distance'].attrs['unit'] = 'h^-1 cMpc'

        dm_group = handle.create_group('DM_diff_Lines')
        dm_group.attrs['unit'] = 'pc cm^-3'
        ne_group = handle.create_group('ne_Lines')
        ne_group.attrs['unit'] = 'cm^-3'

        for i in range(lc_num):
            dm_group.create_dataset(f'LOS_{i}', data=dm_igm[:, i, :].flatten())
            ne_group.create_dataset(f'LOS_{i}', data=ne_igm[:, i, :].flatten())

    print(f'[DONE] saved light-cone products to {out_path}')
    return out_path


if 'DM_IGM_LC_tot' in globals() and 'ne_LC_shifted_tot' in globals():
    export_light_cone_products(
        storage_dir=storage_path,
        agn_info=AGN_info,
        dm_igm_lc_tot=DM_IGM_LC_tot,
        ne_lc_shifted_tot=ne_LC_shifted_tot,
        bin_centers_shifted_tot=Bin_centers_shifted_tot,
        z_list_center=z_list_center,
        lc_num=LC_num,
        h_value=h,
    )
