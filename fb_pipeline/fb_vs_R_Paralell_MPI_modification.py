#######
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
import os
import joblib
from joblib import Parallel, delayed, parallel_backend, Memory
from scipy.spatial import cKDTree
import time
import sys
from tqdm import tqdm
import re
from tqdm_joblib import tqdm_joblib
import logging
import multiprocessing
from multiprocessing import Lock
from filelock import FileLock
import bisect
import ast
from mpi4py import MPI
import gc
tqdm_disabled = True
# 初始化共享资源
#file_lock = Lock()
#os.environ['JOBLIB_TEMP_FOLDER'] = '/home/zhaozhang/tmp_joblib'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# joblib / MPI resource helpers
def _resolve_n_jobs(mpi_size, default_div=4):
    env = os.getenv("JOBLIB_N_JOBS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, (multiprocessing.cpu_count() // max(1, mpi_size)) // default_div)

def _resolve_backend():
    # default to threading to avoid large per-process memory copies
    return os.getenv("JOBLIB_BACKEND", "threading")
# Step 1: 宇宙常数
#Omega_b = 0.04889
h = 0.6774
mp = 1.6726219e-24  # 质子质量 [g]

# Step 2: 恒星形成临界数密度（氢原子/cm^3）
n_H_crit = 0.1  # 可根据模拟中写入的值调整

# Step 3: 临界质量密度（单位 g/cm^3）
rho_crit_starforming = n_H_crit * mp

# Step 4: 宇宙临界密度（单位 g/cm^3）
rho_crit_cosmic = 1.8788e-26 * h**2 * 1e3 / 1e6  # 转为 g/cm^3

# Step 5: 平均重子密度
#rho_b_mean = Omega_b * rho_crit_cosmic

# Step 6: 计算阈值：rho_crit / rho_mean 
#rho_thresh = rho_crit_starforming / rho_b_mean
#log_rho_thresh = np.log10(rho_thresh)
#log_rho_thresh = np.log10(10**3.5)
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("halo_processing.log", mode="w")  # 输出到日志文件
    ]
)
logger = logging.getLogger()
logger = logging.getLogger(__name__)  # 获取记录器

def log_msg(msg, level="info"):
    """
    统一日志输出：写入日志文件，并同时 flush 到 stdout。
    """
    try:
        getattr(logger, level)(msg)
    except Exception:
        logger.info(msg)
    print(msg, flush=True)

tqdm(disable=True, dynamic_ncols=True, ascii=True)
# 字体设置

font_label = {
    'family': 'serif',
    'size': 20
}
tick_size = 18
font_legend = {
    'family': 'serif',
    'size': 16
}

# 基础路径
#snapshots = [
#    "snapdir_015/snapshot_015", "groups_015/fof_subhalo_tab_015",
#    "snapdir_016/snapshot_016", "groups_016/fof_subhalo_tab_016",
#    "snapdir_017/snapshot_017", "groups_017/fof_subhalo_tab_017",
#    "snapdir_018/snapshot_018", "groups_018/fof_subhalo_tab_018",
#    "snapdir_019/snapshot_019", "groups_019/fof_subhalo_tab_019",
#    "snapdir_020/snapshot_020", "groups_020/fof_subhalo_tab_020"
#]



# 基础路径
base_path_noAGN = "/sqfs/work/hp240141/z6b340/Data/L100N1024_NoBH/output/"
base_path_fiducial = "/sqfs/work/hp240141/z6b340/Data/L100N1024_Fiducial_v1/output/"
storage_path = "/sqfs/work/hp240141/z6b340/results/f_IGM_results"
if not os.path.exists(storage_path):
    # 创建目录
    os.makedirs(storage_path)
kdtree_output_path = os.path.join(storage_path, "kdtree_storage")
if not os.path.exists(kdtree_output_path ):
    # 创建目录
    os.makedirs(kdtree_output_path)

fb_output_path = os.path.join(storage_path, "fb_storage")
if not os.path.exists(fb_output_path):
    # 创建目录
    os.makedirs(fb_output_path)
# 函数：检测目录下的 snapdir 和 groups 子目录，并生成 snapshot 列表
snapshots = [
    #"snapdir_000/snapshot_000", "groups_000/fof_subhalo_tab_000",
    "snapdir_001/snapshot_001", "groups_001/fof_subhalo_tab_001",
    #"snapdir_017/snapshot_017", "groups_017/fof_subhalo_tab_017",
    #"snapdir_018/snapshot_018", "groups_018/fof_subhalo_tab_018",
    #"snapdir_019/snapshot_019", "groups_019/fof_subhalo_tab_019",
    #"snapdir_020/snapshot_020", "groups_020/fof_subhalo_tab_020"
]
snapshots_noAGN = snapshots
snapshots_fiducial = snapshots


def get_snapshots(base_path):
    snapshots = []
    # 获取目录中所有以 snapdir_ 和 groups_ 开头的子目录
    snapdirs = [d for d in os.listdir(base_path) if d.startswith("snapdir_")]
    groupdirs = [d for d in os.listdir(base_path) if d.startswith("groups_")]

    # 确保按编号排序，适配零填充格式
    snapdirs = sorted(snapdirs, key=lambda x: int(re.search(r'\d+', x).group()))
    groupdirs = sorted(groupdirs, key=lambda x: int(re.search(r'\d+', x).group()))

    # 确保 snapdir 和 group 子目录对应
    for snapdir, groupdir in zip(snapdirs, groupdirs):
        snap_num = "{:03d}".format(int(re.search(r'\d+', snapdir).group()))  # 补零到 3 位
        if snap_num == "{:03d}".format(int(re.search(r'\d+', groupdir).group())):  # 编号匹配
            snapshots.append("{}/snapshot_{}".format(snapdir, snap_num))
            snapshots.append("{}/fof_subhalo_tab_{}".format(groupdir, snap_num))
        else:
            print("Warning: Mismatch between {} and {}".format(snapdir, groupdir))

    return snapshots

# 获取两个路径下的 snapshots 列表
snapshots_noAGN = get_snapshots(base_path_noAGN)
snapshots_fiducial = get_snapshots(base_path_fiducial)


# 打印结果
print("Snapshots for noAGN:")
print(snapshots_noAGN)
print("\nSnapshots for fiducial:")
print(snapshots_fiducial)
# 定义缩放因子
alias_snap = {
    "rho": "PartType0/Density",
    "T": "PartType0/Temperature",
    "sphpos": "PartType0/Coordinates",
    "dmpos": "PartType1/Coordinates",
    "starpos": "PartType4/Coordinates",
    "bhpos": "PartType5/Coordinates",
    "smoothlen": "PartType0/SmoothingLength"
}
alias_fof = {
    "halomass": "Group/GroupMass",
    "halopos": "Group/GroupPos",
    "halonum": "Group/GroupNsubs",
    "halo0": "Group/GroupFirstSub",
    "halolen": "Group/GroupLen",
    "haloRV" : "Group/Group_R_Crit200",
    "haloMV": "Group/Group_M_Crit200",
    "Groupmasstype": "Group/GroupMassType"
    #"GroupID": "IDs"
}

alias_mass = {
    "sphmass": 0,  # 对应粒子类型 0（气体）的质量
    "dmmass": 1,   # 对应粒子类型 1（暗物质）的质量
    "starmass": 4,  # 对应粒子类型 4（恒星）的质量
    "bhmass": 5    # 对应粒子类型 5（黑洞）的质量
}

a_scaling = {
    "rho": "-3",
    "u": "0",
    "sphpos": "1",
    "dmpos": "1",
    "starpos": "1",
    "bhpos": "1",
    "smoothlen": "1",
    "T": "0",
    "sphmass": "0",
    "dmmass": "0",
    "starmass": "0",
    "bhmass": "0",
    "bhMdot": "0",
    "halopos": "1",
    "halomass": "0",
    "halolen": "0",
    "halonum": "0",
    "halo0": "0",
    "subhalopos": "1",
    "subhalomass": "0",
    "subhalBHMmax": "0",
    #"subhalolen" : "0",
    "haloRV" : "1",
    "haloMV": "0",
    "halobhM": "0",
    "GroupID": "0",
    "Groupmasstype": "0",
    "sphID": "0",
    "dmID": "0",
    "starID": "0"

}    

h_scaling = {
    "rho": "2",
    "u": "0",
    "sphpos": "-1",
    "dmpos": "-1",
    "starpos": "-1",
    "bhpos": "-1",
    "smoothlen": "-1",
    "T": "0",
    "sphmass": "-1",
    "dmmass": "-1",
    "starmass": "-1",
    "bhmass": "-1",
    "bhMdot": "0",
    "halopos": "-1",
    "halolen": "0",
    "halonum": "0",
    "halo0": "0",
    "subhalopos": "-1",
    "subhalomass": "0",
    "subhalBHMmax": "0",
    #"subhalolen" : "0",
    "haloRV" : "-1",
    "halomass": "-1",
    "haloMV": "-1",
    "halobhM": "-1",
    "Groupmasstype": "-1",
    "GroupID": "0",
    "sphID": "0",
    "dmID": "0",
    "starID": "0"

}


        
# 函数：加载分块文件的数据，应用标度因子

def get_scaling_factors(dataset_name, a_scaling, h_scaling):
    """
    获取指定数据集的 a_scaling 和 h_scaling 值。
    """
    if dataset_name in a_scaling and dataset_name in h_scaling:
        return float(a_scaling[dataset_name]), float(h_scaling[dataset_name])
    else:
        raise ValueError(f"Scaling factors for dataset '{dataset_name}' not found.")

        
def load_single_file(file, dataset_path, a_scaling_factor, h_scaling_factor, a, h):
    """
    加载单个文件的数据并应用标度因子。

    参数：
    - file: 单个文件路径
    - dataset_path: 数据集路径
    - a_scaling_factor: a 标度因子
    - h_scaling_factor: h 标度因子
    - a: 当前快照的尺度因子
    - h: 哈勃常数

    返回：
    - numpy 数组，包含加载后的数据，或 None 如果加载失败
    """
    try:
        with h5py.File(file, "r") as f:
            dataset = f[dataset_path][:]
            total_size = dataset.shape[0]
            dataset = dataset * (a ** a_scaling_factor) * (h ** h_scaling_factor)
            return dataset
    except KeyError:
        print(f"Dataset {dataset_path} not found in file {file}.")
    except OSError:
        print(f"Unable to open file {file}.")
    except Exception as e:
        print(f"Error loading {file}: {e}")
    return None

# 函数：加载分块文件的数据，应用标度因子
def load_snapshot_data(file_pattern, dataset_name, a, h):
    """
    无分块加载粒子数据，并动态应用标度因子。
    
    参数：
    - file_pattern: 文件路径模式，例如 "snapdir/snapshot"
    - dataset_name: 数据集名称，例如 "sphpos" 或 "halopos"
    - a: 当前快照的尺度因子
    - h: 哈勃常数
    
    返回：
    - numpy 数组，包含加载后的数据
    """
    # 获取数据集的路径和标度因子
    if dataset_name in alias_snap:
        dataset_path = alias_snap[dataset_name]
    elif dataset_name in alias_fof:
        dataset_path = alias_fof[dataset_name]
    elif dataset_name in alias_mass:
        dataset_path = f"PartType{alias_mass[dataset_name]}/Masses"
    else:
        raise ValueError(f"Dataset '{dataset_name}' not found in aliases.")
    
    a_scaling_factor, h_scaling_factor = get_scaling_factors(dataset_name, a_scaling, h_scaling)

    # 加载数据文件
    file_list = sorted(glob.glob(f"{file_pattern}.*.hdf5"))
    if not file_list:
        file_list = [f"{file_pattern}.hdf5"]

    print(f"Loading files for pattern: {file_pattern}, dataset: {dataset_path}")

    data = []
    for file in file_list:
        try:
            with h5py.File(file, "r") as f:
                dataset = f[dataset_path][:]
                #print(f"Dataset {dataset_path} size: {dataset.shape}")
                #sys.stdout.flush()
                # 应用标度因子
                dataset = dataset * (a ** a_scaling_factor) * (h ** h_scaling_factor)
                data.append(dataset)
        except KeyError:
            print(f"Dataset {dataset_path} not found in file {file}. Skipping.")
            continue
        except OSError:
            print(f"Unable to open file {file}. Skipping.")
            continue

    if data:
        return np.concatenate(data, axis=0)
    else:
        raise ValueError(f"No valid files found for dataset: {dataset_name}")



def parallel_load_snapshot(file_pattern, dataset_name, a, h, num_jobs=-1):
    """
    并行加载粒子数据，并动态应用标度因子。

    参数：
    - file_pattern: 文件路径模式，例如 "snapdir/snapshot"
    - dataset_name: 数据集名称，例如 "sphpos" 或 "halopos"
    - a: 当前快照的尺度因子
    - h: 哈勃常数
    - num_jobs: 并行线程数量，默认值为 4

    返回：
    - numpy 数组，包含加载后的数据
    """
    # 获取数据集的路径和标度因子
    if dataset_name in alias_snap:
        dataset_path = alias_snap[dataset_name]
    elif dataset_name in alias_fof:
        dataset_path = alias_fof[dataset_name]
    elif dataset_name in alias_mass:
        dataset_path = f"PartType{alias_mass[dataset_name]}/Masses"
    else:
        raise ValueError(f"Dataset '{dataset_name}' not found in aliases.")
    
    a_scaling_factor, h_scaling_factor = get_scaling_factors(dataset_name, a_scaling, h_scaling)

    # 获取文件列表
    file_list = sorted(glob.glob(f"{file_pattern}.*.hdf5"))
    if not file_list:
        file_list = [f"{file_pattern}.hdf5"]

    print(f"Parallel loading files for pattern: {file_pattern}, dataset: {dataset_path}")

    # 并行加载
    with tqdm_joblib(tqdm(desc=f"Processing loading {file_pattern}", total=len(file_list), disable=tqdm_disabled)) as progress_bar:
        results = Parallel(n_jobs=num_jobs, prefer="threads")(

            delayed(load_single_file)(file, dataset_path, a_scaling_factor, h_scaling_factor, a, h)
            for file in file_list
        )

    # 过滤掉 None 的结果并合并
    results = [r for r in results if r is not None]
    return np.concatenate(results, axis=0) if results else np.array([], dtype=np.float32)

def calculate_R200_with_particles(halo_center, particle_data, rho_crit_Msun_Mpc3, r_max):
    """
    Calculate R200 and M200 for a given halo using an iterative method with minimal memory usage.
    """
    r_min = max(1e-4, 0.01 * r_max)  # Initial search range (Mpc)
    tolerance = 1e-1  # Convergence tolerance
    max_iterations = 100  # Maximum iterations

    def compute_enclosed_mass(radius):
        """
        Calculate the enclosed mass within a given radius.
        Iterates over particle types to reduce memory usage.
        """
        enclosed_mass = 0.0
        for ptype in particle_data:
            positions = particle_data[ptype]["positions"]
            masses = particle_data[ptype]["masses"]

            if positions.size > 0:  # Skip empty datasets
                distances = np.linalg.norm(positions - halo_center, axis=1)
                enclosed_mass += np.sum(masses[distances <= radius])
        return enclosed_mass

    previous_r_mid = None  # For tracking convergence issues

    # Iteratively search for R200
    for _ in range(max_iterations):
        r_mid = (r_min + r_max) / 2.0
        enclosed_mass = compute_enclosed_mass(r_mid)
        avg_density = enclosed_mass / ((4 / 3) * np.pi * r_mid**3)
        rho_200 = 200 * rho_crit_Msun_Mpc3

        # Debugging information
        print(f"Iteration {_}: r_mid={r_mid:.4f}, enclosed_mass={enclosed_mass:.4e}, "
              f"avg_density={avg_density:.4e}, rho_200={rho_200:.4e}")
        
        # Convergence criteria
        if np.abs(avg_density - rho_200) < tolerance:
            return r_mid, enclosed_mass
        elif avg_density > rho_200:
            r_min = r_mid  # Expand the searching area
        else:
            r_max = r_mid  # Shrink the searching area

        # Early exit for extreme cases (handle no progress)
        if r_mid == previous_r_mid:
            print("No progress in R200 convergence, stopping iteration.")
            return None, None
        previous_r_mid = r_mid

    raise ValueError("R200 did not converge within the maximum number of iterations.")

######################Group
def build_kdtrees_by_region_parallel_and_save_group(
    particle_data, box_size, region_size, halo_positions, output_path, chunk_size=1000000, n_jobs=-1, group_size=5
):
    """
    构建 KDTree，并将多个相邻区域分组存储到同一个 HDF5 文件中。

    参数：
    - particle_data: 包含粒子坐标和质量的数据字典
    - box_size: 模拟盒子的大小
    - region_size: 区域大小
    - halo_positions: Halo 的位置
    - output_path: HDF5 文件存储路径
    - chunk_size: 每次处理的粒子数量
    - n_jobs: 并行线程数
    - group_size: 每组包含的区域数量
    """ 
    
    def save_group_to_disk(group_data, group_key, output_path, region_counts):
        """
        将一个分组的数据写入磁盘。
        """
        # 清理之前生成的所有 HDF5 文件
        group_key_str = "_".join(map(str, group_key))  # 使用连字符连接 group_key
        file_path = f"{output_path}/kdtree_group_{group_key_str}.h5"
        with h5py.File(file_path, "a") as f:  # "a" 模式追加写入
            for region_key, data in group_data.items():
                region_key_str = "_".join(map(str, region_key))  # 使用连字符连接 region_key
                group = f.require_group(f"region_{region_key_str}")  # 确保分组存在
                for ptype, positions_list in data["positions"].items():
                    # 如果有对应的粒子类型数据
                    if ptype in data["masses"]:
                        masses_list = data["masses"][ptype]

                        # 将分块的数据合并
                        positions = np.concatenate(positions_list, axis=0)
                        masses = np.concatenate(masses_list, axis=0)

                        # 创建或更新数据集
                        if f"{ptype}/positions" in group:
                            #print(f"Dataset {ptype}/positions already exists in {region_key_str}. Overwriting.")
                            del group[f"{ptype}/positions"]
                        group.create_dataset(f"{ptype}/positions", data=positions, compression="gzip")

                        if f"{ptype}/masses" in group:
                            #print(f"Dataset {ptype}/masses already exists in {region_key_str}. Overwriting.")
                            del group[f"{ptype}/masses"]
                        group.create_dataset(f"{ptype}/masses", data=masses, compression="gzip")
            f.attrs["region_counts"] = region_counts
    

    def process_chunk(start, end, ptype, positions, masses):
        """
        对单个粒子块进行分区处理。
        """
        chunk_positions = np.mod(positions[start:end], box_size)
        chunk_masses = masses[start:end]
        region_ids = (chunk_positions // region_size).astype(int)
        unique_regions = np.unique(region_ids, axis=0)

        local_kdtrees = {}
        for region_id in unique_regions:
            mask = np.all(region_ids == region_id, axis=1)
            region_positions = chunk_positions[mask]
            region_masses = chunk_masses[mask]

            region_key = tuple(region_id)  # 保持 region_key 为 tuple 格式
            if region_key not in local_kdtrees:
                local_kdtrees[region_key] = {"positions": {}, "masses": {}}
            local_kdtrees[region_key]["positions"].setdefault(ptype, []).append(region_positions)
            local_kdtrees[region_key]["masses"].setdefault(ptype, []).append(region_masses)
        return local_kdtrees



    # 确定 Halo 的区域 ID
    region_counts = int(box_size // region_size)
    group_data = {}

    # 初始化分组数据

    for ptype in particle_data:
        positions = particle_data[ptype]["positions"]
        masses = particle_data[ptype]["masses"]
        num_particles = positions.shape[0]

        print(f"Processing particle type: {ptype}, Total particles: {num_particles}")

        with tqdm_joblib(tqdm(desc=f"Building KDTree for {ptype}", total=num_particles // chunk_size + 1, disable=tqdm_disabled)) as progress_bar:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_chunk)(
                    start,
                    min(start + chunk_size, num_particles),
                    ptype,
                    positions,
                    masses,
                )
                for start in range(0, num_particles, chunk_size)
            )

        # 合并处理结果
        for local_kdtrees in results:
            for region_key, data in local_kdtrees.items():
                group_key = tuple(np.floor(np.array(region_key) / group_size).astype(int))  # 使用 tuple 格式
                if group_key not in group_data:
                    group_data[group_key] = {}
                if region_key not in group_data[group_key]:
                    if "gas" not in data["positions"]:
                        print(f"Region {region_key} does not contain gas particles.")
                    group_data[group_key][region_key] = {"positions": {}, "masses": {}}

                for ptype in data["positions"]:
                    group_data[group_key][region_key]["positions"].setdefault(ptype, []).extend(data["positions"][ptype])
                    group_data[group_key][region_key]["masses"].setdefault(ptype, []).extend(data["masses"][ptype])

        
    # 将分组数据写入磁盘
    for group_key, group in group_data.items():
        save_group_to_disk(group, group_key, output_path, region_counts)

def get_neighbor_regions(halo_center, box_size, region_size, search_radius):
    """
    获取包含中心区域及所有实际与搜索半径覆盖到的区域编号。

    Parameters:
    - halo_center: Halo 中心坐标 (Mpc)。
    - box_size: 模拟盒子的大小 (Mpc)。
    - region_size: 区域的大小 (Mpc)。
    - search_radius: 搜索半径 (Mpc)。

    Returns:
    - neighbor_regions: 包含需要加载的区域编号的列表。
    """
    region_id = (halo_center // region_size).astype(int)
    neighbor_regions = []

    # 确定偏移范围
    min_offset = -np.ceil(search_radius / region_size).astype(int)
    max_offset = np.ceil(search_radius / region_size).astype(int)

    for dx in range(min_offset, max_offset + 1):
        for dy in range(min_offset, max_offset + 1):
            for dz in range(min_offset, max_offset + 1):
                offset = np.array([dx, dy, dz])
                # 计算当前偏移区域的边界
                region_min = (region_id + offset) * region_size
                region_max = region_min + region_size

                # 考虑周期边界条件
                region_min = np.mod(region_min, box_size)
                region_max = np.mod(region_max, box_size)

                # 检查球是否与区域重叠
                overlap = True
                for dim in range(3):
                    # 考虑盒子边界跨越的情况
                    if region_min[dim] < region_max[dim]:  # 正常区域
                        if not (region_min[dim] - search_radius <= halo_center[dim] <= region_max[dim] + search_radius):
                            overlap = False
                            break
                    else:  # 跨越边界区域
                        if not (halo_center[dim] <= region_max[dim] + search_radius or halo_center[dim] >= region_min[dim] - search_radius):
                            overlap = False
                            break

                if overlap:
                    region_key = tuple(np.mod(region_id + offset, box_size // region_size).astype(int))
                    neighbor_regions.append(f"region_{region_key[0]}_{region_key[1]}_{region_key[2]}")

    return neighbor_regions

#####################group
def load_data_from_regions_group(region_kdtree_path, neighbor_regions, group_size=5):
    """
    从分组存储的 HDF5 文件中加载指定的 KDTree 数据。

    Parameters:
    - region_kdtree_path: HDF5 文件路径。
    - neighbor_regions: 邻近区域列表。
    - group_size: 每组包含的区域数量，用于分组。

    Returns:
    - kdtrees: 包含加载的 KDTree 数据的字典。
    """
    # 计算需要加载的分组
    groups_to_load = set(
        tuple(np.floor(np.array([int(x) for x in region.split('_')[1:]]) / group_size).astype(int))
        for region in neighbor_regions
    )

    region_data = {}

    for group_key in groups_to_load:
        # 构造 HDF5 文件路径
        group_file = f"{region_kdtree_path}/kdtree_group_{'_'.join(map(str, group_key))}.h5"
        # 检查文件是否存在
        if not os.path.exists(group_file):
            log_msg(f"Group file {group_file} not found.", level="warning")
            continue

        # 打开 HDF5 文件并加载数据
        with h5py.File(group_file, "r") as f:
            for region_key in f.keys():                                                                                                                                                                                                                                                                                                                                                                                                                                      
                # 确保区域与邻近区域列表对齐
                if region_key in neighbor_regions:
                    if region_key not in region_data:
                        region_data[region_key] = {}

                    # 遍历粒子类型
                    for ptype in f[region_key]:
                        # 检查是否包含 positions 和 masses 数据
                        if "positions" in f[region_key][ptype] and "masses" in f[region_key][ptype]:
                            positions = f[f"{region_key}/{ptype}/positions"][:]
                            masses = f[f"{region_key}/{ptype}/masses"][:]

                            # 创建 KDTree 并存储
                            region_data[region_key][ptype] = {
                                "positions": positions,  # 加载 positions
                                "masses": masses
                            }
                        else:
                            region_data[region_key][ptype] = {
                                "positions": np.empty((0, 3)), # 空矩阵
                                "masses": np.empty((0,))
                            }
    return region_data 

def build_data_for_halos_with_figm(
    particle_data, 
    box_size, 
    halo_data, 
    output_path, 
    region_kdtree_path, 
    slice_indices,
    region_size,
    chunk_size=1000000, 
    n_jobs=-1, 
    C_search=10,
    group_size=5,
    R_max = 10,
    resume=True,
    completion_mode=False,
    completion_start_slice=0,
):
    """
    为每个 Halo 构建一个 KDTree，并利用区域 KDTree 数据计算 f_IGM。
    
    参数：
    - particle_data: 包含粒子坐标和质量的数据字典。
    - box_size: 模拟盒子的大小 (Mpc)。
    - halo_data: 包含 Halo 的位置、R_200 和 ID 的字典。
        e.g., halo_data = {"positions": [...], "R200": [...], "ids": [...]}
    - output_path: HDF5 文件存储路径。
    - region_kdtree_path: 区域 KDTree 数据的存储路径。
    - slice_indices: Halo 划分切片索引。
    - region_size: 区域划分大小 (Mpc)。
    - group_size: 每组包含的区域数量。
    - chunk_size: 每次处理的粒子数量。
    - n_jobs: 并行线程数。
    - resume: True 时按切片续跑（默认从最后一个已有切片之后开始）。
    - completion_mode: True 时逐 halo 检查已有数据完整性，仅补全缺失/无效 halo。
    - completion_start_slice: completion_mode=True 时，从该 slice 编号开始补全。
    """

    def save_halos_to_hdf5(halo_data_by_id, file_index, num_halos, output_path):
        """
        将 Halo 的粒子数据按粒子类型存储到指定的 HDF5 文件中，并保存 Halo 总数为属性。
        completion_mode=True 时按 halo_id 增量覆盖；否则整切片重写。
        """
        file_name = os.path.join(output_path, f"halos_kdtree_{file_index}_{int(R_max)}RV.h5")
        write_mode = "a" if completion_mode else "w"
        with h5py.File(file_name, write_mode) as f:
            for halo_id, halo_data in halo_data_by_id.items():
                halo_key = str(int(halo_id))
                if completion_mode and halo_key in f:
                    del f[halo_key]
                halo_group = f.create_group(halo_key)
                # Explicit completion markers:
                # processed=1 means this halo was computed in this run,
                # even when it has no particles within the search radius.
                halo_group.attrs["processed"] = 1
                halo_group.attrs["has_particle_data"] = 0
                valid_ptype_count = 0
                for ptype, pdata in halo_data.items():
                    if "positions" not in pdata or "masses" not in pdata:
                        continue
                    pos = np.asarray(pdata["positions"])
                    ms = np.asarray(pdata["masses"])
                    ptype_group = halo_group.create_group(ptype)
                    ptype_group.create_dataset("positions", data=pos, compression="gzip")
                    ptype_group.create_dataset("masses", data=ms, compression="gzip")
                    if pos.size > 0 and ms.size > 0:
                        valid_ptype_count += 1
                if valid_ptype_count > 0:
                    halo_group.attrs["has_particle_data"] = 1
            f.attrs["Halo_num"] = num_halos

    def process_halo_tree(halo_id, halo_center, halo_r200, region_kdtree_path, group_size, region_size):
        """
        处理单个 Halo 的粒子数据，加载其覆盖区域的 KDTree 数据并构建 Halo 的整体 KDTree。
        """
        # Step 1: 计算 Halo 覆盖的邻近区域
        neighbor_regions = get_neighbor_regions(halo_center, box_size, region_size, C_search * halo_r200)
        # 检查生成的邻近区域
        if len(neighbor_regions) == 0:
            logging.warning(f"Warning: No neighbor regions found for halo {halo_id}.")
        # Step 2: 加载邻近区域的 KDTree 数据
        region_data = load_data_from_regions_group(region_kdtree_path, neighbor_regions, group_size)
        # 检查加载的区域数据
        if len(region_data) == 0:
             logging.warning(f"Warning: No neighbor regions found for halo {halo_id}.")
        # Step 3: 整合所有相关区域的粒子数据
        combined_positions = {}
        combined_masses = {}
        for region_key in region_data:
            for ptype in region_data[region_key]:
                positions = region_data[region_key][ptype]["positions"]  # 获取粒子位置
                masses = region_data[region_key][ptype]["masses"]        # 获取粒子质量
                if positions.size == 0:
                    logging.warning(f"Region {region_key}, type {ptype} has no positions to combine.")
                if ptype not in combined_positions:
                    combined_positions[ptype] = []
                    combined_masses[ptype] = []

                combined_positions[ptype].append(positions)
                combined_masses[ptype].append(masses)

        # 合并各粒子类型的所有位置和质量
        for ptype in combined_positions:
            try:
                combined_positions[ptype] = np.concatenate(combined_positions[ptype], axis=0)
                combined_masses[ptype] = np.concatenate(combined_masses[ptype], axis=0)
            except ValueError as e:
                log_msg(f"Error combining data for type {ptype}: {e}", level="warning")
                combined_positions[ptype] = np.empty((0, 3))
                combined_masses[ptype] = np.empty((0,))

        # Step 4: 为 Halo 构建整体 KDTree
        halo_particle_data = {}
        for ptype in combined_positions:
            # 创建 KDTree
            positions = combined_positions[ptype]
            masses = combined_masses[ptype]
            if len(positions) > 0:
                halo_kdtree = cKDTree(positions, boxsize=box_size)
                indices = halo_kdtree.query_ball_point(halo_center, C_search * halo_r200)
                if len(indices) > 0:
                    halo_particle_data[ptype] = {
                        "positions": positions[indices],
                        "masses": masses[indices],
                    }
        
        if halo_particle_data:
            return halo_id, halo_particle_data
        else:
            logging.warning(f"Warning: Halo {halo_id} has no particle data.")
            return halo_id, {}



    # 获取 Halo 数据
    halo_positions = halo_data["positions"]
    halo_radii = halo_data["R200"]
    halo_ids = halo_data["HaloIDs"]

    def _halo_group_is_completed(halo_group):
        """
        判断单个 halo group 是否可视为“已完成”：
        1) 新格式：processed=1 -> 完成（即使空 halo 也算完成）
        2) 兼容旧格式：至少一个 ptype 下 positions/masses 非空 -> 完成
        """
        try:
            if int(halo_group.attrs.get("processed", 0)) == 1:
                return True
            for ptype in halo_group.keys():
                pgrp = halo_group[ptype]
                if "positions" not in pgrp or "masses" not in pgrp:
                    continue
                if pgrp["positions"].size > 0 and pgrp["masses"].size > 0:
                    return True
            return False
        except Exception:
            return False

    def _collect_valid_halo_ids(file_path, expected_ids):
        """
        从已有切片文件中收集“有效 halo_id”：
        - halo_id 存在；
        - 新格式下 processed=1；或兼容旧格式下存在非空粒子数据。
        """
        valid_ids = set()
        if not os.path.exists(file_path):
            return valid_ids
        try:
            with h5py.File(file_path, "r") as f:
                for hid in expected_ids:
                    hkey = str(int(hid))
                    if hkey not in f:
                        continue
                    if _halo_group_is_completed(f[hkey]):
                        valid_ids.add(int(hid))
        except Exception as e:
            log_msg(f"[DBG] failed to scan slice file {file_path}: {e}")
        return valid_ids

    # ---- resume policy ----
    num_slices = len(slice_indices) - 1
    existing_indices = set()
    pattern = os.path.join(output_path, f"halos_kdtree_*_{int(R_max)}RV.h5")
    for fp in glob.glob(pattern):
        m = re.search(r"halos_kdtree_(\d+)_", os.path.basename(fp))
        if m:
            existing_indices.add(int(m.group(1)))
    # 过滤掉超出当前切片范围的旧文件编号
    existing_indices = {i for i in existing_indices if 0 <= i < num_slices}
    last_existing = max(existing_indices) if existing_indices else -1
    if completion_mode:
        rerun_start_slice = max(0, int(completion_start_slice))
    elif resume:
        rerun_start_slice = last_existing + 1
    else:
        rerun_start_slice = 0
    if last_existing >= 0:
        log_msg(
            f"[DBG] resume enabled: existing slices={sorted(existing_indices)[:5]}... "
            f"max={last_existing}; start from slice {rerun_start_slice}"
        )
    log_msg(f"[DBG] mode: resume={resume}, completion_mode={completion_mode}")
    # 并行处理 Halo 数据（按切片分块，避免一次性占用过多内存）
    backend = _resolve_backend()
    for i in range(num_slices):
        start_idx, end_idx = slice_indices[i], slice_indices[i + 1]
        if start_idx >= end_idx:
            continue
        if completion_mode and i < rerun_start_slice:
            log_msg(f"[DBG] skip slice {i} (before completion_start_slice={rerun_start_slice})")
            continue
        file_path = os.path.join(output_path, f"halos_kdtree_{i}_{int(R_max)}RV.h5")
        expected_slice_ids = list(map(int, halo_ids[start_idx:end_idx]))

        if completion_mode:
            valid_existing_ids = _collect_valid_halo_ids(file_path, expected_slice_ids)
            to_process_indices = [
                halo_idx
                for halo_idx in range(start_idx, end_idx)
                if int(halo_ids[halo_idx]) not in valid_existing_ids
            ]
            if len(to_process_indices) == 0:
                log_msg(f"[DBG] skip slice {i} (all halo IDs valid)")
                continue
            log_msg(
                f"[DBG] slice {i}: valid={len(valid_existing_ids)}/{len(expected_slice_ids)}, "
                f"rebuild={len(to_process_indices)}"
            )
        else:
            # 纯 resume：不检查历史切片内容，直接从 last_existing+1 开始
            if resume and i < rerun_start_slice and os.path.exists(file_path):
                log_msg(f"[DBG] skip slice {i} (resume-only)")
                continue
            to_process_indices = list(range(start_idx, end_idx))

        with tqdm(
            desc=f"Processing Halos slice {i}",
            total=len(to_process_indices),
            disable=tqdm_disabled,
        ) as progress_bar:
            with parallel_backend(backend):
                results = Parallel(n_jobs=n_jobs, batch_size=5)(
                    delayed(process_halo_tree)(
                        halo_ids[halo_idx],
                        halo_positions[halo_idx],
                        halo_radii[halo_idx],
                        region_kdtree_path,
                        group_size,
                        region_size,
                    )
                    for halo_idx in to_process_indices
                )
            progress_bar.update(len(to_process_indices))

        # 整理并保存当前切片
        halo_data_by_id = {halo_id: data for halo_id, data in results}
        n_with_data = sum(1 for _, data in results if bool(data))
        n_empty = len(results) - n_with_data
        log_msg(
            f"[DBG] slice {i}: processed={len(results)}, "
            f"with_data={n_with_data}, empty={n_empty}"
        )
        save_halos_to_hdf5(halo_data_by_id, i, len(halo_ids), output_path)
        del halo_data_by_id, results, to_process_indices, expected_slice_ids
        gc.collect()

    log_msg("KDTree construction and saving completed.")



def load_kdtree_by_halo_id(output_path, halo_id, slice_indices, R_max = 10, halo_id_to_index=None):
    """
    根据 halo_id 和划分的 slice_indices 加载对应的 KDTree 数据。

    参数：
    - output_path: HDF5 文件的存储路径。
    - halo_id: 当前 Halo 的 ID。
    - slice_indices: Halo 切片索引数组。

    返回：
    - kdtree_data: 当前 Halo 的 KDTree 数据。
    """
    # 先把 halo_id 映射到 halo 序号，再按 slice_indices 定位分区
    hid = int(halo_id)
    if halo_id_to_index is not None:
        halo_idx = halo_id_to_index.get(hid, None)
    else:
        # 向后兼容：当 HaloID 本身就是 0..N-1 序号时可直接使用
        halo_idx = hid

    if halo_idx is None:
        raise KeyError(f"Halo ID {halo_id} not found in halo_id_to_index.")

    file_index = bisect.bisect_right(slice_indices, halo_idx) - 1
    if file_index < 0 or file_index >= (len(slice_indices) - 1):
        raise ValueError(f"Halo index {halo_idx} (from halo_id={halo_id}) does not belong to any slice.")

    # 构造 HDF5 文件路径
    file_name = os.path.join(output_path, f"halos_kdtree_{file_index}_{int(R_max)}RV.h5")
    
    # 检查文件是否存在
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"KDTree file {file_name} not found for Halo ID {halo_id}.")

    # 加载 HDF5 文件中的数据
    with h5py.File(file_name, "r") as f:
        if str(halo_id) not in f:
            raise KeyError(f"Halo ID {halo_id} not found in KDTree file {file_name}.")
        halo_data = f[str(halo_id)]
        kdtree_data = {
            ptype: {
                "positions": halo_data[ptype]["positions"][:],
                "masses": halo_data[ptype]["masses"][:],
            }
            for ptype in halo_data.keys()
        }

    return kdtree_data





# 函数：提取 R_cut 范围内的粒子
def get_particles_within_R200(halo_center, R_cut, particle_data, box_size = 100):
    """
    提取 R200 范围内的粒子。
    """
    result = {}
    for ptype in particle_data:
        if "positions" not in particle_data[ptype] or "masses" not in particle_data[ptype]:
            logging.warning(f"Missing 'positions' or 'masses' for particle type '{ptype}'. Skipping.")
            continue

        positions = particle_data[ptype]["positions"]
        masses = particle_data[ptype]["masses"]

        if positions.ndim != 2 or positions.shape[1] != 3:
            logging.error(f"Invalid positions shape for particle type '{ptype}': {positions.shape}")
            continue

        if len(positions) == 0:
            logging.warning(f"No particles found for particle type '{ptype}'. Skipping.")
            continue

        # 计算 R200 范围内的粒子
        kdtree = cKDTree(positions, boxsize=box_size)
        indices = kdtree.query_ball_point(halo_center, R_cut)

        result[ptype] = {
            "positions": positions[indices],
            "masses": masses[indices],
        }

    return result

def log_mean_without_zeros(arr):
    # 将数组中的零替换为NaN
    arr[arr == 0] = np.nan
    # 计算对数平均值，忽略NaN值
    log_mean = np.nanmean(np.log(arr), axis=0)
    return np.exp(log_mean)

def linear_mean_without_zeros(arr):
    # 将数组中的零替换为NaN
    arr[arr == 0] = np.nan
    # 计算对数平均值，忽略NaN值
    linear_mean = np.nanmean(arr, axis=0)
    return linear_mean

def median_without_zeros(arr):
    # 将数组中的零替换为NaN
    arr[arr == 0] = np.nan
    # 计算对数平均值，忽略NaN值
    median = np.nanmedian(arr, axis=0)
    return median

def calculate_error_bands(data):
    """
    计算上下两条曲线作为误差范围。

    参数：
    data (numpy.ndarray): 数据数组。

    返回：
    (numpy.ndarray, numpy.ndarray): 返回上限曲线和下限曲线。
    """
    # 在对数空间中计算 percentiles
    data[data == 0] = np.nan
    log_data = np.log(data)

    # 计算上下限
    percentiles = [16, 84]  # 16th and 84th percentiles
    percentile_values = np.nanpercentile(log_data, percentiles, axis=0)

    # 上下限曲线
    lower_bound_curve = percentile_values[0]
    upper_bound_curve = percentile_values[1]

    return np.exp(lower_bound_curve), np.exp(upper_bound_curve)

def _write_fb_profile_txt(
    out_txt,
    z,
    C_list,
    halo_prop_by_id,
    target_results,
    header_extra="",
    convert_M200_to_physical=True,  # 你的 haloMV 是 /1e10 的单位时用 True
    write_per_halo_rows=True,
):
    """
    Mode B output:
      one file per (snapshot,label,massbin):
        halo_id log10M200 R200 fb_C0 fb_C0.5 ... fb_Cmax

    Additionally stores per-C summary arrays in header:
      fb_logmean, fb_median, fb_p16, fb_p84  (computed over halos)
    """
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    C_list = np.asarray(C_list, dtype=float)

    # build fb / f_gas matrices (Nh, Nc)
    rows = []
    fb_mat = []
    fgas_mat = []
    for r in target_results:
        hid = int(r["halo_id"])
        prop = halo_prop_by_id.get(hid)
        if prop is None:
            continue
        M200 = float(prop["M200"])
        if convert_M200_to_physical:
            M200 *= 1e10
        rows.append((hid, M200, float(prop["R200"])))
        fb_mat.append(np.asarray(r["fb"], dtype=float))
        if "f_gas" in r and r["f_gas"] is not None:
            fgas_mat.append(np.asarray(r["f_gas"], dtype=float))
        else:
            fgas_mat.append(np.full_like(np.asarray(r["fb"], dtype=float), np.nan, dtype=float))

    if len(rows) == 0:
        with open(out_txt, "w") as f:
            f.write(f"# z = {z:.6f}\n")
            f.write("# N_halo = 0\n")
            if header_extra:
                for line in header_extra.strip().splitlines():
                    f.write(f"# {line}\n")
            f.write("# No halos in selection.\n")
        return

    fb_mat = np.vstack(fb_mat)  # (Nh,Nc)
    fgas_mat = np.vstack(fgas_mat)  # (Nh,Nc)

    # robust stats: ignore zeros and NaNs using your helpers
    fb_logmean = log_mean_without_zeros(fb_mat.copy())
    fb_median  = median_without_zeros(fb_mat.copy())
    fb_p16, fb_p84 = calculate_error_bands(fb_mat.copy())
    fgas_logmean = log_mean_without_zeros(fgas_mat.copy())
    fgas_median  = median_without_zeros(fgas_mat.copy())
    fgas_p16, fgas_p84 = calculate_error_bands(fgas_mat.copy())

    # sort by mass
    order = np.argsort([m for (_, m, _) in rows])

    with open(out_txt, "w") as f:
        f.write(f"# z = {z:.6f}\n")
        f.write(f"# N_halo = {len(rows)}\n")
        if header_extra:
            for line in header_extra.strip().splitlines():
                f.write(f"# {line}\n")

        f.write("# C_list = " + ",".join([f"{c:.6g}" for c in C_list]) + "\n")

        def _arrline(name, arr):
            arr = np.asarray(arr)
            f.write(f"# {name} = " + " ".join([f"{x:.8g}" if np.isfinite(x) else "nan" for x in arr]) + "\n")

        _arrline("fb_logmean", fb_logmean)
        _arrline("fb_median",  fb_median)
        _arrline("fb_p16",     fb_p16)
        _arrline("fb_p84",     fb_p84)
        _arrline("f_gas_logmean", fgas_logmean)
        _arrline("f_gas_median",  fgas_median)
        _arrline("f_gas_p16",     fgas_p16)
        _arrline("f_gas_p84",     fgas_p84)

        if write_per_halo_rows:
            col_fb = [f"fb_C{c:.6g}" for c in C_list]
            col_fgas = [f"f_gas_C{c:.6g}" for c in C_list]
            f.write("# halo_id\tlog10M200\tR200\t" + "\t".join(col_fb) + "\t" + "\t".join(col_fgas) + "\n")

            for ii in order:
                hid, M200, R200 = rows[ii]
                fb_row = fb_mat[ii]
                fgas_row = fgas_mat[ii]
                f.write(
                    f"{hid}\t{np.log10(M200):.6f}\t{R200:.6e}\t" +
                    "\t".join([f"{x:.8f}" if np.isfinite(x) else "nan" for x in fb_row]) + "\t" +
                    "\t".join([f"{x:.8f}" if np.isfinite(x) else "nan" for x in fgas_row]) +
                    "\n"
                )
        else:
            f.write("# per-halo rows are disabled (WRITE_PER_HALO_ROWS=0)\n")


def process_halo_multiR(
    halo_id,
    halo_center,
    halo_r200,
    rho_crit_Msun_Mpc3,
    region_kdtree_path,
    slice_indices,
    C_list,
    C_search=10,
    if_Cal_R200=False,
    box_size=100.0,
    R_max = 10,
    halo_id_to_index=None,
    debug=False,
):
    """
    One-pass per halo:
      - load KDTree once (must cover >= max(C_list)*R200)
      - filter particles once within Rmax = max(C_list)*R200
      - compute Mgas/Mdm/Mstar and fb for all C in C_list

    Returns:
      dict with arrays of shape (Nc,)
    """
    if debug:
        print(
            f"[DBG] halo_id={halo_id} R200={halo_r200:.6g} "
            f"box={box_size} C_search={C_search} C_list(min/max)={np.min(C_list):.3g}/{np.max(C_list):.3g}"
        )
        sys.stdout.flush()
    try:
        kdtree_data = load_kdtree_by_halo_id(
            region_kdtree_path,
            halo_id,
            slice_indices,
            R_max,
            halo_id_to_index=halo_id_to_index,
        )
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Failed to load KDTree for halo_id {halo_id}: {e}")
        if debug:
            print(f"[DBG] halo_id={halo_id} kdtree load failed -> None")
            sys.stdout.flush()
        return None
    except Exception as e:
        logging.exception(f"Unexpected error while loading KDTree for halo_id {halo_id}: {e}")
        if debug:
            print(f"[DBG] halo_id={halo_id} kdtree unexpected error -> None")
            sys.stdout.flush()
        return None

    if not kdtree_data:
        if debug:
            print(f"[DBG] halo_id={halo_id} kdtree_data empty -> None")
            sys.stdout.flush()
        return None

    C_list = np.asarray(C_list, dtype=float)
    C_max = float(np.nanmax(C_list))
    if debug:
        print(f"[DBG] halo_id={halo_id} C_max={C_max:.3g} Rmax={C_max * halo_r200:.6g}")
        sys.stdout.flush()
    # optional R200 recalibration (use the largest radius)
    if if_Cal_R200:
        calc_R200, calc_M200 = calculate_R200_with_particles(
            halo_center, kdtree_data, rho_crit_Msun_Mpc3, C_max * halo_r200
        )
        if calc_R200 is None:
            return None
        halo_r200 = float(calc_R200)

    # load particles once within Rmax
    particles = get_particles_within_R200(
        halo_center, C_max * halo_r200, kdtree_data, box_size
    )
    if debug:
        for ptype in ("gas", "dm", "star"):
            p = particles.get(ptype, {})
            pos = np.asarray(p.get("positions", []))
            m = np.asarray(p.get("masses", []))
            print(f"[DBG] halo_id={halo_id} ptype={ptype} N={len(m)} pos_shape={pos.shape}")
            sys.stdout.flush()
    # --- periodic distance helper
    def _periodic_r(pos):
        dr = pos - halo_center[None, :]
        dr -= box_size * np.round(dr / box_size)
        return np.sqrt(np.sum(dr * dr, axis=1))

    # --- cumulative mass within cuts for one particle type
    def _mass_within_cuts(ptype):
        p = particles.get(ptype, {})
        pos = np.asarray(p.get("positions", []))
        m = np.asarray(p.get("masses", []), dtype=float)
        if m.size == 0:
            return np.zeros_like(C_list, dtype=float)

        r = _periodic_r(pos)
        print("halo_id", halo_id, "r_max=", np.max(r), "R_cut_max=", C_max*halo_r200)

        order = np.argsort(r)
        r_sorted = r[order]
        m_sorted = m[order]
        cum = np.cumsum(m_sorted)

        Rcuts = C_list * halo_r200
        idx = np.searchsorted(r_sorted, Rcuts, side="right") - 1
        out = np.zeros_like(Rcuts, dtype=float)
        ok = idx >= 0
        out[ok] = cum[idx[ok]]
        return out

    Mgas  = _mass_within_cuts("gas")
    Mdm   = _mass_within_cuts("dm")
    Mstar = _mass_within_cuts("star")

    den = Mgas + Mdm + Mstar
    fb = np.where(den > 0, (Mgas + Mstar) / den, np.nan)  # den=0 -> nan
    f_gas = np.where(den > 0, (Mgas) / den, np.nan)
    return {
        "halo_id": int(halo_id),
        "R200": float(halo_r200),
        "C_list": C_list,
        "Mgas": Mgas,
        "Mdm": Mdm,
        "Mstar": Mstar,
        "fb": fb,
        "f_gas": f_gas,
    }



# 主函数：计算 f_IGM
def calculate_f_b_with_regions_and_kdtree(
    base_path,
    snapshots,
    output_file,
    kdtree_output_path,
    label="f",
    box_size=100.0,
    region_size=2.0,
    chunk_size=1000000,
    n_jobs=-1,
    C_search=10,
    # Mode B controls
    R_max=10,
    R_num=10,
    C_list=None,
    filter_M200_low=10,   # e.g. 10 means 1e10
    filter_M200_up=11,    # e.g. 11 means 1e11
    if_Cal_R200=False,
    slice_count=64,
    group_size=5,
    log_T_thre=4,
    log_rho_thre=3.5,
    # new controls for this task
    z_target=0.0,
    z_tol=1e-3,
    only_one_snapshot=True,
    out_txt_dir=None,
    txt_prefix="fb_profile",
    resume=True,
    completion_mode=False,
    completion_start_slice=0,
    skip_halo_build=True,
):
    """Compute fb(<C_R200*R200) for a halo mass bin and (optionally) only z~0.

    Assumptions consistent with your current pipeline:
    - You have already built per-halo KDTree subsets saved under region_kdtree_path.
    - process_halo returns at least:
        gas_mass_within_Rcut, dm_mass_within_Rcut, star_mass_within_Rcut, f_b_within_Rcut
      and is consistent with get_particles_within_R200 using cut radius = C_R200*R200.

    Outputs:
    - Keeps the original HDF5 outputs (gas mass list) in output_file.
    - Additionally writes a TXT file with per-halo fb and masses for the selected mass bin.

    Notes:
    - filter_M200_low/up are interpreted as log10(M200/Msun). Internally, this function
      follows your existing convention: M200 stored in units of 1e10 Msun (or Msun/h)
      so it converts by /1e10 when building the mass range. If your M200 unit differs,
      adjust the M_filter_low/up conversion below.
    """

    # Where to write txt
    if out_txt_dir is None:
        out_txt_dir = os.path.dirname(output_file) or "."
    os.makedirs(out_txt_dir, exist_ok=True)

    # Use provided C_list; fallback to a safe default if not provided
    if C_list is None:
        C_0_default = 0.05
        C_min = min(C_0_default, R_max / R_num)
        C_list = np.linspace(C_min, R_max, R_num)

    # Create HDF5 for per-snapshot gas masses (kept from your original design)
    #with h5py.File(output_file, "w") as hdf5_file:

    # --- iterate snapshot pairs (snapfile, groupfile)
    # Your original code used: for i in range(36, 38, 2)
    # Here we scan all pairs and select z~target.
    snap_indices = list(range(0, len(snapshots), 2))

    wrote_one = False
    for i in snap_indices:
        if i + 1 >= len(snapshots):
            break

        snapfile = base_path + snapshots[i]
        groupfile = base_path + snapshots[i + 1]
        print(f"\nProcessing Snapshot: {snapshots[i]} and {snapshots[i + 1]}")

        # --- read header for redshift
        try:
            with h5py.File(f"{snapfile}.0.hdf5", "r") as f_snap:
                a = float(f_snap["Header"].attrs["Time"])
                z = 1.0 / a - 1.0
                try:
                    h = float(f_snap["Header"].attrs["HubbleParam"])
                    Omega_0 = float(f_snap["Header"].attrs["Omega0"])
                    OmegaLambda = float(f_snap["Header"].attrs["OmegaLambda"])
                    OmegaBaryon = float(f_snap["Header"].attrs["OmegaBaryon"])
                except Exception:
                    h = float(f_snap["Parameters"].attrs["HubbleParam"])
                    Omega_0 = float(f_snap["Parameters"].attrs["Omega0"])
                    OmegaLambda = float(f_snap["Parameters"].attrs["OmegaLambda"])
                    OmegaBaryon = float(f_snap["Parameters"].attrs["OmegaBaryon"])
        except OSError as e:
            print(f"Error reading header for {snapfile}: {e}. Skipping.")
            continue

        # ---- run-context check (after header) ----
        print(
            "[RUN-CHECK] snapfile={0} groupfile={1} a={2:.6g} z={3:.6g} h={4:.6g} "
            "Omega0={5:.6g} OmegaLambda={6:.6g} OmegaBaryon={7:.6g} z_target={8:.6g} z_tol={9:.6g}"
            .format(snapfile, groupfile, a, z, h, Omega_0, OmegaBaryon, OmegaLambda, z_target, z_tol)
        )
        Cosmo_Par = {
            "Omega_0": Omega_0,
            "Omega_Lamda": OmegaLambda,
            "OmegaBaryon": OmegaBaryon,
            "HubbleParam": h

        }
        # If we only want z~0 (or another target z)
        if abs(z - z_target) > z_tol:
            continue
        box_label = "Box_" + snapshots[i].split("_")[-1] + label
        region_kdtree_path = os.path.join(kdtree_output_path, box_label)   
        print(f"region_kdtree_path: {region_kdtree_path}")
        if not os.path.exists(region_kdtree_path):
            os.makedirs(region_kdtree_path)
        # 计算临界密度
        G = 6.67430e-8  # 引力常数 (cm^3 g^-1 s^-2)
        H0_cgs = (h * 100) * 1e5 / 3.08567758e24  # H0 转为 cgs 单位 (s^-1)
        rho_crit = 3 * H0_cgs**2 / (8 * np.pi * G)
        rho_crit_Msun_Mpc3 = rho_crit * (3.08567758e24)**3 / 1.98847e43  # 转为 1e10 Msun/Mpc^3
        # 加载粒子数据并应用标度
        try:
            sphmass = parallel_load_snapshot(snapfile, "sphmass", a, h)
            sphpos = parallel_load_snapshot(snapfile, "sphpos", a, h)
            dmpos = parallel_load_snapshot(snapfile, "dmpos", a, h)
            dmmass = parallel_load_snapshot(snapfile, "dmmass", a, h)
            starpos = parallel_load_snapshot(snapfile, "starpos", a, h)
            starmass = parallel_load_snapshot(snapfile, "starmass", a, h)
            bhpos = parallel_load_snapshot(snapfile, "bhpos", a, h) if label == "f" else None
            bhmass = parallel_load_snapshot(snapfile, "bhmass", a, h) if label == "f" else None
            sphT = parallel_load_snapshot(snapfile, "T", a, h)
            sphrho = parallel_load_snapshot(snapfile, "rho", a, h)

            # 输出 sphpos 和 halopos 的最大值
            print(f"Max sphpos for snapshot {i/2}: {np.max(sphpos):.4f} Mpc")
            print(f"Min sphpos for snapshot {i/2}: {np.min(sphpos):.4f} Mpc")
        except (KeyError, ValueError) as e:
            print(f"Error loading snapshot: {e}")
            continue

        # 加载 Halo 数据并应用标度
        try:
            halopos = parallel_load_snapshot(groupfile, "halopos", a, h)
            haloRV = parallel_load_snapshot(groupfile, "haloRV", a, h)
            haloMV = parallel_load_snapshot(groupfile, "haloMV", a, h)
            halomass = parallel_load_snapshot(groupfile, "halomass", a, h)
            # 将 Halo 数据按照质量 haloMV 降序排序
            sorted_indices = np.argsort(-haloMV)  # 负号表示降序排列

            # 根据排序结果重新排列所有属性
            halopos = halopos[sorted_indices]
            haloRV = haloRV[sorted_indices]
            haloMV = haloMV[sorted_indices]
            halomass = halomass[sorted_indices]
            # 为 Halo 分配统一的 HaloID，按照排序后的顺序
            haloIDs = np.arange(len(halopos), dtype=int)

            # 输出 halopos 的最大值和最小值
            print(f"Max halopos for snapshot {i/2}: {np.max(halopos):.4f} Mpc")
            print(f"Min halopos for snapshot {i/2}: {np.min(halopos):.4f} Mpc")
        except (KeyError, ValueError) as e:
            print(f"Error loading FoF file: {e}")
            continue

        # 物理距离
        box_size_physical = box_size / h * a
        region_size_physical = region_size / h * a
        # 组装粒子数据
        particle_data = {
            "gas": {"positions": sphpos, "masses": sphmass, "T": sphT, "density": sphrho},
            "dm": {"positions": dmpos, "masses": dmmass},
            "star": {"positions": starpos, "masses": starmass},
        }
        if label == "f":
            particle_data["bh"] = {"positions": bhpos, "masses": bhmass}
        # 动态生成粒子种类列表
        particle_types = list(particle_data.keys())

        region_counts = int(box_size_physical // region_size_physical)
        print(f"region counts: {region_counts}")

        # 转换为 g/cm^3
        #density_cgs = density_code * 6.7699e-31

        # 使用前面算出的 rho_mean（g/cm^3）
        rho_mean_0 = 4.2149e-31
        rho_mean = rho_mean_0*(1+z)**3
        # 提取温度与质量
        temperature = particle_data["gas"]["T"]
        masses = particle_data["gas"]["masses"]
        density_cgs = particle_data["gas"]["density"] * 6.7699e-31 # 或 PartType0["Density"]
        # 计算 rho / rho_mean 及其对数
        rho_ratio = density_cgs / rho_mean
        # 定义密度临界值
        rho_ratio_thresh = 10 ** log_rho_thre
        # 温度划分
        cold_mask = sphT < 10**log_T_thre
        hot_mask = sphT >= 10**log_T_thre

        # 密度划分：临界密度为 log10(rho_b / rho_mean) = 5.6

        # 四个子类掩码
        cold_condensed = (temperature < 10**log_T_thre) & (rho_ratio >= rho_ratio_thresh)
        hot_condensed  = (temperature >= 10**log_T_thre) & (rho_ratio >= rho_ratio_thresh)
        cold_IGM       = (temperature < 10**log_T_thre) & (rho_ratio < rho_ratio_thresh)
        hot_IGM        = (temperature >= 10**log_T_thre) & (rho_ratio < rho_ratio_thresh)

        # 分别计算质量
        mass_cold_condensed = np.sum(masses[cold_condensed])
        mass_hot_condensed = np.sum(masses[hot_condensed])
        mass_cold_IGM = np.sum(masses[cold_IGM])
        mass_hot_IGM = np.sum(masses[hot_IGM])
        print(f"z={z:.2f}, mean T={np.mean(temperature):.2e}, median rho_ratio={np.median(rho_ratio):.2e}")
        print(f"Cold condensed gas mass: {mass_cold_condensed:.4e}")
        print(f"Hot condensed gas mass: {mass_hot_condensed:.4e}")
        print(f"Cold IGM gas mass: {mass_cold_IGM:.4e}")
        print(f"Hot IGM gas mass: {mass_hot_IGM:.4e}")
        #####清理kdtree文件
        def clear_output_files(output_path, files):
            """
            清理指定路径中的所有 HDF5 文件。
            """
            if not files:
                log_msg(f"No files found in {output_path} to delete.")
            else:
                for file in files:
                    os.remove(file)
                    log_msg(f"Removed old file: {file}")

        def clear_output_file(file):
            """
            清理单个 HDF5 文件（不影响其他完备文件）。
            """
            if os.path.exists(file):
                os.remove(file)
                log_msg(f"Removed file: {file}")
            else:
                log_msg(f"File not found, skip delete: {file}")
        def check_and_clear_region_kdtree_files(region_kdtree_path, region_counts):
            """
            检查 HDF5 文件中的 Halo 数是否匹配，如果不匹配，则清理 HDF5 文件。
            """
            kdtree_files = glob.glob(f"{region_kdtree_path}/kdtree_*.h5")

            if not kdtree_files:
                log_msg("No KDTree files found. Rebuilding KDTree...")
                return True

            for file in kdtree_files:
                try:
                    with h5py.File(file, "r") as f:
                        if "region_counts" in f.attrs and f.attrs["region_counts"] == region_counts:
                            continue
                        else:
                            log_msg(f"Region counts mismatch in {file}. Clearing all KDTree files...")
                            #clear_output_files(region_kdtree_path, kdtree_files)
                            return True
                except Exception as e:
                    log_msg(f"Error reading {file}: {e}. Clearing all KDTree files...")
                    clear_output_files(region_kdtree_path, kdtree_files)
                    return True

            log_msg("KDTree files are up-to-date.")
            return False

        if check_and_clear_region_kdtree_files(region_kdtree_path, region_counts):
            build_kdtrees_by_region_parallel_and_save_group(
            particle_data, 
            box_size_physical, 
            region_size_physical, 
            halopos, 
            region_kdtree_path, 
            chunk_size=chunk_size, 
            n_jobs=n_jobs, 
            group_size=group_size
            )           
            print("Regional KDTree rebuilt and saved.")
        else:
            print("Regional KDTree files are up-to-date. Skipping KDTree build.")

        ##############检查是否需要清清理之前的kdtree文件
        def check_and_clear_halo_files(
            output_path,
            num_valid_halos,
            valid_halo_ids,
            slice_indices=None,
            required_keys=["positions", "masses"],
            R_max=10,
            resume=False,
        ):
            """
            检查所有 HDF5 文件中的 Halo 数是否匹配，以及是否包含完整的 Halo IDs 和粒子数据。
            如果不匹配或数据缺失，则清理 HDF5 文件。

            参数：
            - output_path: HDF5 文件的存储路径。
            - num_valid_halos: 预期的 Halo 总数。
            - valid_halo_ids: 预期的 Halo ID 列表。
            - required_keys: 每个粒子类型必须包含的键列表。
            """
            halo_kdtree_files = glob.glob(f"{output_path}/halos_kdtree_*{int(R_max)}RV.h5")

            if not halo_kdtree_files:
                log_msg("No KDTree files found. Rebuilding KDTree...")
                return True

            # 用于存储所有切片文件中存在的 Halo IDs
            all_existing_halo_ids = set()

            if resume:
                log_msg("[DBG] resume=True: skip Halo_num and required_keys checks.")

            for file in halo_kdtree_files:
                missing_key_file = None
                missing_key_detail = None
                try:
                    with h5py.File(file, "r") as f:
                        if not resume:
                            # 1. 检查 Halo 数量是否匹配
                            if "Halo_num" not in f.attrs or f.attrs["Halo_num"] != num_valid_halos:
                                log_msg(
                                    f"Halo number mismatch in {file}. Expected {num_valid_halos}, "
                                    f"found {f.attrs.get('Halo_num', 'None')}. Clearing all files..."
                                )
                                clear_output_files(output_path, halo_kdtree_files)
                                log_msg("Clearing finished! Rebuilding...")
                                return True

                        # 2. 收集文件中的 Halo IDs
                        existing_halo_ids = set(map(int, f.keys()))
                        all_existing_halo_ids.update(existing_halo_ids)

                        # 3. 检查每个 Halo 是否包含所需的键（resume=False 时才检查）
                        if not resume:
                            for halo_id in existing_halo_ids:
                                halo_group = f[str(halo_id)]
                                for ptype in halo_group.keys():  # 遍历每种粒子类型
                                    ptype_group = halo_group[ptype]
                                    if not all(key in ptype_group for key in required_keys):
                                        missing_key_file = file
                                        missing_key_detail = f"halo {halo_id}, type {ptype}"
                                        break
                                if missing_key_file:
                                    break
                            if missing_key_file:
                                # defer deletion until after file is closed
                                pass

                    if missing_key_file:
                        log_msg(
                            f"Missing required keys in {missing_key_file}, {missing_key_detail}. "
                            "Clearing this file only..."
                        )
                        clear_output_file(missing_key_file)
                        log_msg("Clearing finished! Rebuilding from this file...")
                        return True

                except Exception as e:
                    log_msg(f"Error reading {file}: {e}. Clearing this file only...")
                    clear_output_file(file)
                    log_msg("Clearing finished! Rebuilding from this file...")
                    return True

            # 4. 检查是否所有预期的 Halo IDs 都存在（允许额外 ID）
            valid_id_set = set(map(int, valid_halo_ids))
            missing_halo_ids = valid_id_set - all_existing_halo_ids
            if missing_halo_ids:
                if len(missing_halo_ids) > 100:
                    log_msg(
                        f"Halo IDs mismatch across all files. {len(missing_halo_ids)} Halo IDs are missing. "
                        f"Showing first 100 IDs: {list(missing_halo_ids)[:100]}"
                    )
                else:
                    log_msg(f"Halo IDs mismatch across all files. Missing Halo IDs: {missing_halo_ids}.")
                if resume:
                    log_msg("[DBG] resume=True: keep existing files, will rebuild missing slices only.")
                    return True
                if slice_indices is None:
                    log_msg("[WARN] slice_indices not provided; will rebuild missing slices without clearing files.")
                    return True

                # map missing halo IDs -> slice file indices, then clear only those files
                halo_id_to_pos = {int(hid): idx for idx, hid in enumerate(valid_halo_ids)}
                missing_slice_indices = set()
                for hid in missing_halo_ids:
                    pos = halo_id_to_pos.get(int(hid))
                    if pos is None:
                        continue
                    # slice_indices are positions; find slice containing pos
                    si = bisect.bisect_right(slice_indices, pos) - 1
                    if 0 <= si < (len(slice_indices) - 1):
                        missing_slice_indices.add(int(si))

                    if missing_slice_indices:
                        log_msg(
                            f"Missing Halo IDs map to slices: {sorted(missing_slice_indices)[:20]}"
                            f"{' ...' if len(missing_slice_indices) > 20 else ''}. "
                            "Clearing only these slice files..."
                        )
                    for si in sorted(missing_slice_indices):
                        fp = os.path.join(output_path, f"halos_kdtree_{si}_{int(R_max)}RV.h5")
                        clear_output_file(fp)
                    log_msg("Clearing finished! Rebuilding from missing slices...")
                else:
                    log_msg("[WARN] Missing Halo IDs could not be mapped to slices; no files cleared.")
                return True

            # 4.5 额外一致性：检查 ID 范围与数量（有助于发现排序/截断问题）
            if valid_id_set:
                vmin = min(valid_id_set)
                vmax = max(valid_id_set)
                if vmin != 0 or vmax != (num_valid_halos - 1):
                    log_msg(
                        f"[WARN] valid_halo_ids are not contiguous 0..N-1 "
                        f"(min={vmin}, max={vmax}, N={num_valid_halos}). "
                        "This can indicate non-stable IDs across runs."
                    )

            log_msg("All KDTree files are consistent and up-to-date.")
            return False

        M_filter_low = 10**filter_M200_low/1e10
        M_filter_up = 10**filter_M200_up/1e10
        print(f"M_filter_low = {M_filter_low:.3e} (10^{filter_M200_low})")
        print(f"M_filter_up  = {M_filter_up:.3e} (10^{filter_M200_up})")
        # 计算有效 Halo 数
        print(f"M_200: {haloMV[:10]}")
        print(f"M_FoF: {halomass[:10]}")
        valid_halos = [
            (HaloID, center, R200, M200)
            for (HaloID, center, R200, M200) in zip(haloIDs, halopos, haloRV, haloMV)
            if M200 >= 0
        ]
        num_valid_halos = len(valid_halos)
        print(f"Number of valid halos: {num_valid_halos}")

        # --- mass bin selection
        # Your original code suggests M200 is stored in units of 1e10.
        # If so, log10(M200_phys) = log10(M200*1e10). Here we implement a selection
        # by converting log10 bounds into that internal unit.

        target_halo_ids = set([
            HaloID for (HaloID, _, _, M200) in valid_halos
            if (M_filter_low <= M200 <= M_filter_up)
        ])
        print(f"Number of halos in mass range: {len(target_halo_ids)}")

        # --- build halo_data arrays (order is same as valid_halos)
        halo_data = {
            "positions": np.array([center for _, center, _, _ in valid_halos]),
            "R200": np.array([R200 for _, _, R200, _ in valid_halos]),
            "HaloIDs": np.array([HaloID for HaloID, _, _, _ in valid_halos]),
        }

        # --- build a dict for halo properties so we can write per-halo M200/R200
        halo_prop_by_id = {
            int(HaloID): {"M200": float(M200), "R200": float(R200), "center": center}
            for (HaloID, center, R200, M200) in valid_halos
        }
        
        # --- slice_indices: keep your existing construction
        # NOTE: IMPORTANT BUGFIX from your storage code:
        # file_halos should be built by halo index range, not halo_id values.
        # So always use start/end as *indices*.
        if num_valid_halos // 100000 == 0:
            slice_indices = np.unique(np.linspace(0, num_valid_halos, slice_count + 1, dtype=int))
        else:
            linear_fraction = 0.00001
            num_linear_slices = int(slice_count * linear_fraction)
            linear_indices = np.linspace(0, int(num_valid_halos * linear_fraction), num_linear_slices, dtype=int)
            log_indices = np.floor(
                np.logspace(
                    np.log10(max(int(num_valid_halos * linear_fraction), 1)),
                    np.log10(num_valid_halos),
                    slice_count - num_linear_slices + 1,
                )
            ).astype(int)
            slice_indices = np.unique(np.concatenate([linear_indices, log_indices]))
        if slice_indices[0] != 0:
            slice_indices = np.insert(slice_indices, 0, 0)
        if slice_indices[-1] != num_valid_halos:
            slice_indices = np.append(slice_indices, num_valid_halos)

        # Conditional tail split:
        # - keep current binning when C_R_max <= 10
        # - when C_R_max > 10, split slices after index 63 into chunks of 20000 halos
        C_R_max = R_max
        if C_R_max > 10:
            split_from_slice = 63
            halos_per_slice_after_63 = 20000
            current_num_slices = len(slice_indices) - 1
            if current_num_slices > split_from_slice:
                tail_start = int(slice_indices[split_from_slice])
                tail_end = int(slice_indices[-1])
                tail_points = np.arange(
                    tail_start + halos_per_slice_after_63,
                    tail_end,
                    halos_per_slice_after_63,
                    dtype=int,
                )
                slice_indices = np.concatenate(
                    [
                        slice_indices[: split_from_slice + 1],
                        tail_points,
                        np.array([tail_end], dtype=int),
                    ]
                )
                slice_indices = np.unique(slice_indices.astype(int))
                print(
                    f"[DBG] C_R_max={C_R_max}>10: re-sliced tail from slice {split_from_slice}, "
                    f"chunk={halos_per_slice_after_63}, num_slices={len(slice_indices)-1}"
                )
        print(slice_indices)
        # ---- debug: slice diagnostics ----
        num_slices = len(slice_indices) - 1
        empty_slices = [
            (i, int(slice_indices[i]), int(slice_indices[i + 1]))
            for i in range(num_slices)
            if slice_indices[i] >= slice_indices[i + 1]
        ]
        print(f"[DBG] slice_count={slice_count}, num_slices={num_slices}")
        print(f"[DBG] first/last slices: {slice_indices[:10]} ... {slice_indices[-10:]}")
        if empty_slices:
            print(f"[DBG] empty slices (start>=end): {empty_slices[:10]}")
        # ---- end debug ----
        # 检查 KDTree 文件是否需要重建
        if skip_halo_build:
            log_msg("[DBG] skip_halo_build=True: force skip halo KDTree build and use existing files.")
            need_rebuild = False
        elif resume and (not completion_mode):
            log_msg("[DBG] resume-only mode: skip pre-check, start from last_existing+1.")
            need_rebuild = True
        else:
            need_rebuild = check_and_clear_halo_files(
                region_kdtree_path,
                num_valid_halos,
                halo_data["HaloIDs"],
                slice_indices=slice_indices,
                R_max=R_max,
                resume=resume,
            )

        if need_rebuild:
            build_data_for_halos_with_figm(
                particle_data=particle_data,
                box_size=box_size_physical,
                halo_data=halo_data,
                output_path=region_kdtree_path,
                region_kdtree_path=region_kdtree_path,
                slice_indices=slice_indices,
                region_size=region_size_physical,
                n_jobs=n_jobs,
                C_search=C_search,
                group_size=group_size,
                R_max = R_max,
                resume=resume,
                completion_mode=completion_mode,
                completion_start_slice=completion_start_slice,
            )
            
            log_msg("Halo kdtree rebuilt and saved.")
        else:
            log_msg("Halo KDTree files are up-to-date. Skipping KDTree build.")
        halo_id_to_index = {int(hid): i for i, hid in enumerate(halo_data["HaloIDs"])}
        # Halo 数据的并行处理
        # 并行处理
        # ---- ONE parallel call for all C ----
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_halo_multiR)(
                halo_id=halo_id,
                halo_center=halo_center,
                halo_r200=true_R200,
                rho_crit_Msun_Mpc3=rho_crit_Msun_Mpc3,
                region_kdtree_path=region_kdtree_path,
                slice_indices=slice_indices,
                C_list=C_list,
                C_search=C_search,
                if_Cal_R200=if_Cal_R200,
                box_size=box_size_physical,
                R_max = R_max,
                halo_id_to_index=halo_id_to_index,
            )
            for (halo_id, halo_center, true_R200, true_M200) in valid_halos
        )
        # Defensive: joblib should return a list, but guard against None to avoid TypeError
        if results is None:
            results = []
        #for (halo_id, halo_center, true_R200, true_M200) in valid_halos:
        #    results=process_halo_multiR(
        #            halo_id=halo_id,
        #            halo_center=halo_center,
        #            halo_r200=true_R200,
        #            rho_crit_Msun_Mpc3=rho_crit_Msun_Mpc3,
        #            region_kdtree_path=region_kdtree_path,
        #            slice_indices=slice_indices,
        #            C_list=C_list,
        #            C_search=C_search,
        #            if_Cal_R200=if_Cal_R200,
        #            box_size=box_size_physical,
        #            R_max = R_max
        #    )

        # ---- select only target halo ids in mass bin ----
        target_results = [
            r for r in results
            if (r is not None) and (int(r.get("halo_id", -1)) in target_halo_ids)
        ]

        sys.stdout.flush()
        # ---- write ONE profile txt (Mode B) ----
        out_profile_txt = os.path.join(
            out_txt_dir,
            f"{txt_prefix}_{label}_z{z:.2f}_M{filter_M200_low}_{filter_M200_up}_Rmax{R_max}_N{R_num}_with_fgas.txt",
        )
        header_extra = (
            f"mass_bin_log10 = [{filter_M200_low}, {filter_M200_up}]\n"
            f"R_max = {R_max}, R_num = {R_num}\n"
            f"C_search = {C_search}\n"
            f"selection_unit_note: assumes M200 stored in units of 1e10 in catalogue\n"
            f"box_size = {box_size_physical}  (same units as positions/R200)\n"
            f"Cosmo_Param: h = {h}, Omega_m = {Omega_0}, Omega_lambda = {OmegaLambda}, Omega_b = {OmegaBaryon}" 
        )
        write_per_halo_rows = str(os.getenv("WRITE_PER_HALO_ROWS", "1")).lower() not in ("0", "false", "no")

        _write_fb_profile_txt(
            out_profile_txt,
            z=z,
            C_list=C_list,
            halo_prop_by_id=halo_prop_by_id,
            target_results=target_results,
            header_extra=header_extra,
            convert_M200_to_physical=True,
            write_per_halo_rows=write_per_halo_rows,
        )

        print(f"[OK] wrote Mode-B profile txt: {out_profile_txt}")


# ============================
# Main / MPI entry (fb(<Rcut>) version) — uses (R_max, R_num) grid
#
# You asked to control cut radii only by:
#   - R_max : maximum radius in units of R200 (e.g. 10)
#   - R_num : number of equal intervals (e.g. 20 -> step 0.5)
# Then C_list = linspace(0, R_max, R_num+1).
#
# Usage examples:
#   mpirun -np 2 python fb_vs_R_Paralell_MPI.py               # defaults: R_max=10, R_num=20, M=1e10-1e11, z~0
#   mpirun -np 2 python fb_vs_R_Paralell_MPI.py 10 20 10 11   # R_max=10, R_num=20, logM=[10,11]
#   mpirun -np 2 python fb_vs_R_Paralell_MPI.py 10 20 10 11 0.0 1e-2
#   mpirun -np 2 python fb_vs_R_Paralell_MPI.py 5  50 12 13  0.0 1e-2
# ============================

start_time = time.time()

# -------------------- MPI setup --------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size < 2:
    raise RuntimeError("This fb(<R) script requires at least 2 MPI ranks (rank0=noBH + rank1=fiducial).")

# -------------------- defaults --------------------
default_R_max = 10.0
default_R_num = 20

default_FM200_low = 10.0
default_FM200_up = 11.0

default_logT = 4.0
default_logrho = 3.5

default_z_target = 0.0
default_z_tol = 1e-3
default_skip_halo_build = True
default_resume = False
default_completion_mode = False
default_completion_start_slice = 63


# -------------------- CLI parsing --------------------
# argv:
#   [R_max] [R_num] [logM_low] [logM_up] [z_target] [z_tol]
# optional (keep compatible with your previous IGM-style knobs):
#   [logT] [logrho] can be appended later if needed
#   [resume] [completion_mode] [completion_start_slice] [skip_halo_build]
if len(sys.argv) == 1:
    R_max = default_R_max
    R_num = default_R_num
    filter_M200_low = default_FM200_low
    filter_M200_up = default_FM200_up
    z_target = default_z_target
    z_tol = default_z_tol
    log_T_thre = default_logT
    log_rho_thre = default_logrho
    resume = default_resume
    completion_mode = default_completion_mode
    completion_start_slice = default_completion_start_slice
    skip_halo_build = default_skip_halo_build
else:
    # required: R_max, R_num
    R_max = float(ast.literal_eval(sys.argv[1])) if len(sys.argv) > 1 else default_R_max
    R_num = int(ast.literal_eval(sys.argv[2])) if len(sys.argv) > 2 else default_R_num

    # optional: mass bin
    filter_M200_low = float(ast.literal_eval(sys.argv[3])) if len(sys.argv) > 3 else default_FM200_low
    filter_M200_up = float(ast.literal_eval(sys.argv[4])) if len(sys.argv) > 4 else default_FM200_up

    # optional: z selection
    z_target = float(ast.literal_eval(sys.argv[5])) if len(sys.argv) > 5 else default_z_target
    z_tol = float(ast.literal_eval(sys.argv[6])) if len(sys.argv) > 6 else default_z_tol

    # optional: thermodynamic thresholds (kept for compatibility; you may ignore in fb)
    log_T_thre = float(ast.literal_eval(sys.argv[7])) if len(sys.argv) > 7 else default_logT
    log_rho_thre = float(ast.literal_eval(sys.argv[8])) if len(sys.argv) > 8 else default_logrho
    resume = bool(ast.literal_eval(sys.argv[9])) if len(sys.argv) > 9 else default_resume
    completion_mode = bool(ast.literal_eval(sys.argv[10])) if len(sys.argv) > 10 else default_completion_mode
    completion_start_slice = int(ast.literal_eval(sys.argv[11])) if len(sys.argv) > 11 else default_completion_start_slice
    skip_halo_build = bool(ast.literal_eval(sys.argv[12])) if len(sys.argv) > 12 else default_skip_halo_build

if R_num <= 0:
    raise ValueError(f"R_num must be positive. Got {R_num}")
if R_max <= 0:
    raise ValueError(f"R_max must be positive. Got {R_max}")

# C_list grid (avoid C=0). Use a small floor, but don't exceed R_max/R_num.
C_0_default = 0.05
C_min = min(C_0_default, R_max / R_num)
C_list = np.linspace(C_min, R_max, R_num)
max_C = float(np.max(C_list))

if rank == 0:
    step = R_max / R_num
    print(f"R_max={R_max}, R_num={R_num} -> step={step} ; C_list[0:5]={C_list[:5]} ... C_list[-1]={C_list[-1]}")
    print(f"mass_bin_log10=[{filter_M200_low},{filter_M200_up}], z_target={z_target}, z_tol={z_tol}")
    print(
        f"resume={resume}, completion_mode={completion_mode}, "
        f"completion_start_slice={completion_start_slice}, "
        f"skip_halo_build={skip_halo_build}"
    )
    sys.stdout.flush()

# -------------------- resources --------------------
#n_jobs = _resolve_n_jobs(size, default_div=1)
n_jobs = max(1, multiprocessing.cpu_count() - 1)

n_jobs = 37

# -------------------- shared params --------------------
box_size = 100.0
region_size = 5.0
massrange_key = f"10e{int(filter_M200_low)}-10e{filter_M200_up}"
fb_output_path_massrange = os.path.join(fb_output_path, massrange_key)
if not os.path.exists(fb_output_path_massrange):
    # 创建目录
    os.makedirs(fb_output_path_massrange)
# Build halo KDTree ONCE with a search radius that covers the largest cut.
# If your KDTree builder uses C_search to decide which particles to store, this is critical.

# -------------------- run tasks per rank (FINAL Mode B) --------------------
# 建议：C_search 覆盖 R_max
C_search = max(5, int(np.ceil(R_max)))
print(f"Search Radius is {C_search} RV")

def _mass_tag(lo, up):
    return f"M{str(lo).replace('.','p')}_{str(up).replace('.','p')}"

mass_tag = _mass_tag(filter_M200_low, filter_M200_up)
r_tag = f"Rmax{R_max}_N{R_num}"

# 每个 rank 一个 logfile（避免循环覆盖 & stdout 漏恢复）
logfile = os.path.join(
    fb_output_path_massrange,
    f"logfile_fb_profile_{mass_tag}_{r_tag}_rank{rank}.txt"
)
old_stdout = sys.stdout
sys.stdout = open(logfile, "w")

try:
    if rank == 0:
        label = "n"
        print(f"[Rank0] noBH Mode-B fb profile: {mass_tag}, {r_tag}")
        calculate_f_b_with_regions_and_kdtree(
            base_path=base_path_noAGN,
            snapshots=snapshots_noAGN,
            output_file=f"{storage_path}/halo_fb_profile_{label}_{mass_tag}_{r_tag}.hdf5",
            kdtree_output_path=kdtree_output_path,
            label=label,
            box_size=box_size,
            region_size=region_size,
            chunk_size=1000000,
            n_jobs=n_jobs,
            C_search=C_search,
            # Mode B controls
            R_max=R_max,
            R_num=R_num,
            C_list=C_list,
            filter_M200_low=filter_M200_low,
            filter_M200_up=filter_M200_up,
            if_Cal_R200=False,
            slice_count=64,
            group_size=5,
            log_T_thre=log_T_thre,
            log_rho_thre=log_rho_thre,
            z_target=z_target,
            z_tol=z_tol,
            only_one_snapshot=True,
            out_txt_dir=fb_output_path_massrange,
            txt_prefix=f"fb_profile_{mass_tag}_{r_tag}",
            resume=resume,
            completion_mode=completion_mode,
            completion_start_slice=completion_start_slice,
            skip_halo_build=skip_halo_build,
        )

    elif rank == 1:
        label = "f"
        print(f"[Rank1] fiducial Mode-B fb profile: {mass_tag}, {r_tag}")
        calculate_f_b_with_regions_and_kdtree(
            base_path=base_path_fiducial,
            snapshots=snapshots_fiducial,
            output_file=f"{storage_path}/halo_fb_profile_{label}_{mass_tag}_{r_tag}.hdf5",
            kdtree_output_path=kdtree_output_path,
            label=label,
            box_size=box_size,
            region_size=region_size,
            chunk_size=1000000,
            n_jobs=n_jobs,
            C_search=C_search,
            R_max=R_max,
            R_num=R_num,
            C_list=C_list,
            filter_M200_low=filter_M200_low,
            filter_M200_up=filter_M200_up,
            if_Cal_R200=False,
            slice_count=64,
            group_size=5,
            log_T_thre=log_T_thre,
            log_rho_thre=log_rho_thre,
            z_target=z_target,
            z_tol=z_tol,
            only_one_snapshot=True,
            out_txt_dir=fb_output_path_massrange,
            txt_prefix=f"fb_profile_{mass_tag}_{r_tag}",
            resume=resume,
            completion_mode=completion_mode,
            completion_start_slice=completion_start_slice,
            skip_halo_build=skip_halo_build,
        )

finally:
    sys.stdout.close()
    sys.stdout = old_stdout

comm.Barrier()

if rank == 0:
    elapsed = (time.time() - start_time) / 3600.0
    print(f"[DONE] Mode-B fb profile finished in {elapsed:.2f} hours")
    print(f"Outputs: one profile TXT per label in {fb_output_path_massrange}")

    # ---- Plot (robust + HPC-safe) ----
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _find_profile_txt(label_char):
        # 你的输出格式: f"{txt_prefix}_{label}_z{z:.3f}_M..._Rmax..._N....txt"
        # 这里用通配 z* 来适配浮点误差
        pattern = os.path.join(
            fb_output_path_massrange,
            f"fb_profile_{mass_tag}_{r_tag}_{label_char}_z*_M{filter_M200_low}_{filter_M200_up}_Rmax{R_max}_N{R_num}.txt",
        )
        hits = sorted(glob.glob(pattern))
        return hits[-1] if hits else None

    def _read_header_arrays(path):
        C_list = fb_median = fb_p16 = fb_p84 = f_b_cosmic = None
        with open(path, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                if line.startswith("# Cosmo_Param"):
                    parts = [p.strip() for p in line.split(",")]
                    Omega_m = float(parts[1].split("=")[1].strip())
                    Omega_b = float(parts[3].split("=")[1].strip())
                    f_b_cosmic = Omega_b / Omega_m
                if line.startswith("# C_list"):
                    C_list = np.array([float(x) for x in line.split("=", 1)[1].strip().split(",")])
                elif line.startswith("# fb_median"):
                    fb_median = np.array([float(x) if x != "nan" else np.nan for x in line.split("=", 1)[1].split()])
                elif line.startswith("# fb_p16"):
                    fb_p16 = np.array([float(x) if x != "nan" else np.nan for x in line.split("=", 1)[1].split()])
                elif line.startswith("# fb_p84"):
                    fb_p84 = np.array([float(x) if x != "nan" else np.nan for x in line.split("=", 1)[1].split()])
        return C_list, fb_median, fb_p16, fb_p84, f_b_cosmic

    txt_n = _find_profile_txt("n")  # noBH
    txt_f = _find_profile_txt("f")  # fiducial

    plt.figure(figsize=(8, 6))
    for sim_name, path in [("noBH", txt_n), ("fiducial", txt_f)]:
        if path is None:
            print(f"[WARN] profile txt for {sim_name} not found; skip plotting.")
            continue

        C_list, fb_med, fb_p16, fb_p84, f_b_cosmic = _read_header_arrays(path)
        if C_list is None or fb_med is None:
            print(f"[WARN] failed to read header arrays from {path}; skip plotting.")
            continue

        plt.plot(C_list, fb_med/f_b_cosmic, label=f"median ({sim_name})")
        if fb_p16 is not None and fb_p84 is not None:
            plt.fill_between(C_list, fb_p16/f_b_cosmic, fb_p84, alpha=0.2, label=f"p16–p84 ({sim_name})")

    plt.xlabel("$R/R_{200}$")
    plt.ylabel("$f_b(<R)/f_{\mathrm{b, cosmic}}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(fb_output_path_massrange, f"fb_profile_{mass_tag}_{r_tag}_z{z_target}_plot.png")
    plt.savefig(out_png, dpi=200)
    print(f"[OK] saved plot: {out_png}")
