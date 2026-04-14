"""Build a rotating-box FRB DM light cone from CROCODILE fullbox snapshots.

Compared with the no-rotation version, this script assigns one shared random
rotation and one shared random translation to each radial shell so that the
shell-to-shell box repetition is suppressed without erasing angular
correlations inside the shell.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.spatial import cKDTree

OMEGA_M_DEFAULT = 0.3090
OMEGA_L_DEFAULT = 0.6910
C_KM_S = 299792.458


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a rotating-box DM light cone from CROCODILE grid data.')
    parser.add_argument('--base-dir', required=True, help='Directory containing fullbox_###_*_lv*.hdf5 snapshots')
    parser.add_argument('--agn-tag', default='f', help='Feedback tag in snapshot filenames, e.g. n or f')
    parser.add_argument('--level', default='8', help='Grid refinement level used in the filenames')
    parser.add_argument('--snap-min', type=int, required=True, help='Minimum snapshot index to include')
    parser.add_argument('--snap-max', type=int, required=True, help='Maximum snapshot index to include')
    parser.add_argument('--nlos', type=int, default=30000, help='Number of lines of sight to generate')
    parser.add_argument('--z-max', type=float, default=1.0, help='Maximum redshift of the cone')
    parser.add_argument('--step-mult', type=float, default=1.0, help='Internal radial step in units of the grid cell size')
    parser.add_argument('--n-profile', type=int, default=300, help='Number of cumulative DM(z) samples stored in the output file')
    parser.add_argument('--overlap-bins', type=int, default=16, help='Number of blending bins across snapshot boundaries')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for observer, directions, and shell transforms')
    parser.add_argument('--no-rotate-shells', action='store_true', help='Disable per-shell random rotations')
    parser.add_argument('--no-translate-shells', action='store_true', help='Disable per-shell random translations')
    parser.add_argument('--output', help='Optional output HDF5 path; defaults to <base-dir>/Ray_tracing/...')
    parser.add_argument('--verbose', action='store_true', help='Print progress information while tracing the cone')
    return parser.parse_args()


def comoving_distance(z: float, omega_m: float, omega_l: float, h0: float) -> float:
    if z <= 0.0:
        return 0.0
    integrand = lambda zp: 1.0 / np.sqrt(omega_m * (1.0 + zp) ** 3 + omega_l)
    integral, _ = quad(integrand, 0.0, z)
    return (C_KM_S / h0) * integral


def find_redshift_from_chi(chi: float, omega_m: float, omega_l: float, h0: float, zmax: float = 10.0) -> float:
    if chi <= 0.0:
        return 0.0
    func = lambda z: comoving_distance(z, omega_m, omega_l, h0) - chi
    sol = root_scalar(func, bracket=[0.0, zmax])
    if not sol.converged:
        raise RuntimeError('root_scalar did not converge while inverting chi(z)')
    return float(sol.root)


def fibonacci_sphere_directions(n_dir: int, rng: np.random.Generator) -> np.ndarray:
    rnd = rng.random() * n_dir
    i = np.arange(n_dir, dtype=np.float64)
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    phi = 2.0 * math.pi * (i + rnd) / golden
    z = 1.0 - 2.0 * (i + 0.5) / n_dir
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack([x, y, z], axis=1)


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    q2 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


@dataclass
class SnapshotGrid:
    path: str
    z: float
    a: float
    h: float
    L_code: float
    cell_size_code: float
    grid_len: int
    kdtree: cKDTree
    ne_cgs: np.ndarray


@dataclass
class OverlapPlan:
    blend: np.ndarray
    snap_a: np.ndarray
    snap_b: np.ndarray
    alpha: np.ndarray


@dataclass
class ShellTransform:
    R: np.ndarray
    t: np.ndarray
    L: float

    def world_to_sim(self, x_world: np.ndarray) -> np.ndarray:
        x = x_world - self.t
        x = (self.R @ x.T).T
        return np.mod(x, self.L)


@dataclass
class RayConeConfig:
    base_dir: str
    agn_tag: str
    level_input: str
    snap_min: int = 15
    snap_max: int = 20
    n_dir: int = 30000
    seed: int = 0
    z_max: float = 1.0
    step_mult: float = 1.0
    n_profile: int = 300
    overlap_bins: int = 16
    rotate_each_shell: bool = True
    translate_each_shell: bool = True
    x_obs: Optional[np.ndarray] = None
    omega_m: Optional[float] = None
    omega_l: Optional[float] = None
    verbose: bool = True


def compute_electron_density_cgs(rho: np.ndarray, xe: np.ndarray, a: float, h: float, a_scaling: float, h_scaling: float, mp_g: float = 1.67262192369e-24) -> np.ndarray:
    ne = rho * 6.77e-31 * xe / mp_g
    ne *= (a ** a_scaling) * (h ** h_scaling)
    return ne


def load_fullbox_snapshot(file_path: str) -> SnapshotGrid:
    with h5py.File(file_path, 'r') as handle:
        gridpos = handle['/Cell/GridPos'][:]
        xe = handle['/Cell/ElectronAbundance'][:]
        rho = handle['/Cell/GasDensity'][:]
        box_size = float(handle['Parameters'].attrs['BoxSize'])
        hubble = float(handle['Parameters'].attrs['HubbleParam'])
        a = float(handle['/Header'].attrs['Time'])
        z = float(handle['/Header'].attrs['Redshift'])
        cell_size = float(handle['Header'].attrs.get('CellSize', np.nan))
        grid_len = int(handle['Header'].attrs.get('GridLen', 0))
        if (not np.isfinite(cell_size)) or cell_size <= 0.0:
            cell_size = (box_size / float(grid_len)) if grid_len > 0 else (box_size / 128.0)
        a_scaling = float(handle['/Cell/GasDensity'].attrs.get('a_scaling', 0.0))
        h_scaling = float(handle['/Cell/GasDensity'].attrs.get('h_scaling', 0.0))
    ne = compute_electron_density_cgs(rho, xe, a, hubble, a_scaling, h_scaling)
    return SnapshotGrid(
        path=file_path,
        z=z,
        a=a,
        h=hubble,
        L_code=box_size,
        cell_size_code=cell_size,
        grid_len=grid_len,
        kdtree=cKDTree(gridpos),
        ne_cgs=ne,
    )


def sample_ne_kdtree(grid: SnapshotGrid, x_sim: np.ndarray) -> np.ndarray:
    x_flat = x_sim.reshape(-1, 3)
    _, idx = grid.kdtree.query(x_flat, k=1)
    return grid.ne_cgs[idx].reshape(x_sim.shape[:-1])


def build_overlap_plan(snap_idx: np.ndarray, overlap_bins: int) -> OverlapPlan:
    nbin = int(snap_idx.size)
    blend = np.zeros(nbin, dtype=bool)
    snap_a = np.full(nbin, -1, dtype=np.int32)
    snap_b = np.full(nbin, -1, dtype=np.int32)
    alpha = np.zeros(nbin, dtype=np.float64)
    m = int(max(0, overlap_bins))
    if m == 0:
        return OverlapPlan(blend=blend, snap_a=snap_a, snap_b=snap_b, alpha=alpha)
    switches = np.where(snap_idx[1:] != snap_idx[:-1])[0] + 1
    for ks in switches:
        k0 = int(ks)
        k1 = int(min(nbin, ks + m))
        if k0 >= k1:
            continue
        a_idx = int(snap_idx[ks - 1])
        b_idx = int(snap_idx[ks])
        t = np.arange(k0, k1, dtype=np.float64) - float(k0)
        a = (t + 0.5) / float(k1 - k0)
        blend[k0:k1] = True
        snap_a[k0:k1] = a_idx
        snap_b[k0:k1] = b_idx
        alpha[k0:k1] = a
    return OverlapPlan(blend=blend, snap_a=snap_a, snap_b=snap_b, alpha=alpha)


def discover_snapshots(cfg: RayConeConfig) -> List[str]:
    pattern = re.compile(rf'fullbox_([0-9]+)_({cfg.agn_tag})_lv{cfg.level_input}[.]hdf5$')
    files = []
    for name in os.listdir(cfg.base_dir):
        match = pattern.match(name)
        if match is None:
            continue
        number = int(match.group(1))
        if cfg.snap_min <= number <= cfg.snap_max:
            files.append((number, os.path.join(cfg.base_dir, name)))
    files.sort(key=lambda item: item[0])
    return [path for _, path in files]


def load_snapshots(cfg: RayConeConfig) -> List[SnapshotGrid]:
    paths = discover_snapshots(cfg)
    if not paths:
        raise RuntimeError('No snapshots found. Check base_dir, agn_tag, level_input, and the snapshot range.')
    grids = [load_fullbox_snapshot(path) for path in paths]
    grids.sort(key=lambda grid: grid.z, reverse=True)
    return grids


def assign_snapshot_indices(z_mid: np.ndarray, grids: List[SnapshotGrid]) -> np.ndarray:
    z_snaps = np.array([grid.z for grid in grids], dtype=np.float64)
    return np.argmin(np.abs(z_mid[:, None] - z_snaps[None, :]), axis=1).astype(np.int32)


def find_mw_like_observer_from_fullbox(grid: SnapshotGrid, rng: np.random.Generator) -> np.ndarray:
    gridpos = grid.kdtree.data
    ne = grid.ne_cgs
    mask = (ne > 1e-4) & (ne < 1e-3)
    candidates = gridpos[mask]
    if len(candidates) == 0:
        return rng.uniform(0.0, grid.L_code, size=3)
    return candidates[rng.integers(0, len(candidates))]


def build_dm_cone(cfg: RayConeConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    rng = np.random.default_rng(cfg.seed)
    grids = load_snapshots(cfg)
    h = float(grids[0].h)
    h0 = 100.0 * h
    omega_m = float(cfg.omega_m) if cfg.omega_m is not None else float(OMEGA_M_DEFAULT)
    omega_l = float(cfg.omega_l) if cfg.omega_l is not None else float(OMEGA_L_DEFAULT)
    box_size = float(grids[0].L_code)
    cell_size_code = float(grids[0].cell_size_code)
    x_obs = find_mw_like_observer_from_fullbox(grids[0], rng) if cfg.x_obs is None else np.asarray(cfg.x_obs, dtype=np.float64)
    n_hat = fibonacci_sphere_directions(cfg.n_dir, rng)
    dchi_step_mpc = cfg.step_mult * (cell_size_code / h)
    chi_max = comoving_distance(cfg.z_max, omega_m, omega_l, h0)
    nbin = max(cfg.n_profile, int(np.ceil(chi_max / dchi_step_mpc)))
    chi_edges = np.linspace(0.0, chi_max, nbin + 1)
    chi_mid = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    dchi_mpc = chi_edges[1:] - chi_edges[:-1]
    z_mid = np.array([
        find_redshift_from_chi(float(chi), omega_m, omega_l, h0, zmax=max(2.0, cfg.z_max * 2.0))
        for chi in chi_mid
    ], dtype=np.float64)
    out_idx = np.unique(np.linspace(0, nbin - 1, max(1, cfg.n_profile), dtype=np.int64))
    dm_out = np.zeros((cfg.n_dir, out_idx.size), dtype=np.float32)
    z_out = z_mid[out_idx]
    snap_idx = assign_snapshot_indices(z_mid, grids)
    plan = build_overlap_plan(snap_idx, int(cfg.overlap_bins))
    shell_transforms = []
    for _ in range(nbin):
        rotation = random_rotation_matrix(rng) if cfg.rotate_each_shell else np.eye(3)
        shift = rng.uniform(0.0, box_size, size=3) if cfg.translate_each_shell else np.zeros(3)
        shell_transforms.append(ShellTransform(R=rotation, t=shift, L=box_size))
    dm_running = np.zeros(cfg.n_dir, dtype=np.float64)
    out_ptr = 0
    progress_step = max(1, nbin // 10)
    for k in range(nbin):
        chi_code = chi_mid[k] * h
        x_world = x_obs[None, :] + chi_code * n_hat
        x_sim = shell_transforms[k].world_to_sim(x_world)
        if not plan.blend[k]:
            ne = sample_ne_kdtree(grids[int(snap_idx[k])], x_sim)
        else:
            ne_a = sample_ne_kdtree(grids[int(plan.snap_a[k])], x_sim)
            ne_b = sample_ne_kdtree(grids[int(plan.snap_b[k])], x_sim)
            alpha = float(plan.alpha[k])
            ne = (1.0 - alpha) * ne_a + alpha * ne_b
        dl_pc = (dchi_mpc[k] * 1.0e6) / (1.0 + z_mid[k]) ** 2
        dm_running += ne * dl_pc
        if k == out_idx[out_ptr]:
            dm_out[:, out_ptr] = dm_running.astype(np.float32)
            out_ptr += 1
            if out_ptr == out_idx.size:
                out_ptr = out_idx.size - 1
        if cfg.verbose and (k % progress_step == 0):
            print(f'[PROG] k={k:4d}/{nbin} z~{z_mid[k]:.3f} DM_mean~{dm_running.mean():.2f} pc cm^-3')
    meta = {
        'N_los': int(cfg.n_dir),
        'N_bin': int(nbin),
        'N_out': int(z_out.size),
        'z_max': float(cfg.z_max),
        'h': float(h),
        'H0': float(h0),
        'Omega_m': float(omega_m),
        'Omega_L': float(omega_l),
        'pos_obs': np.asarray(x_obs, dtype=np.float64),
        'cell_size_code': float(cell_size_code),
        'cell_size_Mpc': float(dchi_mpc[0]),
        'overlap_bins': int(cfg.overlap_bins),
        'snap_redshifts': [float(grid.z) for grid in grids],
        'snap_paths': [grid.path for grid in grids],
        'mapping': 'shared_random_shell_transform',
        'dm_units': 'pc cm^-3',
        'rotate_each_shell': bool(cfg.rotate_each_shell),
        'translate_each_shell': bool(cfg.translate_each_shell),
    }
    return dm_running, dm_out, z_out, n_hat, meta


def save_dm_cone_hdf5(out_path: str, cfg: RayConeConfig, dm_tot: np.ndarray, dm_cum: np.ndarray, z_prof: np.ndarray, n_hat: np.ndarray, meta: Dict) -> str:
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    h = float(meta.get('h', np.nan))
    h0 = float(meta.get('H0', np.nan))
    omega_m = float(meta.get('Omega_m', np.nan))
    omega_l = float(meta.get('Omega_L', np.nan))
    hubble_param = np.nan
    box_size_code = np.nan
    time_a = np.nan
    grid_len = 0
    first_path = meta.get('snap_paths', [None])[0]
    if first_path is not None:
        try:
            with h5py.File(first_path, 'r') as handle:
                box_size_code = float(handle['Parameters'].attrs.get('BoxSize', np.nan))
                hubble_param = float(handle['Parameters'].attrs.get('HubbleParam', np.nan))
                time_a = float(handle['/Header'].attrs.get('Time', np.nan))
                grid_len = int(handle['Header'].attrs.get('GridLen', 0))
        except Exception:
            pass
    if not np.isfinite(hubble_param) and np.isfinite(h):
        hubble_param = h
    with h5py.File(out_path, 'w') as handle:
        g_header = handle.create_group('Header')
        g_data = handle.create_group('Data')
        g_meta = handle.create_group('Meta')
        g_header.attrs['BaseDir'] = str(cfg.base_dir)
        g_header.attrs['FeedbackTag'] = 'Fiducial' if str(cfg.agn_tag).lower() == 'f' else 'NoBH'
        g_header.attrs['AGNTag'] = str(cfg.agn_tag)
        g_header.attrs['Level'] = str(cfg.level_input)
        g_header.attrs['HubbleParam'] = float(hubble_param) if np.isfinite(hubble_param) else np.nan
        g_header.attrs['H0_km_s_Mpc'] = float(h0) if np.isfinite(h0) else np.nan
        g_header.attrs['Omega_m'] = float(omega_m) if np.isfinite(omega_m) else np.nan
        g_header.attrs['Omega_L'] = float(omega_l) if np.isfinite(omega_l) else np.nan
        g_header.attrs['Time_a'] = float(time_a) if np.isfinite(time_a) else np.nan
        g_header.attrs['BoxSize_ckpc_h'] = float(box_size_code) if np.isfinite(box_size_code) else np.nan
        g_header.attrs['GridLen'] = int(grid_len)
        g_header.attrs['z_max'] = float(cfg.z_max)
        g_header.attrs['overlap_bins'] = int(cfg.overlap_bins)
        g_header.attrs['N_los'] = int(meta.get('N_los', dm_tot.size))
        g_header.attrs['N_bin'] = int(meta.get('N_bin', z_prof.size))
        g_header.attrs['RotateEachShell'] = bool(cfg.rotate_each_shell)
        g_header.attrs['TranslateEachShell'] = bool(cfg.translate_each_shell)
        g_header.attrs['dm_units'] = 'pc cm^-3'
        g_header.attrs['angle_units'] = 'radian'
        g_header.attrs['distance_units'] = 'Mpc (comoving)'
        g_data.create_dataset('dm_tot', data=np.asarray(dm_tot, dtype=np.float64), compression='gzip', compression_opts=4)
        g_data.create_dataset('dm_cum', data=np.asarray(dm_cum, dtype=np.float64), compression='gzip', compression_opts=4)
        g_data.create_dataset('z_prof', data=np.asarray(z_prof, dtype=np.float64))
        g_data.create_dataset('n_hat', data=np.asarray(n_hat, dtype=np.float64))
        if 'snap_redshifts' in meta:
            g_meta.create_dataset('snap_redshifts', data=np.asarray(meta['snap_redshifts'], dtype=np.float64))
        if 'snap_paths' in meta:
            dt = h5py.string_dtype(encoding='utf-8')
            g_meta.create_dataset('snap_files', data=np.array([os.path.basename(path) for path in meta['snap_paths']], dtype=object), dtype=dt)
        if 'pos_obs' in meta:
            g_meta.create_dataset('pos_obs_code', data=np.asarray(meta['pos_obs'], dtype=np.float64))
    return out_path


def default_output_path(cfg: RayConeConfig) -> Path:
    z_str = f'{cfg.z_max:.2f}'.replace('.', 'p')
    return Path(cfg.base_dir) / 'Ray_tracing' / f'dm_cone_rotating_box_zmax{z_str}_lv{cfg.level_input}_{cfg.agn_tag}_Nlos{cfg.n_dir}.hdf5'


def main() -> None:
    args = parse_args()
    cfg = RayConeConfig(
        base_dir=args.base_dir,
        agn_tag=args.agn_tag,
        level_input=args.level,
        snap_min=args.snap_min,
        snap_max=args.snap_max,
        n_dir=args.nlos,
        seed=args.seed,
        z_max=args.z_max,
        step_mult=args.step_mult,
        n_profile=args.n_profile,
        overlap_bins=args.overlap_bins,
        rotate_each_shell=not args.no_rotate_shells,
        translate_each_shell=not args.no_translate_shells,
        verbose=args.verbose,
    )
    dm_tot, dm_cum, z_prof, n_hat, meta = build_dm_cone(cfg)
    out_path = args.output if args.output else default_output_path(cfg)
    save_dm_cone_hdf5(out_path, cfg, dm_tot, dm_cum, z_prof, n_hat, meta)
    print(f'[DONE] saved to {out_path}')


if __name__ == '__main__':
    main()
