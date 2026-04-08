from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def solve_linear_system(
    K: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """Solve linear system K @ U = F."""
    # 形状检查
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K 必须是方阵，当前形状为 {K.shape}")

    n = K.shape[0]

    F = np.asarray(F, dtype=float)
    if F.ndim == 2 and F.shape[1] == 1:
        F = F.ravel()  # (n,1) -> (n,)
    if F.ndim != 1 or F.shape[0] != n:
        raise ValueError(f"F 维度必须是长度为 {n} 的一维向量，当前形状为 {F.shape}")

    try:
        U = np.linalg.solve(K, F)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"线性方程组求解失败：{e}. "
            f"可能是刚度矩阵奇异(约束不充分)"
        )

    return U

def solve_linear_system_sparse(
    K: csr_matrix,
    F: np.ndarray,
) -> np.ndarray:
    """Solve sparse linear system K @ U = F."""
    if not isinstance(K, csr_matrix):
        raise TypeError(
            f"solve_linear_system_sparse 期望 K 是 csr_matrix 类型，收到 {type(K)}"
        )

    if K.shape[0] != K.shape[1]:
        raise ValueError(f"K 必须是方阵，当前形状为 {K.shape}")

    n = K.shape[0]

    F = np.asarray(F, dtype=float)

    if F.ndim == 2 and F.shape[1] == 1:
        F = F.ravel()

    if F.ndim != 1 or F.shape[0] != n:
        raise ValueError(
            f"F 必须是长度为 {n} 的一维向量，当前形状为 {F.shape}"
        )

    try:
        U = spsolve(K, F)
    except Exception as e:
        raise RuntimeError(
            f"稀疏线性方程组求解失败：{e}。"
            f"可能是刚度矩阵奇异(约束不充分)"
        )

    return U