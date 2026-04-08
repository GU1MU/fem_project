from typing import Callable, Iterable, Sequence, List, Tuple
from scipy.sparse import coo_matrix
import numpy as np

def assemble_global_stiffness(
    num_dofs: int,
    elements: Iterable,
    get_element_dofs: Callable[[object], Sequence[int]],
    compute_element_stiffness: Callable[[object], np.ndarray],
) -> np.ndarray:
    """Assemble a dense global stiffness matrix."""
    K = np.zeros((num_dofs, num_dofs), dtype=float)

    for elem in elements:
        # 单元刚度矩阵
        Ke = compute_element_stiffness(elem)  

        # 单元对应的全局 DOF 编号
        elem_dofs = list(get_element_dofs(elem))
        nd = len(elem_dofs)

        if Ke.shape != (nd, nd):
            raise ValueError(
                f"单元刚度矩阵 Ke 维度 {Ke.shape} 与 DOF 数量 {nd} 不匹配"
            )

        for a, I in enumerate(elem_dofs):
            for b, J in enumerate(elem_dofs):
                K[I, J] += Ke[a, b]

    return K

def assemble_global_stiffness_sparse(
    num_dofs: int,
    num_elements: int,
    get_element_dofs: Callable[[int], Sequence[int]],
    compute_element_stiffness: Callable[[int], np.ndarray],
) -> Tuple[List[int], List[int], List[float]]:
    """Assemble a sparse global stiffness matrix."""
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for eid in range(num_elements):
        Ke = compute_element_stiffness(eid)
        dofs = list(get_element_dofs(eid))

        nd = len(dofs)
        if Ke.shape != (nd, nd):
            raise ValueError(
                f"单元 {eid} 的 Ke 形状 {Ke.shape} 与 DOF 数量 {nd} 不匹配"
            )

        for a in range(nd):
            I = dofs[a]
            if I < 0 or I >= num_dofs:
                raise IndexError(f"单元 {eid} DOF 索引 I={I} 越界 [0, {num_dofs})")

            for b in range(nd):
                J = dofs[b]
                if J < 0 or J >= num_dofs:
                    raise IndexError(f"单元 {eid} DOF 索引 J={J} 越界 [0, {num_dofs})")

                rows.append(I)
                cols.append(J)
                data.append(float(Ke[a, b]))

    K_sparse = coo_matrix((data, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr()

    return K_sparse