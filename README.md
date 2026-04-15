# FEM Python

这是一个面向教学和实验的有限元项目，当前仓库以 `src/fem` 作为核心库，以 `examples` 中的脚本作为使用入口。项目覆盖了网格读取、自由度管理、单元刚度计算、全局装配、边界条件处理、线性求解，以及结果导出。

## 当前支持

- 读取 Abaqus `.inp` 和部分 CSV 网格
- 支持 `Truss2D`、`Beam2D`、`Tri3`、`Quad4`、`Quad8`、`Hex8`、`Tet4`、`Tet10`
- 支持位移边界、集中力、体力、边力/面力、重力
- 支持稀疏刚度矩阵装配与线性静力求解
- 导出节点位移、单元/节点应力以及 VTK 可视化文件

## 仓库结构

- `src/fem/`
  核心有限元库
- `examples/`
  示例模型、输入网格和材料数据
- `README.md`
  项目说明
- `requirements.txt`
  Python 依赖
- `.python-version`
  本地开发时使用的 Python 版本标记

## 核心模块

- `src/fem/mesh.py`
  定义节点、单元和不同类型网格容器
- `src/fem/dof_manager.py`
  管理节点到全局自由度编号的映射
- `src/fem/mesh_io.py`
  读取 Abaqus/CSV 网格和材料数据
- `src/fem/stiffness.py`
  计算各类单元刚度矩阵
- `src/fem/assemble.py`
  装配全局刚度矩阵
- `src/fem/boundary.py`
  构造载荷向量并施加边界条件
- `src/fem/solve.py`
  求解线性方程组
- `src/fem/post.py`
  导出位移、应力和 VTK 结果
- `src/fem/helper.py`
  按几何位置选择节点或边，用于施加载荷和约束

## 运行环境

推荐使用 Python 3.13；至少应为 Python 3.10 及以上。

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

说明：

- 当前仓库没有 `pyproject.toml` 或 `setup.py`，因此示例脚本依赖 `PYTHONPATH=src` 才能导入 `fem`
- 如果在 IDE 中运行示例，也需要把 `src` 标记为源码目录，或加入解释器搜索路径

## 示例脚本

当前仓库中已有 4 个主示例：

- `examples/plate_with_hole_quad4.py`
  2D `Quad4` 开孔板
- `examples/plate_with_hole_quad8.py`
  2D `Quad8` 开孔板
- `examples/cantilever_beam_quad8.py`
  2D `Quad8` 悬臂梁
- `examples/cantilever_beam_hex8.py`
  3D `Hex8` 悬臂梁

运行示例时，在已激活虚拟环境且设置好 `PYTHONPATH` 后执行，例如：

```powershell
python examples\plate_with_hole_quad4.py
python examples\cantilever_beam_hex8.py
```

## 标准求解流程

示例脚本遵循同一条主线：

1. 用 `mesh_io` 读取网格和材料
2. 用 `stiffness` 计算单元刚度
3. 用 `assemble` 装配全局刚度矩阵
4. 用 `boundary` 定义约束和载荷
5. 用 `solve` 求解位移
6. 用 `post` 导出位移、应力和可视化结果
