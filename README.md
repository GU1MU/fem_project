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

## 核心模块

- `src/fem/core/`
  定义节点、单元、网格容器、自由度映射和标准模型数据对象，包含集合、面、材料、截面和分析步框架
- `src/fem/abaqus/`
  读取 Abaqus 输入文件并构造 `FEMModel`
- `src/fem/io/`
  读取Abaqus/CSV网格数据和独立材料表，入口为 `io.inp`、`io.csv`、`io.materials`
- `src/fem/elements/`
  计算各类单元刚度矩阵
- `src/fem/materials/`
  定义材料本构矩阵，并把材料按element set赋给模型
- `src/fem/steps/`
  定义分析步、约束、载荷和输出请求
- `src/fem/assemble/`
  装配全局刚度矩阵
- `src/fem/boundary/`
  定义边界条件、构造载荷向量并施加约束
- `src/fem/solvers/`
  求解线性方程组，并提供静力线性求解流程
- `src/fem/post/`
  导出位移、应力和 VTK 结果
- `src/fem/selection/`
  按几何位置选择节点、边或面，并可生成模型中的集合和面对象

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

当前仓库中已有2个主示例：

- `examples/cantilever_beam_hex8.py`
  读取inp网格后的手写`mesh-model-solve-result`流程示例
- `examples/cantilever_beam_hex8_abaqus.py`
  从Abaqus输入文件读取完整模型的3D `Hex8`悬臂梁

运行示例时，在已激活虚拟环境且设置好 `PYTHONPATH` 后执行，例如：

```powershell
python examples\cantilever_beam_hex8.py
python examples\cantilever_beam_hex8_abaqus.py
```

## 标准求解流程

手写流程示例遵循同一条主线：

1. 用`fem.io`读取网格，并创建纯数据结构`FEMModel`
2. 用`selection`构造set，直接写入`model.node_sets`和`model.element_sets`
3. 用`materials`定义材料并按element set赋值
4. 用`steps`定义step、位移约束和载荷
5. 用`solvers.static_linear.solve()`求解并生成`ModelResult`
6. 用`post`导出位移、应力和可视化结果
