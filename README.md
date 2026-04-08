# FEM Python

这是一个用于 2D 有限元分析的教学项目，覆盖网格读取、单元刚度、载荷/约束、求解与后处理导出。

## 功能概览
- 读取 Abaqus .inp 与 CSV 网格
- 支持 Truss2D、Beam2D、Tri3、Quad4、Quad8 单元
- 组装刚度矩阵并求解线性方程组
- 施加位移/力、体力与面力
- 导出位移、应力与 VTK 可视化结果

## 项目结构
- `src/fem/`：核心算法与数据结构
  - `mesh.py`：网格与节点/单元定义
  - `mesh_io.py`：网格与材料读入
  - `stiffness.py`：单元刚度与本构
  - `assemble.py`：全局装配
  - `boundary.py`：约束与载荷
  - `solve.py`：线性求解
  - `post.py`：后处理与导出
- `examples/`：示例脚本
- `for_thesis/`：论文相关的数据处理与绘图脚本
- `results/`：输出结果

*上述部分由AI生成*

## 配置环境
在项目文件夹下输入：

```bash
python -m venv .venv
pip install -r requirements.txt
```

需要3.10以上的python版本

## 示例代码

在examples有三个不同模型的示例代码，代码中有比较详细的注释