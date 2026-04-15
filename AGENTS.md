# AGENT.md

本文件用于约束本仓库中的 AI 协作行为。

## 基本原则

- 所有改动必须优先遵守现有项目结构、模块边界和数据流。
- 不要为了局部修改随意重构整体架构、改写模块职责，或引入与现有风格不一致的新层次。
- 在开始设计或修改前，先阅读 `docs/project_guides/architecture.md`，并以其中约束为准。

## 运行与导入约束

- 这个项目目前不是可直接安装的包。
- 直接执行 `python -c "import fem"` 会失败。
- 只有设置 `PYTHONPATH=src` 后才能导入 `fem`。
- 运行示例时应明确使用类似 `PYTHONPATH=src python examples/...` 的方式，或在 IDE / 终端中把 `src` 加入解释器搜索路径。
- agent 在编写、修改或说明运行步骤时，必须遵守这一约束，不能假设项目已经完成安装配置。

## AI 生成代码要求

- 任何由 AI 生成、补全或大幅改写的代码，都必须明确提醒协作者进行额外代码审查。
- 不要默认把 AI 生成代码视为可直接合入的最终结果。
- 如果改动涉及核心求解链路、单元刚度、边界条件、后处理或 I/O，必须进一步提高审查强度。

## AI 代码审查要求

- 对 AI 生成代码进行审查时，必须遵循 `docs/project_guides/review.md`。
- 审查重点应至少覆盖：正确性、架构一致性、边界条件、接口兼容性、数值逻辑、回归风险和测试覆盖。
- 如果 `review.md` 中的要求与当前实现存在冲突，应先暂停提交并与协作者确认。
- 如果 agent 在编写代码时暴露出新的典型错误，或在 review 中识别出具有复用价值的错误模式，可以询问协作者是否将该错误补充编入 `docs/project_guides/review.md`，作为后续审查规则的一部分。

## 架构遵循要求

- 任何新增功能、重构或修复都必须与 `docs/project_guides/architecture.md` 保持一致。
- 不要绕过已有主流程：网格读取、自由度管理、单元刚度、装配、边界条件、求解、后处理。
- 非必要情况下，不新增平行的替代实现，不复制已有模块职责。

## 提交前检查

- 明确提醒协作者按 `docs/project_guides/review.md` 进行额外审查。
- 确认改动没有破坏 `docs/project_guides/architecture.md` 中定义的约束。

## 文档位置

- 架构约束：`docs/project_guides/architecture.md`
- 审查规范：`docs/project_guides/review.md`
