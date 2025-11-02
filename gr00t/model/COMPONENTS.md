# 模型核心组件关系

此目录下的三个关键文件协同实现 GR00T N1.5 的模型加载与推理流水线：

- **gr00t_n1.py**  
  - 定义 GR00T N1.5 的实际神经网络结构，继承自 Hugging Face `PreTrainedModel`。  
  - 组装视觉语言主干 `EagleBackbone` 与动作头 `FlowmatchingActionHead`，并负责加载权重、校验输入输出形状、执行前向传播。  
  - 是训练和推理时调用的核心模型类。

- **policy.py**  
  - 提供对外友好的 `Gr00tPolicy` 封装。  
  - 负责下载/定位模型权重、加载 `experiment_cfg` 元数据、构建模态配置与变换，并对外暴露 `get_action`、`apply_transforms` 等接口。  
  - 内部会实例化 `GR00T_N1_5` 模型和必要的变换。

- **transforms.py**  
  - 定义模型输入输出所需的模态变换及数据整理工具。  
  - 包含 `GR00TTransform`、`DefaultDataCollator` 及相关辅助函数，负责将原始视频/语言/状态整理为模型期望的张量格式，并支持动作的反归一化。  
  - 被 `Gr00tPolicy` 与训练流程复用，以确保数据处理一致。

协作关系：`transforms.py` 决定数据如何进入和离开模型；`gr00t_n1.py` 提供具体的神经网络；`policy.py` 将两者整合为易用的推理接口，并处理权重加载与元数据管理。***
