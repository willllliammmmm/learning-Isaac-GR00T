# 实验管线组件关系

`gr00t/experiment` 目录下的三个文件负责配置、调度与执行 GR00T 的训练流程：

- **data_config.py**  
  - 描述各种训练/推理场景的数据形态。  
  - 每个 `BaseDataConfig` 子类声明要读取的模态键、时间窗口以及对应的变换序列，并可通过 `load_data_config` 按名称或外部模块加载。  
  - 输出的模态配置与 `ComposedModalityTransform` 被训练和推理流程复用。

- **runner.py**  
  - 提供高层封装 `TrainRunner`，负责拼装模型、数据集与训练参数。  
  - 利用 `data_config` 构建 `LeRobot` 数据集，实例化 `GR00T_N1_5` 模型和自定义 `DualBrainTrainer`，并管理实验目录、元数据保存、日志系统等。  
  - 是启动训练脚本时的主要入口。

- **trainer.py**  
  - 在 Hugging Face `transformers.Trainer` 基础上实现 `DualBrainTrainer`。  
  - 定制数据采样器、优化器参数分组、梯度累积、模型保存与断点恢复，确保双脑架构训练的稳定性与可控性。  
  - 被 `TrainRunner` 调用以执行实际的训练循环。

整体关系：`data_config.py` 定义数据如何准备，`runner.py` 用这些配置串联数据集与模型，并交给 `trainer.py` 的 `DualBrainTrainer` 执行训练，三者共同构成完整的实验管线。***
