# Isaac-GR00T 项目上下文

## 高层架构
- 仓库将 NVIDIA 的 GR00T N1.5 人形机器人策略封装为 Python 库（`gr00t`），并附带脚本、示例和部署工具。
- 标准流程：加载符合 LeRobot 规范的数据集 → 应用模态特定变换 → 整理为 Eagle VLM 输入 → 运行 GR00T 主干和流匹配动作头 → 反归一化动作以驱动下游控制。
- 官方检查点托管在 Hugging Face；`gr00t.model.policy.Gr00tPolicy` 负责拉取快照、加载元数据并执行推理。
- 微调沿用相同数据管线，通过自定义的 `DualBrainTrainer` 嵌入 Hugging Face `Trainer`，可选 LoRA 适配器以控制可训练参数规模。

## 核心包结构（`gr00t/`）
- **data**：数据集实现（`LeRobotSingleDataset`、`LeRobotMixtureDataset`），读取 LeRobot 模式、统计特征并通过可插拔后端抓取视频帧。变换算子位于 `transform/`，组合成被数据配置复用的 `ComposedModalityTransform`。`embodiment_tags.py` 与 `schema.py` 描述不同机体的观测/动作布局。
- **model**：封装预训练 GR00T 网络。`backbone/eagle_backbone.py` 对接 Eagle 2.5 VLM，可选择开启语言或视觉塔的梯度。`action_head/` 实现流匹配动作头，带有机体感知编码器。`gr00t_n1.py` 负责拼接主干与动作头、校验张量形状、暴露 Hugging Face 风格配置。`transforms.py` 提供 `GR00TTransform`、`DefaultDataCollator` 以及语言-视频批次格式化工具。
- **experiment**：训练辅助模块。`data_config.py` 罗列内置模态配置并支持自定义导入（`module:ClassName`）。`runner.py` 将数据集、模型与 `transformers.TrainingArguments` 连接到 `DualBrainTrainer`，处理检查点元数据导出与日志配置。`trainer.py` 定制 Hugging Face Trainer，提供确定性采样、优化器分组和安全的断点恢复。
- **eval**：策略评估的客户端/服务端抽象。`robot.py` 定义 ZeroMQ 推理端点，`service.py` 提供 RPC 协议，`simulation.py` 支持多步仿真回放和可选视频录制。
- **utils**：PEFT 集成（`utils/peft.py`）、实验记录、评测辅助与视频 IO（`utils/video.py`），可在 `decord`、`torchcodec`、OpenCV 之间自动切换。

## 脚本与入口
- `scripts/inference_service.py`：将 GR00T 作为 ZMQ/HTTP 服务或简单客户端运行，接受 Hugging Face 模型 ID、数据配置和去噪步数等参数。
- `scripts/gr00t_finetune.py`：用于监督微调或 LoRA 适配的 CLI，处理数据集混合、梯度累积、优化器配置及分布式检查点。
- `scripts/eval_policy.py`：离线评估脚本，可绘制预测与真值曲线。
- `scripts/load_dataset.py`：数据加载教程笔记本的 CLI 版本，便于快速验证。
- `scripts/simulation_service.py`：独立封装，暴露策略的 ZMQ 服务并运行多环境仿真客户端。
- `deployment_scripts/`：TensorRT 导出（`export_onnx.py`）、引擎编译脚本及 Jetson 部署说明，对应 `orin.Dockerfile`、`thor.Dockerfile`。

## 文档、示例与素材
- `README.md`：核心入门文档，涵盖环境依赖、数据格式、快速入门命令、机体头选择、基准结果与部署提示。
- `getting_started/`：系列笔记本与 Markdown，讲解数据加载、推理、微调、新机体适配和部署流程，补充 LeRobot 兼容模式说明。
- `examples/`：任务模板（Libero、RoboCasa、Unitree G1、SO-100、SimplerEnv），提供自定义数据配置、模态清单与评测脚本，便于复现实验。
- `demo_data/robot_sim.PickNPlace`：示例数据集（含 `meta/modality.json`），用于验证数据加载与微调脚本。
- `media/`：文档引用的图像与视频（架构图、性能图、示例演示等）。

## 配置与依赖
- `pyproject.toml`：声明 `gr00t` 包、核心依赖（transformers 4.51、accelerate、ray、peft、tyro、Torch 2.5.1 等）以及面向开发、部署、Jetson 的可选分组，构建信息依赖 `setuptools_scm`。
- `Makefile`：提供 `run-checks`（isort、black、ruff、pytest）、`format`、`build` 等便捷目标。
- `Dockerfile`、`thor.Dockerfile`、`orin.Dockerfile`：分别面向 x86 训练、Jetson Thor、Jetson Orin 的基础容器。

## 测试与验证
- `tests/`：基于 pytest 的测试集，覆盖数据加载、模态处理与视频读取（`test_dataset.py`、`test_load.py`、`test_load_video.py`），`media/labeled_frames_video.mp4` 为视频测试素材。
- CI：`README.md` 中的 GitHub Actions 徽章显示自动化格式检查与测试状态。
- 建议本地验证：提交前执行 `make run-checks`，并用示例数据进行脚本冒烟测试。

## 扩展建议
- 新数据源：编写自定义 `BaseDataConfig` 子类（可放在外部通过 `module:ClassName` 引入），描述模态键、时域范围与变换流程，可复用 `ComposedModalityTransform` 组合视频/状态/动作处理。
- 新机体：扩展 `EmbodimentTag` 并更新 `EMBODIMENT_TAG_MAPPING`，同时在数据集元数据中提供匹配的动作与状态形状。
- 自定义主干或动作头：遵循 `GR00T_N1_5_Config` 约定，以保持与现有检查点和脚本的序列化兼容。
- 部署：Jetson 优先采用 TensorRT 流程（`deployment_scripts/build_engine.sh`）；对接服务时可先运行 `inference_service.py --http-server` 暴露 REST 接口。
