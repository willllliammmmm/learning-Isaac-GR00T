# 动作头组件关系

本目录内的三个核心文件共同构成 GR00T N1.5 的动作预测头：

- **action_encoder.py**  
  - 定义将动作序列编码为统一隐藏表征的模块。  
  - `MultiEmbodimentActionEncoder` 和一系列 `CategorySpecificLinear/MLP` 根据机体标签挑选对应参数分支，使不同机器人动作维度映射到同一语义空间。  
  - 编码结果作为后续 Diffusion Transformer 的输入之一。

- **cross_attention_dit.py**  
  - 实现 Diffusion Transformer（DiT）主干，包括自注意力与交叉注意力结构。  
  - 负责在联合的视觉、状态和动作特征上执行去噪建模，是动作头的核心时序推理网络。  
  - 被动作头调用以迭代生成或修正动作潜表示。

- **flow_matching_action_head.py**  
  - 顶层动作头封装，组合动作编码器与 DiT，并提供流匹配推理流程。  
  - 初始化时读取配置以选择机体特定的编码/解码分支、设定噪声调度与推理步数。  
  - 对外暴露 `FlowmatchingActionHead`，训练与推理时调用该类即可完成动作生成。

三者关系：`flow_matching_action_head.py` 作为入口，先调用 `action_encoder.py` 将动作编码，再使用 `cross_attention_dit.py` 提供的 DiT 主干进行流匹配推理，最后解码得到目标动作。***
