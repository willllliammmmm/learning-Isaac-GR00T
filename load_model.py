# 本地加载权重
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

# 1. 选择官方内置的数据配置（这里以四ier GR1 手臂数据为例）
data_config = load_data_config("fourier_gr1_arms_only")
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

# 2. 指定本地模型目录或 Hugging Face ID
MODEL_PATH = "models/GR00T-N1.5-3B"  # 若未下载，可改成 "nvidia/GR00T-N1.5-3B"

# 3. 加载策略模型
policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag="gr1",          # 根据你的机器人形态选择或自定义
    modality_config=modality_config,
    modality_transform=modality_transform,
    device="cuda",                 # 或 "cpu"
)

print("GR00T N1.5 模型加载完成")

# # 从云端拉取权重
# from gr00t.experiment.data_config import load_data_config
# from gr00t.model.policy import Gr00tPolicy

# data_config = load_data_config("fourier_gr1_arms_only")

# policy = Gr00tPolicy(
#     model_path="nvidia/GR00T-N1.5-3B",      # 直接给出云端模型 ID
#     embodiment_tag="gr1",
#     modality_config=data_config.modality_config(),
#     modality_transform=data_config.transform(),
#     device="cuda",                          # 或 "cpu"
# )

# print("模型已加载（权重自动缓存在 ~/.cache/huggingface/hub）")