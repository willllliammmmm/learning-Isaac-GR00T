# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 根据 Apache License, Version 2.0 许可协议进行授权；
# 除非符合协议，否则禁止使用此文件。
# 你可以在以下地址获取协议副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件“按原样”分发，
# 不提供任何明示或暗示的担保或保证。
# 有关许可证下的具体权限和限制，请参阅许可证内容。

"""
GR00T 推理服务（Inference Service）

此脚本同时提供基于 ZMQ 和 HTTP 的服务端/客户端实现，
可用于部署 GR00T 模型进行远程推理。

HTTP 模式下提供 REST API，便于与网页应用或其他系统集成。

---

1. 默认模式为 ZMQ 服务端。

运行服务器：
    python scripts/inference_service.py --server

运行客户端：
    python scripts/inference_service.py --client

---

2. 运行 HTTP 服务器模式：

HTTP 模式依赖以下库：
    - 服务端（运行 GR00T 模型，需 GPU 支持）：
        pip install uvicorn fastapi json-numpy
    - 客户端：
        pip install requests json-numpy

HTTP 服务端启动示例：
    python scripts/inference_service.py --server --http-server --port 8000

HTTP 客户端示例（假设服务器在 0.0.0.0:8000 运行）：
    python scripts/inference_service.py --client --http-server --host 0.0.0.0 --port 8000

如需远程访问，可使用 bore 进行端口转发（bore.pub 提供中继）：
    bore local 8000 --to 159.223.171.199
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


@dataclass
class ArgsConfig:
    """推理服务的命令行参数配置"""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """模型检查点目录或 Hugging Face 模型 ID"""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """要使用的机器人机体标签"""

    data_config: str = "fourier_gr1_arms_waist"
    """
    使用的数据配置名称，例如：
        so100、fourier_gr1_arms_only、unitree_g1 等。
    也可以指定自定义路径，格式为 "module:ClassName"，
    具体可参考 gr00t/experiment/data_config.py。
    """

    port: int = 5555
    """服务器监听端口"""

    host: str = "localhost"
    """服务器主机地址"""

    server: bool = False
    """是否以服务器模式运行"""

    client: bool = False
    """是否以客户端模式运行"""

    denoising_steps: int = 4
    """动作生成时使用的去噪步数"""

    api_token: str = None
    """API 访问令牌（若未提供，则关闭认证）"""

    http_server: bool = False
    """是否以 HTTP 服务器模式运行（默认使用 ZMQ 模式）"""


#####################################################################################


def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    示例：通过 ZMQ 客户端调用推理服务。
    """
    # 使用 ZMQ 客户端模式
    # 创建策略客户端封装
    policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)

    print("可用的模态配置：")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())

    time_start = time.time()
    action = policy_client.get_action(obs)
    print(f"从服务器获取动作总耗时：{time.time() - time_start} 秒")
    return action


def _example_http_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    示例：通过 HTTP 客户端调用推理服务。
    """
    import json_numpy

    json_numpy.patch()
    import requests

    print("正在测试 HTTP 服务器...")

    time_start = time.time()
    response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
    print(f"从 HTTP 服务器获取动作总耗时：{time.time() - time_start} 秒")

    if response.status_code == 200:
        action = response.json()
        return action
    else:
        print(f"错误：{response.status_code} - {response.text}")
        return {}


def main(args: ArgsConfig):
    if args.server:
        # ---------------------------
        # 服务端模式
        # ---------------------------
        # 创建一个策略对象（Gr00tPolicy）
        # 该对象封装了模型路径、数据变换名称、机体标签与去噪步数，
        # 用于在服务器模式下执行推理请求。
        #
        # 若指定了新的 data_config，则需自行构建模态配置与变换；
        # 可参考 gr00t/utils/data.py。
        data_config = load_data_config(args.data_config)
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )

        # 启动服务器
        if args.http_server:
            from gr00t.eval.http_server import HTTPInferenceServer  # noqa: F401

            server = HTTPInferenceServer(
                policy, port=args.port, host=args.host, api_token=args.api_token
            )
            server.run()
        else:
            server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
            server.run()

    elif args.client:
        # ---------------------------
        # 客户端测试模式
        # ---------------------------
        # 该模式会向服务器发送一条随机生成的观测数据，
        # 并接收对应的动作预测结果，用于测试通信与推理是否正常。

        # 示例观测数据（各通道形状）：
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.left_arm: (1, 7)
        # - obs: state.right_arm: (1, 7)
        # - obs: state.left_hand: (1, 6)
        # - obs: state.right_hand: (1, 6)
        # - obs: state.waist: (1, 3)
        #
        # 示例动作输出：
        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)
        obs = {
            "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }

        if args.http_server:
            action = _example_http_client_call(obs, args.host, args.port, args.api_token)
        else:
            action = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

        for key, value in action.items():
            print(f"动作输出：{key}: {value.shape}")

    else:
        raise ValueError("请指定 --server 或 --client 参数之一")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
