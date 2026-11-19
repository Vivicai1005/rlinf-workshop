# 数据卷：只读挂载到 /models/rlinf-workshop（共约 33 GB）

需要提前下载以下 4 项，全部来自 Hugging Face，请按指定的 revision 下载。

## (1) 策略模型

```
Provider:   Hugging Face
Repository: robbyant/lingbot-vla-4b-posttrain-robotwin
Revision:   3e0c7c476bde3daaac00f79f3741a292a299f60a
Files:      model.safetensors
            config.json
            lingbotvla_cli.yaml
```

放到：`/models/rlinf-workshop/lingbot-vla-4b-posttrain-robotwin/`（16.8 GB）

## (2) tokenizer / processor

```
Provider:   Hugging Face
Repository: Qwen/Qwen2.5-VL-3B-Instruct
Revision:   66285546d2b821cf421d4f5eb2576359d3770cd3
Files:      config.json
            generation_config.json
            chat_template.json
            preprocessor_config.json
            tokenizer_config.json
            tokenizer.json
            vocab.json
            merges.txt
```

放到：`/models/rlinf-workshop/Qwen2.5-VL-3B-Instruct/`（11 MB）

## (3) 模型配置

```
Provider:   Hugging Face
Repository: robbyant/lingbot-vla-4b
Revision:   87712223fb463f0f6a6565f8feba64c35418e393
Files:      config.json
```

放到：`/models/rlinf-workshop/lingbot-vla-4b/`（294 B）

## (4) RoboTwin 仿真资源（注意：这是 dataset 仓库，并且下载后需要解压）

```
Provider:   Hugging Face（dataset）
Repository: TianxingChen/RoboTwin2.0
Revision:   981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042
Files:      background_texture.zip   sha256 54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070
            objects.zip              sha256 6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27
            embodiments.zip          sha256 6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6
```

处理：三个 zip 都解压到 `/models/rlinf-workshop/robotwin-assets/` 下，得到 `background_texture/`、`objects/`、`embodiments/` 三个目录（共 16 GB）。解压后可以删掉 zip 和 `__MACOSX` 目录。解压之后不需要任何其他处理，资源以只读方式使用即可。

## 最终目录结构

目录名必须完全一致；`robotwin-assets` 的位置是镜像里固定写死的。

```
/models/rlinf-workshop/
├── lingbot-vla-4b-posttrain-robotwin/   model.safetensors, config.json, lingbotvla_cli.yaml
├── Qwen2.5-VL-3B-Instruct/              上面列出的 8 个文件
├── lingbot-vla-4b/                      config.json
└── robotwin-assets/
    ├── background_texture/
    ├── objects/
    └── embodiments/
```

## 省事的做法

在任意一台能访问 Hugging Face（或 hf-mirror）的机器上运行本目录下的 [`prepare_mount.sh`](prepare_mount.sh)。它会按上面的 revision 下载全部 4 项，逐个校验 sha256 并解压好，生成的目录直接挂载即可：

```bash
git clone https://github.com/Vivicai1005/rlinf-workshop.git
bash rlinf-workshop/mount/prepare_mount.sh /path/to/rlinf-workshop-data
# 默认通过 hf-mirror.com 下载；用 HF_ENDPOINT 可以换成别的地址。重复运行时会跳过已经下载好且校验正确的文件。
```
