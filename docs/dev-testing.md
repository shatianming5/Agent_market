# 开发依赖与完整测试（含 RL）

本项目的 RL 测试用例依赖可选包 `gymnasium` 与 `stable-baselines3`。建议使用专用的开发依赖文件安装：

```powershell
# 创建本地虚拟环境
python -m venv venv
.\\venv\\Scripts\\Activate.ps1

# 安装项目基础依赖与开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 含 gymnasium / stable-baselines3 / pytest

# 运行完整测试（包含 RL）
python -m pytest -q -rs
```

说明
- `requirements-dev.txt` 包括：`pytest`, `gymnasium`, `stable-baselines3`（会拉取 `torch` CPU 版本）。
- 未安装这些可选依赖时，RL 相关测试将被跳过；安装后可全部通过。

常用命令
- 只运行某个测试文件：`python -m pytest tests\test_freqai_pipeline.py -q -k rl`
- 显示详细输出：`python -m pytest -vv -rs`

