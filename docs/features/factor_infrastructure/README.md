# Factor Compiler / Factor Hub / Factor Memory

[返回主 README](../../../README.md) · [返回功能索引](../README.md)

这一组功能负责因子规格化、编译、评分、注册、部署和历史记忆，是 LLM 因子生成与研究闭环的基础设施。

## Factor Compiler

代码：

```text
src/agent_market/factor_compiler/
```

能力：

- 输入结构化 `FactorSpec`
- 支持 `ExprNode` AST、元数据、约束、测试用例
- DSL 解析与序列化
- 复杂度、安全性、数据可用性、schema 等静态检查
- 因子评分与 artifacts 写入

API / Flow 对应：

```text
POST /run/factor_compile
POST /run/factor_eval
agent_flow step: factor_compile
agent_flow step: factor_eval
```

## Factor Hub

代码：

```text
src/agent_market/factor_hub/
```

能力：

- SQLite 因子注册表
- Evaluation / Deployment / Event 存储
- Python client
- FastAPI REST + WebSocket server
- Streamlit dashboard
- 迁移已有 JSON 因子库

启动：

```bash
python scripts/factor_lab.py hub serve --host 127.0.0.1 --port 8765
python -m agent_market.factor_hub.server --host 127.0.0.1 --port 8765
```

Python client 示例：

```python
from agent_market.factor_hub import Client

fh = Client()
fh.init_db()
fh.migrate_json("user_data/freqai_expressions.json", lib_name="g-factors")
top = fh.query(ic_gt=0.05, status="active", limit=20)
```

## Factor Memory

代码：

```text
src/agent_market/factor_memory.py
```

能力：

- 从表达式评分结果生成 factor cards
- 保存失败样本、lineage 和检索上下文
- 每个 run 写本地 memory
- 合并进全局 control-plane memory
- 给 LLM / agent / 后续挖掘提供历史上下文

常见产物：

```text
artifacts/runs/<run_id>/factor_memory/
  factor_memory.json
  factor_cards.json
  factor_failure_cards.json
  factor_lineage.json

artifacts/control_plane/factor_memory/
```

## 相关入口

```bash
python scripts/factor_compile.py --help
python scripts/factor_eval.py --help
python scripts/factor_lab.py hub --help
python scripts/factor_lab.py memory-audit --help
```

## 注意事项

- Factor Compiler 是结构化因子输入的边界，避免 LLM 直接输出无法检查的自由文本。
- Factor Hub 是 registry/API 层，不替代 Factor Lab mining。
- Factor Memory 是运行产物和 control-plane 状态，默认落在 `artifacts/`，不要手动改。

