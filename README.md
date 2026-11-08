# 🇦🇺🇺🇸 US & ASX AI-Powered Quantitative Research Platform (`stock-colab`)

面向 **美股（NYSE / Nasdaq）** 与 **澳股（ASX 200 / 300）** 的生产级模块化量化投研与 AI 辅助决策平台，专为 **Google Colab（AI Ultra 5x ~1,000 Compute Units/月）+ Google Drive 持久化** 与 **本地极速投研（DuckDB + Parquet）** 双环境协同设计。

---

## 核心特性（Key Capabilities）

1. **Google Colab 原生工作流与算力预算管控（`quant_platform.colab`）**
   - 自动检测 Google Colab 运行环境并挂载 Google Drive（`/content/drive/MyDrive/stock_colab_store`），确保 DuckDB 数据库、Parquet 列式文件与实验 Checkpoint 跨 Colab 会话持久保存。
   - 自动识别加速器（CPU / Tesla T4 / L4 / A100）并实时核算每月 1,000 Compute Units (CU) 消耗。
   - 支持在 Colab 单元格内 **直接内嵌渲染 Apache ECharts 交互式看板（`render_inline_colab_dashboard`）**，或通过 `google.colab.output.serve_kernel_port_as_iframe(8000)` 打开完整 10 模块 FastAPI 看板。
2. **跨市场精确时区对齐与零前视偏差（Zero Look-Ahead Bias）**
   - 精确转换 `America/New_York`（16:00 ET 收盘）与 `Australia/Sydney`（10:00 AEDT/AEST 开盘、16:10 收盘）至 UTC 时间戳。
   - `align_us_overnight_to_asx()` 严格保证 $T$ 日澳股开盘仅能访问 $T-1$ 日已完成收盘的美股、大宗商品与汇率数据。
   - 财务基本面表（`fundamentals_pit`）强制分离报告期期末（`period_end_date`）与实际披露时间戳（`available_at_utc`），杜绝财报前视偏差；历史成分股快照表（`universe_membership`）防止幸存者偏差。
3. **美股与澳股制度差异精确建模（Franking Credits & Withholding Tax）**
   - **澳股（ASX）**：精确计算澳洲 30% 公司税率下的 **Franking Credits（红利完税抵免）** 税前等效股息率：$\text{Grossed-up Yield} = \text{Cash Yield} \times \left(1 + f \cdot \frac{\tau}{1 - \tau}\right)$。
   - **美股（US）**：支持扣除美澳税收协定（W-8BEN）下的 **15% 股息预提税（Dividend Withholding Tax）**。
   - **双币种（AUD / USD）计价**：基于历史 `AUDUSD=X` 汇率序列自动折算跨市场组合净值与收益。
4. **6 大类 22 个量化因子引擎与统计评价**
   - **Value**：`earnings_yield` (1/PE), `book_to_price` (1/PB), `ev_ebitda_inv`, `fcf_yield`
   - **Quality**：`roe`, `roic`, `profit_margin`, `earnings_stability`, `balance_sheet_strength`
   - **Momentum**：`mom_1m`, `mom_3m`, `mom_6m`, `mom_12m`, `risk_adj_momentum`
   - **Risk**：`volatility_20d`, `downside_volatility`, `beta_60d`, `max_drawdown_120d`
   - **Liquidity**：`adv_20d`, `turnover_ratio`, `spread_tightness`
   - **Market-Specific**：`franking_grossed_yield`, `commodity_sensitivity`, `audusd_sensitivity`, `us_overnight_signal`
   - 支持截面去极值（Winsorisation）、Z-Score 标准化、GICS 行业中性化（Sector Neutralisation）及 IC / Rank IC / IC IR / 分层收益（Quantiles Q1~Q5）/ 衰减与换手率检验。
5. **6 种组合构建优化器与 Purged Walk-Forward 回测引擎**
   - 支持 `equal_weight`、`market_cap`、`factor_weight`、`risk_parity` (ERC)、`min_variance`（Ledoit-Wolf 收缩协方差）、`mean_variance` 及个股/行业权重上限约束。
   - 严格区分 **In-Sample（训练集）**、**Validation（验证集）** 与 **Out-of-Sample（样本外测试集）**，并在切分边界强制执行 Purge & Embargo 隔离期。
6. **AI 多智能体投研系统（7 Specialized Agents）**
   - 包含 Market Data、Fundamental、Quantitative Research、News & Sentiment、Options Analysis（含 Black-Scholes 定价与 Greeks）、Risk Management 与 Research Coordinator 七大角色，所有数字均来自确定性工具并附带数据表溯源。

---

## 在 Google Colab 中使用（Recommended Colab Workflow）

仓库内置两个开箱即用的 Google Colab Notebooks（位于 `notebooks/` 目录）：

1. **`notebooks/01_Colab_Quickstart_End_to_End_Pipeline.ipynb`**
   - 一键挂载 Google Drive、拉取/生成美股与澳股数据、计算因子、运行美股/澳股选股器、检验跨市场隔夜传导、执行回测，并在 Colab 单元格内直接渲染交互式图表看板。
2. **`notebooks/02_Colab_GPU_ML_Factor_Ranking_and_Regimes.ipynb`**
   - 面向 Colab GPU/CPU 的机器学习实验工作台：对比线性统计基线（Ridge）与非线性多因子交互模型（GBDT / LightGBM）、执行市场状态分类（Market Regime Classification）与 20 日前瞻波动率预测，并将实验 Checkpoint 自动保存至 Google Drive。

---

## 本地快速启动（Local Setup with `uv`）

```bash
# 1. 创建 Python 3.12 虚拟环境并安装全部依赖
uv sync --extra dev

# 2. 运行端到端全流程演示（数据入库 -> 因子计算 -> 选股 -> 跨市场分析 -> 回测 -> AI 报告）
uv run quant-platform demo --mode fixture

# 3. 运行自动化测试套件与代码检查
uv run pytest -v
uv run ruff check src tests

# 4. 启动 10 模块交互式 FastAPI + ECharts 投研看板 (http://127.0.0.1:8000)
uv run quant-platform serve --port 8000
```
