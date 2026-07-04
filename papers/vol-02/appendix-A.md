# 付録A　統計/ML Skill テンプレート集

> **本付録の目的**：第4〜13章で作った Skill の**再利用可能な雛形**を、`SKILL.md` + `provenance schema` + `minimal example` の 3 点セットで提供します。vol-01 の provenance を **統計/ML 拡張**した完全スキーマも本付録末尾に置きます。

> [!IMPORTANT]
> テンプレートは**そのままコピーして使うためのもの**ではなく、**「何を書き、何を検証すべきか」を思い出すためのチェックリスト**です。実際のデータ・目的に合わせて調整してください。

---

## A.1 本付録の使い方

| 目的 | 参照するテンプレート | 該当章 |
|---|---|---|
| 表形式データで回帰／分類 Skill を作る | A.2（教師あり） | 第5章 |
| PCA・クラスタリング・異常検知 Skill を作る | A.3（教師なし） | 第6章 |
| ベイズ校正・回帰 Skill を作る | A.4（Bayesian） | 第10章 |
| 階層モデル（反復測定・ロット差） Skill を作る | A.5（Hierarchical） | 第11章 |
| capstone（sklearn → PyMC 連結） Skill を作る | A.6（Multi-layer） | 第13章 |
| provenance schema の全項目を確認する | A.7（完全スキーマ） | 第4・10・11章 |

**テンプレの共通要素**（6 要素）は第4章 §4.2 に準拠：①目的 / ②入力条件 / ③出力形式 / ④成功条件 / ⑤禁止事項（severity 付き）/ ⑥再現性条件。

---

## A.2 教師あり学習 Skill テンプレート（第5章）

### A.2.1 `SKILL.md` 雛形

```markdown
---
skill_id: sklearn_supervised_<domain>_v1
skill_version: 1.0.0
task_type: regression | classification
domain: <matbench_expt_gap | matbench_steels | ...>
maintainer: <ORCID or team>
created: YYYY-MM-DD
---

## ① 目的
- 何を予測するか（例：組成 → バンドギャップ eV）
- どう使うか（意思決定文脈：スクリーニング / 校正 / etc.）

## ② 入力条件（data contract 参照：Ch4）
- 特徴量：`X: DataFrame (n, p)` — 列名・単位・型
- 目的変数：`y: Series (n,)` — 単位・型
- グループ列：`group_key: Series | None` — 装置/ロット/試料 ID
- split_unit: {sample | lot | instrument | patient | none}
- 前処理：単位統一済み、欠損なし、外れ値検査済み

## ③ 出力形式
- `posterior/point_estimate`: 予測ベクトル
- `cv_scores.json`: fold ごと MAE/RMSE/R²
- `feature_importance.json`: SHAP または permutation importance
- `provenance.yaml`: A.7 スキーマ準拠
- `figures/`: 学習曲線、residual plot、feature importance

## ④ 成功条件（3 点セット）
- 予測誤差：CV RMSE ≤ <target> — 根拠：<baseline / task standard>
- 安定性：CV fold 間 RMSE の CV of Variance ≤ 20%
- 予測分布：外挿域（applicability_domain 外）で警告

## ⑤ 禁止事項（severity 付き）
- **fatal**：Pipeline 外での fit（データリーク） / test 複数回参照 / 目的変数の特徴量への混入
- **error**：group_key ありで KFold 使用（GroupKFold 必須）
- **warning**：p >> n で正則化パラメータ未指定

## ⑥ 再現性条件
- random_seed 固定、package_versions ピン、CV 分割 hash 記録
- provenance の `cv_scheme.group_key` と `data_split` を必須化
```

### A.2.2 最小コード骨格

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.linear_model import Ridge

pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
cv = GroupKFold(n_splits=5)   # groups が必要な場合のみ
res = cross_validate(pipe, X, y, cv=cv, groups=group_key,
                     scoring=("neg_root_mean_squared_error", "r2"),
                     return_train_score=True, return_indices=True)
```

### A.2.3 チェックリスト（PR 前）
- [ ] `check_split_contract(train_idx, test_idx, group_key)` パス
- [ ] SHAP または permutation importance を出力
- [ ] `applicability_domain` を provenance に記録
- [ ] 過学習の目安：train R² − CV R² ≤ 0.1

---

## A.3 教師なし学習 Skill テンプレート（第6章）

### A.3.1 `SKILL.md` 雛形（PCA / clustering / anomaly を単一 template で扱う）

```markdown
---
skill_id: sklearn_unsupervised_<domain>_v1
task_type: dimensionality_reduction | clustering | anomaly_detection
---

## ④ 成功条件（task_type 別）

### dimensionality_reduction (PCA)
- 個別 loading 安定率：individual_agree ≥ 0.9
- 部分空間安定率：subspace_agree ≥ 0.9（縮退時）
- 累積寄与率：≥ 0.8 で採用成分数決定

### clustering
- 内部指標：silhouette ≥ 0.5（k-means）
                 or DBCV / cluster_persistence（HDBSCAN）
- 安定性：bootstrap ARI ≥ 0.7
- 外部評価：既知ラベルがあれば ARI / NMI

### anomaly_detection
- 感度・特異度：既知の異常例で TPR ≥ 0.9, FPR ≤ 0.1
- 閾値の provenance 記録
```

### A.3.2 安定性測定関数（第6章 §6.2）

```python
def bootstrap_stability(estimator, X, n_boot=50, seed=0):
    """PCA loading / cluster assignment の bootstrap 安定性"""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    base = estimator.fit(X)
    scores = []
    for i in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        est_i = estimator.fit(X[idx])
        # PCA: 主成分 subspace 一致率 / Clustering: ARI
        scores.append(measure(base, est_i))
    return np.mean(scores)
```

---

## A.4 Bayesian（PyMC）Skill テンプレート（第10章）

### A.4.1 `SKILL.md` 雛形

```markdown
---
skill_id: pymc_bayesian_<domain>_v1
task_type: bayesian_calibration | bayesian_regression | ...
backend: pymc-nuts | numpyro-nuts-cpu | numpyro-nuts-gpu
---

## ② 入力条件
- X, y のスケール範囲（prior 設計に必要）
- applicability_domain: prior predictive で検証済み範囲

## ③ 出力形式
- posterior_artifact: `.nc`（InferenceData NetCDF）
- diagnostics_summary: R-hat, ESS, divergences, BFMI, treedepth
- prior_specification: prior 定義（式 + 数値）
- prior_predictive_range: 95% 区間
- posterior_predictive_check: BPV, PIT calibration

## ④ 成功条件
- 診断ゲート：diagnostics_pass == true（14 項目チェックリスト）
- Prior sensitivity：main / weaker / stronger の 3 分岐でロバスト
- PPC：観測データが posterior predictive 95% 区間内

## ⑤ 禁止事項
- **fatal**：diagnostics_pass == false での運用
- **fatal**：backend 変更後の tolerance 未検証（Ch10 §10.9）
- **error**：prior_predictive_check 未実施
```

### A.4.2 骨格コード（線形回帰のベイズ版）

```python
import pymc as pm
with pm.Model() as m:
    beta = pm.Normal("beta", 0.0, 5.0)
    alpha = pm.Normal("alpha", 0.0, 5.0)
    sigma = pm.HalfNormal("sigma", 1.0)
    mu = alpha + beta * x_obs
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
    idata = pm.sample(draws=2000, tune=2000, chains=4, cores=4,
                      nuts_sampler="numpyro", random_seed=42,
                      target_accept=0.95)
    idata.extend(pm.sample_prior_predictive(random_seed=42))
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=42))
```

### A.4.3 診断関数（第12章 §12.9）

```python
def check_diagnostics(idata, thr=None):
    """14 項目チェックリスト（診断 10 + provenance 4）"""
    thr = thr or {"rhat": 1.01, "ess_bulk": 400, "ess_tail": 400,
                  "divergences": 0, "bfmi": 0.3, "treedepth": None}
    rhat = az.rhat(idata)
    ess = az.ess(idata, method="bulk")
    # ...（Ch12 の実装参照）
    return {"diagnostics_pass": bool(...), "rhat_max": ..., ...}
```

---

## A.5 階層モデル Skill テンプレート（第11章・Capstone Layer 3）

### A.5.1 `SKILL.md` 雛形

```markdown
---
skill_id: pymc_hierarchical_<domain>_v1
task_type: hierarchical_regression | variance_decomposition
hierarchy: [lab, instrument, lot, sample]
backend: numpyro-nuts-cpu
---

## ② 入力条件（追加）
- group_id 列：`lab_id`, `inst_id`, `lot_id` の対応表
- 各レベルの group 数：n_lab, n_inst, n_lot
- **applicability_domain の階層版**：inst_count_min, lot_count_min

## ③ 出力形式（追加）
- 各分散成分の事後：sigma_lab, sigma_inst, sigma_lot, sigma_y
- 各レベルの random effect：lab_eff, inst_eff, lot_eff
- 予測分解：total = grand_mean + lab_eff + inst_eff + lot_eff

## ④ 成功条件（追加）
- **非中心化必須**：centered だと divergences 頻発（第11章 §11.3）
- sum-to-zero 制約：lab_eff.mean() ≈ 0, inst_eff.mean() ≈ 0
- 小 group 数警告：inst_count < 5 のとき strong prior + sensitivity 必須
```

### A.5.2 非中心化 + sum-to-zero テンプレート

```python
coords = {"lab": lab_names, "inst": inst_names, "lot": lot_names,
          "obs": np.arange(n_obs)}
with pm.Model(coords=coords) as m_h:
    mu_grand = pm.Normal("mu_grand", 0.0, 5.0)
    sigma_lab = pm.HalfNormal("sigma_lab", 1.0)
    sigma_inst = pm.HalfNormal("sigma_inst", 1.0)
    sigma_lot = pm.HalfNormal("sigma_lot", 1.0)
    sigma_y = pm.HalfNormal("sigma_y", 1.0)

    # sum-to-zero via ZeroSumNormal（PyMC 5.7+）
    lab_eff = pm.ZeroSumNormal("lab_eff", sigma=sigma_lab, dims="lab")
    inst_eff = pm.ZeroSumNormal("inst_eff", sigma=sigma_inst, dims="inst")

    # non-centered lot effect（ZeroSumNormal 不要な階層）
    lot_raw = pm.Normal("lot_raw", 0.0, 1.0, dims="lot")
    lot_eff = pm.Deterministic("lot_eff", lot_raw * sigma_lot, dims="lot")

    mu = (mu_grand + lab_eff[lab_idx] + inst_eff[inst_idx]
          + lot_eff[lot_idx])
    pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y, dims="obs")

    idata = pm.sample(draws=2000, tune=2000, chains=4,
                      nuts_sampler="numpyro", target_accept=0.95,
                      random_seed=0)
```

---

## A.6 Multi-layer Capstone Skill テンプレート（第13章）

Layer 1（sklearn）→ Layer 2（PyMC 校正）→ Layer 3（階層モデル）を **provenance chain で連結**します。

### A.6.1 layer 間の受け渡し契約

```yaml
layer_1:
  skill_id: sklearn_regression_matbench_v1
  outputs:
    y_pred_raw: DataFrame(n, 1)
    y_pred_sha256: <hash>
layer_2:
  skill_id: pymc_calibration_v1
  inputs:
    y_pred_raw: layer_1.outputs.y_pred_raw
    y_pred_sha256: layer_1.outputs.y_pred_sha256   # verify chain
  outputs:
    y_pred_calib: DataFrame(n, 3)   # mean, lower95, upper95
    y_pred_calib_sha256: <hash>
layer_3:
  skill_id: pymc_hierarchical_v1
  inputs:
    y_pred_calib_sha256: layer_2.outputs.y_pred_calib_sha256
  outputs:
    variance_decomposition: {sigma_lab, sigma_inst, sigma_lot}
```

### A.6.2 chain 検証（第13章 §13.6）

```python
def verify_provenance_chain(prov: dict) -> bool:
    for layer_id, layer in prov["layers"].items():
        for input_key, expected_hash in layer.get("input_hashes", {}).items():
            upstream = layer["upstream"]
            actual = prov["layers"][upstream]["output_hashes"][input_key]
            if actual != expected_hash:
                raise AssertionError(
                    f"chain break at {layer_id}.{input_key}: "
                    f"{expected_hash[:8]} != {actual[:8]}")
    return True
```

---

## A.7 統計/ML 拡張 provenance 完全スキーマ

vol-01 の provenance を**統計/ML 拡張**したスキーマ完全版です。**必須項目**（★）と**推奨項目**（○）を明示。

```yaml
# ---------- vol-01 継承（★★★） ----------
run_id: <UUIDv7>                       # ★
run_timestamp: <ISO 8601 UTC>          # ★
operator: <ORCID or team_id>           # ★
skill_id: <namespace_snake_case>       # ★
skill_version: <SemVer>                # ★
code_sha256: <64 hex>                  # ★（Skill dir tree hash）
git_commit_sha: <40 hex>               # ★

input_sha256: <64 hex>                 # ★（データ）
data_contract_hash: <64 hex>           # ★（契約定義）
package_versions:                      # ★
  python: 3.11.7
  sklearn: 1.9.0
  pymc: 5.28.5
  arviz: 0.23.4
  numpyro: 0.21.0
  jax: 0.10.2
random_seed: <int>                     # ★（top-level）

# ---------- 統計/ML 拡張：CV / データ分割（★） ----------
data_split:
  strategy: KFold | GroupKFold | StratifiedGroupKFold | TimeSeriesSplit
  n_splits: 5
  shuffle: true
  random_state: 0
  group_key: inst_id | lot_id | sample_id | null
  split_manifest_hash: <64 hex>        # ★ Ch7
  split_unit: {sample|lot|instrument|patient|none}

cv_scheme:                             # ★ Ch7
  outer: {type: GroupKFold, n_splits: 5, group_key: inst_id}
  inner: {type: GroupKFold, n_splits: 3, group_key: lot_id}
  purge: null | {gap: 5}

# ---------- モデル設定（★） ----------
model_config:
  estimator: Ridge | RandomForestRegressor | pm.Model
  hyperparameters: {alpha: 1.0, ...}
  pipeline_steps: [scaler, model]      # sklearn の場合
  applicability_domain:
    x_range: [[-3, 3], ...]
    y_range: [0, 10]
    inst_count_min: 3                  # 階層モデル用
    lot_count_min: 5

hyperparam_search:                     # ○ nested CV の場合
  n_trials: 20
  search_space: {alpha: [0.01, 100]}
  selection_metric: neg_rmse
  test_set_access_count: 0
  selection_metric_changed: false      # ★ p-hacking 防止

# ---------- Bayesian 追加項目（★ Ch10-12） ----------
sampler_config:
  draws: 2000
  tune: 2000
  chains: 4
  cores: 4
  target_accept: 0.95
  random_seeds: [0, 1, 2, 3]

backend_config:
  backend: numpyro-nuts-cpu
  jax_enable_x64: true
  jax_platform: cpu | gpu
  hardware: <CPU/GPU model>
  cross_backend_tested: true           # ★ Ch14 §14.7

prior_specification:
  beta: {dist: Normal, mu: 0, sigma: 5}
  sigma: {dist: HalfNormal, sigma: 1}
  sigma_group:
    dist: HalfNormal
    sigma: 1
    sensitivity_alternatives:          # ★ Ch11 (n_group < 5 のとき必須)
      - {sigma: 0.3}
      - {sigma: 3.0}

prior_predictive_range:                # ★ Ch10 §10.3
  y_95ci: [-2.5, 2.5]

diagnostics_summary:                   # ★ Ch12 §12.9
  diagnostics_pass: true
  rhat_max: 1.005
  ess_bulk_min: 850
  ess_tail_min: 720
  n_divergences: 0
  bfmi_min: 0.5
  treedepth_max: 8
  reached_max_treedepth_pct: 0.0

posterior_predictive_check:            # ★ Ch12 §12.5
  bpv_pass: true
  pit_calibration_ks_p: 0.42
  rank_uniformity_pass: true

posterior_artifact:                    # ★ Ch10
  path: artifacts/posterior_v1.0.0.nc
  sha256: <64 hex>
  format: netcdf4

# ---------- 階層モデル追加項目（Ch11） ----------
hierarchical_structure:
  levels: [lab, instrument, lot, sample]
  n_groups: {lab: 3, inst: 9, lot: 45, sample: 180}
  parameterization: non_centered       # ★ Ch11 §11.3
  sum_to_zero: [lab_eff, inst_eff]
  applicability_domain:
    inst_count: 9

# ---------- Multi-layer chain（Ch13） ----------
upstream_run_ids: [<UUIDv7>, ...]      # ○（Layer 2+ のみ）
input_hashes: {y_pred_raw: <sha256>, ...}
output_hashes: {y_pred_calib: <sha256>, ...}

# ---------- 監査・失敗管理（Ch14-15） ----------
data_source: real | synthetic          # ★ Ch14 §14.8
synthetic_production_override:         # data_source=synthetic 時のみ必須
  approved_by: <ORCID>
  justification: <text>

known_issues_snapshot_hash: <64 hex>   # ★ Ch14 §14.9
acknowledged_known_issue_ids: []

certification_pass: true               # ★ Ch13 §13.7

# ---------- 決定紐付け（Ch15 §15.5.4） ----------
decision:                              # ○（production 実行）
  type: pilot_go_nogo | lot_release | ...
  value: go | hold
  rationale_link: <URI>
  reviewer: <ORCID>

# ---------- ログ完全性（Ch15 §15.5.5） ----------
integrity:
  log_sha256: <64 hex>
  prev_log_sha256: <64 hex>
  signature: <PKI signature>
access_policy_id: <ID>
retention_policy_id: <ID>
```

---

## A.8 テンプレートを新規 Skill に適用する手順

1. **対象 task_type を A.2〜A.6 から選ぶ**（複数該当なら Multi-layer）
2. **`SKILL.md` 雛形をコピー**、①〜⑥ をドメイン固有に書き換え
3. **④成功条件を数値化**（第4章 §4.4 の 3 点セット準拠）
4. **⑤禁止事項に severity を付与**（第4章 §4.5、fatal/error/warning）
5. **provenance schema を A.7 から抽出**（task_type に該当する項目のみ）
6. **CI ゲート**：`check_split_contract`, `check_diagnostics`, `verify_provenance_chain` を追加
7. **pytest**：第14章 §14.10 のテストを実装

---

## A.9 まとめ

- Skill 雛形は **6 要素 × task_type 別**（教師あり / 教師なし / Bayesian / 階層 / Multi-layer）
- provenance は **vol-01 継承 + 統計/ML 拡張** の 2 層構造
- **A.7 完全スキーマ**は必須★・推奨○を明示。task_type に応じて必要項目を抽出
- テンプレは「コピペ元」ではなく「チェックリスト」として使う

**次の付録**：付録B（cheatsheet + PyMC↔Stan 対応表）、付録C（トラブルシューティング + 実データ候補）。

---

## 参考資料

### 本書内の該当章

- 第4章 §4.2：Skill 設計原則（6 要素）
- 第4章 §4.4：成功条件 3 点セット
- 第4章 §4.5：禁止事項 severity
- 第5-8章：sklearn Skill 実装
- 第7章：CV 設計・データリーク検知
- 第10-12章：Bayesian Skill 実装・診断
- 第11章：階層モデル・非中心化
- 第13章：Multi-layer capstone・provenance chain
- 第14章 §14.9：known_issues schema
- 第15章 §15.5：監査ログ 4 レイヤ

### 外部参考

- Kapoor, S. & Narayanan, A. *Leakage and the Reproducibility Crisis in ML-Based Science*. Patterns (2023). — Leakage taxonomy
- Gelman, A., et al. *Bayesian Workflow*. arXiv:2011.01808 (2020). — Bayesian Skill 設計の指針
- Vehtari, A., et al. *Rank-Normalization, Folding, and Localization: An Improved R̂*. Bayesian Analysis (2021). — 診断指標
