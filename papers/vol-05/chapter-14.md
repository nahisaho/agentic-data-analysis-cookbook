# 第14章　総合ハンズオン (Advanced Capstone) — 因果 DAG × 階層 GP × Human 承認 BO

> [!IMPORTANT]
> **本章の到達目標**
>
> 1. **14a（基本）**：vol-04 の因果 DAG から effective search space を導出し、単一装置 GP surrogate で single-objective BO を 3 iteration 回せる
> 2. **14b（発展）**：階層 GP でマルチ装置対応、multi-objective + constrained + batch BO で 10 iteration まで拡張できる
> 3. `budget_remaining` / `stop_condition` を operational 化できる
> 4. 各 iteration で `experiment_launch_authorization` を取得する Human ループを設計できる
> 5. Batch × Multi-objective × Constrained の統合ケースで **契約の合成規則** を書ける
>
> **本章で扱わないこと**
>
> - 各要素の理論的骨格 → **第6章 (GP)、第7章 (acquisition)、第9章 (多目的)、第10章 (制約)、第11章 (MTGP/HierGP)、第12章 (batch)、第13章 (AL)**
> - 14b の後半 5 iteration スクリプト → **付録 D**
> - 失敗パターン → **第15章**
> - 組織運用 → **第16章**

---

## 14.1 Capstone のシナリオ

**課題**：ARIM 施設で **新規セラミック材料の合成条件最適化**。3 台の装置（A, B, C）で並行実験、単位実験コスト 4 時間 / 実験、予算 60 実験。

**目的**：

- 硬度 $y_1$ を最大化（GPa）
- 密度 $y_2$ を最小化（g/cm³）
- 制約：yield $y_3 \geq 60$ %（歩留まり）、温度 $\leq 800$ °C（安全）

**入力空間**（初期宣言）：

- $x_1$：合成温度（°C, 200–900）
- $x_2$：圧力（MPa, 0.1–5.0）
- $x_3$：保持時間（min, 30–180）
- $x_4$：原料 A の質量分率（0.1–0.6）
- $x_5$：原料 B の質量分率（0.1–0.5）
- $x_6$：焼結雰囲気（categorical: [Ar, N2, Air]）
- facility_id（categorical: [A, B, C]）

**14a**：装置 A のみ、single-objective（硬度最大化のみ、密度・yield は制約外にキャッシュ）、batch_size=1、3 iteration

**14b**：3 装置並行、multi-objective (硬度 max, 密度 min)、制約 (yield ≥ 60、温度 ≤ 800)、batch_size=4（装置あたり）、10 iteration（後半 5 は付録 D）

---

## 14.2 14a — 基本 BO ループ（3 iteration）

### 14.2.1 Phase 1 — 因果 DAG から effective search space を導出

vol-04 の因果 DAG 分析（付録 C 参照）から、以下が判明済みとする：

- **$x_5$（原料 B 質量分率）は $x_4$（原料 A）と強い負の相関（$x_4 + x_5 \leq 0.85$、残りが原料 C）**
- **$x_3$（保持時間）は硬度に対して $x_1$（温度）を経由する mediator**（$x_1 \to x_3 \to y_1$ ではなく $x_1 \to y_1$ 直接効果と、$x_1 \to x_3$ の間接的関係）
- **$x_6$（焼結雰囲気）は硬度に交絡変数**として作用

契約：

```yaml
search_space_bounds:
  version: "v0.2"
  derived_from: "vol-04:appendix-c:causal_dag_ceramic_v1"
  dag_of_record_uri: "vol-04:appendix-c:causal_dag_ceramic_v1"
  dag_of_record_sha256: "sha256:aaaa..."
  dag_approval_id: "approval:vol04-DAG-20260501-01"
  dag_revision_policy:
    immutable_within_run: true
    on_dag_hash_mismatch: "fatal_new_run_required"
    on_new_approved_dag_available: "continue_current_run_with_pinned_dag"
    requires_human_approval_for_rederive: true
  variables:
    x_temperature:   {type: "continuous",  low: 200.0, high: 800.0, unit: "degC"}   # 安全制約で 900 → 800 に絞込
    x_pressure:      {type: "continuous",  low: 0.1,   high: 5.0,   unit: "MPa"}
    x_hold_time:     {type: "continuous",  low: 30.0,  high: 180.0, unit: "min"}
    x_mass_frac_A:   {type: "continuous",  low: 0.1,   high: 0.6,   unit: "1"}
    x_mass_frac_B:   {type: "continuous",  low: 0.1,   high: 0.5,   unit: "1"}
    x_atmosphere:    {type: "categorical", choices: ["Ar", "N2", "Air"]}
  linear_constraints:
    - {name: "mass_frac_sum", expr: "x_mass_frac_A + x_mass_frac_B", operator: "<=", threshold: 0.85, source: "causal_dag_derivation"}
  hard_constraints_ref: "vol-05:ch10:hard_constraints_ceramic_v0_2"
  bo_applicability:
    status: "mixed_supported"                # Ch12 判定規則 4（連続 + 少数 categorical）
    reason: "5 continuous + 1 categorical (K=3), effective dim ~7"
    rule_applied: 4
```

### 14.2.2 Phase 2 — 単一装置 GP surrogate

装置 A のみで開始。初期実験 8 点（Latin Hypercube + `x_mass_frac_A + x_mass_frac_B ≤ 0.85` の rejection sampling）：

```yaml
surrogate:
  version: "v0.1"
  model_family: "fixed_noise_gp"               # Ch5 canonical。実験ノイズが既知（yield 測定は再現性 2-3%）
  kernel_spec:
    name: "matern52"                          # Ch6 canonical
    nu: 2.5
    ard: true                                 # 各次元別 length_scale
    mixed_kernel:                             # Ch6/Ch8 混合カーネル
      continuous_kernel: "matern52"
      categorical_kernel: "hamming"           # x_atmosphere に適用
    likelihood:
      type: "fixed_noise"
      noise_variance_source: "empirical_replicate_variance"
  hrd_ref: "vol-05:ch07:hrd_surrogate_v0_1"   # HRD による外挿検出
  fit_diagnostics:
    log_likelihood: -12.3
    length_scales_by_dim:
      x_temperature: 180.5                     # bounds の 30% → 妥当
      x_pressure: 1.2                          # bounds の 24% → 妥当
      x_hold_time: 50.0                        # bounds の 33% → 妥当
      x_mass_frac_A: 0.15                      # bounds の 30% → 妥当
      x_mass_frac_B: 0.10                      # bounds の 25% → 妥当
    length_scale_health: "normal"             # Ch15 §15.x 参照
```

### 14.2.3 Phase 3 — single-objective BO を 3 iteration

**Iteration 1**：

```yaml
iteration_index: 1
run_id: "run-2026-1201-01"
objective_mode:                                # Ch13 契約
  mode: "optimization"
  declared_by: "human:project_lead_XXXX"
  approval_id: "approval:HITL-20261201-OM01"
acquisition_spec:
  name: "qLogNEI"                              # Ch7/Ch9 canonical
  best_f_source: "observed_max_y1"
  x_baseline_ref: "observed_dataset_v1"
batch_size: 1                                  # 14a は sequential
optimization:
  library: "botorch"
  num_restarts: 20
  raw_samples: 512
candidate:
  x_temperature: 620.5
  x_pressure: 2.8
  x_hold_time: 95.0
  x_mass_frac_A: 0.35
  x_mass_frac_B: 0.30
  x_atmosphere: "Ar"
  acquisition_value: 1.42
constraint_precheck:                           # Ch10 Layer 3
  linear_constraints_ok: true                  # 0.35 + 0.30 = 0.65 <= 0.85
  hard_constraints_ok: true                    # temp 620.5 <= 800
  hrd_status: "in_domain"
experiment_launch_authorization:              # Ch5
  approval_id: "approval:HITL-20261201-EXP01"
  approved_by: "human:project_lead_XXXX"
  scope: {candidate_id: "cand-2026-1201-01"}
  approved_at: "2026-12-01T10:15:00Z"
result:
  y_hardness: 8.3
  y_density: 4.1
  y_yield: 72.5                                # 制約は満たすが 14a では non-objective
  observed_at: "2026-12-01T14:30:00Z"
budget_remaining:                              # 下記 §14.2.4
  experiments_remaining: 51                    # 60 - (8 init + 1)
  wall_clock_remaining_hours: 204              # 51 * 4 hour
```

**Iteration 2 と 3**：surrogate を再 fit（Iter 1 の結果を追加）、qLogNEI で次候補を選び、Human 承認 → 実行 → 結果取込。

**3 iteration 後の状況**（例）：

| Iter | Candidate temp | Candidate pressure | y_hardness | Status |
|---|---|---|---|---|
| 1 | 620.5 | 2.8 | 8.3 | ok |
| 2 | 690.0 | 3.2 | 9.1 | ok（新 best） |
| 3 | 720.0 | 3.5 | 9.4 | ok（新 best） |

### 14.2.4 `budget_remaining` と `stop_condition` を operational 化

BO Skill が予算・停止条件を機械的に守るための契約：

```yaml
budget_remaining:                              # 毎 iteration 更新（append-only history あり）
  version: "v0.2"
  total_experiments: 60
  experiments_completed: 9                     # 8 init + 1 BO iter
  experiments_remaining: 51                    # total - completed
  pending_experiments: 0                       # Ch12 pending_experiments.entries[status=in_progress] の count
  wall_clock_total_hours: 240
  wall_clock_used_hours: 36
  wall_clock_remaining_hours: 204
  cost_accounting:                             # 実験コスト
    unit_cost_hours: 4
    reserved_for_confirmation_experiments: 5   # 最終確認用に取り置き
  # 派生フィールド（Skill が計算・機械検証）
  experiments_available_for_bo: 46             # remaining - pending - reserved_for_confirmation = 51 - 0 - 5
  batch_size_ceiling: 46                        # proposed batch_size は experiments_available_for_bo を超えてはならない
  overrun_policies:
    on_proposed_batch_over_available: "fatal"
    on_pending_plus_proposed_over_budget: "fatal"
    on_confirmation_reserve_encroachment: "fatal"
  fatal_on_overrun: true                        # Skill が予算超過を提案したら fatal

stop_condition:
  version: "v0.2"
  type: "composite"                             # enum: max_iterations | budget_exhausted | improvement_stall | acquisition_below_threshold | composite | human_stop
  # clauses は OR 結合。いずれか一つが成立すれば次候補生成を停止
  clauses:
    - {name: "iterations",    condition: "iteration_index >= 3",                         reason: "14a scope"}
    - {name: "budget",        condition: "budget_remaining.experiments_available_for_bo <= 0",  reason: "confirmation reserve 保護"}
    - {name: "acquisition",   condition: "max(acquisition_value_history[-3:]) < 0.05",   reason: "改善余地なし"}
    - {name: "improvement",   condition: "max_y_history[-3:] の変化率 < 0.01",           reason: "収束"}
    - {name: "human_stop",    condition: "human_stop_signal == true",                     reason: "Human override"}
  combination_rule: "any"                       # enum: any (どれか成立で停止) | all (全成立で停止) | weighted
  clause_semantics: "any_satisfied_clause_stops_candidate_generation"
  approval_to_relax: "fatal"                    # Skill が clause を勝手に緩めるのは fatal
```

**運用**：

- 各 iteration 開始時に `budget_remaining` を再計算し、`stop_condition.clauses` を評価
- **どれか 1 つの clause が満たされたら Skill は次候補生成を停止**、Human に完了報告
- `human_stop_signal` は Human が随時発信可能（`stop_condition.type: human_stop` として独立にも使える）

---

## 14.3 14b — 発展（階層 GP × multi-objective × constrained × batch × 10 iteration）

### 14.3.1 Phase 1 — 3 装置の並行実験に拡張

14a の初期 8 点に加えて、装置 B・C で各 6 点の初期実験を追加（計 20 点）。装置差の存在を仮定：

- 装置 A: 硬度がやや高めに出る（bias +0.3 GPa）
- 装置 B: 密度測定に高精度
- 装置 C: yield 変動が大きい

### 14.3.2 Phase 2 — 階層 GP (HierGP) surrogate

Ch11 §11.4 に従い、hierarchical GP を使う。共分散：

$$
\mathrm{cov}((x, i), (x', j)) = k_\text{global}(x, x') + \mathbf{1}[i=j] \cdot k_\text{dev}^{(i)}(x, x')
$$

```yaml
surrogate:
  version: "v0.2"
  model_family: "hierarchical_gp"              # Ch5/Ch11 canonical
  task_index_spec:                              # Ch11 §11.3
    task_map: {"A": 0, "B": 1, "C": 2}
    validation_rules:
      exhaustiveness: "closed_set"
      unknown_facility_policy: "requires_human_approval"
      new_task_policy: "fatal_if_untracked"
  kernel_spec:
    input_kernel:
      name: "matern52"
      nu: 2.5
      ard: true
      mixed_kernel: {continuous_kernel: "matern52", categorical_kernel: "hamming"}
    global_covariance: {name: "matern52"}
    deviation_covariance: {name: "matern52", sigma_dev: {distribution: "log_normal", params: {loc: -1.5, scale: 0.5}}}
    length_scale_prior:
      parameterization: "shared_across_tasks_with_deviation_offset"
      formula: "ell_i = ell_shared * exp(delta_i), delta_i ~ Normal(0, sigma_delta)"
      sigma_delta: 0.3
  outputs:
    - {name: "y_hardness", direction: "maximize"}
    - {name: "y_density",  direction: "minimize"}
    - {name: "y_yield",    direction: "constraint"}
  hrd_ref:
    schema_version: "v0.2"
    scale_source:
      type: "gp_lengthscale"
      # HierGP では kernel_spec.input_kernel が **全 task 共有の global input kernel**（§14.3.2 の
      # k_global に相当）。deviation covariance は per-task 偏差の kernel で、HRD スケールには不採用。
      input_kernel_is_global_shared_kernel: true
      source_path: "kernel_spec.input_kernel.length_scale"
      exclude_variables: ["facility_id", "task_index"]  # task index は距離から除外（Ch11 §11.3）
```

### 14.3.3 Phase 3 — Multi-objective + Constrained + Batch BO の統合

契約の合成規則（各章契約を単一 iteration にまとめる）：

```yaml
iteration_index: 4                              # 14a 3 iter に続く
run_id: "run-2026-1210-04"

# ---- Ch9 multi-objective ----
objectives_spec:
  version: "v0.2"
  objectives:
    - {name: "y_hardness", direction: "maximize"}
    - {name: "y_density",  direction: "minimize"}
  reference:
    reference_point: [7.0, 5.0]                 # 硬度下限、密度上限
    source: "human_declared"
    approval_id: "approval:HITL-20261210-RP01"
  scalarization:
    allowed: false                               # デフォルト。エージェント自律スカラー化禁止

# ---- Ch10 constraints ----
constraints_spec:
  version: "v0.2"
  hard_constraints:
    - {name: "yield_min",   expr: "y_yield >= 60",       enforcement_layer: [1, 2, 3]}
    - {name: "temp_max",    expr: "x_temperature <= 800", enforcement_layer: [1, 2, 3]}
  soft_constraints:
    - {name: "yield_target", expr: "y_yield >= 75",       violation_penalty: "cEI_multiplier"}
  feasibility_model:
    y_yield: {model_family: "single_task_gp", output_type: "regression"}
  feasibility_probability_threshold: 0.85       # Ch10 Step 4

# ---- Ch12 batch ----
batch_diversity_policy:
  version: "v0.2"
  batch_size: 4                                 # 装置 A/B/C の並行度に合わせ、内 2 は装置 A
  strategy: "joint_acquisition"
  minimum_pairwise_distance:
    metric: "standardized_euclidean"
    threshold: 0.3
    threshold_source: "human_declared"
    enforcement: "post_hoc_skill_validation"
    distance_feature_set:                       # ★ 距離計算の feature 範囲を明示
      exclude_variables: ["facility_id", "task_index"]  # 同一 recipe × 異装置を "多様" と誤判定しない
      categorical_handling: "gower_or_one_hot_declared"
      normalization_contract_ref: "pending_experiments.tensor_encoding_contract"
  fallback_on_violation: "sequential_greedy_replan"

pending_experiments:                            # Ch12 §12.3
  version: "v0.2"
  entries:
    - {experiment_id: "exp-A-041", facility_id: "A", x: {x_temperature: 720, x_pressure: 3.5, x_hold_time: 100, x_mass_frac_A: 0.35, x_mass_frac_B: 0.28, x_atmosphere: "Ar"}, submitted_at: "2026-12-09T14:00:00Z", authorization_id: "approval:HITL-20261209-A41"}
  fantasize_policy: {method: "botorch_internal_fantasize", num_samples: 128}
  cancellation_policy: "requires_human_approval"
  tensor_encoding_contract:
    derivation_deterministic: true

# ---- Ch11 multi-task pending ----
parallel_facility_policy:
  version: "v0.1"
  mode: "cross_facility_joint_fantasize"
  per_facility_batch_size: {A: 2, B: 1, C: 1}
  pending_experiments_visibility: "cross_facility"

# ---- Ch7-10 acquisition ----
acquisition_spec:
  name: "qLogNEHVI"                             # Ch9 canonical
  ref_point_source: "objectives_spec.reference.reference_point"
  constraints: "constraints_spec.hard_constraints (via callables)"

# ---- Ch5 experiment_launch_authorization ----
experiment_launch_authorization:
  approval_id: "approval:HITL-20261210-EXP04"
  approved_by: "human:project_lead_XXXX"
  scope:
    batch_size: 4
    per_facility_batch_size: {A: 2, B: 1, C: 1}   # 承認時に facility 配分も pin
    authorized_candidate_ids: ["cand-2026-1210-A1", "cand-2026-1210-A2", "cand-2026-1210-B1", "cand-2026-1210-C1"]
    candidate_facility_map:                       # candidate_id → facility_id の bijective map
      cand-2026-1210-A1: A
      cand-2026-1210-A2: A
      cand-2026-1210-B1: B
      cand-2026-1210-C1: C
    max_experiments_authorized: 4
  approved_at: "2026-12-10T09:30:00Z"

# ---- iteration 結果 ----
candidates:
  - {candidate_id: "cand-2026-1210-A1", facility_id: "A", x: {...}, acquisition_value: 1.8}
  - {candidate_id: "cand-2026-1210-A2", facility_id: "A", x: {...}, acquisition_value: 1.5}
  - {candidate_id: "cand-2026-1210-B1", facility_id: "B", x: {...}, acquisition_value: 1.6}
  - {candidate_id: "cand-2026-1210-C1", facility_id: "C", x: {...}, acquisition_value: 1.4}
batch_diversity_metrics:                        # Group C
  min_pairwise_distance: 0.42                   # threshold 0.3 満たす
  max_pairwise_distance: 1.85
  post_hoc_validation_passed: true
constraint_precheck:                            # Ch10 Layer 3 full schema
  all_candidates_passed: true
  constraints_declared_hash: "sha256:cccc..."
  checker_version: "v0.2"
  per_candidate:
    - {candidate_id: "cand-2026-1210-A1", candidate_hash: "sha256:1111...", passed: true, evaluated_layers: [1,2,3]}
    - {candidate_id: "cand-2026-1210-A2", candidate_hash: "sha256:2222...", passed: true, evaluated_layers: [1,2,3]}
    - {candidate_id: "cand-2026-1210-B1", candidate_hash: "sha256:3333...", passed: true, evaluated_layers: [1,2,3]}
    - {candidate_id: "cand-2026-1210-C1", candidate_hash: "sha256:4444...", passed: true, evaluated_layers: [1,2,3]}

# ---- Group C event streams（毎 iteration append-only、Ch10/Ch12 hash chain）----
constraint_violation_events: []                 # 本 iteration では違反なし
duplicate_candidate_events: []                  # Ch12 §12.7
stale_pending_events: []                        # Ch12 §12.7
batch_drift_events: []                          # Ch12 §12.7

task_correlation_matrix:                        # Ch11 Group C
  tilde_B_normalized: [[1.00, 0.72, 0.65], [0.72, 1.00, 0.58], [0.65, 0.58, 1.00]]
  diagnostic: "healthy_partial_pooling"         # all-ones でも identity でもない
```

### 14.3.4 Iteration 進行と契約更新の流れ

- 各 iteration で `surrogate` を最新観測で再 fit（`hyperparameters_refit_per_fold: true` 相当）
- `pending_experiments.entries` を append、完了したものは event streams で status を更新
- `budget_remaining` と `stop_condition` を評価（下記 14.3.5）
- Pareto front / hypervolume を Group C に記録

### 14.3.5 拡張 stop_condition（multi-objective 対応）

```yaml
stop_condition:
  version: "v0.3"
  type: "composite"
  clauses:
    - {name: "iterations",         condition: "iteration_index >= 10",                                          reason: "14b scope"}
    - {name: "budget",             condition: "budget_remaining.experiments_remaining <= 5",                    reason: "confirmation reserve"}
    - {name: "hypervolume",        condition: "hypervolume_improvement_last_3_iter < 0.02 * hypervolume_scale", reason: "Pareto 収束"}
    - {name: "constraint_stall",   condition: "feasibility_probability_average < 0.60 for 2 iter",              reason: "実行可能領域を外れ続けている"}
    - {name: "task_diagnostic",    condition: "task_correlation_matrix.tilde_B_normalized.diagnostic in [all_ones_degenerate, identity_degenerate] for consecutive_iters >= 2",  reason: "階層 GP の pooling 失敗"}
    - {name: "human_stop",         condition: "human_stop_signal == true",                                       reason: "Human override"}
  combination_rule: "any"
  clause_semantics: "any_satisfied_clause_stops_candidate_generation"
  hypervolume_scale: "ref_point_bounding_box"   # Ch9 canonical enum
  task_diagnostic_thresholds:                    # ★ task_diagnostic clause の numeric 閾値
    all_ones_offdiag_min: 0.95                   # 対角以外の最小値がこれ以上なら all_ones_degenerate
    identity_offdiag_max_abs: 0.10               # 対角以外の絶対値の最大値がこれ以下なら identity_degenerate
    min_observations_per_task: 5
    require_two_consecutive_iters: true
  approval_to_relax: "fatal"
```

### 14.3.6 Iteration 5 での task_correlation_matrix 診断とエスカレーション

仮想シナリオ：iteration 5 で `tilde_B_normalized` を評価したところ：

```
[[1.00, 0.99, 0.98],
 [0.99, 1.00, 0.99],
 [0.98, 0.99, 1.00]]
```

診断：`all_ones` に近い → **装置差を surrogate が捉えていない**（前処理でノーマライズしすぎた可能性）。

**Skill の応答**：

1. `stop_condition.clauses.task_diagnostic` が発火 → Skill は自律的に次候補生成を停止
2. Human に **`hierarchical_model_degeneracy_alert`** を発行：診断内容、疑わしい前処理、推奨対応（スケーリング再検討、prior 変更）
3. Human 承認あるまで `iteration_index` を進めない
4. Skill が自律的に prior を変更するのは fatal（Ch11 §11.4 準拠）

### 14.3.7 後半 5 iteration のスクリプト

Iteration 6-10 の実装スクリプトは **付録 D**（planned / forthcoming、vol-05 完成時に収録）に収録予定。以下を扱う：

- **MTGP/HierGP posterior を source として `qLogNoisyExpectedHypervolumeImprovement` を使う**（BoTorch は「multi-task 専用の qLogNEHVI」を持たない。posterior が MTGP/HierGP から供給されれば multi-task 挙動になる）
- `X_pending` に task_index を含めた tensor 構築（Ch11 §11.5 + Ch12 §12.3 tensor_encoding_contract）
- 制約 callable を multi-output feasibility model として構築（Ch10）
- `optimize_acqf(q=4, sequential=True, fixed_features={task_dim: 0/1/2})` で装置固定 batch
- iteration 6-10 の Pareto front 進展と task_correlation_matrix の推移
- 全 10 iteration 完了時の deliverables 一覧（Pareto front、確認実験計画、provenance bundle）

> [!NOTE]
> 付録 D 未収録の間は、iteration 6-10 の運用は本章 §14.3.3 の契約を反復適用する形で **代替可能**。Skill テンプレートを本章契約に固定し、Skill 実行の provenance を Group C event streams に記録することで、後日付録 D が公開されても機械互換性を保つ。

---

## 14.4 Capstone 契約合成規則

14b で示した通り、各章契約は独立に定義され、iteration 単位で **合成**される。合成規則：

### 14.4.1 Iteration 実行パイプライン（明示的順序）

Skill は各 iteration で以下の順で pipeline を実行する。順序は fixed、Skill が並べ替えるのは fatal：

1. **contract_version_pin snapshot + seed pin**（version と seed を run_id と紐付け）
2. **stop_condition と budget_remaining の precheck**（実行前に停止判定を先に評価）
3. **pending_experiments の取込と fantasize**（Ch12 tensor_encoding_contract 経由）
4. **acquisition 最適化**（hard_constraints callable を評価器として渡す）
5. **post-hoc diversity 検証**（Ch12 §12.2、distance_feature_set に従う）
6. **Ch10 Layer 3 constraint_precheck**（constraint_precheck ブロックを機械検証）
7. **experiment_launch_authorization 取得**（scope が candidates と bijective であることを検証）
8. **Group C event streams への append**（constraint_violation / duplicate_candidate / stale_pending / batch_drift / task_correlation_matrix / hypervolume_history）

### 14.4.2 Ref-only、conflict resolution

- **Ref-only**：他章契約は `_ref` 参照で示す（`hrd_ref`, `experiment_launch_authorization_ref`）。inline copy 禁止（drift の温床）
- **上書き禁止**：`hard_constraints`、`pending_experiments`、`objective_mode` などの上位契約は他契約から上書きされない。Skill が上書きしたら fatal
- **Version pin**：iteration 開始時に全契約の version を snapshot（`contract_version_pin` として run_id と紐付け）
- **Approval scope match**：`experiment_launch_authorization.scope` が `candidates` および `candidate_facility_map` と bijective
- **Idempotent iteration**：同じ input + 同じ seed で iteration を再実行すると同じ candidates が出る（`sequential_seed_provenance` 参照）

```yaml
contract_version_pin:                           # iteration 開始時に snapshot
  version: "v0.2"
  iteration_index: 4
  run_id: "run-2026-1210-04"
  pinned_at: "2026-12-10T08:00:00Z"
  contracts:
    search_space_bounds:                "v0.2"
    surrogate:                          "v0.2"
    objectives_spec:                    "v0.2"
    constraints_spec:                   "v0.2"
    batch_diversity_policy:             "v0.2"
    pending_experiments:                "v0.2"
    parallel_facility_policy:           "v0.1"
    acquisition_spec:                   "qLogNEHVI"
    experiment_launch_authorization:    "v0.1"
    stop_condition:                     "v0.3"
    budget_remaining:                   "v0.2"
    hrd:                                "v0.2"
    task_index_spec:                    "v0.1"
    tensor_encoding_contract:           "v0.1"

sequential_seed_provenance:                     # ★ Idempotency のための seed pin
  version: "v0.1"
  initial_seed: 20261210
  derivation_rule: "seed_k = hash(run_id || iteration_index || component)"
  components:
    lhs_initial_design:      2026121001
    acquisition_raw_samples: 2026121002
    mc_sampler:              2026121003
    fantasize_sampler:       2026121004
    diversity_replan:        2026121005
  seed_change_policy: "fatal"                   # iteration 途中で seed 変更は fatal
```

---

## 14.5 Skill 契約チェックリスト（Capstone）

- [ ] Phase 1：`search_space_bounds.dag_of_record_sha256` + `dag_approval_id` + `dag_revision_policy` あり
- [ ] `linear_constraints` が DAG 由来で明示
- [ ] Phase 2：14a は `fixed_noise_gp`（or `single_task_gp`）、14b は `hierarchical_gp` に切替。**同一 Skill で兼務しない**（別 Skill 登録）
- [ ] `kernel_spec.likelihood.type` が明示（fixed_noise / homoskedastic 等）
- [ ] `task_index_spec` の `unknown_facility_policy` と `new_task_policy` が明示（Ch11）
- [ ] `objective_mode.mode = optimization`、`declared_by` あり
- [ ] `objectives_spec.reference.reference_point` が human_declared、Skill 自律変更禁止
- [ ] `constraints_spec.hard_constraints` が 3 層すべてに登録（enforcement_layer: [1,2,3]）
- [ ] `feasibility_probability_threshold` が明示、Skill 自律緩和禁止
- [ ] `batch_diversity_policy.enforcement: post_hoc_skill_validation` かつ `distance_feature_set.exclude_variables` に task_index/facility_id
- [ ] `pending_experiments.tensor_encoding_contract.derivation_deterministic: true`
- [ ] `parallel_facility_policy.mode: cross_facility_joint_fantasize`
- [ ] `acquisition_spec.name: qLogNEHVI` かつ `ref_point_source` で objectives_spec 参照
- [ ] `experiment_launch_authorization.scope` が candidates と bijective、`per_facility_batch_size` + `candidate_facility_map` あり
- [ ] `constraint_precheck` に per_candidate schema、`constraints_declared_hash`、`checker_version` を記録
- [ ] Group C event streams (`constraint_violation_events`, `duplicate_candidate_events`, `stale_pending_events`, `batch_drift_events`) を毎 iteration append-only 記録
- [ ] `budget_remaining` に `experiments_available_for_bo` 派生フィールド、overrun_policies を含む
- [ ] `stop_condition` clauses は any 結合、`task_diagnostic_thresholds` に numeric 閾値
- [ ] `contract_version_pin` + `sequential_seed_provenance` を iteration 開始時に snapshot
- [ ] iteration pipeline は §14.4.1 の 8 ステップ固定順、Skill による並べ替えは fatal
- [ ] `task_correlation_matrix` を毎 iteration 記録、`task_diagnostic_thresholds` で degeneracy 診断
- [ ] `hierarchical_model_degeneracy_alert` の発行フロー明確
- [ ] 14a → 14b の Skill 切替（single_task_gp/fixed_noise_gp → hierarchical_gp）は Human 承認付き

---

## 14.6 章末演習

**問 1**：14a の iteration 2 で候補が `x_mass_frac_A: 0.6, x_mass_frac_B: 0.4` （合計 1.0）だった。Skill の応答は？

**問 2**：14b の iteration 5 で `tilde_B_normalized` の対角以外がほぼ 1 になっていた。Skill と Human はどう対応すべきか。

**問 3**：iteration 3 で `pending_experiments` の 1 件が 24 時間経っても completed にならない。Skill は自律的にキャンセルしていいか？

**問 4**：14b の Human が「reference_point を [7.5, 4.8] に変えたい」と言ってきた。契約上の正しい手順は？

**問 5**：iteration 8 で `stop_condition.hypervolume clause` が発火。Skill は次候補を提案せず停止したが、Human は「もう 2 iteration 続けたい」と言ってきた。契約に沿った正しい対応は？

**問 6**：14b で使う `contract_version_pin` の目的を説明し、pin なしで運用した場合に起きる問題を挙げよ。

---

## 14.7 参考資料

### 本書内

- 第3章：Skill 逸脱パターン全般
- 第5章：`experiment_launch_authorization`, `surrogate_model_family`, provenance 21 フィールド
- 第6章：GP surrogate、Matérn kernel、混合カーネル
- 第7章：acquisition (qLogNEI)、HRD
- 第8章：BNN / DKL（本 Capstone では非採用、代替として言及）
- 第9章：多目的、qLogNEHVI、objectives_spec
- 第10章：constraints_spec、3 層防御、cEI、safe BO
- 第11章：MTGP / HierGP、task_index_spec、task_correlation_matrix
- 第12章：batch_diversity_policy、pending_experiments、tensor_encoding_contract
- 第13章：Active Learning（本 Capstone では objective_mode: optimization に固定）
- 第15章（planned）：Capstone で起きうる失敗（duplicate、stale pending、Pareto 疑似収束）
- 付録 D：iteration 6-10 スクリプト
- vol-04 付録 C：因果 DAG 導出手順

### 外部参考

- Daulton et al., "Multi-objective Bayesian Optimization over High-Dimensional Search Spaces", arXiv:2109.10964 — high-dim + multi-objective
- Balandat et al., "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization", NeurIPS 2020
- Journel & Huijbregts, "Mining Geostatistics" — 因果 DAG と surrogate の橋渡しに参考
- Emukit tutorial for mixed BO/AL workflows

> [!NOTE]
> 次章（第15章）では、本 Capstone で起きうる失敗パターンを **BO 一般** と **Agentic 特有** の 2 セクションで整理します。GP 外挿誤用、length scale 暴走、Pareto 疑似収束、duplicate candidates、stale pending、prior 勝手変更、stop condition 勝手緩和など。
