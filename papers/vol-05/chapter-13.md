# 第13章　Active Learning と逐次実験計画 — BO 以外の逐次戦略

> [!IMPORTANT]
> **本章の到達目標**
>
> 1. Active learning (AL) の骨格（uncertainty sampling / query by committee / expected model change）を Skill 契約で説明できる
> 2. **BO vs Active Learning の判断分岐**（"最適化" と "学習" の目的分離）を書ける
> 3. Emukit / modAL の Skill 化テンプレートを持てる
> 4. **エージェントが目的を "最適化" と誤解しない契約**（`objective_mode` enum、`stop_condition_for_learning`）を設計できる
> 5. AL と BO を組み合わせる場合の切替 gate（`objective_mode_switch_events`）を書ける
>
> **本章で扱わないこと**
>
> - BO の骨格（surrogate / acquisition / stop condition） → **第6〜10章**
> - Batch BO → **第12章**（batch AL でも似た多様性契約を再利用）
> - Advanced Capstone → **第14章**（BO 側で AL を活用）
> - 失敗パターン全般 → **第15章**

---

## 13.1 なぜ Active Learning か — BO と何が違うのか

BO は **「目的関数を最適化」** する逐次戦略。Active Learning は **「モデルを効率よく学習」** する逐次戦略。同じ「次の実験点を選ぶ」でも、**目的が違えば選ぶべき点も違います**。

- **BO の目的**：$\arg\min_x f(x)$（あるいは $\arg\max$）を見つける。**exploit と explore のバランス**
- **AL の目的**：モデル $\hat{f}$ の誤差 $\|\hat{f} - f\|$ を減らす。**explore 主体**、exploit なし
- **共通点**：GP や BNN などの surrogate を使い、acquisition-like な選択規則で次の点を選ぶ
- **決定的違い**：**AL は「良い点を選ばない」**。むしろ「モデルが自信のない点」＝性能の悪そうな点を積極的に選ぶ

材料 R&D の例：

- **BO**：「最高硬度の合金組成を見つけたい」→ 有望領域を集中探索
- **AL**：「合金組成 → 硬度の予測モデルを作りたい」→ 探索空間全体をカバー、後で別の目的にも使える
- **混在**：「まず全体のマップを作り、その後で最適点を絞る」→ AL 期間 → BO 期間の切替

> [!WARNING]
> **エージェントの typical 失敗**：ユーザが「モデルを作りたい」と言っているのに、エージェントが「じゃあ最高値を探しますね」と BO を回してしまう。これは目的の取り違えで、**AL で得られるべき均等サンプルが得られない**。Skill 契約で `objective_mode` を明示する必要がある。

---

## 13.2 Active Learning の acquisition 骨格

### 13.2.1 Uncertainty Sampling — GP variance / BNN entropy

回帰の場合、predictive variance が大きい点を選ぶ：

$$
x^* = \arg\max_x \sigma^2(x)
$$

分類の場合、entropy が大きい点を選ぶ：

$$
x^* = \arg\max_x H[p(y|x)] = -\sum_c p(y=c|x)\log p(y=c|x)
$$

**契約**（BO の acquisition_spec を AL 用に転用）：

**canonical enum：`active_learning_acquisition_spec.name`**（vol-05 Ch13 §13.2 SoT）

| Value | 戦略族 | 適用タスク | 参照節 |
|:---|:---|:---|:---|
| `predictive_variance` | 予測の pointwise variance が最大の点を選ぶ（回帰の epistemic uncertainty 狙い） | regression | §13.2.1 |
| `predictive_entropy` | 予測分布の entropy が最大の点を選ぶ（分類の総 uncertainty） | classification | §13.2.1 |
| `query_by_committee` | 複数モデル（committee）の予測が食い違う点を選ぶ（vote entropy / KL 等） | regression / classification | §13.2.2 |
| `expected_model_change` | 候補点を追加した際のモデルパラメタ変化の期待値を評価 | regression / classification | §13.2.3 |
| `mutual_information_bald` | パラメタ θ と予測 y の相互情報量（total entropy − E[aleatoric entropy]、epistemic のみ） | classification（主） / regression（副） | §13.2.3 |
| `integrated_variance_reduction` | 候補点追加による領域全体の variance 積分減少を評価（global uncertainty reduction） | regression | §13.2.4 |

> [!NOTE]
> - **制約付き**バリアントは canonical name + `_constrained` suffix（例：`predictive_variance_constrained`）
> - **cost-aware** バリアントは canonical name + `_cost_aware` suffix（例：`mutual_information_bald_cost_aware`）
> - 実装で使う場合は、上記 canonical name を必ずそのまま用いること（表記ゆれ禁止）。エイリアス（例：`BALD` / `MI-BALD`）は display 用としてのみ許可、machine-readable YAML では canonical name を用いる

```yaml
active_learning_acquisition_spec:
  version: "v0.2"
  name: "predictive_variance"                 # enum: §13.2 canonical table 参照
  # 補足：predictive_variance は回帰の pointwise variance、predictive_entropy は分類の entropy、
  # mutual_information_bald は total entropy - E[aleatoric entropy] で epistemic uncertainty を狙う。
  # integrated_variance_reduction は候補点追加による領域全体の variance 積分減少を評価
  task_type: "regression"                     # enum: regression | classification | ranking
  batch_size: 4                                # AL でも batch 提案あり
  candidate_generation_mode: "pool_based"     # enum: pool_based | continuous_synthesis | finite_feasible_set
  diversity_policy_ref: "vol-05:ch12:batch_diversity_policy_v0_2"  # 多様性は Ch12 と共有
```

### 13.2.2 Query by Committee (QBC)

複数モデル（committee）の予測が **食い違う点** を選ぶ：

$$
x^* = \arg\max_x \mathrm{Disagreement}\bigl(\{\hat{f}_k(x)\}_{k=1}^K\bigr)
$$

- 回帰：committee 予測の分散
- 分類：Vote entropy / KL divergence

**Skill 契約**：committee を明示：

```yaml
committee_spec:
  version: "v0.1"
  members:
    - {model_id: "gp_matern52_seed1", weight: 1.0}
    - {model_id: "gp_matern32_seed2", weight: 1.0}
    - {model_id: "random_forest_seed3", weight: 1.0}
  disagreement_metric: "committee_mean_variance"  # enum: committee_mean_variance | committee_total_variance | vote_entropy | mean_kl_divergence | jensen_shannon
  # committee_mean_variance: 各 member の予測平均のばらつき（epistemic のみ）
  # committee_total_variance: 各 member の predictive 分布を混合した total variance（epistemic + aleatoric）
  refresh_policy: "every_iteration"           # enum: every_iteration | fixed_after_init | on_drift
  fit_failure_policy: "fatal"                 # enum: fatal | drop_member_and_alert | retry_with_backup_prior
```

### 13.2.3 Expected Model Change (EMC) / BALD

**EMC**：候補点 $x$ を追加したら、モデルパラメタがどれだけ変わるかを期待値評価

**BALD (Bayesian Active Learning by Disagreement)**：モデルパラメタ $\theta$ と予測 $y$ の相互情報量：

$$
\mathrm{BALD}(x) = H[y|x, \mathcal{D}] - \mathbb{E}_{\theta|\mathcal{D}}\bigl[H[y|x,\theta]\bigr]
$$

BNN や deep ensembles で扱いやすい。Ch8 の DKL / BNN と組合わせる。

> [!NOTE]
> `predictive_entropy` は total predictive uncertainty（epistemic + aleatoric）を測る一方、`mutual_information_bald` は expected aleatoric entropy を差し引き、**epistemic（モデルの知識不足）だけを狙う**。同種観測ノイズの GP 回帰では BALD が variance-based selection に近づくことがある。

---

## 13.3 BO vs Active Learning の判断分岐

**契約**：新規プロジェクト開始時、Skill は Human に「目的モード」を必ず宣言させる：

```yaml
objective_mode:
  version: "v0.2"
  mode: "optimization"                        # enum: optimization | learning | exploration_only | sensitivity_analysis | hybrid_learn_then_optimize | hybrid_utility_mixture
  learning_goal: null                          # optional、mode ∈ {learning, exploration_only, sensitivity_analysis} の場合。enum: prediction_model | space_coverage | sensitivity_analysis
  declared_by: "human:project_lead_XXXX"
  declared_at: "2026-12-01T09:00:00Z"
  approval_id: "approval:HITL-20261201-OM01"
  rationale: "最高硬度の合金組成を探索するため BO を選択"
  # hybrid_learn_then_optimize の場合、切替 gate を明示（operational に定義）
  switch_gate:                                # optional、hybrid_learn_then_optimize の場合のみ
    from: "learning"
    to: "optimization"
    metric: "normalized_rmse"                  # enum: normalized_rmse | normalized_mae | r2 | negative_log_predictive_density
    validation_method: "leave_one_out_cv"     # enum: leave_one_out_cv | k_fold | held_out_test | fixed_validation_pool
    validation_distribution: "declared_operational_domain"  # 検証分布の宣言（Skill が勝手に in-sample を使わないため）
    threshold: 0.15
    confidence_rule: "upper_95_ci_below_threshold"  # enum: point_estimate_below_threshold | upper_95_ci_below_threshold | bootstrap_ci_below_threshold
    min_observations: 30
    hyperparameters_refit_per_fold: true       # LOO-CV の refit ポリシーを明示
    approval_required: true
  # hybrid_utility_mixture の場合、AL 効用と BO 効用の混合を明示
  hybrid_policy:                              # optional、hybrid_utility_mixture の場合のみ
    bo_utility: "qLogEI"                       # BO acquisition（Ch7-10 canonical enum）
    learning_utility: "integrated_variance_reduction"  # AL acquisition（§13.2 canonical enum）
    combination_rule: "human_approved_weighted_sum"    # enum: human_approved_weighted_sum | pareto_over_utilities | epsilon_greedy
    weights: {bo: 0.7, learning: 0.3}          # human_approved_weighted_sum の場合
    # 注意：これは「候補選定効用の Pareto」であり、物理目的（硬度・密度等）の Pareto ではない
  budget_split:                               # optional、hybrid_learn_then_optimize の場合のみ
    learning_budget: 30                       # 実験数
    optimization_budget: 20
    total_budget: 50
```

### 13.3.1 判断表

| 状況 | 推奨モード | 理由 |
|---|---|---|
| 「最高値を見つけたい」 | `optimization` | BO 単体、exploit + explore |
| 「関係性を理解したい／予測モデルを作りたい」 | `learning` (learning_goal: prediction_model) | AL 単体、全域探索 |
| 「空間を均等にカバーしたい／初期実験計画」 | `exploration_only` (learning_goal: space_coverage) | space-filling design、accuracy target なし |
| 「入力変数の感度・重要度を測りたい」 | `sensitivity_analysis` | Sobol 指数、Morris 法など。予測精度目標なし |
| 「まず全体を把握してから絞りたい」 | `hybrid_learn_then_optimize` | AL 期 → BO 期、budget split、switch_gate 経由 |
| 「予測精度と最適化を同時最適化」 | `hybrid_utility_mixture` | 候補選定効用（AL 効用 + BO 効用）の重み付き和。**物理目的の Pareto ではない** |
| 「装置差を理解しつつ最高値を探す」 | `hybrid_learn_then_optimize` + MTGP (Ch11) | 学習段で装置差を捉え、BO 段で MTGP surrogate として利用 |

> [!WARNING]
> `hybrid_utility_mixture` は **候補選定 utility の混合**であって、多目的 BO の Pareto ではない。物理目的（硬度・密度・コストなど）の Pareto は Ch9 の `objectives_spec` を使う。AL 効用（variance / information gain）を EHVI/NEHVI の目的ベクトル成分に入れてはいけない（fatal）。

### 13.3.2 目的モード誤認の防止契約

> [!WARNING]
> **Skill が objective_mode を自律的に変更するのは fatal**（第3章 §3.5 逸脱と同種）。切替は必ず Human 承認付きの `objective_mode_switch_events` として記録。

```yaml
objective_mode_switch_events:                 # append-only、hash chain（Ch10 event_hash schema; Ch12 cancellation/pending events も同 schema を再利用）
  - event_id: "evt-mode-switch-20261215-001"
    event_hash: "sha256:..."
    previous_event_hash: "sha256:0000..."
    switched_at: "2026-12-15T14:00:00Z"
    from_mode: "learning"
    to_mode: "optimization"
    approved_by: "human:project_lead_XXXX"
    approval_id: "approval:HITL-20261215-OM02"
    switch_gate_evaluation:
      metric: "normalized_rmse"
      measured_value: 0.12
      confidence_interval: [0.09, 0.145]      # upper 95% CI
      threshold: 0.15
      confidence_rule: "upper_95_ci_below_threshold"
      threshold_met: true                      # upper CI (0.145) < 0.15
      n_observations: 32
      validation_method: "leave_one_out_cv"
      validation_distribution: "declared_operational_domain"
      hyperparameters_refit_per_fold: true
      gate_condition_met: true
    run_id: "run-2026-1215-01"
    skill_execution_id: "skill-exec-ABC"
```

---

## 13.4 Emukit / modAL の Skill 化

### 13.4.1 Emukit — 実験計画寄りの AL / BO 統合ライブラリ

Emukit は Amazon 製の decision-making under uncertainty library。BO / AL / sensitivity analysis を統一 API で提供。**連続空間の synthesis-style AL**（`ParameterSpace` 上で候補を生成）が既定。pool-based ではないため、`candidate_generation_mode: continuous_synthesis`。

```python
import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.model_wrappers import GPyModelWrapper
import GPy

space = ParameterSpace([
    ContinuousParameter("x_temperature", 200.0, 800.0),
    ContinuousParameter("x_pressure", 0.1, 5.0),
])

# 初期データ
X_init = np.array([[300, 1.0], [500, 2.5], [700, 4.0]])
Y_init = np.array([[expensive_measurement(x)] for x in X_init])

# GP surrogate
gpy_model = GPy.models.GPRegression(X_init, Y_init)
emukit_model = GPyModelWrapper(gpy_model)

# Active Learning loop（uncertainty sampling）
acquisition = ModelVariance(emukit_model)
al_loop = ExperimentalDesignLoop(space, emukit_model, acquisition)

def user_function(X):
    # Emukit は (n, d) を渡し、(n, 1) を期待する
    return np.array([[expensive_measurement(x)] for x in X])

al_loop.run_loop(user_function=user_function, stopping_condition=10)
```

> [!NOTE]
> Emukit / GPy は比較的低アクティビティな依存。長期運用ではバージョンを pin し、Skill インターフェースで実装をラップして交換可能にする（Ax / BoTorch への切り替えも検討）。

### 13.4.2 modAL — scikit-learn ベースの AL

modAL は scikit-learn model なら何でも AL に載せられる汎用ライブラリ。分類タスクに強い。**pool-based AL**（有限 pool から候補を選ぶ）が既定。`candidate_generation_mode: pool_based`。

```python
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier
import numpy as np

learner = ActiveLearner(
    estimator=RandomForestClassifier(n_estimators=100),
    query_strategy=uncertainty_sampling,
    X_training=X_init,
    y_training=y_init,
)

# 1 iteration: query → label → teach → pool 更新（重複回避）
query_idx, query_x = learner.query(X_pool)
y_new = human_label(query_x)                  # ラベル取得（Human 承認 = Ch5 experiment_launch_authorization）
learner.teach(query_x, y_new)
X_pool = np.delete(X_pool, query_idx, axis=0)  # ★ pool から除去、次 iteration の重複回避
```

### 13.4.3 Skill 契約テンプレート — Active Learning Skill

```yaml
active_learning_skill:
  version: "v0.2"
  skill_name: "active_learning_regression_v1"
  objective_mode: "learning"                  # Ch13 §13.3、この Skill は learning 専用
  surrogate:
    model_family: "single_task_gp"            # Ch5 canonical enum
    kernel_spec: {name: "matern52", nu: 2.5}  # Ch6
  acquisition:
    name: "predictive_variance"               # §13.2 canonical enum v0.2
  batch_size: 4
  candidate_generation_mode: "pool_based"     # enum: pool_based | continuous_synthesis | finite_feasible_set
  diversity_policy_ref: "vol-05:ch12:batch_diversity_policy_v0_2"
  stop_condition_for_learning:                # BO の stop_condition とは別、学習用
    type: "generalization_error_threshold"    # enum: generalization_error_threshold | max_iterations | pool_exhausted | budget_exhausted
    metric: "normalized_rmse"                  # enum: normalized_rmse | normalized_mae | r2 | negative_log_predictive_density
    threshold: 0.15
    validation_method: "leave_one_out_cv"     # enum: leave_one_out_cv | k_fold | held_out_test | fixed_validation_pool
    validation_distribution: "declared_operational_domain"
    confidence_rule: "upper_95_ci_below_threshold"
    hyperparameters_refit_per_fold: true
    approval_to_stop: "requires_human_approval"
  experiment_launch_authorization_ref: "vol-05:ch05:experiment_launch_authorization_v0_1"
  # AL でも Ch5 の承認 gate は必須。詳細は §13.4.4
```

### 13.4.4 AL と Ch5 experiment_launch_authorization の関係

AL Skill が候補を提案しても、**物理実験の実行には Ch5 の `experiment_launch_authorization` が必須**（vol-05 の安全 invariant）：

- `batch_size = 1` の場合：**per-experiment 承認**
- `batch_size > 1` の場合：**per-batch 承認**。ただし `approval_scope` に `authorized_candidate_ids`、`batch_size`、`max_experiments_authorized` を明示。承認スコープが AL 提案 batch と一致する必要あり
- 承認後の pool 変更 / diversity policy 変更 / candidate_generation_mode 変更は **承認を無効化**（新規承認が必要、fatal 相当）
- objective_mode の `learning → optimization` 切替（§13.3.2）は、次回 experiment_launch_authorization を **新規 mode で再取得**する必要あり

---

## 13.5 Agentic 特有の失敗パターン契約

- **目的の取り違え**：Human が「モデルを作りたい」と言ったのに、Skill が BO を実行 → `objective_mode` の宣言を必須化、宣言なしでの Skill 実行は fatal
- **勝手なモード切替**：AL 中に「良さそうな領域があった」と Skill が BO に切り替える → §13.3.2 の switch_events で防護
- **stop_condition_for_learning の勝手な緩和**：generalization error threshold を Skill が自律的に上げる → Ch10 と同じく fatal
- **committee の勝手な入れ替え**：QBC で committee メンバーを Skill が変更 → `committee_spec.refresh_policy` に従うことを機械検証、逸脱は fatal
- **候補生成の勝手な絞込み**：
  - pool_based の場合、candidate pool を Skill が事前フィルタしてしまうと bias 発生 → `pool_modification_events` で追跡
  - continuous_synthesis の場合、`search_space_bounds` の縮小を Skill が自律実施 → `search_space_modification_events` で追跡（下記）

```yaml
pool_modification_events:                     # candidate_generation_mode: pool_based の場合のみ、append-only hash chain
  - event_id: "evt-pool-mod-20261220-001"
    event_hash: "sha256:..."
    previous_event_hash: "sha256:0000..."
    modified_at: "2026-12-20T10:00:00Z"
    modification_type: "filter"               # enum: filter | expand | resample | replace
    approved_by: "human:project_lead_XXXX"
    approval_id: "approval:HITL-20261220-PM01"
    rationale: "実験不可能な組成領域を除外"
    pool_size_before: 10000
    pool_size_after: 8500

search_space_modification_events:             # candidate_generation_mode: continuous_synthesis / finite_feasible_set の場合、append-only hash chain
  - event_id: "evt-space-mod-20261220-001"
    event_hash: "sha256:..."
    previous_event_hash: "sha256:0000..."
    modified_at: "2026-12-20T10:00:00Z"
    modification_type: "shrink"               # enum: shrink | expand | reparametrize | add_constraint
    approved_by: "human:project_lead_XXXX"
    approval_id: "approval:HITL-20261220-SM01"
    rationale: "装置制約で温度上限を 800→700 に変更"
    bounds_before: {x_temperature: [200, 800]}
    bounds_after:  {x_temperature: [200, 700]}
```

---

## 13.6 Skill 契約チェックリスト

- [ ] `objective_mode.mode` が Human 宣言済み、`approval_id` あり
- [ ] `learning_goal` が learning/exploration_only/sensitivity_analysis で明示
- [ ] `objective_mode` の変更は `objective_mode_switch_events` で追跡、Skill 自律変更禁止
- [ ] hybrid_learn_then_optimize の `switch_gate` は metric / validation_method / confidence_rule / min_observations がすべて明示
- [ ] hybrid_utility_mixture の場合、AL 効用を BO の目的ベクトルに混入していないこと
- [ ] `active_learning_acquisition_spec.name` が canonical enum v0.2
- [ ] `candidate_generation_mode` が明示（pool_based / continuous_synthesis / finite_feasible_set）
- [ ] `task_type` が明示（regression / classification / ranking）
- [ ] QBC 使用時、`committee_spec.members`、`disagreement_metric`、`fit_failure_policy` が明示
- [ ] `batch_size >= 2` の場合、`diversity_policy_ref` で Ch12 の post-hoc 検証を利用
- [ ] `stop_condition_for_learning` に metric / confidence_rule / validation_distribution が明示、Skill 自律緩和禁止
- [ ] `experiment_launch_authorization_ref` で Ch5 承認 gate を機械参照、per-experiment / per-batch のスコープ一致検証
- [ ] Pool 変更は `pool_modification_events`、search_space 変更は `search_space_modification_events` で Human 承認付き
- [ ] Hybrid モードの場合、`budget_split` または `hybrid_policy.weights` が明示
- [ ] BO Skill (Ch7-12) と切り分けて別 Skill として登録（同一 Skill で兼務しない）

---

## 13.7 章末演習

**問 1**：Human は「合金組成の予測モデルを作りたい」と言っている。Skill が受けるべき正しい行動は？

**問 2**：AL 実行中に「有望領域が見つかった」と Skill が判断した。何をすべきか。

**問 3**：Query by Committee で committee のうち 1 つが GP fit に失敗した。契約上の応答は？

**問 4**：hybrid_learn_then_optimize モードで、learning phase の budget 30 のうち 25 実施して generalization error が 0.10 になった。次に何をすべきか？

**問 5**：Skill が candidate pool から「危険そうな領域」を勝手にフィルタしていた。監査で何をチェックすべきか？

---

## 13.8 参考資料

### 本書内

- 第3章 §3.5：Skill 逸脱パターン（objective_mode の自律変更禁止）
- 第5章 §5.2：`experiment_launch_authorization`, `surrogate_model_family` の初出
- 第6章：GP surrogate（AL でも同じ surrogate を使う）
- 第8章：BNN / DKL（BALD の基盤）
- 第10章：event_hash / previous_event_hash schema
- 第11章：MTGP（hybrid モードで装置差を捉える surrogate）
- 第12章：batch_diversity_policy（AL でも再利用）
- 第14章（planned）：Capstone（BO 側で AL を活用するケース）
- 第15章（planned）：objective_mode 取り違え、committee 勝手な変更の失敗パターン

### 外部参考

- Settles, "Active Learning Literature Survey", Univ. Wisconsin TR, 2010 — AL の baseline surveys
- Houlsby et al., "Bayesian Active Learning for Classification and Preference Learning", arXiv:1112.5745, 2011 — BALD 原論文
- MacKay, "Information-based objective functions for active data selection", Neural Computation 1992 — expected model change 系
- Emukit documentation (https://emukit.github.io/) — Amazon の AL/BO 統合 lib
- modAL documentation (https://modal-python.readthedocs.io/) — scikit-learn ベース AL

> [!NOTE]
> 次章（第14章）では、これまで積み上げた第6〜13章の Skill 契約を用いて、**因果 DAG による search space 絞込 → 単一装置 GP → BO 3 iteration**（14a：基本）と、**階層 GP × multi-objective × constrained × batch × 10 iteration**（14b：発展）を統合した Advanced Capstone を実装します。
