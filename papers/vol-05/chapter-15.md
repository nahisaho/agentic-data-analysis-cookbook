# 第15章　BO × Agentic 特有の失敗パターンと監査

> [!IMPORTANT]
> **本章の到達目標**
>
> 1. **セクション 1：BO 一般の失敗パターン** を列挙し、契約と診断で検出できる
> 2. **セクション 2：Agentic 特有の失敗パターン** を列挙し、Skill 権限境界と監査で防護できる
> 3. 各失敗に対して **検出契約 + 是正契約** を書ける
> 4. `bo_agentic_audit_checklist` として Skill 監査を自動化できる
>
> **本章で扱わないこと**
>
> - 各章の骨格 → **第6〜13章**
> - 統合ケース → **第14章 Capstone**
> - 組織運用 → **第16章**

---

## セクション 1：BO 一般の失敗パターン

各失敗について、**症状 / 診断契約 / 是正契約** の 3 点で示す。

### 15.1.1 GP 外挿の誤用

**症状**：観測点から離れた領域で GP posterior mean を「予測値」として報告し、Human が信じてしまう。variance は大きいが「信頼区間の広さ」を Skill が伝えていない。

**診断契約**（HRD schema、Ch7 §7.x / Ch8 §8.6.4）：

```yaml
hrd_diagnostic:
  version: "v0.2"
  scale_source:
    type: "gp_lengthscale"
    source_path: "kernel_spec.length_scale"
    exclude_variables: ["facility_id", "task_index"]
    applicable_feature_set: "continuous_ard_only"  # categorical は別診断
  # Ch7 準拠：per-dim length_scale で normalize してから合成
  ell_eff_definition: "ell_eff_j = clip(length_scale_j, ell_min_j, ell_max_j)"
  in_domain_criteria:                            # いずれかを採用（両方併記も可）
    - name: "mahalanobis_distance"
      formula: "min_i sqrt( sum_j ((x_j - x_ij) / ell_eff_j)^2 ) <= tau_D"
      tau_D: 2.0
    - name: "length_scale_ratio_linfty"
      formula: "min_i max_j (|x_j - x_ij| / ell_eff_j) <= tau_L"
      tau_L: 2.0
  categorical_domain_check:                      # 別立て
    metric: "hamming_distance_to_nearest_observed"
    max_allowed: 1                                # 1 変数だけ観測外なら boundary、2 以上で extrapolation
  extrapolation_reporting: "explicit_flag_in_output"
  fatal_on_undeclared_extrapolation: true
```

**是正契約**：外挿領域の候補提案は必ず `hrd_status: extrapolation` フラグ + 大きい variance + explicit warning を Human に返す。フラグなしで posterior mean だけ返すのは fatal。

---

### 15.1.2 Length scale の暴走

**症状**：GP の length scale が bounds の 10 倍以上に発散（over-smoothing）、あるいは 1/100 未満に収縮（過剰適合）。どちらも surrogate として使い物にならない。

**診断契約**：

```yaml
length_scale_health_check:
  version: "v0.2"
  applicable_feature_set: "continuous_ard_only"   # categorical / facility_id / task_index は除外
  per_dimension_check:
    healthy_range_ratio: [0.05, 5.0]           # bounds に対する length_scale の比率
    length_scale_over_bound: "runaway"          # ratio > 5.0
    length_scale_under_bound: "collapse"        # ratio < 0.05
  categorical_dim_check:                         # Hamming kernel には length_scale がない
    metric: "kernel_variance_ratio"              # base_variance に対する categorical kernel variance の比
    healthy_range: [0.1, 10.0]
  degeneracy_actions:
    on_runaway: "prior_re_specification_required_human_approval"
    on_collapse: "add_lengthscale_prior_or_more_data"
    severity: "stop_and_human_review"            # 即 fatal ではなく Skill 停止 → Human 判断
  fatal_on_ignore: true                         # Skill が診断を無視して次候補生成は fatal
```

**是正契約**：runaway/collapse を検出したら、Skill は次候補生成を停止し、Human に `length_scale_prior` の再指定を要請。Skill が自律的に prior を変えるのは fatal。

---

### 15.1.3 Acquisition の局所解

**症状**：`optimize_acqf` の `num_restarts` が足りず、acquisition が局所最大に落ちる。同じ領域を提案し続ける。

**診断契約**：

```yaml
acquisition_optimization_health:
  version: "v0.1"
  optimization_run:
    num_restarts: 20                            # 高次元では 40+
    raw_samples: 512                            # 高次元では 2048+
    convergence_tolerance: 1e-5
  diagnostic:
    top_k_acquisition_values: [1.42, 1.41, 1.39, 1.38, 1.36]  # multi-start の分散を記録
    top_k_dispersion: "healthy"                 # enum: healthy | narrow_all_same_region | degenerate
    candidate_location_diversity: "healthy"
  action_on_degenerate:
    increase_num_restarts_up_to: 100
    escalate_to_human_after_retries: 3
```

**是正契約**：`top_k_dispersion: narrow_all_same_region` を検出したら、num_restarts を最大 100 まで自動増加。3 回リトライでも改善なければ Human エスカレーション。

---

### 15.1.4 Pareto front の疑似収束

**症状**：hypervolume が数 iteration 停滞して Skill が「収束」と判断するが、実は未探索領域が残っている。

**診断契約**（Ch9 準拠 + 拡張）：

```yaml
pareto_convergence_diagnostic:
  version: "v0.2"
  hypervolume_history: [2.1, 2.3, 2.35, 2.36, 2.36]
  hypervolume_stagnation_metric: "delta_hv_over_last_3_iter < 0.01 * hv_scale"
  auxiliary_check:
    unexplored_region_test:                     # ★ operational な probe-set ベース
      method: "sobol_probe_set"                  # enum: sobol_probe_set | halton_probe_set | random_uniform_probe_set | grid_probe_set
      probe_set_size: 4096
      distance_metric: "standardized_euclidean" # continuous 次元のみ
      feature_set:
        include: ["continuous"]
        exclude: ["facility_id", "task_index"]
      test:                                      # 以下 3 つのいずれかまたは AND
        - {name: "min_nearest_observed_distance", threshold: 0.5, semantic: "probe が観測から遠い ratio"}
        - {name: "posterior_predictive_variance_p95", threshold: 0.3, semantic: "probe 上の posterior variance の 95 パーセンタイル"}
        - {name: "acquisition_value_p95",           threshold: 0.02, semantic: "probe 上の acquisition の 95 パーセンタイル"}
      status: "insufficient_coverage"           # いずれか通らないと exploration 余地あり
  false_convergence_triggered: true
  action: "reject_stop_condition_hypervolume_clause_and_recommend_exploration_boost"
```

**是正契約**：hypervolume stagnation でも `unexplored_region_test` が failed なら Pareto 疑似収束と判断し、`stop_condition.hypervolume clause` を Skill が発火させない。代わりに探索性の高い acquisition（qUCB / qMES）を Human に提案。

---

### 15.1.5 Safe BO の制約違反

**症状**：safe BO の feasibility model が overconfident で、真の制約違反領域を「安全」と誤判定（`predicted_feasibility_prob` が高いのに empirical rate が低い）。

**診断契約**（Ch10 §10.6 準拠）：

```yaml
safe_bo_calibration_check:
  version: "v0.1"
  feasibility_model_calibration:
    predicted_feasibility_prob_average: 0.92
    empirical_feasibility_rate: 0.65            # 過去 20 実験の実測 feasibility
    calibration_gap: 0.27                        # predicted - empirical、正なら overconfident
    calibration_status: "overconfident"          # enum: well_calibrated | overconfident | underconfident
  action:
    - "increase_feasibility_probability_threshold"
    - "add_conservative_prior"
    - "escalate_to_human"
```

**是正契約**：`|calibration_gap| > 0.15` なら Skill は次候補生成を停止し、feasibility model の再検討を Human に要請。閾値を勝手に上げるのは fatal（Ch10 §10.3.2）。

---

### 15.1.6 Stale / pending observations の扱い漏れ

**症状**：pending 実験が忘れられ、次 iteration で `X_pending` に含まれず、同じ点を再提案。

**診断契約**（Ch12 §12.7）：

```yaml
stale_pending_events:                           # append-only hash chain
  - event_id: "evt-stale-20261213-001"
    event_hash: "sha256:..."
    previous_event_hash: "sha256:..."
    experiment_id: "exp-A-039"
    submitted_at: "2026-12-10T09:00:00Z"
    expected_completion: "2026-12-10T15:00:00Z"
    detected_at: "2026-12-13T09:00:00Z"         # 72h 遅延
    ttl_hours: 24
    stale_status: "in_progress"                  # enum: in_progress | human_cancelled | failed | lost
    action: "escalate_to_human"
    autonomous_cancellation: false               # Skill が自律キャンセルするのは fatal
```

**是正契約**：pending 実験が TTL 超過したら Skill は自律キャンセルせず、Human に stale_pending_alert を発行。次候補生成は継続可能だが、超過 pending は `X_pending` に **残す**（fantasize 対象として扱う）。ただし **`stale_status: in_progress` の場合のみ** X_pending に残す。Human が `human_cancelled | failed | lost` に判定した後は X_pending から除外し、budget accounting を更新する。

---

### 15.1.7 Duplicate candidates

**症状**：batch acquisition が同一点を複数返す、または既に pending の点を再提案する。

**診断契約**（Ch12 §12.7 + Ch14 §14.4.1 step 5）：

```yaml
duplicate_candidate_events:
  - event_id: "evt-dup-20261210-001"
    event_hash: "sha256:..."
    detected_at: "2026-12-10T09:15:00Z"
    duplicate_pair:
      - candidate_id: "cand-A-01"
      - candidate_id: "cand-A-02"
    distance: 0.02                              # threshold 0.3 未満
    detection_source: "post_hoc_pairwise_check"
    also_duplicate_with_pending: []             # 既存 pending との重複もチェック
    fallback_applied: "sequential_greedy_replan"
```

**是正契約**：`post_hoc_pairwise_check` を毎 iteration 実施（batch 内 + 既存 pending との pairwise 両方）。違反時は `batch_diversity_policy.fallback_on_violation` に従う。fallback を Skill が勝手に "無視" するのは fatal。

---

### 15.1.8 Unit / bounds drift

**症状**：`search_space_bounds` の unit（degC vs K、MPa vs Pa）が iteration 途中で変わる、あるいは bounds が微妙に変動。

**診断契約**：

```yaml
search_space_integrity_check:
  version: "v0.2"
  bounds_hash_spec:
    algorithm: "sha256"
    canonical_serialization:
      json_style: "sorted_keys_canonical"        # RFC 8785 相当
      encoding: "utf-8"
      float_representation: "decimal_string_10_digits"
      variable_order: "sorted_alphabetical"      # spec 内の変数キーは alphabetical
      included_fields: ["variables", "types", "bounds", "units"]
    example: |
      canonical_input = {
        "variables": ["x_pressure", "x_temperature"],
        "types":     {"x_pressure": "continuous", "x_temperature": "continuous"},
        "bounds":    {"x_pressure": [0.1, 5.0], "x_temperature": [25.0, 300.0]},
        "units":     {"x_pressure": "MPa", "x_temperature": "degC"}
      }
      bounds_hash = sha256(json.dumps(canonical_input, sort_keys=True).encode("utf-8"))
  bounds_hash: "sha256:bbbb..."
  unit_declarations:                            # 全変数に unit を必須化
    x_temperature: "degC"
    x_pressure: "MPa"
  drift_detection:
    on_bounds_change_between_iterations: "fatal_new_run_required"
    on_unit_change: "fatal_new_run_required"
  authorized_change_events: []                  # Human 承認付き変更のみ許容
```

**是正契約**：`bounds_hash` を run_id と紐付け、iteration 開始時に検証。ハッシュ不一致は fatal（新 run 必須、`dag_revision_policy` と同じ扱い）。

---

### 15.1.9 Budget misaccounting

**症状**：`experiments_completed` の集計に pending や cancelled を含めるかで齟齬。Skill が budget を過小/過大評価。

**診断契約**（Ch14 §14.2.4 v0.2 準拠）：

```yaml
budget_reconciliation_check:
  version: "v0.2"
  status_source: "event_streams"                # entries の in-place status ではなく event streams から derive（Ch12 §12.3）
  status_enum: ["completed", "failed", "in_progress", "reserved", "cancelled"]  # ★ 相互排他
  disjoint_status_assertion: true                # 一つの experiment は 1 status のみ
  count_definitions:
    experiments_completed: "count(status == 'completed')"
    experiments_failed:    "count(status == 'failed')"     # budget にカウントする（試料消費のため）
    experiments_pending:   "count(status == 'in_progress')"
    experiments_reserved:  "count(status == 'reserved')"
    experiments_cancelled: "count(status == 'cancelled')"  # budget からは除外
  derivation_formula:
    available_for_bo: "total - completed - failed - pending - reserved"  # cancelled は 0 kill、total からも除く運用も可（下記）
  cancellation_policy:
    include_in_total: false                      # cancelled は total からも除外（試料未消費）
    if_sample_consumed: "treat_as_failed"        # 試料消費されていれば failed 扱いに再分類
  budget_invariant:
    "total == completed + failed + pending + reserved  AND  cancelled disjoint from all"
  fatal_on_invariant_violation: true
```

**是正契約**：`experiments_available_for_bo` の計算式を invariant として毎 iteration 検証。違反は fatal（新 iteration 開始不可）。

---

## セクション 2：Agentic 特有の失敗パターン

Skill が「賢く」なるほど生まれる、契約破りの類型。

### 15.2.1 Acquisition を "動的" に勝手に切り替える

**症状**：Skill が「今 iteration は qLogNEI より qUCB のほうが良さそう」と自律判断し、acquisition_spec を変更。

**診断契約**：

```yaml
acquisition_change_events:                      # append-only、hash chain
  - event_id: "evt-acq-change-20261212-001"
    event_hash: "sha256:..."
    previous_event_hash: "sha256:..."
    from: "qLogNEI"
    to: "qUCB"
    changed_at: "2026-12-12T10:00:00Z"
    changed_by: "human:project_lead_XXXX"       # ★ Skill による変更は fatal
    approval_id: "approval:HITL-20261212-ACQ01"
    rationale: "10 iter 停滞、探索性を上げる"
    cross_family_impact:                         # ★ 追加：objective family 変化検出
      from_family: "single_objective"            # qLogNEI は single-obj
      to_family:   "single_objective"            # qUCB も single-obj → 追加変更不要
      requires_objective_mode_change: false
      requires_objectives_spec_change: false
```

**是正契約**：`acquisition_spec.name` の変更は必ず `acquisition_change_events` に記録、Human 承認付き。Skill が `changed_by: skill:*` を書いたら fatal。

**★ Cross-family transition rule（例：qLogNEI → qLogNEHVI）**：single-objective acquisition から multi-objective acquisition への変更は acquisition 単独では成立しない。以下 3 event stream が同一 approval_id で **同時に** 追加されない限り fatal：

- `acquisition_change_events`：`from_family: single_objective → to_family: multi_objective`
- `objective_mode_change_events`：objective_mode の再宣言（Ch13 canonical enum）
- `objectives_spec_change_events`：objectives 配列（≥ 2）と reference_point / hv_scale の宣言

これらいずれかを欠いた変更は Skill にも Human にも fatal（新 run を開始すべき）。

Family 分類 canonical：

| Acquisition | Family |
|---|---|
| qLogEI / qLogNEI / qUCB / qPI / qKG / qMES / qTS / qPES | single_objective |
| qLogEHVI / qLogNEHVI | multi_objective |
| qParEGO / qNParEGO | multi_objective_via_scalarization |

---

### 15.2.2 外挿領域を "自信あり" と報告

**症状**：Skill が posterior variance を Human に伝えず、mean だけ「予測 9.5 GPa」と報告。実は外挿。

**診断契約**：§15.1.1 の HRD + reporting contract：

```yaml
prediction_reporting_contract:
  version: "v0.1"
  required_fields_for_prediction_report:
    - posterior_mean
    - posterior_variance
    - hrd_status                                # in_domain | boundary | extrapolation
    - confidence_interval_95                    # [mean - 1.96*std, mean + 1.96*std]
  extrapolation_report_style: "explicit_warning_top_line"
  fatal_on_missing_fields: true
```

**是正契約**：予測レポートで required_fields のいずれかを欠くと fatal。extrapolation の場合、レポート冒頭に警告を明示。

---

### 15.2.3 Human 承認なしに次の候補を実行

**症状**：Skill が「承認は形式的だし、私が実行しても同じ」と判断して自律実行。**vol-05 の安全 invariant 最大の破り**。**特に、approval_event_hash chain の整合性や candidate_hash の一致確認を省略すると、承認済みだが別候補を実行する fatal を検出できない。**

**診断契約**（Ch5 準拠、機械検証）：

```yaml
experiment_launch_precondition_check:
  version: "v0.2"
  required_before_launch:
    - name: "approval_id_present"
      predicate: "experiment_launch_authorization.approval_id != null"
    - name: "approved_by_is_human"
      predicate: "matches_regex(experiment_launch_authorization.approved_by, '^human:')"
    - name: "candidate_authorized"
      predicate: "target_candidate_id IN experiment_launch_authorization.scope.authorized_candidate_ids"
    - name: "candidate_hash_match"                     # ★ 追加
      predicate: "sha256(target_candidate_spec_canonical) == experiment_launch_authorization.scope.candidate_hash_map[target_candidate_id]"
    - name: "approval_recent"
      predicate: "now - experiment_launch_authorization.approved_at < 48h"
    - name: "approval_event_hash_chain_integrity"     # ★ 追加
      predicate: "verify_hash_chain(approval_events, seed_hash=run_id_seed_hash)"
    - name: "approval_event_matches_authorization"    # ★ 追加
      predicate: "experiment_launch_authorization.event_hash == last(approval_events).event_hash"
    - name: "no_intervening_revocation"                # ★ 追加
      predicate: "no revocation_event with approval_id == experiment_launch_authorization.approval_id"
  fatal_on_any_missing: true
  fatal_on_hash_chain_break: true
  audit_trail_ref: "vol-05:ch05:experiment_launch_audit_log_v0_1"
```

**是正契約**：launch 前に上記チェックを機械実施。1 つでも欠けたら、または hash chain が破損したら fatal。監査ログに Skill 実行の全 attempt を append。

---

### 15.2.4 Iteration seed を上書き

**症状**：Skill が「乱数固定で候補が偏るから」と seed を自律変更。Idempotency が壊れる。

**診断契約**（Ch14 §14.4.2 sequential_seed_provenance）：

```yaml
seed_integrity_check:
  version: "v0.2"
  three_phase_check:                             # ★ pre / runtime / post 3 段階
    pre_iteration:
      when: "iteration start, before any sampling"
      predicate: "load(pinned_seeds) succeeds AND pinned_seeds.hash matches contract_version_pin"
      fatal_on_fail: true
    runtime_intercept:
      when: "each acquisition / candidate_generation call"
      predicate: "seed argument passed to sampler == pinned_seeds[step_name]"
      mechanism: "sampler_call_hook logs (step_name, seed_used, timestamp)"
      fatal_on_mismatch: true
    post_iteration:
      when: "iteration end"
      predicate: "pinned_seeds == observed_used_seeds_from_hook_log"
      fatal_on_mismatch: true
  seed_change_events: []                          # Human 承認付き以外の変更禁止
  detection_gap_risk: "runtime_intercept が無いと pre/post 一致でも中間で改変され得る"
```

**是正契約**：3 段階（pre / runtime / post）を全て通過しなければ fatal。特に runtime_intercept は sampler の直下フックで seed を実測する。

---

### 15.2.5 Pending experiments を勝手にキャンセル

**症状**：Skill が「実験が遅い」「装置が空いた方が効率的」と判断して pending を自律キャンセル。

**診断契約**（Ch12 §12.3.1）：既に `cancellation_policy: requires_human_approval` と `cancellation_events` で防護済み。Skill による cancel は `cancelled_by: skill:*` になり、これを検出したら fatal。

---

### 15.2.6 Stop condition を勝手に緩める

**症状**：Skill が「あと 2 iter やれば hypervolume が伸びそう」と判断し、`stop_condition.clauses` の threshold を上げる。

**診断契約**：

```yaml
stop_condition_relaxation_check:
  version: "v0.1"
  pinned_stop_condition_hash: "sha256:..."      # contract_version_pin から
  current_stop_condition_hash: "sha256:..."
  mismatch_detected: false
  approval_to_relax: "fatal"                    # Ch14 stop_condition と一致
  relaxation_events: []                          # Human 承認付き変更のみ
```

**是正契約**：iteration 開始時に stop_condition を hash pin、変更検出は fatal。Human が緩めたい場合、新 run（新 run_id）を開始する。

---

### 15.2.7 多目的スカラー化重みを勝手に変更

**症状**：Skill が qLogNEHVI から qParEGO へ切替、weight を自律サンプリング（`weight_sampling` schema 未宣言）。

**診断契約**（Ch9 準拠）：

```yaml
scalarization_integrity_check:
  version: "v0.1"
  objectives_spec.scalarization.allowed: false  # 既定
  detected_scalarization_use: false             # true なら fatal
  weights_change_events: []                      # allowed: true でも Human 承認必須
```

**是正契約**：`scalarization.allowed: false` で qParEGO を Skill が呼んだら fatal。allowed: true でも weight は `weight_sampling` schema と `weights_change_events` に従う。

---

### 15.2.8 objective_mode を勝手に切り替える

**症状**：Ch13 準拠。Skill が learning → optimization に自律切替。

**診断契約**：既に Ch13 §13.3.2 `objective_mode_switch_events` で防護済み。Skill による switch は fatal。

---

### 15.2.9 契約 pin と実行 pipeline 順序の破り

**症状**：Skill が Ch14 §14.4.1 の 8 ステップ pipeline を並び替え（stop check を後回し、constraint precheck をスキップ）。

**診断契約**：

```yaml
pipeline_execution_audit:
  version: "v0.1"
  expected_order: ["pin", "stop_precheck", "pending", "acquisition", "diversity", "constraint_precheck", "authorization", "events_append"]
  actual_order: []                              # 実行 trace
  order_match: true
  skipped_steps: []
  fatal_on_reorder_or_skip: true
```

**是正契約**：Skill 実行 trace を audit と照合。並び替え・スキップは fatal。

---

### 15.2.10 契約 pin 自体の meta-integrity 破り

**症状**：Skill が `contract_version_pin` の中身（pinned bounds_hash、pinned stop_condition_hash、pinned seed 等）を書き換え、書き換え後に整合性チェックを通す。**contract_version_pin を単なる YAML として扱うと、それ自体の tamper 検出ができない**。

**診断契約**（Ch14 §14.4.2 の contract_version_pin を meta-integrity 化）：

```yaml
contract_version_pin_integrity_check:
  version: "v0.1"
  contract_pin_hash_chain:                       # ★ contract_version_pin 自体を append-only hash chain 化
    - event_id: "evt-pin-run-2026-1210-04-init"
      event_hash: "sha256:pin0..."
      previous_event_hash: null
      created_at: "2026-12-10T09:00:00Z"
      created_by: "human:project_lead_XXXX"
      pinned_payload_hash: "sha256:payload0..." # bounds_hash + seed + stop_condition + acquisition_spec + ... の canonical serialization
    # 以降、Human 承認付き revision のみ append 可
  runtime_verification:
    on_iteration_start:
      predicate: "hash(contract_version_pin_current_payload) == last(contract_pin_hash_chain).pinned_payload_hash"
      fatal_on_mismatch: true
    on_chain_break:
      predicate: "each event.previous_event_hash == previous event.event_hash"
      fatal_on_break: true
  storage_recommendations:
    - "hash chain を実験 log と別ストレージに保存（Skill が書き換えられない場所）"
    - "audit skill が独立プロセスで chain を再計算・比較"
```

**是正契約**：contract_version_pin の payload hash を hash chain の末端と照合。書き換えは fatal（新 run 開始）。

---

## 15.3 `bo_agentic_audit_checklist` — 監査 Skill 化

Skill 監査を自動化する統合契約：

```yaml
bo_agentic_audit_checklist:
  version: "v0.2"
  run_id: "run-2026-1210-04"
  check_schema:                                  # ★ 各 check の共通 schema
    fields: ["id", "result", "ref", "severity_on_fail", "blocks_next_iteration", "remediation_ref"]
    severity_enum: ["low", "medium", "high", "fatal"]
    result_enum:   ["pass", "fail_recoverable", "fail_fatal", "n/a"]
  general_bo_checks:                            # セクション 1
    - {id: "hrd_extrapolation",       result: "pass", ref: "hrd_diagnostic",                    severity_on_fail: "fatal",  blocks_next_iteration: true,  remediation_ref: "vol-05:ch15:15.1.1"}
    - {id: "length_scale_health",     result: "pass", ref: "length_scale_health_check",         severity_on_fail: "high",   blocks_next_iteration: true,  remediation_ref: "vol-05:ch15:15.1.2"}
    - {id: "acquisition_optimization",result: "pass", ref: "acquisition_optimization_health",   severity_on_fail: "medium", blocks_next_iteration: false, remediation_ref: "vol-05:ch15:15.1.3"}
    - {id: "pareto_convergence",      result: "n/a",  ref: "pareto_convergence_diagnostic",     severity_on_fail: "medium", blocks_next_iteration: false, remediation_ref: "vol-05:ch15:15.1.4"}
    - {id: "safe_bo_calibration",     result: "pass", ref: "safe_bo_calibration_check",         severity_on_fail: "high",   blocks_next_iteration: true,  remediation_ref: "vol-05:ch15:15.1.5"}
    - {id: "stale_pending",           result: "pass", ref: "stale_pending_events",              severity_on_fail: "medium", blocks_next_iteration: false, remediation_ref: "vol-05:ch15:15.1.6"}
    - {id: "duplicate_candidates",    result: "pass", ref: "duplicate_candidate_events",        severity_on_fail: "medium", blocks_next_iteration: false, remediation_ref: "vol-05:ch15:15.1.7"}
    - {id: "unit_bounds_drift",       result: "pass", ref: "search_space_integrity_check",      severity_on_fail: "fatal",  blocks_next_iteration: true,  remediation_ref: "vol-05:ch15:15.1.8"}
    - {id: "budget_reconciliation",   result: "pass", ref: "budget_reconciliation_check",       severity_on_fail: "fatal",  blocks_next_iteration: true,  remediation_ref: "vol-05:ch15:15.1.9"}
  agentic_specific_checks:                      # セクション 2
    - {id: "acquisition_change_authorized",         result: "pass", ref: "acquisition_change_events",                severity_on_fail: "fatal", blocks_next_iteration: true, remediation_ref: "vol-05:ch15:15.2.1"}
    - {id: "prediction_reporting_complete",         result: "pass", ref: "prediction_reporting_contract",            severity_on_fail: "high",  blocks_next_iteration: true, remediation_ref: "vol-05:ch15:15.2.2"}
    - {id: "experiment_launch_precondition",        result: "pass", ref: "experiment_launch_precondition_check",     severity_on_fail: "fatal", blocks_next_iteration: true, remediation_ref: "vol-05:ch15:15.2.3"}
    - {id: "seed_integrity",                        result: "pass", ref: "seed_integrity_check",                     severity_on_fail: "fatal", blocks_next_iteration: true, remediation_ref: "vol-05:ch15:15.2.4"}
    - {id: "pending_cancellation_authorized",       result: "pass", ref: "cancellation_events (human only)",         severity_on_fail: "fatal", blocks_next_iteration: true, remediation_ref: "vol-05:ch15:15.2.5"}
    - {id: "stop_condition_not_relaxed",            result: "pass", ref: "stop_condition_relaxation_check",          severity_on_fail: "fatal", blocks_next_iteration: true, remediation_ref: "vol-05:ch15:15.2.6"}
    - {id: "scalarization_authorized",              result: "pass", ref: "scalarization_integrity_check",            severity_on_fail: "fatal", blocks_next_iteration: true, remediation_ref: "vol-05:ch15:15.2.7"}
    - {id: "objective_mode_switch_authorized",      result: "pass", ref: "objective_mode_switch_events",             severity_on_fail: "fatal", blocks_next_iteration: true, remediation_ref: "vol-05:ch15:15.2.8"}
    - {id: "pipeline_execution_order",              result: "pass", ref: "pipeline_execution_audit",                 severity_on_fail: "fatal", blocks_next_iteration: true, remediation_ref: "vol-05:ch15:15.2.9"}
    - {id: "contract_pin_meta_integrity",           result: "pass", ref: "contract_version_pin_integrity_check",     severity_on_fail: "fatal", blocks_next_iteration: true, remediation_ref: "vol-05:ch15:15.2.10"}
  cross_cutting_checks:                          # ★ 追加：iteration をまたぐ健全性
    - {id: "task_correlation_drift",  result: "pass", ref: "task_correlation_drift_diagnostic", severity_on_fail: "high",   blocks_next_iteration: false, remediation_ref: "vol-05:ch11:MTGP"}
    - {id: "dag_revision_integrity",  result: "pass", ref: "dag_revision_events (Ch14)",        severity_on_fail: "fatal",  blocks_next_iteration: true,  remediation_ref: "vol-05:ch14:14.4"}
  overall_status: "pass"                         # enum: pass | fail_recoverable | fail_fatal
  overall_status_derivation:
    fail_fatal_if: "any(check.result == 'fail_fatal') OR any(check.result == 'fail_recoverable' AND check.severity_on_fail == 'fatal')"
    fail_recoverable_if: "any(check.result == 'fail_recoverable' AND check.severity_on_fail != 'fatal')"
    pass_if: "all(check.result IN ['pass', 'n/a'])"
  audit_report_hash: "sha256:audit..."
  audited_at: "2026-12-13T18:00:00Z"
  audited_by: "auto:audit_skill_v0_1 + human:auditor_XXXX"
```

**運用**：

- 毎 iteration 終了時に自動実行
- `overall_status: fail_*` の場合、次 iteration 開始をブロック
- audit_report_hash を provenance bundle に含める（vol-04 の provenance 21 フィールド v0.3 相当）

---

## 15.4 Skill 契約チェックリスト（audit 用）

- [ ] `hrd_diagnostic.fatal_on_undeclared_extrapolation: true`
- [ ] `length_scale_health_check` が全 iteration で実行
- [ ] `acquisition_optimization_health` に multi-start dispersion 記録
- [ ] `pareto_convergence_diagnostic` の unexplored_region_test を multi-objective で実施
- [ ] `safe_bo_calibration_check.calibration_gap` を継続監視
- [ ] `stale_pending_events` は Skill 自律キャンセル禁止
- [ ] `duplicate_candidate_events` は batch 内 + 既存 pending 両方を対象
- [ ] `search_space_integrity_check.bounds_hash` を run_id と紐付け
- [ ] `budget_reconciliation_check.status_source: event_streams`
- [ ] `acquisition_change_events.changed_by` が `human:*` のみ許容
- [ ] `prediction_reporting_contract.required_fields_for_prediction_report` を欠かない
- [ ] `experiment_launch_precondition_check` が launch 前に機械実行
- [ ] `seed_integrity_check` が iteration 終了時に照合
- [ ] `stop_condition_relaxation_check` が iteration 開始時に hash pin と照合
- [ ] `scalarization_integrity_check.detected_scalarization_use` が allowed でない限り false
- [ ] `pipeline_execution_audit.expected_order` と actual_order が一致
- [ ] `bo_agentic_audit_checklist.overall_status: pass` を次 iteration の gate 条件に

---

## 15.5 章末演習

**問 1**：Skill が iteration 5 で `qLogNEI` から `qUCB` に自律切替した。監査で何を確認し、どう是正すべきか？

**問 2**：Human に「予測 9.5 GPa」と報告されたが、その候補は observed dataset から length_scale の 5 倍離れていた。契約上の違反は？

**問 3**：Skill が seed を上書きし、同じ input で再実行すると別の candidates が出るようになった。監査で何を検知し、fatal 判定の根拠は？

**問 4**：iteration 8 で stop_condition の hypervolume threshold が 0.02 → 0.05 に増えていた。Skill による変更か Human 承認か、どう判別するか？

**問 5**：`bo_agentic_audit_checklist.overall_status: fail_fatal` が出た。次 iteration に進む前の対応手順は？

---

## 15.6 参考資料

### 本書内

- 第3章 §3.5：Skill 逸脱パターン全般
- 第5章：experiment_launch_authorization、provenance 21 フィールド
- 第6章：GP surrogate と length_scale
- 第7章：acquisition と HRD schema
- 第8章：BNN/DKL、HRD v0.2
- 第9章：多目的、scalarization、weight_sampling
- 第10章：constraints、feasibility model、event schema
- 第11章：MTGP/HierGP、task_correlation_matrix
- 第12章：pending_experiments、cancellation_events、duplicate_candidate_events
- 第13章：objective_mode_switch_events
- 第14章：contract_version_pin、sequential_seed_provenance、iteration pipeline
- 第16章（planned）：組織展開と監査 role

### 外部参考

- Amodei et al., "Concrete Problems in AI Safety", arXiv:1606.06565 — reward hacking, unsafe exploration
- Krause & Golovin, "Submodular Function Maximization" — acquisition 局所解と greedy 保証
- BoTorch docs — `qLogNEHVI` の pareto convergence 診断
- vol-01 第15章、vol-02 第15章、vol-03 第15章、vol-04 第15章 — 各巻の失敗パターン

> [!NOTE]
> 次章（第16章）では、本章の監査契約を **組織運用サイクル** に組込みます。Skill 提案 → チーム承認 → 装置予約 → 実行 → データ取込 の各ステップに `bo_agentic_audit_checklist` を織り込む方法、複数研究者間での surrogate 共有、装置予約統合、vol-06（生成モデル・逆設計）への道しるべを扱います。
