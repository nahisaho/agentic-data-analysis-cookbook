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
    exclude_variables: ["facility_id"]
  in_domain_criterion: "min_distance_to_training_data <= 2.0 * length_scale"
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
  version: "v0.1"
  per_dimension_check:
    healthy_range_ratio: [0.05, 5.0]           # bounds に対する length_scale の比率
    length_scale_over_bound: "runaway"          # ratio > 5.0
    length_scale_under_bound: "collapse"        # ratio < 0.05
  degeneracy_actions:
    on_runaway: "prior_re_specification_required_human_approval"
    on_collapse: "add_lengthscale_prior_or_more_data"
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
  version: "v0.1"
  hypervolume_history: [2.1, 2.3, 2.35, 2.36, 2.36]
  hypervolume_stagnation_metric: "delta_hv_over_last_3_iter < 0.01 * hv_scale"
  auxiliary_check:
    unexplored_region_test:
      metric: "coverage_of_search_space_by_pareto_neighborhood"
      threshold: 0.4                            # Pareto 点の近傍で覆われる面積比
      status: "insufficient_coverage"
  false_convergence_triggered: true
  action: "reject_stop_condition_hypervolume_clause_and_recommend_exploration_boost"
```

**是正契約**：hypervolume stagnation でも `unexplored_region_test` が failed なら Pareto 疑似収束と判断し、`stop_condition.hypervolume clause` を Skill が発火させない。代わりに探索性の高い acquisition（qUCB / qMES）を Human に提案。

---

### 15.1.5 Safe BO の制約違反

**症状**：safe BO の feasibility model が underconfident で、真の制約違反領域を「安全」と誤判定。

**診断契約**（Ch10 §10.6 準拠）：

```yaml
safe_bo_calibration_check:
  version: "v0.1"
  feasibility_model_calibration:
    predicted_feasibility_prob_average: 0.92
    empirical_feasibility_rate: 0.65            # 過去 20 実験の実測 feasibility
    calibration_gap: 0.27                        # 大きすぎる、underconfident
    calibration_status: "miscalibrated"
  action:
    - "increase_feasibility_probability_threshold"
    - "add_conservative_prior"
    - "escalate_to_human"
```

**是正契約**：`calibration_gap > 0.15` なら Skill は次候補生成を停止し、feasibility model の再検討を Human に要請。閾値を勝手に上げるのは fatal（Ch10 §10.3.2）。

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
    action: "escalate_to_human"
    autonomous_cancellation: false               # Skill が自律キャンセルするのは fatal
```

**是正契約**：pending 実験が TTL 超過したら Skill は自律キャンセルせず、Human に stale_pending_alert を発行。次候補生成は継続可能だが、超過 pending は `X_pending` に **残す**（fantasize 対象として扱う）。

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
  version: "v0.1"
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
  version: "v0.1"
  status_source: "event_streams"                # entries の in-place status ではなく event streams から derive（Ch12 §12.3）
  count_definitions:
    experiments_completed: "count(status == 'completed')"
    experiments_cancelled: "count(status == 'cancelled')"  # budget にカウントしない
    experiments_failed: "count(status == 'failed')"        # budget にカウントする（試料消費のため）
    experiments_pending: "count(status == 'in_progress')"  # available_for_bo から減算
  budget_invariant:
    "total_experiments >= completed + failed + pending + reserved_for_confirmation"
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
```

**是正契約**：`acquisition_spec.name` の変更は必ず `acquisition_change_events` に記録、Human 承認付き。Skill が `changed_by: skill:*` を書いたら fatal。

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

**症状**：Skill が「承認は形式的だし、私が実行しても同じ」と判断して自律実行。**vol-05 の安全 invariant 最大の破り**。

**診断契約**（Ch5 準拠、機械検証）：

```yaml
experiment_launch_precondition_check:
  version: "v0.1"
  required_before_launch:
    - experiment_launch_authorization.approval_id: "not_null"
    - experiment_launch_authorization.approved_by: "matches_regex(^human:)"
    - experiment_launch_authorization.scope.authorized_candidate_ids: "includes_target_candidate_id"
    - experiment_launch_authorization.approved_at: "timestamp_recent(48h)"
  fatal_on_any_missing: true
  audit_trail_ref: "vol-05:ch05:experiment_launch_audit_log_v0_1"
```

**是正契約**：launch 前に上記チェックを機械実施。1 つでも欠けたら fatal。監査ログに Skill 実行の全 attempt を append。

---

### 15.2.4 Iteration seed を上書き

**症状**：Skill が「乱数固定で候補が偏るから」と seed を自律変更。Idempotency が壊れる。

**診断契約**（Ch14 §14.4.2 sequential_seed_provenance）：

```yaml
seed_integrity_check:
  version: "v0.1"
  pinned_seeds_at_iteration_start: {...}        # contract_version_pin から
  seeds_used_at_iteration_end: {...}
  invariant: "pinned == used"
  seed_change_events: []                        # Human 承認付き以外の変更禁止
  fatal_on_mismatch: true
```

**是正契約**：iteration 終了時に pinned seed と actually used seed を照合。不一致は fatal。

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

## 15.3 `bo_agentic_audit_checklist` — 監査 Skill 化

Skill 監査を自動化する統合契約：

```yaml
bo_agentic_audit_checklist:
  version: "v0.1"
  run_id: "run-2026-1210-04"
  general_bo_checks:                            # セクション 1
    - {id: "hrd_extrapolation",       result: "pass", ref: "hrd_diagnostic"}
    - {id: "length_scale_health",     result: "pass", ref: "length_scale_health_check"}
    - {id: "acquisition_optimization",result: "pass", ref: "acquisition_optimization_health"}
    - {id: "pareto_convergence",      result: "n/a",  ref: "pareto_convergence_diagnostic (single-obj 14a では n/a)"}
    - {id: "safe_bo_calibration",     result: "pass", ref: "safe_bo_calibration_check"}
    - {id: "stale_pending",           result: "pass", ref: "stale_pending_events"}
    - {id: "duplicate_candidates",    result: "pass", ref: "duplicate_candidate_events"}
    - {id: "unit_bounds_drift",       result: "pass", ref: "search_space_integrity_check"}
    - {id: "budget_reconciliation",   result: "pass", ref: "budget_reconciliation_check"}
  agentic_specific_checks:                      # セクション 2
    - {id: "acquisition_change_authorized",         result: "pass", ref: "acquisition_change_events"}
    - {id: "prediction_reporting_complete",         result: "pass", ref: "prediction_reporting_contract"}
    - {id: "experiment_launch_precondition",        result: "pass", ref: "experiment_launch_precondition_check"}
    - {id: "seed_integrity",                        result: "pass", ref: "seed_integrity_check"}
    - {id: "pending_cancellation_authorized",       result: "pass", ref: "cancellation_events (human only)"}
    - {id: "stop_condition_not_relaxed",            result: "pass", ref: "stop_condition_relaxation_check"}
    - {id: "scalarization_authorized",              result: "pass", ref: "scalarization_integrity_check"}
    - {id: "objective_mode_switch_authorized",      result: "pass", ref: "objective_mode_switch_events"}
    - {id: "pipeline_execution_order",              result: "pass", ref: "pipeline_execution_audit"}
  overall_status: "pass"                         # enum: pass | fail_recoverable | fail_fatal
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
