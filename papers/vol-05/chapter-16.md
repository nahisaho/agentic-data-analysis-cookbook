# 第16章　組織展開と終章 — 実験リソースの割当・BO 運用・次巻への道しるべ

> [!NOTE]
> **章の位置付け**：本章は vol-05 の終章として、Ch5-Ch15 で確立した **BO Skill × Human-in-the-loop 3 段階権限 × 21-field provenance × RFC 8785 event_hash chain × 15.3 監査 Skill (`bo_agentic_audit_checklist`)** を **単一 BO プロジェクト内の技術契約** から **組織横断の運用ガバナンス** に持ち上げる。技術詳細の追加は行わず、既に canonical 化された仕組み（Ch5 §5.2 provenance、Ch5 §5.3 `experiment_launch_authorization` の vol-04 L3 parent chain、Ch10 §10.5 `event_canonical_serialization`、Ch15 §15.3 監査 Skill）を **どの役割が・いつ・どの権限で** 実施するかを整理する。**新しい fatal / gate / provenance field は定義しない**。

## 16.1 vol-05 の到達点 — 5 つの canonical レイヤ

vol-05 が読者に手渡した canonical schema は以下の 5 層に整理できる：

| Layer | Canonical 要素 | 定義章 | 運用主体 |
|---|---|---|---|
| **L0: Surrogate 契約** | `surrogate_model_family` / `kernel_spec` / `tensor_encoding_contract` / hierarchical GP hyperprior | Ch5 §5.2 / Ch6 / Ch8 / Ch11 | Skill (fit) + BO Reviewer |
| **L1: Acquisition 契約** | `acquisition_spec.name` (canonical enum) / `hallucinated_recommendation_detection` (Ch7 operational 定義) / `batch_diversity_policy_v0_2` (Ch12) | Ch5 §5.2 / Ch7 / Ch9 / Ch12 | Skill (compute) + Research Lead (approve) |
| **L2: 実験実行承認** | `experiment_launch_authorization` (vol-04 L3 に `parent_authorization_id` で chain) / `^human:<id>` 承認 identity / `sequential_seed_provenance` (上書き禁止) | Ch5 §5.3 / vol-04 Ch4 §4.6.2 | Facility Experiment Board (Accountable) |
| **L3: 制約・安全契約** | `constraints_declared` / `hard_constraints_ceramic_v0_2` (Ch10 §10.6) / safe BO `feasibility_probability_threshold` | Ch10 | Skill (verify) + Safety Officer |
| **L4: 監査 (Audit) と施設昇格** | `bo_agentic_audit_checklist` / `event_hash` chain (Ch10 §10.5 `event_canonical_serialization` RFC 8785 JCS) / `experiment_reservation_events` (Ch14 §14.4.0) / `dag_revision_policy` | Ch10 §10.5 / Ch12 / Ch14 / Ch15 §15.3 | Facility BO Review Board |

**vol-05 の中心命題**：これら 5 層は **event_hash chain**（RFC 8785 JCS + `sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8'))`）によって **推移的に接続** されており、任意の下流主張（次候補の推奨・batch 実行・施設装置の予約・vol-06 生成モデルへの surrogate 譲渡）は **上流の全 event_hash を再計算して verify できる**。**逐次的な決定の連鎖を hash-verifiable に定義した点** が vol-05 の技術的達成である。

### 16.1.1 Canonical Schema クイックリファレンス

読者が canonical 名を素早く参照できるよう、vol-05 で確定した主要 canonical 要素を 1 表に集約：

| Canonical 要素 | SoT 章 | 関連 canonical 名 | 参照章 |
|---|---|---|---|
| `surrogate_model_family` enum | Ch5 §5.2.X | single_task_gp / fixed_noise_gp / heteroskedastic_gp / multi_task_gp / hierarchical_gp / random_forest / bayesian_neural_network / deep_kernel_learning | Ch6 / Ch8 / Ch11 |
| `acquisition_spec.name` enum（単目的） | Ch5 §5.2.X | qLogEI / qLogNEI / qUCB / qPI / qKG / qMES / qTS / qPES | Ch7 |
| `acquisition_spec.name` enum（多目的・スカラー化） | Ch5 §5.2.X | qLogEHVI / qLogNEHVI / qParEGO / qNParEGO | Ch9 |
| `active_learning_acquisition_spec.name` enum | Ch13 §13.2 (canonical enum table L63-68) | 6 canonical values: predictive_variance / predictive_entropy / query_by_committee / expected_model_change / mutual_information_bald / integrated_variance_reduction | Ch13 |
| `experiment_launch_authorization` schema | Ch5 §5.3 | `authorization_id: elauth_YYYYMMDD_HHMMSS_iter<n>` (Ch5 §5.3 canonical、H-R2-1 note 参照) / `parent_authorization_id: "vol-04:L3_intervention_execution_authorization:<id>"` / `status` enum (Ch5 §5.2.Y canonical、H7 note 参照) / `approver_id: ^human:<id>` | Ch12 / Ch14 / Ch15 |

> [!IMPORTANT]
> **H7 note — `status` enum の 4 値化**：本章 §16.1.1〜§16.7 は canonical `status` を `pending | approved | denied | revoked` の 4 値として扱います。これは vol-04 Ch4 §4.6.2 および vol-05 Ch5 §5.2.Y の canonical と一致します。ただし **Ch5 §5.3 L306 の inline example は現時点で 3 値 (`pending | approved | denied`, `revoked` 抜け) のまま**であり、施設 registry から見ると upstream 不整合です。次回 vol-05 patch（v0.2.1）で Ch5 §5.3 の inline comment を 4 値に揃える予定です。それまでの間、施設運用では **canonical SoT である Ch5 §5.2.Y の 4 値 enum を採用**してください。

> [!IMPORTANT]
> **H-R2-1 note — `authorization_id` spelling の upstream drift**：本章は `elauth_YYYYMMDD_HHMMSS_iter<n>` を canonical form として使用します。これは Ch5 §5.3 の inline examples（L305, L383, L398, L433）と一致します。ただし **Ch5 §5.2.Y の canonical 表 (L166)** は `l3_auth_YYYYMMDD_HHMMSS_iter<n>` (vol-04) / **`exp_launch_YYYYMMDD_HHMMSS_iter<n>` (vol-05)** を宣言しており、spelling が異なります（`elauth_` vs `exp_launch_`）。施設 registry から見ると upstream 不整合です。次回 vol-05 patch（v0.2.1）で Ch5 §5.2.Y を `elauth_` に揃える予定です。それまでの間、施設運用では **本章の `elauth_` を canonical として採用**してください（`exp_launch_` は Ch5 §5.2.Y の旧 spelling として alias 扱い）。
| `hallucinated_recommendation_detection` 合成基準 | Ch7 | GP 予測分散閾値 × Mahalanobis 距離 × length scale 越え検知の AND | Ch8 (BNN/RF に拡張) / Ch15 |
| `sequential_seed_provenance` | Ch5 §5.2 | seed の上書き禁止契約（v0.2 field） | Ch15 §15.2 (Agentic 失敗) |
| `hard_constraints_ceramic_v0_2` | Ch10 §10.6 | 温度上限 / 圧力上限 / 組成範囲の canonical 例 | Ch14 capstone |
| `batch_diversity_policy_v0_2` | Ch12 | batch 内の類似候補排除ポリシー | Ch14 |
| `event_hash` / `previous_event_hash` schema | Ch10 §10.5 anchor `event_canonical_serialization` | RFC 8785 JCS + `sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8'))` | Ch12 / Ch13 / Ch14 / Ch15 |
| `experiment_status_transition_events` | Ch12 §12.7.1 | pending / approved / launched / completed / cancelled / superseded の状態遷移 event | Ch15 |
| `dag_revision_policy` | Ch14 §14.4.0 | 因果 DAG の逐次改訂ポリシー（BO 途中で DAG を変えるときの契約） | Ch14 capstone |
| `experiment_reservation_events` | Ch14 §14.4.0-b | 装置予約の event stream | Ch14 capstone / 本章 §16.3 |
| `bo_agentic_audit_checklist` (監査 Skill) | Ch15 §15.3 | Section 1 + Section 2 の 2 セクション構成、BO 一般失敗 + Agentic 特有失敗の統合 checklist | Ch15 |
| 21-field BO × 逐次 × Agentic provenance | Ch5 §5.2 | Core 18 + 拡張 3（`bo_library_stack_metadata` / `sequential_seed_provenance` / `batch_size`） | 全章 |

上表を単一 SoT として運用し、施設 registry / audit manifest はこの命名に従うこと（drift 防止）。vol-05 内での旧名からの alias は原則として認めず、canonical 名で統一する。

## 16.2 責任分担マトリクス — RACI と BO 3 段階権限の対応

> [!NOTE]
> **Glossary — 施設側 3 主体の役割区分**：
> - **Facility Experiment Board**：**個別実験単位** の実験実行承認（L2 `experiment_launch_authorization` 発行）。BO iteration ごとに動く小規模承認体で、Research Lead + BO Reviewer + Facility Ops 責任者の常設ミーティング。
> - **Facility BO Review Board**：**四半期ガバナンス** 主体（L4 装置予約優先度・L5 監査・cross-project 共有承認・canonical schema versioning）。§16.9 原則 2 で常設化。
> - **Facility Ops**：**装置スケジューリング実務**（L4 の日常運用 R、`experiment_reservation_events` の実装）。Facility BO Review Board の accountable 下で動く。

Ch5 §5.3 で確立した BO 3 段階権限（**surrogate 学習・acquisition 計算は自律 / 候補の絞り込み提案は準自律 / 実験実行判断は必ず Human 承認**）を RACI（Responsible / Accountable / Consulted / Informed）で展開：

| 承認レイヤ | Skill | Research Lead | BO Reviewer | Facility Experiment Board | Facility BO Review Board | Facility Ops | Safety Officer |
|---|---|---|---|---|---|---|---|
| **L0 Surrogate fit / 再学習** | **R** (autonomous) | **A** (surrogate 選択の妥当性) | C (kernel 選択・技術品質) | I | I | — | — |
| **L1 Acquisition 計算 / 候補提示** | **R** (propose_only) | **A** | C | I | I | — | — |
| **L2 実験実行承認** (`experiment_launch_authorization`) | R (propose_only) | C | R (BO 適合性) | **A** | I | I | C (safety 制約時) |
| **L3 制約 gate 検証** | **R** (verify) | C | C | I | I | — | **A** |
| **L4 施設装置予約 / batch 実行** | R (propose) | C | C | I | **A** | R (実務) | I |
| **L5 監査 (`bo_agentic_audit_checklist`)** | R (audit_manifest 生成) | C | C | I | **A** | C | C |

**運用原則**：

- **A (Accountable) は常に人間**。Skill は **A になれない**（vol-01 Ch6 / vol-04 Ch4 §4.6.3 の autonomy 境界と一致）
- **L1 / L2 / L4 / L5 の A 分離**：L1 は "どの候補を提案するか" の妥当性（**単一 BO ループ内、Research Lead**）、L2 は "実際に装置を動かして実験するか" の妥当性（**単一実験単位、Facility Experiment Board**）、L4/L5 は "施設装置の割当と監査結果集約" の妥当性（**四半期ガバナンス、Facility BO Review Board**）。Facility Experiment Board と Facility BO Review Board は **同一組織の別 body**（前者は個別実験承認、後者は四半期ガバナンス）
- **L3 の Safety Officer**：`hard_constraints_ceramic_v0_2` の制約破りは **単純な BO の失敗** ではなく **装置損傷・人的被害** に直結する。Safety Officer が独立の A を持ち、Skill は verify のみ（Ch10 §10.6）
- **Skill の R は原則 "propose_only"**：L0 の surrogate fit のみが完全自律だが、その fit も **`sequential_seed_provenance` の seed 上書き禁止契約** によって Human 追跡可能。Ch5 §5.2 provenance field `agent_action_log` に **全 propose 履歴 + surrogate 再学習履歴** が記録される
- **L4 の Facility Ops による実務 R**：`experiment_reservation_events`（Ch14 §14.4.0-b）が単一の event stream として運用される。複数プロジェクトの候補提案が施設装置を争う場合、**予約優先度は Skill が決めない**（本章 §16.3.3 で施設運用ポリシーを定義）。Facility Ops は日常運用の R、Facility BO Review Board が優先度 conflict 解消の A を持つ

## 16.3 施設運用のガバナンスサイクル

Ch15 の `bo_agentic_audit_checklist` を単一プロジェクトの gate から **施設運用の周期的レビュー機構** に持ち上げる：

### 16.3.1 BO サイクルの組織的運用フロー

以下フローの `event stream:` 表記は **canonical event stream への集約** を示す。角括弧 `[...]` は本章の教育的 label（canonical ではない）であり、実運用では対応する canonical stream に append される。

```
[Iteration i]
  1. Skill: surrogate 再学習 (L0, autonomous)
        → [教育label: surrogate_refit] を authorization_events_stream (Ch5) に append
  2. Skill: acquisition 計算 → 候補 k 個を propose (L1, propose_only)
        → [教育label: acquisition_proposal] を authorization_events_stream (Ch5) に append
  3. Research Lead: 候補レビュー → k' 個に絞り込む (L1 承認)
        → [教育label: candidate_shortlist] を authorization_events_stream (Ch5) に append
  4. Safety Officer: hard_constraints 検証 (L3)
        → [教育label: constraint_verification] を constraint_violation_events (Ch10 §10.5.4) に append
        （違反時のみ event 発火、pass 時は verification_log の照合のみ）
  5. Facility Experiment Board: experiment_launch_authorization 発行 (L2)
        → authorization_events_stream (Ch5、canonical stream)
        → authorization_id: elauth_YYYYMMDD_HHMMSS_iter<n> (Ch5 §5.3 canonical、H-R2-1 note 参照)
  6. Facility Ops: 装置予約
        → experiment_reservation_events (Ch14 §14.4.0-b、canonical stream)
  7. 実験実行 → データ取得 → observation ingestion
        → experiment_status_transition_events (Ch12 §12.7.1、canonical stream)
  8. Skill: budget_remaining / stop_condition 更新
        → experiment_status_transition_events (Ch12) の state snapshot として reflect
        （state field 更新は event stream ではなく state を伴う transition event）
[Next iteration or Stop]
```

各 event は `previous_event_hash` で前 event に chain されており、**BO サイクル全体が RFC 8785 JCS canonical serialization による hash chain として verifiable**（Ch10 §10.5 anchor `event_canonical_serialization` の帰結）。

> [!NOTE]
> **event stream map**：本章の教育 label と canonical stream の対応は以下：`surrogate_refit` / `acquisition_proposal` / `candidate_shortlist` → `authorization_events_stream` (Ch5)、`constraint_verification` → `constraint_violation_events` (Ch10 §10.5.4)、`budget/stop_condition 更新` → `experiment_status_transition_events` (Ch12 §12.7.1) の state snapshot。**新規 canonical event stream は Ch16 では定義しない**（章冒頭宣言通り）。

### 16.3.2 周期的レビュー — Quarterly BO Review Board

施設内で複数の BO プロジェクトが並行して走る場合、以下の周期的レビューを推奨する：

| レビュー種別 | 頻度 | 主催 | 入力 | 出力 |
|---|---|---|---|---|
| **プロジェクト単位 BO Review** | iteration 5 回ごと | Research Lead | `bo_agentic_audit_checklist` (Ch15 §15.3) の Section 2（Agentic 特有）出力 = §15.2.1〜§15.2.10 の 10 subsection 集計 | 継続 / 中断 / 方針変更 |
| **施設 BO Review Board** | 四半期 | Facility BO Review Board | event: `experiment_reservation_events` (Ch14 §14.4.0-b) / state snapshot: `budget_remaining` 集計（`experiment_status_transition_events` 由来） / 各プロジェクトの `bo_agentic_audit_checklist` Section 2 集約 | 装置優先度 rebalance / 予算再配分 |
| **Cross-Project Surrogate Sharing Review** | 半期 | Facility BO Review Board + Research Leads | 各プロジェクトの surrogate `dag_of_record_uri` + `kernel_spec` の overlap 分析 | 共有 surrogate 候補の特定（§16.4） |
| **年次 Safety Review** | 年次 | Safety Officer | `constraint_verification_events` の失敗率 + `hard_constraints_ceramic_v0_2` の実効性 | 制約 gate 更新 / 装置運用制限強化 |

### 16.3.3 装置予約優先度の決定 — Skill が決めない領域

複数プロジェクトの候補提案が同一装置を争う場合、優先度決定は **Skill の権限外** とする。以下を運用ポリシーとする：

1. **明示的優先度メタデータ**：各プロジェクトは `experiment_reservation_events` に `project_priority_tier: {critical, high, normal, exploratory}` を宣言する（施設運用ドキュメント側の field、canonical schema には含めない）
2. **conflict 発生時の解消手順**：
   - (a) 同一 tier 内では先着（`event_hash` chain の `created_at` timestamp（Ch10 §10.5.1）による先着順）
   - (b) 異 tier 間では tier 順（critical > high > normal > exploratory）
   - (c) tier 昇格の要求は **必ず人間** が施設 BO Review Board に申請
3. **Skill が禁止される行動**：
   - 他プロジェクトの予約を勝手に取り消す（**Ch15 §15.2.5 の "自プロジェクト pending キャンセル" を他プロジェクトに拡張した Agentic 失敗**。Ch15 §15.2 には未列挙、施設運用側で明示禁止）
   - 自プロジェクトの tier を prompt injection で書き換える
   - `experiment_reservation_events` の event_hash chain を再計算せずに reservation を上書き

## 16.4 Cross-Project Surrogate Sharing — 複数研究者間での再利用

ARIM 施設として複数研究者が同一装置・類似 search space で BO を回す場合、surrogate（および観測データ）の共有が候補となる。canonical schema を破壊せずに共有を実現するための運用契約を以下に整理：

### 16.4.1 共有可能条件

以下を **すべて** 満たすときのみ、surrogate を cross-project 共有できる：

| 条件 | canonical 参照 | 確認方法 |
|---|---|---|
| (a) `search_space_bounds` が overlap（片方が subset） | Ch5 §5.2 | 数値比較 |
| (b) `constraints_declared` が compatible（片方が superset） | Ch10 | 制約集合の含意関係 |
| (c1) `surrogate_model_family` が一致 | Ch5 §5.2.X | canonical enum 照合 |
| (c2) `kernel_spec` が一致 | Ch6 | kernel spec 照合 |
| (d) 装置差の hierarchical prior 構造が互換（Ch11 の multi-task GP を使う場合） | Ch11 | prior hyperparameter の shape 一致 |
| (e) 観測データの `tensor_encoding_contract` が一致 | Ch5 §5.2 Group A | encoding version 照合 |
| (f) 元プロジェクトの `sharing_authorization_ledger` に対応 entry が存在する（`sharing_authorization_id: shauth_YYYYMMDD_HHMMSS_seq<n>` で参照可能、後述 §16.4.3） | 施設 registry 側 | ledger entry の存在検証 |
| (g) `hard_constraints_ceramic_v0_2` に **共有制限フラグが立っていない**（例：企業秘密条項） | Ch10 §10.6 | 施設 registry 側 metadata |

> [!NOTE]
> 条件 (f) は **canonical status enum を拡張しない**方針。`experiment_launch_authorization.status` は Ch5 §5.2.Y の 4 値（pending | approved | denied | revoked）のまま、cross-project 共有承認は施設運用側の `sharing_authorization_ledger` エントリと `sharing_authorization_id` で表現する。

条件 (f) の共有承認は **Skill 側では発行できない**。必ず Research Lead × Facility BO Review Board × 元データ所有者の 3 者承認が必要（後述 §16.4.3）。

### 16.4.2 共有時の provenance chain

共有された surrogate を下流プロジェクトが使う場合、下流の provenance には以下を必ず記録：

```yaml
surrogate_provenance:
  origin_project_id: "proj_2026_arim_ceramic_hard_v2"
  origin_experiment_launch_authorization_id: "elauth_20261115_093015_iter3"  # 元プロジェクトの L2 承認 (Ch5 §5.3 canonical form、H-R2-1 note 参照)
  origin_last_event_hash: "sha256:def456..."                                   # 元プロジェクトの authorization_events_stream (Ch5) の tail event の event_hash
  sharing_authorization_id: "shauth_20261201_143022_seq1"                      # 3 者承認の identity (施設 sharing_authorization_ledger 側 field)
  sharing_authorization_approvers:                                             # 承認順は §16.4.3 の申請フローに従う
    - {approver_id: "^human:research_lead_downstream", role: "downstream_pi", order: 1}
    - {approver_id: "^human:origin_data_owner", role: "data_owner", order: 2}
    - {approver_id: "^human:facility_bo_review_board_chair", role: "board_chair", order: 3}
  compatibility_verification:                                                  # §16.4.1 の (a)-(g) 検証結果
    search_space_bounds_check: passed
    constraints_declared_check: passed
    surrogate_family_check: passed
    kernel_spec_check: passed
    hierarchical_prior_check: passed
    tensor_encoding_contract_check: passed
    sharing_ledger_entry_check: passed                                         # (f) sharing_authorization_ledger 存在検証
    sharing_flag_check: passed                                                 # (g) 制限フラグ
  compatibility_verification_event_hash: "sha256:..."                          # 検証 event
```

> [!NOTE]
> 上記 `surrogate_provenance` block は **施設運用ドキュメント側の推奨 metadata** であり、canonical Ch5 §5.2 の 21 fields には含まれない（章冒頭宣言）。`origin_last_event_hash` は元プロジェクトの `authorization_events_stream` (Ch5) の tail event を参照する運用契約。`shauth_YYYYMMDD_HHMMSS_seq<n>` は施設 sharing ledger の identity format で、canonical schema ではない。

**重要**：下流プロジェクトの `sequential_seed_provenance`（Ch5 §5.2）は **元プロジェクトから継承しない**。下流は独立の seed を発行し、**元プロジェクトの seed を上書きすることも禁止**（**Ch15 §15.2.4 の "Iteration seed を上書き" Agentic 失敗の cross-project 変種**、cross-project 共有時にも同等の禁止が施設運用側で明示される）。

### 16.4.3 3 者承認プロセス

surrogate 共有承認は以下を経る：

```
[申請] 下流 Research Lead → sharing_authorization_request
  ↓
[技術検証] BO Reviewer → §16.4.1 (a)-(e)（(c) は (c1)/(c2) を含む計 6 sub-conditions）を機械的検証
  ↓
[権利検証] 元データ所有者 → `sharing_authorization_ledger` に entry を発行 (f)、`hard_constraints_ceramic_v0_2` の共有制限フラグを再確認 (g)
  ↓
[施設承認] Facility BO Review Board → 3 者承認を統合
  ↓
sharing_authorization_id 発行（`shauth_YYYYMMDD_HHMMSS_seq<n>` 形式、施設 sharing_authorization_ledger 側 identity）
  ↓
event_hash chain に register（施設 registry 側の event stream、authorization_events_stream に append）
```

Skill が禁止される行動：**§16.4.1 の (a)-(e)（(c1)/(c2) を含む 6 sub-conditions）を "自動 pass" と報告して 3 者承認を bypass する**（**Ch15 §15.2.3 の "Human 承認なしに次の候補を実行" Agentic 失敗の cross-project 共有変種**、Ch15 §15.2 には未列挙、施設運用側で明示禁止）。

## 16.5 因果構造の再利用と BO の連結 — vol-04 との橋渡し

vol-04 Ch13 capstone で作られた因果 DAG（例：`causal_dag_ceramic_v1`、vol-04 appendix-c §C.12）は vol-05 の Ch14 capstone で **search space の絞り込み** に使われる。この cross-vol 連結の運用契約を整理：

### 16.5.1 DAG-derived search space の canonical 記録

vol-05 BO が vol-04 DAG を利用する場合、以下 `search_space_derivation` block を provenance の **施設運用ドキュメント側 metadata** として推奨する（**canonical Ch5 §5.2 の 21 fields には追加しない**、章冒頭宣言）：

```yaml
search_space_derivation:
  source: "vol04_dag"
  source_dag_uri: "vol-04:appendix-c:causal_dag_ceramic_v1"                     # vol-04 appendix-c §C.12 canonical URI (Ch14 §14.2.1 と同一形式)
  dag_of_record_sha256: "sha256:abc..."                                          # vol-04 の canonical field 名 (Ch4 §4.4)
  decision_variables:                                                            # BO の decision variable として operate する変数
    - "x_temperature"     # continuous, 200-800 degC
    - "x_pressure"        # continuous, 0.1-5.0 MPa
    - "x_hold_time"       # continuous, 30-180 min
    - "x_mass_frac_A"     # continuous, 0.1-0.6
    - "x_mass_frac_B"     # continuous, 0.1-0.5
    - "x_atmosphere"      # categorical, {Ar, N2, Air}
  context_variables_from_backdoor_adjustment:                                    # 測定・条件付けの対象、BO decision には含めない
    - "x_precursor_lot"                                                          # 前駆体ロット差 (measurable confounder)
    - "x_operator_shift"                                                         # オペレータ差 (measurable confounder)
  derivation_rationale: |
    vol-04 DAG の backdoor adjustment set 要素は confounder として測定・条件付けする pre-treatment 変数であり、
    BO の decision variable（介入対象）には含めない。search space からは除外し、observation の context
    として記録する。上記 decision_variables は Ch14 capstone §14.2.1 の canonical search space と一致。
  derivation_event_hash: "sha256:..."                                            # このイベントの hash (authorization_events_stream に append)
```

> [!NOTE]
> 上記は施設運用側の推奨 metadata（canonical schema ではない）。`context_variables_from_backdoor_adjustment` は例示であり、実プロジェクトの DAG に応じて変数集合は変わる。**本節の `context_variables_from_backdoor_adjustment` の例示（`x_precursor_lot` / `x_operator_shift`）は Ch14 §14.2.1 の canonical には列挙されていない**（施設運用 metadata 側で扱う context variable の例）。Ch14 §14.2.1 の canonical search space（`x_temperature` / `x_pressure` / `x_hold_time` / `x_mass_frac_A` / `x_mass_frac_B` / `x_atmosphere` の 6 変数、`x_atmosphere` は decision variable として categorical kernel で扱う）と本節の `decision_variables` は完全一致する。

### 16.5.2 DAG 改訂時の伝播

BO 進行中に vol-04 DAG が改訂された場合（例：vol-04 の新しい observational study で追加変数が発見）、以下を運用契約とする：

1. **vol-05 側の Skill は DAG を勝手に更新しない**（Ch14 §14.4.0 `dag_revision_policy`）
2. DAG 更新の伝播は **Research Lead が明示承認**（`dag_revision_events` を発行）
3. 既実施の observation を破棄せず、新 DAG での再解釈を `search_space_derivation.previous_derivation_event_hash` で chain
4. 新 DAG との整合性が取れない observation は `observation_status: quarantined` として保持（削除禁止、Ch12 §12.7.1 の状態遷移）

## 16.6 BO 特有の対外開示ポリシー

BO 結果を対外発表（論文・特許・製品仕様）する際の開示範囲：

| 開示項目 | 通常論文 | 施設内部レポート | 特許明細書 |
|---|---|---|---|
| `surrogate_model_family` / `kernel_spec` | ✅ 開示推奨 | ✅ | ✅ |
| `acquisition_spec.name` | ✅ 開示推奨 | ✅ | ✅ |
| **`search_space_bounds`** | ⚠️ 部分開示 or 正規化値 | ✅ | ✅ (実施可能性のため) |
| **`hard_constraints_ceramic_v0_2` の具体値** | ⚠️ 装置固有情報の含意注意 | ✅ | ⚠️ 装置固有 vs 発明本質の切り分け |
| **`experiment_launch_authorization` identity** | ❌ 個人 identity は開示禁止 | ✅ (role 単位で開示) | ❌ |
| **観測データの生値** | ⚠️ 匿名化・集計 | ✅ | ⚠️ 実施例として一部開示 |
| **`sequential_seed_provenance`** | ⚠️ 再現性のため開示推奨、ただし seed 値そのものは注意 | ✅ | ⚠️ 実施可能性のため開示 |
| **`bo_agentic_audit_checklist` 結果** | ✅ Section 1 のみ開示、Section 2（Agentic 特有失敗）は施設内 | ✅ 全体 | ❌ |
| **event_hash chain 全体** | ❌ (rebuild manifest URI のみ開示可、施設 registry 発行の `arim://facility/registries/bo-canonical-<ver>/run_id/<uuid>` 等) | ✅ | ❌ |

**原則**：
- **再現性優先の場合**：BO methodology（surrogate + acquisition + iteration schedule）は再現性のため開示推奨
- **装置固有情報保護**：`hard_constraints_ceramic_v0_2` の具体値、`sequential_seed_provenance` の seed 値は装置損傷リスクや競合模倣リスクとの兼ね合いで判断
- **個人 identity 保護**：`^human:<id>` の実 identity は施設内部運用のみ、対外開示時は role 名（Research Lead / BO Reviewer 等）に置換

## 16.7 Skill 更新と canonical schema のバージョニング

vol-05 の canonical schema は `v0.2` として発行された（Ch10 §10.5 event_hash の canonical 化、Ch10 §10.6 `hard_constraints_ceramic_v0_2`、Ch12 `batch_diversity_policy_v0_2`、Ch14 §14.4.0 `dag_revision_policy` + `experiment_reservation_events`）。今後の改訂に備えた運用契約：

### 16.7.1 breaking vs additive の判定

| 変更種別 | 例 | バージョン記法 | 移行手続き |
|---|---|---|---|
| **additive**（既存 provenance emitter を壊さない） | 新規 field 追加 / enum に新 value 追加 | v0.2 → v0.3 | 施設内周知のみ |
| **compat rename**（旧名 alias を残す） | 例：`acquisition_function` → `acquisition_spec.name`（**この rename は既に v0.1 → v0.2 移行で完了**。将来の compat rename の template として例示） | v0.2 → v0.3 | 施設 registry に alias 表を追加 |
| **breaking**（既存 provenance を壊す） | field 削除 / enum value 削除 / hash 計算方法変更 | v0.2 → **v1.0** | Facility BO Review Board 承認 + 3 か月 grace period |

### 16.7.2 canonical Schema の semver 運用

- **メジャー版アップ (v0.2 → v1.0)**：event_hash 計算方法の変更、`^human:` identity 形式の変更、既存 canonical enum value の削除
- **マイナー版アップ (v0.2 → v0.3)**：新規 field 追加、新規 enum value 追加、記法の緩和
- **パッチ版アップ (v0.2 → v0.2.1)**：typo 修正、docstring 更新、テーブル追加

### 16.7.3 施設 registry での運用

```yaml
# 施設運用ドキュメント（非 canonical, 例示）
facility_bo_canonical_registry:
  registry_uri: "arim://facility/registries/bo-canonical-v0.2"
  registry_sha256: "sha256:..."
  current_version: "v0.2"
  supported_versions: ["v0.2"]                          # v0.1 は grace period 終了で dropped
  breaking_change_grace_period_months: 3
  canonical_enums:
    surrogate_model_family:
      version: "v0.2"
      values: [single_task_gp, fixed_noise_gp, heteroskedastic_gp, multi_task_gp,
               hierarchical_gp, random_forest, bayesian_neural_network, deep_kernel_learning]
    acquisition_spec_name_single_objective:
      version: "v0.2"
      values: [qLogEI, qLogNEI, qUCB, qPI, qKG, qMES, qTS, qPES]
    acquisition_spec_name_multi_objective:
      version: "v0.2"
      values: [qLogEHVI, qLogNEHVI]
    acquisition_spec_name_scalarization:
      version: "v0.2"
      values: [qParEGO, qNParEGO]
    active_learning_acquisition_spec_name:
      version: "v0.2"
      values: [predictive_variance, predictive_entropy, query_by_committee,
               expected_model_change, mutual_information_bald,
               integrated_variance_reduction]                                 # Ch13 §13.2 (canonical enum table L63-68) 6 values
  next_review_date: "2026-10-01"                                              # ISO 8601 date (次期 Facility BO Review Board)
  next_review_period: "2026-Q4"                                               # 参考: 四半期 label
```

## 16.8 vol-06 への道しるべ — 生成モデル・逆設計・組織横断ガバナンス

vol-05 が扱わなかったスコープを vol-06 に引き渡す。**vol-06 はまだ執筆計画段階** であり、以下は道しるべである：

### 16.8.1 vol-06 の想定スコープ

| 領域 | vol-05 での扱い | vol-06 で扱う |
|---|---|---|
| **生成モデル** | 扱わない | 材料合成条件の逆設計、Diffusion / VAE / Normalizing Flow を Skill 化 |
| **逆設計** | 扱わない | 望ましい物性から合成条件を逆算する Skill、vol-04 因果 DAG + vol-05 BO surrogate を制約に |
| **組合せ空間 BO (BOCS)** | Ch12 で判断表のみ言及 | 完全実装、離散変数・グラフ構造への拡張 |
| **強化学習ベースの実験計画** | 扱わない | RL による長期報酬最大化、BO との比較 |
| **LLM ファインチューニング** | 扱わない | 材料ドメイン特化 LLM、vol-03 深層 Skill との連結 |
| **組織横断ガバナンス** | 施設内運用（本章） | ARIM 全体 / 異施設間 / 国際協業でのガバナンス、data sovereignty |

### 16.8.2 vol-05 canonical を継承する予定の要素

vol-06 は vol-05 の canonical schema を破壊しない：

- `surrogate_model_family` enum は vol-06 で **拡張**（generative_surrogate / diffusion_conditional / vae_latent_bo が追加候補）
- `acquisition_spec.name` enum は vol-06 で **拡張**（generative_acquisition / inverse_design_acquisition が追加候補）
- `event_hash` chain は vol-06 の逆設計提案にも継承（`inverse_design_events`）
- `experiment_launch_authorization` の parent chain は vol-06 で **より深く**（vol-06 生成 → vol-05 BO 承認 → **vol-04 L3 (`intervention_execution_authorization`)** → vol-01 baseline HITL の 4 段 chain）

### 16.8.3 vol-06 に持ち越す論点

- **生成モデルの hallucination 検知**：vol-05 の `hallucinated_recommendation_detection`（Ch7）を生成モデルにどう拡張するか
- **逆設計の因果的妥当性**：生成された合成条件が vol-04 因果 DAG の adjustment set 上で正当か
- **組合せ空間の探索**：連続 BO と離散最適化のハイブリッド
- **異施設間 provenance**：hash chain の cross-facility 互換性

## 16.9 ARIM 施設としての BO 運用 — 3 つの実装原則

vol-04 §15.7 の因果推論運用 3 原則に対応する、vol-05 の **BO 運用 3 原則**：

### 原則 1：Surrogate Sharing と 3 者承認の統合

複数研究者が類似 search space で BO を回す場合、surrogate 共有は施設リソース効率化の最大のレバー。しかし §16.4.1 の (f) 元プロジェクト所有者承認が **技術的に自動化できない**（データ所有権と競合防止の判断を要する）ため、施設として **3 者承認プロセスを常設化** する。

原則：
- **BO Reviewer による技術検証（§16.4.1 (a)-(e)、(c1)/(c2) を含む 6 sub-conditions）は自動化可能**：これを Skill 化して機械的 pass/fail を返す
- **元データ所有者の `sharing_authorization_ledger` entry 発行は自動化しない**：施設 registry に紙 or e-signature として保管、`sharing_authorization_id` にリンク
- **Facility BO Review Board の統合承認**：四半期レビューで承認候補を bulk 処理し、`shauth_YYYYMMDD_HHMMSS_seq<n>` を発行

### 原則 2：Facility BO Review Board の常設化

Facility Causal Review Board（vol-04 §15.7 原則 2）に対応する **BO 版 review board** を常設する。役割：

```yaml
# 施設運用ドキュメント（非 canonical, 例示）
facility_bo_review_board:
  members:
    chair: "^human:facility_bo_chair"
    members:
      - "^human:senior_bo_researcher_1"
      - "^human:senior_bo_researcher_2"
      - "^human:facility_ops_manager"
      - "^human:safety_officer"
      - "^human:external_advisor"                      # 半期に 1 回参加
  meeting_frequency: "quarterly"
  standing_agenda:
    - project_bo_reviews                                # §16.3.2 各プロジェクト
    - surrogate_sharing_authorizations                  # §16.4.3 の統合承認
    - device_reservation_conflicts                      # §16.3.3 の tier 昇格申請
    - canonical_schema_versioning                       # §16.7 semver 判定
    - facility_bo_audit_summary                         # §16.9 原則 3
  # バンドル内包（Ch15 canonical には含まれない、施設 registry 管理）:
  #   - facility_bo_review_board_minutes.md (Board 議事録)
  #   - device_reservation_policy.yaml
  #   - sharing_authorization_ledger.jsonl
```

### 原則 3：Skill 契約の Facility-wide Registry

vol-04 §15.7 原則 3 の Skill 契約 registry を vol-05 BO Skill に拡張。**canonical schema は Ch5 §5.2 が SoT** であるため、registry は **参照メタデータ + 施設運用契約** のみを保持：

```yaml
# 施設運用ドキュメント（非 canonical, 例示）
facility_bo_skill_registry:
  registry_uri: "arim://facility/registries/bo-skill-v0.2"
  registered_skills:
    - skill_id: "arim.bo.single_objective_gp_ei.v0.2"
      canonical_reference: "vol-05/Ch5-§5.2, appendix-a (未執筆)"
      surrogate_model_family: "single_task_gp"
      acquisition_spec_name: "qLogEI"
      pilot_project_id: "proj_2026_arim_ceramic_hard_v2"
      pilot_completion_event_hash: "sha256:..."
      facility_approval_id: "shauth_20260401_140030_seq7"
      approved_devices: ["device_XRD_01", "device_XRD_02"]
      approved_search_space_regions: ["ceramic_hard_v0_2_region_A"]
    - skill_id: "arim.bo.multi_objective_qehvi.v0.2"
      # ...同形式
    - skill_id: "arim.bo.safe_bo_ceramic_v0_2.v0.2"
      canonical_reference: "vol-05/Ch10 + Ch10 §10.6 hard_constraints_ceramic_v0_2"
      hard_constraints_id: "hard_constraints_ceramic_v0_2"
      safety_officer_approval_id: "^human:safety_officer_2026Q1"
      # ...
```

## 16.10 章末：vol-05 が読者に手渡すもの

vol-05 を通読した読者は、以下を **手を動かして構築** できる：

1. **単目的 BO Skill**（GP surrogate + acquisition + Human 承認ループ、Ch5-8）：seed 上書き禁止、外挿検知の合成基準、event_hash chain による全 iteration の再現性
2. **多目的 / 制約付き / safe BO Skill**（Ch9-10）：Pareto front 管理、`hard_constraints_ceramic_v0_2`、safe BO の probability threshold
3. **階層 BO Skill**（Ch11）：Multi-task GP、装置差 hierarchical prior、複数装置の並行探索
4. **Batch BO Skill**（Ch12）：`batch_size` / `pending_experiments` / `batch_diversity_policy_v0_2`、q-EHVI batch 拡張
5. **Active Learning Skill**（Ch13）：BO と AL の判断分岐、`active_learning_acquisition_spec.name`
6. **Advanced Capstone**（Ch14）：因果 DAG × 階層 GP × Human 承認 batch BO の統合、`dag_revision_policy` + `experiment_reservation_events`
7. **BO × Agentic 監査 Skill**（Ch15）：`bo_agentic_audit_checklist` の Section 1 + Section 2、canonical fatal 名 registry
8. **施設運用ガバナンス**（本章）：3 段階権限の RACI、Surrogate Sharing 3 者承認、Facility BO Review Board、canonical schema semver

vol-01〜04 が「予測 → 因果 → DoE 一括生成」のラダーを Skill 化したのに対し、vol-05 は「**逐次的な意思決定の連鎖を hash-verifiable にする**」ラダーを Skill 化した。この連鎖は vol-06（生成モデル・逆設計）でさらに深くなる（vol-06 生成 → vol-05 BO → **vol-04 L3 (`intervention_execution_authorization`)** → vol-01 baseline HITL の 4 段 chain）。

**BO Skill は "最適化を任せる" ものではない**。**"次にどこを打つかの判断過程を hash-verifiable に記録する"** 契約である。エージェントは surrogate を fit し、acquisition を計算し、候補を提示するが、**実験を実行するのは常に人間** であり、その判断過程は event_hash chain として **後日誰でも検証できる** ように残る。これが vol-05 が読者に手渡す原理である。

## 章末チェックリスト（BO 運用実装のセルフチェック）

- [ ] 自プロジェクトの `experiment_launch_authorization` は vol-04 L3 に `parent_authorization_id` で chain されているか
- [ ] `event_hash` は Ch10 §10.5 `event_canonical_serialization`（RFC 8785 JCS + `sha256(json.dumps(..., sort_keys=True, separators=(',', ':')))`）に準拠しているか
- [ ] `sequential_seed_provenance` の seed 上書き禁止契約は施設 registry で強制されているか
- [ ] `hallucinated_recommendation_detection` は Ch7 の合成基準（予測分散 × Mahalanobis × length scale）を AND で運用しているか
- [ ] `hard_constraints_ceramic_v0_2` は Safety Officer が accountable として承認したか
- [ ] Cross-project surrogate 共有時、§16.4.1 の (a)-(g) 8 条件（(c) は (c1)/(c2) に分割）をすべて検証したか
- [ ] `experiment_reservation_events` の tier は Skill が prompt injection で書き換えていないか
- [ ] Facility BO Review Board の四半期レビュー入力と、プロジェクト単位 BO Review（iteration 5 回ごと）での `bo_agentic_audit_checklist` Section 2 集約が §16.3.2 の役割分担どおりに運用されているか
- [ ] canonical schema バージョンは施設 registry の `current_version` と一致しているか
- [ ] vol-04 DAG を search space 絞り込みに使う場合、`search_space_derivation` を provenance に記録したか
- [ ] vol-06 に持ち越す論点（生成モデル、BOCS、cross-facility provenance）を認識しているか

## 参考資料

### 本書内参照

- **Ch5 §5.2**：21-field BO × 逐次 × Agentic provenance schema の SoT
- **Ch5 §5.3**：`experiment_launch_authorization` と vol-04 L3 の parent chain
- **Ch7**：`hallucinated_recommendation_detection` の operational 合成基準
- **Ch10 §10.5**（anchor `event_canonical_serialization`）：RFC 8785 JCS canonical serialization
- **Ch10 §10.6**：`hard_constraints_ceramic_v0_2`
- **Ch12 §12.7.1**：`experiment_status_transition_events`
- **Ch14 §14.4.0**：`dag_revision_policy`
- **Ch14 §14.4.0-b**：`experiment_reservation_events`
- **Ch15 §15.3**：`bo_agentic_audit_checklist`（監査 Skill）
- **Ch15 §15.2.3**：Human 承認なしに次の候補を実行（本章 §16.4.3 で cross-project 共有 bypass の変種を明示）
- **Ch15 §15.2.4**：Iteration seed を上書き（本章 §16.4.2 で cross-project 変種を明示）
- **Ch15 §15.2.5**：Pending experiments を勝手にキャンセル（本章 §16.3.3 で他プロジェクト予約取消の拡張変種を明示）

### 巻をまたぐ参照

- **vol-01 Ch6**：Human-in-the-loop の baseline SoT（本章 §16.10 の 4 段 chain 末端「vol-01 baseline HITL」の canonical 参照）
- **vol-04 Ch4 §4.6.2**：`^human:<id>` identity canonical + L3 `intervention_execution_authorization`（vol-05 L2 の parent）
- **vol-04 appendix-c §C.12**：`causal_dag_ceramic_v1`（vol-05 Ch14 capstone で使用、canonical URI `vol-04:appendix-c:causal_dag_ceramic_v1`）
- **vol-04 Ch15**：因果推論運用 3 原則（本章 §16.9 の対応原型）
- **vol-03 appendix-a §A.9**：canonical Gate & Policy Field Reference（本章の gate 契約設計の参考）

### 外部文献（組織運用・ガバナンス）

- Frazier, P. I. "A Tutorial on Bayesian Optimization." arXiv:1807.02811（BO 理論）
- Shahriari, B. et al. "Taking the Human Out of the Loop: A Review of Bayesian Optimization." Proceedings of the IEEE 104(1), 2016（本書は逆に **Human を戻す** 立場）
- Garnett, R. *Bayesian Optimization*. Cambridge University Press, 2023（BO の教科書）
- NIST AI Risk Management Framework 1.0（2023）— Manage 段階の運用契約の参考
- ISO/IEC 42001:2023（AI Management System）— canonical schema バージョニングの参考

### 次巻への案内

**vol-06「AI エージェント時代の生成モデル・逆設計入門 — ARIM データで動く Agentic Skill」**（企画中）：
- 生成モデル（Diffusion / VAE / Normalizing Flow）による材料合成条件の逆設計 Skill
- vol-04 因果 DAG + vol-05 BO surrogate を制約に組み込んだ逆設計
- 組合せ空間の探索（BOCS の完全実装）
- LLM ファインチューニングと材料ドメイン特化
- 異施設間 / 国際協業での provenance と data sovereignty

vol-05 の canonical schema（`surrogate_model_family` / `acquisition_spec.name` / `event_hash` / `experiment_launch_authorization` の parent chain）は vol-06 で拡張されるが破壊されない。vol-05 で作った BO Skill と provenance はそのまま vol-06 の 4 段 chain の 1 段として組み込まれる。
