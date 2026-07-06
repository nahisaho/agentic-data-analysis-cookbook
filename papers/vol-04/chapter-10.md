# 第10章 古典的実験計画の Skill 化 — Full factorial / Fractional factorial / Randomization / Blocking

> **本章の到達目標**
> - **観測データからの因果推定（第I-II部）と、能動的介入による効果推定（DoE）**の違いを、identification 仮定の観点で対比できる
> - **完全要因計画（full factorial）** と **一部実施要因計画（fractional factorial）** の実行数・交絡パターン・**resolution** を Skill 契約項目として書き下せる
> - **完全ランダム化（CRD）** と **ブロックランダム化（RCBD / split-plot）** の使い分けを、`nuisance_factor_declaration` と `randomization_scope` として契約に落とし込める
> - **randomization seed の provenance**——生成時刻・アルゴリズム・スキーマバージョン——を Ch4 §4.9 の `experimental_design_provenance` に沿って記録できる
> - **pyDOE2 / OApackage** による設計行列生成と、契約ハッシュ（`design_matrix_sha256`）による immutability を実装できる
> - **`design_proposal_skill` / `design_approval_skill`** を Ch5-7 の proposal/approval 分離パターンで設計し、**エージェントによる情報量 vs 実行コストのトレードオフ提案**をどこまで自律に委ねてよいかを判断できる
>
> **本章で扱わないこと**
> - 応答曲面（Central Composite Design / Box-Behnken / Kriging surrogate）と品質工学（タグチ SN 比）→ 第11章
> - Bayesian D-/A-optimality と情報利得ベース設計 → 第12章
> - 逐次 BO（acquisition function / Thompson sampling / EI / UCB）→ vol-05 に完全委譲
> - randomization の破綻・blocking 失敗・応答曲面外挿の失敗パターン → 第14章 §14.2

第I部・第II部では **観測データ**（`treatment` が既に割当済み、あるいは自然発生の準実験）から因果効果を推定してきました。ここから第III部では **能動的介入**——研究者が処置を割り当てる——による効果推定に移ります。DoE は「実験による因果推論」であり、**randomization** が SUTVA・consistency・exchangeability の 3 仮定を **設計時点で担保**する強力な武器になります（`checked` に格上げできる仮定が増えます）。

しかし ARIM のような共用施設では「装置予約は 3 日限り」「特定条件では危険物取扱い許可が必要」「blocking factor（装置・オペレータ・ロット）は自然に発生する」といった **実運用制約**があり、教科書的な CRD が使えない場面が多発します。本章では **エージェントに設計を提案させ、Human が承認する** 2 段階を通じて、実施可能性と統計的効率を両立させる Skill 契約を構築します。

---

## 10.1 観測 vs 介入 — identification 仮定の格上げ

### 10.1.1 randomization が「見えない仮定」を「見える仮定」に変える

第0章 §0.6 で導入した 4 識別仮定を、観測データと介入データで並べます：

| 仮定 | 観測データ（第I-II部） | ランダム化介入（本章以降） |
|---|---|---|
| **exchangeability** | assessed（DAG から backdoor 判定 + E-value / Rosenbaum で頑健性） | **checked**（randomization seed + assignment log で監査可能） |
| **positivity** | checked（`positivity_by_stratum`、Ch6） | **checked**（設計行列内で全 treatment cell に $n \geq 1$） |
| **consistency** | assessed（介入定義の明確さ） | **checked**（介入プロトコル + 実行ログ） |
| **SUTVA** | assessed（no spillover の議論） | assessed（依然として理論仮定、ただし blocking + spatial separation で工学的に担保） |

**変化のポイント**：
- **3 つの仮定が `checked` に昇格**——randomization が emprical に検証可能な形で仮定を担保
- **SUTVA だけは依然 assessed**——化学実験の相互汚染・装置間の熱移動・オペレータの学習効果は randomization では消えない

> [!IMPORTANT]
> **「ランダム化したから causal 推論は自動的に正当化される」は誤り**。SUTVA 破綻・非遵守（noncompliance）・脱落（attrition）が起きれば ITT vs ATT の区別が必要になり、Ch7 の IV / SC の枠組みに戻る場面があります。**randomization は「識別を強化する」であって「識別を保証する」ではない**——Skill 契約は依然として refutation gate（Ch9）を required にします。

### 10.1.2 DoE Skill の位置づけ

vol-04 の因果 × DoE Skill は 2 系統に分岐します：

- **観測 Skill**（Ch5-9）：`identification_strategy: backdoor | frontdoor | iv | did | rdd | synthetic_control`
- **DoE Skill**（本章 - Ch12）：`identification_strategy: randomized_experiment` + `experimental_design_provenance`

DoE Skill は Ch4 §4.3 Table 4.2 の **6 点セット中の 5-6（`doe_efficiency` / `randomization_integrity`）を追加で満たす**必要があります（第I-II部 Skill では `not_applicable`）。

---

## 10.2 Full factorial design — 全条件を試すことの意味

### 10.2.1 完全要因計画の骨格

$k$ 個の因子（factor）それぞれが $L_j$ 水準（level）を持つとき、**完全要因計画**は全ての水準組合せを実行：

$$
N_{\text{full}} = \prod_{j=1}^{k} L_j
$$

- **$2^k$ 設計**：全因子 2 水準 → $N = 2^k$ 回。$k=3$ で 8 回、$k=5$ で 32 回、$k=7$ で **128 回**（現実的な上限）
- **$3^k$ 設計**：全因子 3 水準（中心点含む）→ $k=3$ で 27 回、$k=4$ で 81 回
- **mixed-level**：$2 \times 3 \times 4 = 24$ 回など

**利点**：**全ての交互作用が独立に推定できる**——main effect（主効果）だけでなく、$k$-way interaction まで直交して推定。ANOVA 分解は各項の平方和として綺麗に分離します。

**欠点**：**指数爆発**——ARIM で「温度 3 水準 × 圧力 3 水準 × 前処理時間 3 水準 × 装置 4 台 × オペレータ 5 名 = 540 回」は現実的に不可能。**因子数と水準数の管理**が Skill の第一責務です。

### 10.2.2 Skill 契約項目（pyDOE2 / OApackage）

```yaml
skill_type: full_factorial_design_skill
authorization_level: agent_autonomous              # 設計提案までは自律、実行は intervention_execution_authorization
identification_strategy: randomized_experiment

experimental_design_provenance:
  design_type: full_factorial
  factors:                                          # 因子定義（順序と符号は immutable）
    - name: temperature
      unit: celsius
      levels: [400, 500, 600]
      level_encoding: [-1, 0, 1]                    # coded levels
      role: primary                                 # primary | nuisance | blocking
    - name: pressure
      unit: bar
      levels: [1.0, 5.0, 10.0]
      level_encoding: [-1, 0, 1]
      role: primary
    - name: pretreatment_time
      unit: minutes
      levels: [15, 30]
      level_encoding: [-1, 1]
      role: primary
  factors_frozen_at: <timestamp>                    # 因子・水準の post-hoc 追加は fatal
  n_runs_total: 18                                  # 3 × 3 × 2
  replication_policy:
    n_replicates: 2                                 # 各 cell の反復数
    replicate_scope: within_run | across_run        # 独立反復の粒度
    total_experimental_units: 36
  design_matrix_uri: "artifact://designs/<name>.parquet"
  design_matrix_sha256: <string>                    # 実行前に凍結、以後 immutable
  generation:
    library: pyDOE2
    library_version: <string>
    algorithm: fullfact
    random_seed: 42                                 # 実行順序シャッフル用（設計自体は決定的だが実行順は randomize）
    random_seed_generation_method: os_urandom_at_design_time  # seed の来歴（§10.5）
  randomization_scope:
    run_order_randomization: complete               # complete | restricted_by_block
    randomization_seed: 42                          # 上記と同一 seed または別 seed（記録要）
    assignment_log_uri: "artifact://designs/<name>_assignment.jsonl"

identification_validity:
  strategy: randomized_experiment
  exchangeability:
    status: checked
    evidence: randomization_seed_and_assignment_log
    evidence_uri: <string>
  positivity_by_stratum:                            # Ch4 §4.4.1 canonical shape
    status: checked
    strata:
      - key: full_factorial_cell
        min_treated: 1                              # 各 cell に最低 1 unit
        min_control: not_applicable                 # 各 cell が independent treatment
        violation_policy: fail_close
        stratum_report_uri: <string>
  consistency:
    status: checked
    intervention_protocol_uri: <string>
    execution_log_uri: <string>
  sutva_declared:
    status: assessed                                 # randomize しても spillover は消えない
    evidence_uri: <string>

prohibited_actions:
  - modify_factors_or_levels_after_design_freeze     # fatal
  - reduce_n_runs_after_partial_execution            # fatal
  - reassign_units_to_different_cells                # fatal
  - use_execution_order_correlated_with_outcome      # fatal（temporal confounding）
  - ignore_missing_cells_in_analysis                 # fatal（silent listwise deletion 禁止）
```

### 10.2.3 交互作用の推定可能性

$2^3$ full factorial（$N=8$）では以下が全て推定可能：

| 効果 | 自由度 | 推定 |
|---|---|---|
| 平均（intercept） | 1 | 全 8 runs の平均 |
| 主効果 A / B / C | 3 | 各因子の水準差 |
| 2-way interaction AB / AC / BC | 3 | 因子ペアの積 |
| 3-way interaction ABC | 1 | 3 因子の積 |
| **合計** | **8** | full factorial の自由度と一致 |

**契約含意**：Skill が「全交互作用を推定した」と主張する場合、`interaction_estimation_report_uri` に **各項の平方和 + 標準誤差 + p 値 + 効果量** を出力必須。**エージェントが「3-way は無視した」「有意でないので落とした」と post-hoc 判断することは fatal**（変数選択の隠れた自由度）。

### 10.2.4 ARIM ケース：組成 × 温度 × 前処理

例：ペロブスカイト太陽電池の open-circuit voltage $V_{OC}$ を outcome、A-site 組成（Cs / MA / FA の 3 水準）× 焼成温度（100 / 150 °C の 2 水準）× 前処理時間（0 / 30 分の 2 水準）で $3 \times 2 \times 2 = 12$ 条件を 2 反復 → **24 runs**。装置枠 3 日、1 日 8 runs で完了想定。

エージェントは以下を提案：
1. `design_matrix_uri` の生成（`pyDOE2.fullfact([3, 2, 2])`）
2. `randomization_seed` で run 順を shuffle
3. `assignment_log_uri` に seed + 順序 + タイムスタンプを記録
4. **主効果 3 + 2-way 3 + 3-way 1 = 7 効果**の推定可能性を示す

Human 承認（`intervention_execution_authorization`）後、実験室で実行。

---

## 10.3 Fractional factorial design — 一部実施と resolution

### 10.3.1 なぜ fractional か

$2^7 = 128$ runs は多くの ARIM 実験で非現実的。**$2^{k-p}$ fractional factorial** は完全計画の $1/2^p$ を実行：

- $2^{7-3} = 16$ runs で 7 因子を screening（$p=3$）
- $2^{5-1} = 16$ runs で 5 因子（$p=1$）

**代償は交絡（aliasing）**——高次交互作用と主効果が同じ列で表現され、区別できなくなる：

$$
\text{main effect A} = \text{2-way BC} \quad \text{(if the design confounds them)}
$$

### 10.3.2 Resolution（R）— 交絡の深刻さ

| Resolution | 意味 | 実用性 |
|---|---|---|
| **III** | 主効果 vs 2-way interaction が交絡 | **危険**——2-way が主効果に化ける、screening 初期のみ |
| **IV** | 主効果 vs 3-way が交絡、2-way vs 2-way が交絡 | 標準的な screening、主効果は clean だが 2-way は解釈注意 |
| **V** | 主効果 vs 4-way、2-way vs 3-way が交絡 | **推奨**——主効果 + 2-way が clean、3-way 以上は仮定「無視できる」 |

**契約項目**：

```yaml
experimental_design_provenance:
  design_type: fractional_factorial
  fraction: "1/8"                                    # 2^(k-p) の 1/2^p
  resolution: V                                       # III | IV | V | VI+
  generator_string: "D=ABC; E=BCD; F=ACD; G=ABD"    # 定義生成子
  defining_relation_uri: <string>                    # I = ABCD = ...（完全リスト）
  alias_structure_report_uri: <string>               # 各効果と alias の対応表
  alias_structure_frozen_at: <timestamp>
  interactions_assumed_negligible:                   # 「無視する」仮定を明示
    - 3-way
    - 4-way
    - higher
  domain_justification_for_negligibility_uri: <string>  # ドメイン知識で正当化
  factors: [...]                                     # 上記と同構造
  n_runs_total: 16
  design_matrix_uri: <string>
  design_matrix_sha256: <string>
  generation:
    library: pyDOE2                                  # or OApackage
    algorithm: fracfact
    generator: "D=ABC; E=BCD; F=ACD; G=ABD"
    library_version: <string>
prohibited_actions:
  - claim_effect_free_of_aliasing_without_higher_order_negligibility_assumption  # fatal
  - post_hoc_resolve_alias_by_running_extra_cells_without_estimator_contract_change_gate  # fatal
  - report_2way_estimate_from_resolution_iii_design                              # fatal
  - use_defining_relation_that_breaks_declared_resolution                        # fatal
```

### 10.3.3 alias structure の可視化と Skill 出力

resolution IV の $2^{7-3}$ 設計では、alias table を Skill 出力に含めます：

```
A + BDE + CDF + BCG + ABEF + ACEG + ADFG + BCDEG + BDFG + CEFG + ...
B + ADE + CDG + ACF + ABEF + BCEG + BDFG + ACDEG + ADFG + CEFG + ...
...
AB + CE + FG + DFH + ... (only if additional factors present)
```

**エージェントが「A は有意」と報告するとき、契約は "A + higher-order aliases assumed negligible" と明示**——Skill 出力に **省略記法禁止**、`effect_reported_with_alias_disclosure: true` を必須化。

### 10.3.4 Plackett-Burman（PB）設計 — 主効果 screening 特化

**PB 設計**は $n = 12, 20, 24, ...$（4 の倍数）runs で最大 $n-1$ 因子を screening。**主効果のみ検出**、2-way 以上は全て交絡（resolution III 相当）。

```yaml
design_type: plackett_burman
n_runs_total: 12                                     # PB12 で最大 11 因子
factors: [...]
interactions_assumed_negligible: [2-way, 3-way, higher]
domain_justification_for_negligibility_uri: <string> # PB は 2-way 無視が前提
prohibited_actions:
  - report_interaction_effect_from_pb_design         # fatal（PB は主効果 only）
```

**用途**：ARIM の**上流 screening**——「7 因子のうちどれが $Y$ に効くか」だけを知りたい、その後の詳細実験は $2^{k-p}$ / 応答曲面（Ch11）に委ねる場面。

---

## 10.4 直交表とタグチ流の位置づけ（Ch11 予告）

**直交表（orthogonal array, OA）** は fractional factorial の一般化。OA(N, s^k, t) は $N$ runs で $k$ 因子（各 $s$ 水準）を **strength $t$** で直交化。

- **strength 2**：全ての 2 因子ペアが balanced（resolution III / IV 相当）
- **strength 3**：全ての 3 因子 tripleが balanced（resolution V 相当）

**田口メソッド（タグチ流）** は OA を SN 比（signal-to-noise ratio）分析と組合わせた品質工学の方法論。**本章では OA の生成と直交性の検証**までを扱い、SN 比・inner/outer array・タグチ流のロバスト設計は **Ch11 §11.4** に委ねます（応答曲面と併せて扱うのが自然）。

```yaml
design_type: orthogonal_array
oa_notation: "L12(2^11)"                             # 12 runs, 11 factors, 2 levels each
strength: 2                                          # 全 pair balanced
generation:
  library: OApackage                                 # or pyDOE2 (limited)
  library_version: <string>
  algorithm: reduced_form_lookup                     # 標準 OA テーブル参照
orthogonality_check_uri: <string>                    # 直交性の empirical 検証
```

---

## 10.5 Randomization seed の provenance — 監査可能性の要

**randomization は「seed + アルゴリズム + アサインメント log」の 3 点セット**で監査可能でなければ、`exchangeability: checked` は成立しません。

### 10.5.1 seed の来歴（origin）

```yaml
randomization_seed_provenance:
  seed_value: 42
  seed_generation_method:                            # 選択肢
    - os_urandom_at_design_time                      # /dev/urandom 系、デフォルト推奨
    - hardware_rng_at_design_time                    # HRNG（TPM 等）
    - user_specified_deterministic                   # 論文再現・教育用のみ
    - blockchain_derived_at_design_time              # 事後改変不能性が求められる高規制環境
  seed_generation_timestamp: <ISO8601>
  seed_generator_library: python_secrets             # or numpy_default_rng | os_urandom
  seed_generator_library_version: <string>
  seed_registration_uri: <string>                    # seed の事前登録先（immutable ledger）
prohibited_actions:
  - regenerate_seed_after_design_freeze              # fatal
  - use_user_specified_seed_in_production_experiment_without_justification  # warning
  - reuse_seed_across_different_experiments          # fatal（unless intentional replication）
```

### 10.5.2 assignment log の必須項目

```jsonl
{"run_id": "R001", "unit_id": "sample_042", "assigned_cell": {"temperature": 500, "pressure": 5.0}, "assignment_timestamp": "2026-07-06T14:22:11Z", "randomization_seed": 42, "operator": "op_007", "block_id": "day_1_morning"}
{"run_id": "R002", ...}
```

- **run_id**：実行順序（randomization 後の実際の順序）
- **unit_id**：試料 ID（ARIM 資材台帳と連携）
- **assigned_cell**：処置の完全記述
- **block_id**：blocking factor の値
- **operator**：オペレータ ID（後述 blocking で使用）

### 10.5.3 seed 上書きの検知（Ch4 Table 4.4 item 10 の operational 定義）

Ch4 §4.8 Table 4.4 item 10（`randomization_seed_override: fatal`）は本節で操作的に定義：

```yaml
detection:
  compare:
    - source: experimental_design_provenance.generation.random_seed
    - source: assignment_log_uri[*].randomization_seed
  policy: all_records_must_match_provenance_declared_seed
  mismatch_action: fatal_stop_and_notify_facility_causal_review_board
```

---

## 10.6 Blocking — 説明できない変動を「見える化」する

### 10.6.1 なぜ blocking か

ARIM 実験には **structural nuisance variation** が必ずあります：

- **装置差**：XRD-A と XRD-B の baseline がずれる
- **オペレータ差**：習熟度・技巧
- **時期差**：室温・湿度・試薬ロット
- **試料バッチ差**：合成日、原料メーカー

これらを **randomize で "平均化" するのではなく、blocking factor として明示的にモデルに入れる**——分散が「説明可能な項」に吸収され、主効果の検出力が上がります。

### 10.6.2 CRD vs RCBD vs Split-plot

| 設計 | ランダム化スコープ | 用途 |
|---|---|---|
| **CRD（Completely Randomized Design）** | 全 run を completely random に順序化 | blocking factor 無し / 影響無視できる場合 |
| **RCBD（Randomized Complete Block Design）** | ブロック内でランダム化、ブロック間は blocking factor で分離 | 装置差・オペレータ差・日次差を制御 |
| **Split-plot** | ブロック内で "hard-to-change" 因子と "easy-to-change" 因子で 2 段階ランダム化 | 温度（変更に時間かかる）と圧力（即変更可）を混ぜる場合 |
| **Latin Square** | 2 つの nuisance を同時制御（行 × 列 = 装置 × 日） | 装置 × 時期の完全交絡回避 |

### 10.6.3 Skill 契約項目

```yaml
experimental_design_provenance:
  design_type: randomized_complete_block            # or split_plot | latin_square | crd
  randomization_scheme:
    scope: within_block                             # complete | within_block | hierarchical_two_stage
    randomization_seed: 42
    randomization_algorithm: fisher_yates_within_block

  blocking_factors:                                 # 明示宣言（fatal に発火する検知フィールド）
    - name: instrument_id
      type: nuisance                                # nuisance | strategic
      values: [XRD-A, XRD-B, XRD-C]
      assignment_policy: balanced_within_block      # balanced | proportional | complete_confounding
      rationale: baseline_offset_between_instruments
    - name: operator_id
      type: nuisance
      values: [op_007, op_012, op_019]
      assignment_policy: latin_square_with_instrument   # instrument × operator の完全交絡回避
      rationale: operator_skill_variation

  block_size: 6                                     # 各 block 内の run 数
  n_blocks: 4                                       # ブロック数
  n_runs_total: 24

  # split-plot 特有
  split_plot_structure:
    whole_plot_factors: [temperature]               # 変更困難因子
    subplot_factors: [pressure, pretreatment_time]  # 変更容易因子
    whole_plot_randomization_seed: 42
    subplot_randomization_seed: 43                  # 別 seed 推奨

  # analysis stage への契約
  mixed_model_specification_uri: <string>           # 分析時の random effect 指定
                                                      # RCBD: block を random effect、
                                                      # split-plot: whole plot も random effect
  degrees_of_freedom_report_uri: <string>           # 各 error stratum の df を事前算出

prohibited_actions:
  - ignore_blocking_factor_in_analysis              # fatal（block を無視すると SE 過小）
  - treat_split_plot_as_crd_in_analysis             # fatal（whole plot error を過小評価）
  - assign_blocking_factor_post_hoc                 # fatal（実行後に "実は装置差あった" は禁止）
  - unbalance_blocks_without_documented_reason      # warning → fatal if effect estimation
  - use_block_as_fixed_effect_when_generalization_intended  # warning
```

### 10.6.4 実験室でのブロッキング（ARIM 実例）

**シナリオ**：焼成温度 3 水準 × 前処理時間 2 水準 = 6 条件を、3 装置 × 2 オペレータで 3 日間実行。

- **完全交絡回避**：単純に「1 日目は装置 A、2 日目 B、3 日目 C」だと **日 × 装置が完全交絡**——分離不能
- **Latin square**：
  - Day 1: A(op1), B(op2), C(op1)
  - Day 2: B(op1), C(op1), A(op2)
  - Day 3: C(op2), A(op1), B(op2)
  → 装置 × 日 × オペレータの 3-way 交絡を回避しつつ、各条件を装置ごとに 2 反復

エージェントは：
1. Latin square + within-block factorial の hybrid 設計を提案
2. `mixed_model_specification_uri` で「装置・オペレータ・日 = random effect、温度・前処理時間 = fixed effect」を宣言
3. Human 承認後実行、事後分析は事前登録の mixed model で

---

## 10.7 情報量 vs 実行コストのトレードオフ — エージェントの提案範囲

### 10.7.1 効率指標

DoE の効率は **information gain per run** で測ります：

| 指標 | 定義 | 意味 |
|---|---|---|
| **D-efficiency** | $\|X^T X\| / n^p$ の関数、値域 [0, 1] | パラメータ共分散の行列式の逆——**小さい共分散 = 高効率** |
| **A-efficiency** | trace($(X^T X)^{-1}$) の逆数 | 平均予測分散の逆 |
| **G-efficiency** | $\max_x \text{Var}(\hat{y}(x))$ の逆数 | worst-case 予測分散 |
| **V-efficiency** | 予測分散の空間平均の逆数 | 全域平均予測分散 |

**契約項目**：

```yaml
success_criteria:
  doe_efficiency:
    metric: d_efficiency                             # or a_efficiency | g_efficiency | v_efficiency
    value: 0.85
    threshold: 0.80                                  # required 下限
    computation_method: exact | monte_carlo
    computation_evidence_uri: <string>
    comparison_baseline:                             # 比較対象 design（proposal 妥当性の証跡）
      baseline_design_type: full_factorial
      baseline_efficiency: 1.00
      efficiency_ratio: 0.85
      n_runs_ratio: 0.25                             # 1/4 の実行数で 0.85 の効率
```

### 10.7.2 エージェントの提案範囲（Ch4 §4.6 承認ゲート適用）

| 提案内容 | エージェント自律 | Human 承認必須 |
|---|---|---|
| pyDOE2 API 呼び出し・設計行列生成 | ✅ | — |
| resolution / fraction の提案 | ✅（複数案提示） | ✅（採択） |
| **factor の追加・削除** | — | ✅（`variable_selection_authorization`） |
| **blocking factor の追加・削除** | — | ✅（`variable_selection_authorization`） |
| **efficiency threshold の緩和** | — | ✅（`estimator_contract_change_gate` L3） |
| 実行順の randomization | ✅（seed provenance 遵守下） | — |
| **実際の実験装置起動** | — | ✅（`intervention_execution_authorization`） |
| **failure（危険物取扱・budget over・機材競合）検知時の停止** | ✅（fail-close） | — |

### 10.7.3 `design_proposal_skill` と `design_approval_skill` の分離

Ch5-7 の proposal/approval パターンを DoE に適用：

```yaml
skill_type: doe_design_proposal_skill
authorization_level: agent_autonomous
inputs:
  - research_question_uri                            # 何を推定したいか
  - factor_universe_uri                              # ドメインエキスパートが列挙した候補因子
  - factor_universe_frozen_at: <timestamp>
  - budget_constraint:
      max_runs: 32
      max_days: 3
      max_cost: 500000                               # JPY
  - efficiency_target:
      metric: d_efficiency
      minimum: 0.80
  - facility_constraints_uri                         # 装置予約・危険物制限・オペレータ稼働
outputs:
  - proposed_designs_uri                             # 複数案の設計行列
  - efficiency_comparison_report_uri
  - execution_cost_estimate_uri
  - alias_structure_reports_uri                      # fractional の場合
  - blocking_strategy_report_uri
  - proposal_bundle_sha256
prohibited_actions:
  - propose_factor_not_in_factor_universe            # fatal
  - modify_factor_universe_after_freeze              # fatal
  - exceed_budget_constraint_silently                # fatal
  - assume_pilot_data_availability_not_declared      # fatal（agent hallucination）
  - propose_design_below_efficiency_target           # fatal（threshold 未達を隠して提案）
```

```yaml
skill_type: doe_design_approval_skill
authorization_level: human_required
authorization_gate: variable_selection_authorization    # 因子・blocking の approve
                                                          # + intervention_execution_authorization は実行時に別途発動
inputs:
  - proposal_bundle_sha256
  - proposed_designs_uri
  - efficiency_comparison_report_uri
  - reviewer_comments_uri
outputs:
  - approved_design_uri
  - approved_design_sha256
  - approval_evidence:
      - reviewer_signatures
      - approval_timestamp
      - selected_design_id                           # 複数案から選択
      - trade_off_decision_uri                       # 効率 vs コストの意思決定記録
approver: research_lead                               # 個別実験の設計 approve
approver_independence:                                # Ch4 §4.6.2 継承
  conflict_policy: independent_reviewer_required_if_approver_is_data_producer
  fallback_approver: facility_causal_review_board
  facility_scope_escalation:
    applies_to:
      - approved_facility_design_template            # 施設共通のテンプレ化
    default_approver: facility_causal_review_board
prohibited_actions:
  - approve_without_reviewer_signatures              # fatal
  - approve_design_that_violates_budget_constraint   # fatal
  - approve_design_below_efficiency_target_without_documented_justification  # fatal
  - approve_design_with_undocumented_blocking_structure  # fatal
```

---

## 10.8 DoE Skill 全体契約テンプレート

Ch4 §4.9 のテンプレートを DoE 用に完全展開：

```yaml
skill:
  name: full_factorial_perovskite_voc_skill
  version: 0.1.0
  purpose: |
    ペロブスカイト太陽電池の V_OC に対する組成 × 温度 × 前処理時間の主効果 + 2-way + 3-way interaction を推定する。

  # === ① 目的 ===
  estimand_type: ate                                  # DoE では全 cell 平均が基本
  mediation_role: not_applicable
  causal_question_type: type_e_doe                    # 第1章の 5 Type

  # === ② 入力条件（L1: 識別レイヤ） ===
  identification_strategy: randomized_experiment      # DoE 特有
  dag_of_record_uri: "artifact://dags/doe_voc.dot"    # 介入 do(X) を含む DAG
  dag_of_record_sha256: <sha256>
  adjustment_set_approval_uri: <string>               # blocking factor 承認証跡
  confounders_declared: []                             # randomization により empty
  mediators_declared: []
  colliders_declared: []
  blocking_factors_declared:                          # DoE 特有（Ch4 provenance に追加）
    - instrument_id
    - operator_id

  # === ③ 出力形式 ===
  outputs:
    estimate: main_effects_and_interactions_with_ci
    anova_decomposition_uri: <string>
    test_results: dict
    sensitivity_analysis: dict

  # === ④ 成功条件（Ch4 §4.3 の 6 点セット） ===
  success_criteria:
    identification_validity: required
    refutation_pass: required
    positivity_check: required
    external_validity: required                       # 実験域外への外挿は counterfactual_scope_gate
    doe_efficiency: required                          # DoE 特有（Ch4 で not_applicable → required）
    randomization_integrity: required                 # DoE 特有

  identification_validity:
    checker: randomized_design_review
    report_uri: <string>

  # --- refutation gate (Ch9 §9.7.1) ---
  declared_required_tests:
    - e_value                                          # unmeasured confounder への頑健性（DoE では低リスクだが required）
    - random_common_cause
    - data_subset_validation                          # block 別 subset 検証
    - scope_gate_reverification                       # 実験域外への外挿判定
  applicability_manifest_uri: <string>
  applicability_manifest_sha256: <string>

  sensitivity_analysis:
    method: e_value
    effect_direction: harmful | protective
    ci_bound_closest_to_null: <bound>
    smd_to_rr_conversion: vanderweele_2020
    threshold:
      minimum_e_value: 1.5
    provenance_field: sensitivity_report_uri

  counterfactual_scope_gate:                          # 実験域外への予測に必須
    mahalanobis_check:
      threshold: 3.0
      cluster_assignment_uri: <string>
    variance_check:
      threshold: 0.25
    knn_density_check:
      k: 20
      knn_min: 5
    support_envelope_check:
      envelope_report_uri: <string>
    threshold_calibration:
      method: leave_one_out_empirical
      calibration_evidence_uri: <string>
      calibration_approved_at: <timestamp>
    aggregate_policy:
      pass_requires: all_four_pass

  # === ⑤ 禁止事項 ===
  prohibited_actions:
    - modify_factors_or_levels_after_design_freeze
    - reduce_n_runs_after_partial_execution
    - randomization_seed_override
    - ignore_blocking_factor_in_analysis
    - report_2way_estimate_from_resolution_iii_design
    - assign_blocking_factor_post_hoc
    - ignore_missing_cells_in_analysis
    - use_execution_order_correlated_with_outcome

  # === ⑥ 再現性条件（L3: 実行レイヤ） ===
  library_stack:
    design_generation: pyDOE2==1.3.0
    orthogonal_array_lookup: OApackage==2.7.0
    analysis: statsmodels==0.14.0
  environment_lock_uri: <string>
  random_seed: 42
  container_sha256: <string>

  # === ⑦ 承認ゲート ===
  authorization_gates:
    dag_authorization:
      required_for: [dag_of_record_uri_change, dag_of_record_sha256_mismatch]
      approver: research_lead
    variable_selection_authorization:
      required_for:
        - factor_universe_change
        - blocking_factors_declared_change
        - design_parameter_change                     # resolution / fraction / block size
      approver: research_lead
    intervention_execution_authorization:
      required_for:
        - physical_experiment_execution
      approver: pi_and_facility_manager

    approver_independence:
      conflict_policy: independent_reviewer_required_if_approver_is_data_producer
      fallback_approver: facility_causal_review_board
      facility_scope_escalation:
        applies_to:
          - approved_facility_design_template
        default_approver: facility_causal_review_board

  # === ⑧ Estimator provenance reference（Ch9 refutation_gate 用）===
  estimator_provenance_reference:
    estimator_contract_sha256: <string>
    dag_of_record_sha256: <string>
    design_matrix_sha256: <string>                    # DoE 特有

  # === ⑨ DoE 拡張（Ch4 §4.9 ⑧ を展開）===
  experimental_design_provenance:
    design_type: full_factorial                       # or fractional_factorial / plackett_burman /
                                                       #    orthogonal_array / randomized_complete_block /
                                                       #    split_plot / latin_square
    factors: [...]                                    # §10.2.2 参照
    factors_frozen_at: <timestamp>
    n_runs_total: 24
    replication_policy: {...}
    randomization_scope: {...}
    blocking_factors: [...]                           # §10.6.3 参照
    randomization_seed_provenance: {...}              # §10.5.1 参照
    design_matrix_uri: <string>
    design_matrix_sha256: <string>
    assignment_log_uri: <string>
    information_gain_metric: d_efficiency
    information_gain_threshold: 0.80
    generation:
      library: pyDOE2
      library_version: <string>
      algorithm: fullfact | fracfact | pbdesign | lhs
      random_seed: 42
      random_seed_generation_method: os_urandom_at_design_time
```

---

## 10.9 pyDOE2 / OApackage の実装スニペット

### 10.9.1 full factorial

```python
from pyDOE2 import fullfact
import numpy as np
import hashlib
import json

# 3 factors: 3 levels, 2 levels, 2 levels
levels_per_factor = [3, 2, 2]
design_coded = fullfact(levels_per_factor)   # shape: (12, 3), coded as 0, 1, 2 / 0, 1 / 0, 1

# 実際の水準値へマッピング
level_maps = [
    [400, 500, 600],       # temperature
    [1.0, 10.0],           # pressure
    [15, 30],              # pretreatment_time
]
design_actual = np.array([
    [level_maps[j][int(design_coded[i, j])] for j in range(len(levels_per_factor))]
    for i in range(design_coded.shape[0])
])

# randomize run order with seed provenance
rng = np.random.default_rng(seed=42)
run_order = rng.permutation(design_actual.shape[0])
design_randomized = design_actual[run_order]

# hash for immutability
design_bytes = json.dumps(design_randomized.tolist(), sort_keys=True).encode()
design_sha256 = hashlib.sha256(design_bytes).hexdigest()

# assignment log
assignment_log = [
    {
        "run_id": f"R{i+1:03d}",
        "assigned_cell": {
            "temperature": int(design_randomized[i, 0]),
            "pressure": float(design_randomized[i, 1]),
            "pretreatment_time": int(design_randomized[i, 2]),
        },
        "original_row_index": int(run_order[i]),
        "randomization_seed": 42,
    }
    for i in range(design_randomized.shape[0])
]
```

### 10.9.2 fractional factorial（$2^{7-3}$ resolution IV）

```python
from pyDOE2 import fracfact

# generator: D=ABC, E=BCD, F=ACD, G=ABD
design = fracfact("a b c abc bcd acd abd")   # shape: (16, 7), coded as -1, +1
# resolution IV
```

### 10.9.3 Plackett-Burman（PB12）

```python
from pyDOE2 import pbdesign

design = pbdesign(11)   # 12 runs, up to 11 factors, resolution III
```

### 10.9.4 Latin Hypercube（LHS）— 予告

LHS は連続因子の空間充填設計として **応答曲面（Ch11）**で主に使用。本章では設計行列生成 API のみ紹介：

```python
from pyDOE2 import lhs

design = lhs(n=3, samples=20, criterion='maximin')   # 3 factors, 20 points, maximin distance
```

---

## 10.10 章末チェックリスト

- [ ] **§10.1 randomization による識別仮定の格上げ**——exchangeability / positivity / consistency が checked に、SUTVA は assessed のまま——を説明できる
- [ ] **§10.2 full factorial の runs 数と交互作用推定可能性**——$N = \prod L_j$、$2^k$ で $2^k$ 効果全て clean——を計算できる
- [ ] **§10.3 fractional factorial の resolution**——III / IV / V の意味と alias structure を書き下せる
- [ ] **§10.3.4 Plackett-Burman**——主効果 only、2-way 以上禁止——の適用場面を説明できる
- [ ] **§10.5 randomization seed の 3 点セット**（seed value / 生成 method / assignment log）と `randomization_seed_provenance` を作成できる
- [ ] **§10.6 blocking と CRD/RCBD/split-plot/Latin square** の使い分けを、`blocking_factors_declared` + `randomization_scope` で書ける
- [ ] **§10.7 D/A/G/V-efficiency** の定義と、`information_gain_metric` + threshold の設定ができる
- [ ] **§10.7.3 design proposal/approval の 2 Skill 分離**——エージェント自律 vs Human 承認の境界を説明できる
- [ ] **§10.8 DoE Skill 契約テンプレート**を、自研究テーマで完全に埋められる
- [ ] **§10.9 pyDOE2 API**（fullfact / fracfact / pbdesign / lhs）と design_matrix_sha256 の凍結手順を実装できる

---

## 章末演習

### 演習 10.1：自研究テーマで full factorial 設計

第2章 演習 2.1 の因果的問いを DoE で検証する立場で、以下を作成：

1. 因子 3〜5 個の候補リスト（水準含む）と `factor_universe_uri` の内容
2. full factorial での $N$ runs 計算
3. budget constraint（例：3 日、24 runs 上限）と、超えた場合の fractional への切替判断
4. §10.8 テンプレートを埋めた Skill 仕様書

### 演習 10.2：alias structure の解読

$2^{7-3}$ resolution IV 設計で、generator を `D=AB, E=AC, F=BC, G=ABC` としたとき：

1. defining relation を書き下せ（例：`I = ABD = ACE = ...`）
2. 主効果 A の aliases を列挙せよ
3. 2-way interaction AB の aliases を列挙せよ
4. resolution が実際に IV であることを確認せよ

### 演習 10.3：blocking factor の設計

以下のシナリオで最適な blocking 設計を提案：

- 温度 3 水準 × 前処理時間 2 水準 = 6 条件
- 装置 3 台、オペレータ 2 名、実行日 3 日
- 各条件を 2 反復（合計 12 runs）

`blocking_factors_declared` と `randomization_scope` を書き下し、装置 × 日 × オペレータの完全交絡を回避する Latin square + within-block factorial の hybrid 設計を作成。

### 演習 10.4：Skill 設計の 2 段階分離

`doe_design_proposal_skill` と `doe_design_approval_skill` を、自研究の因子リストで埋め、以下を明示：

1. proposal skill が **自律に決定してよいこと**（algorithm 選択・実行順 randomization 等）
2. approval skill で **Human が判断すること**（factor 選定・blocking factor・efficiency threshold 緩和）
3. 危険物取扱い制約が factor_universe に含まれる場合の facility_causal_review_board 経由承認フロー

---

## 参考資料

### 内部 cross-reference

- **第0章**：4 識別仮定（本章 §10.1 で randomization による格上げ）
- **第1章**：causal question の 5 Type（本章は Type E: DoE）
- **第2章**：DAG 記法（介入 do(X) を含む DAG、本章 §10.8 で `dag_of_record_uri`）
- **第4章**：`experimental_design_provenance` 契約定義（本章で operational 実装）、Ch4 §4.9 テンプレート（本章 §10.8 で展開）、3 層承認（本章 §10.7.3 で proposal/approval Skill に適用）
- **第5-7章**：DAG proposal/approval Skill、IV / donor pool proposal/approval Skill（本章 §10.7.3 の design proposal/approval Skill のパターン源泉）
- **第9章**：refutation_gate、`counterfactual_scope_gate` 再検証（本章 DoE Skill でも estimand の外挿判定に required）
- **第11章**：応答曲面（CCD / Box-Behnken / Kriging）、タグチ SN 比（OA からの発展）
- **第12章**：Bayesian D-/A-optimality、one-shot Bayesian DoE
- **第13章**：capstone で本章の設計を実行、Ch5-9 の観測 Skill と統合
- **第14章**：randomization 破綻・blocking 失敗・応答曲面外挿の失敗パターン
- **vol-05**：逐次実験計画（BO の acquisition function）に完全委譲

### 外部参考文献

- Montgomery, D. C. (2019). *Design and Analysis of Experiments* (10th ed.). Wiley. — 古典的 DoE の標準教科書
- Wu, C. F. J., & Hamada, M. S. (2021). *Experiments: Planning, Analysis, and Optimization* (3rd ed.). Wiley. — resolution / alias structure の詳細
- pyDOE2 GitHub: https://github.com/clicumu/pyDOE2 — Python 実装（fullfact / fracfact / pbdesign / lhs / ccdesign / bbdesign）
- OApackage: https://oapackage.readthedocs.io/ — 直交表・DoE 効率計算
- Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters* (2nd ed.). Wiley. — 実験計画の思想と実践

### API チートシート（詳細は付録 B）

| ライブラリ | 主要関数 | 用途 |
|---|---|---|
| pyDOE2 | `fullfact(levels)` | 完全要因計画 |
| pyDOE2 | `fracfact(generator)` | 一部実施要因計画 |
| pyDOE2 | `pbdesign(n_factors)` | Plackett-Burman |
| pyDOE2 | `lhs(n, samples, criterion)` | Latin Hypercube |
| pyDOE2 | `ccdesign(n_factors)` | Central Composite (Ch11) |
| pyDOE2 | `bbdesign(n_factors)` | Box-Behnken (Ch11) |
| OApackage | `oa_from_txt("L12.oa")` | 標準 OA テーブル読込 |
| OApackage | `Deff(oa)` | D-efficiency 計算 |
| statsmodels | `ols(...).fit()` | ANOVA / factorial 分析 |
| statsmodels | `mixedlm(...)` | RCBD / split-plot の mixed model |
