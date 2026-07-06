# 第7章　Acquisition function の Skill 化と外挿検知の operational 定義

> [!IMPORTANT]
> **本章の到達目標**
>
> 1. 7 種の acquisition function（EI / UCB / PI / KG / MES / TS / PES）を判断表から選べる
> 2. exploration vs exploitation の trade-off を Skill 契約の言葉で説明できる
> 3. **エージェントが acquisition を "動的切替する" 場合の妥当性判定基準**を Skill 契約に組み込める
> 4. **外挿検知（`hallucinated_recommendation_detection`）の operational 定義**を、GP の予測分散閾値・Mahalanobis 距離・length scale 越えの合成基準として書ける
> 5. **vol-04 第12章（one-shot Bayesian D/A-optimality）との boundary** を説明できる
>
> **本章で扱わないこと**
>
> - GP kernel 選択 → **第6章**
> - BNN / RF surrogate での外挿検知 → **第8章 §8.6**
> - 多目的 acquisition（qEHVI）→ **第9章**
> - 制約付き acquisition（cEI）と safe BO → **第10章**
> - Batch acquisition（q-EI / q-EHVI）→ **第12章**
> - Active learning（uncertainty sampling / QBC）→ **第13章**

---

## 7.1 Acquisition function とは何か（Skill 契約から見た定義）

Acquisition function $\alpha(x)$ は、**GP posterior の情報を使って「次の観測点として最も価値の高い候補」を数値化する関数**です。BO ループでは：

$$
x_{\text{next}} = \arg\max_{x \in \mathcal{X}} \alpha\bigl(x;\ \mathcal{D}_{1:t},\ \theta_t\bigr)
$$

ここで $\mathcal{D}_{1:t}$ が Group B の `observed_data`、$\theta_t$ が Group A の `kernel_spec` + iteration $t$ で fit された hyperparameter。**acquisition function の選択（Group A `acquisition_spec.name`）と hyperparameter（同 `acquisition_spec.hyperparameters`）は Skill 起動時に固定**——iteration の途中で切り替えるのは第3章 §3.5 逸脱 2、Skill fatal で止めます。

### 符号規約（本章の acquisition すべてに共通）

- **BoTorch を含む本書の実装は原則「$\alpha$ を最大化」に正規化**します
- **最小化問題**は Skill 起動時に `objective_direction: "minimize"` を宣言し、outcome transform で $y_{\text{model}} = -y_{\text{raw}}$（または `Standardize` の一部として符号反転）を適用してから acquisition を計算する
- したがって以降の閉形式は **最大化問題** の形で書きます

> [!NOTE]
> エージェントが「今の iteration ではもっと exploratory な UCB に切り替えたい」と提案した場合、これは **新 Skill run として Human 承認を経る**か、**§7.6 の "動的切替契約" で事前に許可された切替パターンに限定する**か、の 2 択です。暗黙の切替は禁止。

---

## 7.2 7 種の acquisition function 概観

### 7.2.1 Expected Improvement (EI)

最大化問題での閉形式：

$$
\alpha_{\text{EI}}(x) = \mathbb{E}\bigl[\max(0,\ \mu(x) - y^*)\bigr]
= (\mu(x) - y^*)\Phi(z) + \sigma(x)\phi(z), \quad z = \tfrac{\mu(x) - y^*}{\sigma(x)}
$$

ここで $y^*$ は現在の best（最大化なので最大値）。最小化問題は §7.1 の符号規約で最大化に変換してから適用。

- **強み**：exploration と exploitation を自動 balance、計算安価、閉形式
- **弱み**：$y^*$ が noisy な場合に不安定 → **noisy EI**（`qLogNoisyExpectedImprovement`）を使う
- **BoTorch 推奨**：`qLogExpectedImprovement`（数値的に安定な log 版、第4章 §4.6）
- **材料 BO デフォルト**：これで迷わない

### 7.2.2 Upper Confidence Bound (UCB)

$$
\alpha_{\text{UCB}}(x) = \mu(x) + \beta_t \sigma(x)
$$

- **強み**：$\beta_t$ を明示的に制御できる、理論保証あり（Srinivas et al. 2010）
- **弱み**：$\beta_t$ の設定に依存、実務では試行錯誤
- **推奨 $\beta_t$**：$\beta_t = 2 \log(t^{d/2+2} \pi^2 / (3 \delta))$（理論値）、実務では 2-3 の定数で始める
- **BoTorch**：`qUpperConfidenceBound(model, beta=2.0)`

### 7.2.3 Probability of Improvement (PI)

最大化問題での閉形式：

$$
\alpha_{\text{PI}}(x) = \Phi\!\left(\tfrac{\mu(x) - y^* - \xi}{\sigma(x)}\right)
$$

- **強み**：直感的（「best を超える確率」）
- **弱み**：$\xi = 0$ だと exploitation に寄りすぎる、$\xi$ 依存が強い
- **推奨場面**：Skill 契約に「target value 到達確率」を明示したいとき（regret_threshold 型 stop condition との相性がよい）

### 7.2.4 Knowledge Gradient (KG)

$$
\alpha_{\text{KG}}(x) = \mathbb{E}\bigl[\max_{x'} \mu_{t+1}(x')\bigr] - \max_{x'} \mu_t(x')
$$

- **強み**：**次の観測後に posterior mean の最良値がどれだけ改善するかを直接評価**——「情報価値」を測る
- **弱み**：計算コスト大（inner optimization が入る）
- **推奨場面**：**noisy 観測**、**derivatives なし** の状況、**terminal-value 型の目的**（vol-04 第12章の one-shot Bayesian DoE と対比：Ch12 は **一括に $N$ 個の点を選ぶ**、KG は**逐次に 1 個ずつ選ぶ**）
- **BoTorch**：`qKnowledgeGradient(model, num_fantasies=64)`

### 7.2.5 Max-Value Entropy Search (MES)

最適値 $f^*$（未知の確率変数、EI の incumbent $y^*$ とは区別）に関する mutual information：

$$
\alpha_{\text{MES}}(x) = I(y_x;\ f^* \mid \mathcal{D}_{1:t}) = H\bigl[p(y_x | x, \mathcal{D}_{1:t})\bigr] - \mathbb{E}_{f^*}\!\bigl[H[p(y_x | x, \mathcal{D}_{1:t}, f^*)]\bigr]
$$

- **強み**：**最適値の分布に関する情報獲得** を最大化、multi-modal な posterior に強い
- **弱み**：KG と同じく計算コスト大、実装が複雑
- **推奨場面**：**目的関数の最良値そのもの** を知りたい（材料設計で「最高性能はどこまで到達できるか」を推定したい）

### 7.2.6 Thompson Sampling (TS)

- **やること**：GP posterior から関数 $\tilde{f}$ を 1 サンプルし、$x_{\text{next}} = \arg\max \tilde{f}(x)$
- **強み**：**自然に exploration**、batch BO と相性がよい（第12章）、実装が簡単
- **弱み**：GP から関数サンプルするコスト（Fourier feature や pathwise で高速化）
- **推奨場面**：batch BO（q-TS）、**大規模並列**（第12章）

### 7.2.7 Predictive Entropy Search (PES)

- **考え方**：MES と近いが、$p(x^* | \mathcal{D})$ の entropy 減少を測る
- **強み**：$x^*$（最適点の位置）に関する情報を直接測る
- **弱み**：実装が最も難しい、近似の精度依存が強い
- **推奨場面**：MES の代替案、実装は付録 B / Emukit 経由

---

## 7.3 Skill としての acquisition 選択判断表

Skill 契約に貼る acquisition 選定フロー：

```
[Q1] 観測ノイズは？
  ├─ 低い（実験精度が高い） → EI（qLogExpectedImprovement が第一選択）
  └─ 高い（実験ばらつき大）  → qLogNoisyExpectedImprovement / KG

[Q2] exploration の強さを陽に制御したい？
  ├─ YES → UCB（beta を Skill 契約に明記）
  └─ NO  → EI / KG / MES

[Q3] Batch（q 個同時提案）にする？
  ├─ YES → 第12章の batch acquisition 契約に委譲
  └─ NO  → 通常版

[Q4] 目的関数の最良値そのものを知りたい？
  ├─ YES → MES / KG
  └─ NO（最適点を見つけたい）→ EI / UCB / PES

[Q5] 目標値到達確率を契約に組み込みたい？
  ├─ YES → PI（stop_condition.regret_threshold と組み合わせ）
  └─ NO  → EI がデフォルト

[Q6] 制約付き？（実行不可領域あり）
  → 第10章 cEI / cUCB へ

[Q7] 多目的？
  → 第9章 qEHVI へ

[Q8] Human 承認との相性を優先したい？
  → EI（数値が「期待改善量」として直感的、Human review しやすい）
```

**迷ったら EI（`qLogExpectedImprovement`）**。材料 BO の 8 割はこれで足ります。

---

## 7.4 acquisition_spec の Skill 契約

第5章 §5.2 で導入した `acquisition_spec` の完全な内部構造：

```yaml
acquisition_spec:
  # --- 必須フィールド（canonical enum）---
  name: "qLogEI"                   # enum: qLogEI | qLogNoisyEI | qUCB | qPI | qKG | qMES | qTS | qPES | cEI | qEHVI | ...
  objective_direction: "maximize"  # maximize | minimize
  
  # --- 実装マッピング（canonical enum → 実装クラス）---
  implementation:
    library: "botorch"
    class: "qLogExpectedImprovement"       # BoTorch クラス名
    required_runtime_inputs:               # acquisition 生成時に model から供給する引数
      - "best_f"                           # qLogEI: 現在の best
      # qLogNoisyEI: ["X_baseline"]
      # qKG: []（内部で fantasize）+ hyperparameters.num_fantasies
      # qUCB: []（hyperparameters.beta）
      # qMES: ["candidate_set"]
      # qTS: []（内部で sample）
  
  # --- optimizer 設定（acquisition の数値最適化）---
  optimizer:
    method: "L-BFGS-B"             # BoTorch 標準
    num_restarts: 10               # multi-start restart 数
    raw_samples: 128               # 初期サンプル数
    max_iter: 200                  # 最適化の最大 iteration
  
  # --- acquisition 固有 hyperparameter ---
  hyperparameters:
    # qUCB の場合: { beta: 2.0 }
    # qKG / qMES の場合: { num_fantasies: 64 }
    # qPI の場合: { xi: 0.01 }
    {}
  
  # --- 動的切替契約（§7.6）---
  dynamic_switching:
    allowed: false                 # デフォルトは禁止
    allowed_transitions: []        # allowed: true のとき許可される切替パターン
    switch_reason_field: null      # 切替時に Human が記入する reason field
```

> [!IMPORTANT]
> `name`（canonical enum）と `implementation.class`（実装クラス名）を分離する理由：（1）canonical enum を Skill 契約の一級市民として保つ、（2）BoTorch/Emukit 等のライブラリ切替時に契約側を変更せず `implementation` ブロックだけ差し替えられる、（3）audit で「同じ acquisition 意図」を横断的に検索できる。`qMES` の実装クラスは BoTorch では `qMaxValueEntropy` になる、というような命名差もここで吸収します。

---

## 7.5 exploration vs exploitation の trade-off

BO の教科書的トピックですが、Skill 契約の言葉に翻訳すると：

| 目標 | どうすればよいか | Skill 契約での表現 |
|---|---|---|
| exploration を強くしたい | UCB の $\beta$ を大きく、EI で $\xi$ を大きく、TS を使う | `hyperparameters.beta: 3.0`（or larger）|
| exploitation を強くしたい | UCB の $\beta$ を小さく、$\xi = 0$ の EI、Best-of 戦略 | `hyperparameters.beta: 1.0`（or smaller）|
| iteration で徐々に exploitation にシフト | 動的切替（§7.6）が必要 | `dynamic_switching.allowed: true` + 事前定義パターン |

> [!WARNING]
> 「iteration が進むにつれて $\beta$ を自動で減らす」ような**自動 scheduling は Skill 契約で明示的に宣言**する必要があります。宣言なしでの $\beta$ 変更は第3章 §3.5 逸脱 2、Skill fatal です。

---

## 7.6 エージェントが acquisition を "動的切替する" 場合の妥当性判定

**Agentic BO の中心的問題**：エージェントが「今の iteration では exploration が足りないから UCB に切り替えたい」と自発的に判断して acquisition を変えると、実験の provenance が破壊されます（第3章 §3.5 逸脱 2）。

しかし、動的切替が**運用的に必要**な場面はあります：

- 初期 $k$ iteration は exploration 重視の TS、以降は EI に切替（cold start 戦略）
- convergence が疑われたら UCB で穴埋め探索
- hallucination_flag 発火時に UCB → PI に切り替え（次候補の予測ばらつきを抑える）

これを **契約として許可する** 場合の要件：

### 契約要件

**immutable / append-only / effective の 3 層分離：**

1. **Group A の `acquisition_spec.name` は Skill 起動時の初期値として immutable**——直接更新は fatal
2. **切替 event は Group B の append-only ログ `acquisition_switch_events` に記録**（下記）
3. **各 iteration の "effective" acquisition は Group C に `effective_acquisition_name` として都度記録**——これが実際に使われた acquisition
4. Group C の `effective_acquisition_name` は Group A の初期値と Group B のイベントログから deterministic に導出可能でなければならない（audit 対応）

```yaml
acquisition_spec:
  name: "qLogEI"                   # 初期値、immutable
  dynamic_switching:
    allowed: true
    allowed_transitions:
      - from: "qLogEI"
        to: "qUCB"
        trigger:
          condition: "convergence_stalled"
          detection: "no improvement in best_y over N=5 iterations"
        hyperparameters_at_switch: { beta: 2.5 }
        max_uses_per_run: 1
      - from: "qLogEI"
        to: "qPI"
        trigger:
          condition: "hallucination_flag_raised"
          detection: "hallucination_flag.flag == true in previous iteration"
        hyperparameters_at_switch: { xi: 0.01 }
        max_uses_per_run: 3
    switch_reason_field: "acquisition_switch_events"
```

そして Group B に append-only ログ：

```yaml
acquisition_switch_events:
  - from_iteration: 5
    to_iteration: 6
    from_acquisition: "qLogEI"
    to_acquisition: "qUCB"
    trigger_matched: "convergence_stalled"
    switched_at: "2026-11-20T14:30:00Z"
    approver_id: "human:staff_id_XXXX"  # human approval required for switch
    reason: "5 iteration 連続で best_y の改善なし、exploration 増強を Human 判断で承認"
```

### 妥当性判定チェックリスト

- [ ] `allowed_transitions` が Skill 起動時に定義されている（Group A immutable）
- [ ] 各 transition に `trigger.condition` と `trigger.detection` が明示的に書かれている
- [ ] `hyperparameters_at_switch` で新 acquisition の hyperparameter が固定されている
- [ ] `max_uses_per_run` で切替回数に上限がある（無制限切替は禁止）
- [ ] Group B の `acquisition_switch_events` は append-only、`approver_id` に Human 承認が記録される
- [ ] Switch 時にも `experiment_launch_authorization` は毎回必要（§5.3）、物理実験では `parent_authorization_id` として **vol-04 L3 承認 ID を必ず含む**

**宣言されていない transition** は fatal、これが第3章 §3.5 逸脱 2 の中核的な防御機構。

---

## 7.7 外挿検知の operational 定義 — `hallucinated_recommendation_detection`

第5章 §5.6 で宣言レベルで導入した `hallucinated_recommendation_detection` を、本節で **operational な合成基準として実装** します。3 判定の OR 合成：

$$
\text{flag}(x) = \bigl[V_{\text{norm}}(x) > \tau_V\bigr] \lor \bigl[D_{\text{Mah}}(x) > \tau_D\bigr] \lor \bigl[L_{\text{ratio}}(x) > \tau_L\bigr]
$$

### 7.7.1 判定 1：予測分散閾値（$V_{\text{norm}}$）

- **やること**：候補 $x$ での GP posterior 予測分散 $\sigma^2(x)$ を、観測点上の中央値分散 $\text{median}_i \sigma^2(x_i)$ で正規化
- **式**：$V_{\text{norm}}(x) = \sigma^2(x) / \text{median}_i \sigma^2(x_i)$
- **推奨閾値**：$\tau_V = 4.0$（分散が観測点上中央値の 4 倍を超えたら flag）
- **意味**：「観測点から遠い（GP が自信を持てない）領域」を検出

### 7.7.2 判定 2：Mahalanobis 距離（$D_{\text{Mah}}$）

- **やること**：候補 $x$ の、近傍学習点との Mahalanobis 距離を計算
- **式**：$D_{\text{Mah}}(x) = \min_{i} \sqrt{(x - x_i)^\top \text{diag}(1/\ell^2)(x - x_i)}$
- **推奨閾値**：$\tau_D = 3.0$（length scale で正規化した「近傍距離」が 3 を超えたら flag）
- **意味**：「観測点から length scale 単位で遠い候補」を検出、外挿の直接的な指標
- **注意**：$\ell$ は kernel の length scale（ARD の場合は per-dim）

### 7.7.3 判定 3：Length scale で正規化した最近傍距離（$L_{\text{ratio}}$）

- **やること**：候補 $x$ から**最近傍観測点**までの per-dimension normalized distance の**最大値**を測る
- **式**：$L_{\text{ratio}}(x) = \min_i \max_j \bigl(|x_j - x_{i,j}| / \ell_j^{\text{eff}}\bigr)$
  - $\min_i$：$N$ 個の観測点のうち最も近い 1 個を選ぶ
  - $\max_j$：その最近傍点との「最も遠い次元」の乖離を length scale で正規化して評価
- **推奨閾値**：$\tau_L = 1.5$（最近傍点まで、いずれかの次元で length scale の 1.5 倍超）
- **意味**：判定 2（Mahalanobis）が「$L^2$ 集約」で全次元平均を見るのに対し、判定 3 は「$L^\infty$ 集約」で最も乖離した次元を見る。**boundary 上の有望候補**（探索空間の端だが最近傍観測点は近い）を誤検出しにくい
- **$\ell_j^{\text{eff}}$ の clamping**：後述 §7.7.4 で length scale の暴走を防ぐ

> [!IMPORTANT]
> **旧版で「探索空間中心からの距離 / length scale」を使っていた実装は誤り**——boundary 上の有望点をすべて hallucination と誤判定してしまうため、必ず**最近傍観測点との距離**を使うこと。「center から遠い」は外挿の指標ではありません。

### 7.7.4 length scale の circular dependency 対策（$\ell^{\text{eff}}$ の clamping）

判定 2/3 は共に GP の learned length scale $\ell$ に依存します。GP が sparse data で $\ell$ を過大推定すると外挿を見逃し、過小推定すると全候補を flag する **circular problem**——「hallucination しうる surrogate 自身の hyperparameter に検知が依存する」構造。

**対策**：Skill 契約に `ell_min` / `ell_max` を宣言し、判定時は clamp した実効値を使う：

$$
\ell_j^{\text{eff}} = \text{clip}\bigl(\ell_j^{\text{learned}},\ \ell_j^{\text{min}},\ \ell_j^{\text{max}}\bigr)
$$

- $\ell_j^{\text{min}}$：探索範囲幅 $(b_j^{\text{high}} - b_j^{\text{low}})$ の 5-10% 程度
- $\ell_j^{\text{max}}$：探索範囲幅の 200-500% 程度（`ARD` で「効かない次元」判定に相当する上限を超えない）

これを `hallucinated_recommendation_detection.operational_definition` に記録します。

### 7.7.5 実装スケッチ

```python
def hallucinated_recommendation_detection(x_cand, model, X_obs, bounds,
                                            ell_min, ell_max,
                                            tau_V=4.0, tau_D=3.0, tau_L=1.5):
    # length scale の clamping（circular dependency 対策）
    ell_learned = model.covar_module.base_kernel.lengthscale.detach().squeeze()
    ell_eff = ell_learned.clamp(min=ell_min, max=ell_max)  # (d,)

    # 判定 1: 予測分散閾値
    posterior_cand = model.posterior(x_cand.unsqueeze(0))
    var_cand = posterior_cand.variance.squeeze().item()
    posterior_obs = model.posterior(X_obs)
    var_obs_median = posterior_obs.variance.median().item()
    V_norm = var_cand / (var_obs_median + 1e-12)
    
    # 判定 2: Mahalanobis 距離（L2、length scale で正規化）
    diffs = (x_cand.unsqueeze(0) - X_obs) / ell_eff  # (N, d)
    D_Mah = diffs.pow(2).sum(dim=-1).sqrt().min().item()
    
    # 判定 3: 最近傍観測点までの L_inf 正規化距離
    per_dim_dist = (x_cand.unsqueeze(0) - X_obs).abs() / ell_eff  # (N, d)
    L_ratio = per_dim_dist.max(dim=-1).values.min().item()        # min_i max_j
    
    flag = (V_norm > tau_V) or (D_Mah > tau_D) or (L_ratio > tau_L)
    reasons = []
    if V_norm > tau_V: reasons.append("variance_exceeded")
    if D_Mah > tau_D:  reasons.append("mahalanobis_exceeded")
    if L_ratio > tau_L: reasons.append("length_scale_ratio_exceeded")
    
    # 戻り値は Group C の hallucination_flag と同じ shape
    return {
        "hallucination_flag": {
            "flag": flag,
            "reasons": reasons,
            "scores": {
                "variance_ratio": V_norm,
                "mahalanobis_distance": D_Mah,
                "length_scale_ratio": L_ratio,
            },
        }
    }
```

### 7.7.6 Skill 契約への埋め込み

```yaml
hallucinated_recommendation_detection:
  version: "v0.1"
  declared: true
  detection_criteria_ref: "vol-05:ch07:hrd_operational_v0_1"
  operational_definition:
    logic: "OR"
    length_scale_clamping:
      ell_min:  [10.0, 0.1, 0.5]   # per-dim (d,)、探索範囲幅の 5-10%
      ell_max:  [500.0, 5.0, 25.0] # per-dim、探索範囲幅の 200-500%
    criteria:
      - name: "variance_ratio"
        formula: "posterior_variance(x_cand) / median(posterior_variance(X_obs))"
        threshold: 4.0
      - name: "mahalanobis_distance"
        formula: "min_i sqrt(sum_j ((x_cand_j - x_ij) / ell_j_eff)^2)"
        threshold: 3.0
      - name: "length_scale_ratio"
        formula: "min_i max_j |x_cand_j - x_ij| / ell_j_eff"
        threshold: 1.5
  action_on_flag: "return_needs_human_review"
  logged_events_field: "hallucination_events"
```

### 7.7.7 閾値のチューニング指針

- **合成 OR で false positive を許容**：flag → Human review の設計なので、拾いすぎでも安全側
- **iteration 数と観測数に応じて厳しくする**：観測が増えたら $\tau_V$ を 3.0 まで、$\tau_D$ を 2.5 まで下げる
- **絶対に緩めてはいけない**：Skill 実行中に閾値を上げる = fatal（第3章 §3.5 逸脱 3、hallucination を「flag しなくする」逸脱）

> [!TIP]
> 3 判定は independent の情報源になるように設計されています。$V_{\text{norm}}$ は GP 内部の自信、$D_{\text{Mah}}$ は近傍学習点の物理的近さ、$L_{\text{ratio}}$ は bounds の端。いずれか 1 つでも flag すれば Human review。

---

## 7.8 vol-04 第12章 (Bayesian D/A-optimality) との boundary

vol-04 第12章の Bayesian D/A-optimality と本章の KG / MES は数学的に似ているため、**責務の分離** を明示：

| 論点 | vol-04 Ch12（Bayesian D/A-opt）| vol-05 Ch7（KG / MES）|
|---|---|---|
| タイミング | **BO 開始前の一括 DoE**（`N` 個をまとめて選ぶ）| **BO の各 iteration での逐次選択**（1 個ずつ、または batch $q$ 個）|
| 目的関数 | posterior の determinant / trace（**モデル推定精度**）| $\alpha_{\text{KG}}$（**posterior mean の改善**）または $\alpha_{\text{MES}}$（**$y^*$ の entropy 減少**）|
| iteration_index | なし（1 回で終わり）| あり（Group A で管理）|
| retraining_policy | なし | あり（每 iteration / 每 N / threshold）|
| provenance フィールド | vol-04 の 15 フィールド | vol-05 の 21 フィールド |
| Human 承認 | vol-04 L1/L2/L3 | vol-04 L3 + `experiment_launch_authorization` |

> [!IMPORTANT]
> **エージェントが「BO の代わりに vol-04 Ch12 の Bayesian D-opt を毎 iteration 呼ぶ」ような戦略は不適切**——iteration 概念を持たない DoE 手法を逐次的に流用すると、pending experiments や `retraining_policy` の provenance が失われます。**iteration 内で $N$ 個の候補を選ぶ場合は本章の Batch acquisition（第12章 q-EI / q-KG / q-MES）を使う**のが正しい設計です。

---

## 7.9 Skill 契約チェックリスト

- [ ] `acquisition_spec.name` が §7.3 の 7 種から明示的に選ばれている
- [ ] `objective_direction` が明示されている（maximize / minimize）
- [ ] `hyperparameters` に acquisition 固有の設定（beta / xi / num_fantasies）が入っている
- [ ] `optimizer.num_restarts` と `raw_samples` が設定されている（BoTorch 推奨 10 / 128）
- [ ] `dynamic_switching.allowed` が明示的に決まっている（デフォルト false）
- [ ] true の場合、`allowed_transitions` に trigger と hyperparameters_at_switch が完全に書かれている
- [ ] `hallucinated_recommendation_detection.operational_definition` が §7.7 の 3 判定を含む
- [ ] Group C の `hallucination_flag.scores` が 3 判定の値を全て記録する
- [ ] Group B の `hallucination_events` が append-only になっている

---

## 7.10 章末演習

**問 1**：自分のケースで、§7.3 判断表を辿って acquisition を 1 つ選び、根拠を書けますか？

**問 2**：観測ノイズが大きく iteration 数が限られる場合（20 iteration まで）、EI / KG / TS のどれを選びますか？ 理由を書けますか？

**問 3**：iteration 8 で `hallucination_flag.reasons = ["mahalanobis_exceeded"]` が発火した。他 2 判定は閾値内。Skill が返す `action_on_flag` はどれになりますか？（return_needs_human_review / reject / accept_anyway のいずれか）

**問 4**：エージェントが「iteration 3 まで TS で exploration、以降は EI」と提案してきた。この動的切替を契約に組み込むには、§7.6 のどの要素をどう書き込めばよいか？

**問 5**：vol-04 Ch12 の Bayesian D-optimality を、本章の逐次 BO と組み合わせて使う場面はありますか？（一括 DoE で初期観測を選び、以降を BO で回すシナリオを想定）

---

## 7.11 参考資料

### 本書内

- [第3章](./chapter-03.md) §3.5：5 逸脱パターン（逸脱 2 = acquisition drift、逸脱 3 = 外挿を自信あり報告）
- [第5章](./chapter-05.md) §5.2 / §5.6：`acquisition_spec` / `hallucinated_recommendation_detection` の宣言レベル
- [第6章](./chapter-06.md) §6.7：GP surrogate の実装（本章の acquisition が受け取る model）
- [第8章](./chapter-08.md)（planned）：BNN / RF surrogate での外挿検知（本章 §7.7 の合成基準を拡張）
- [第9章](./chapter-09.md)（planned）：qEHVI（多目的 acquisition）
- [第10章](./chapter-10.md)（planned）：cEI / safe BO（制約付き acquisition）
- [第12章](./chapter-12.md)（planned）：Batch acquisition（q-EI / q-TS）
- [第15章](./chapter-15.md)（planned）：BO 一般失敗 + Agentic 特有失敗

### vol-04 との連携

- vol-04 第12章：one-shot Bayesian D/A-optimality（§7.8 で boundary 明示）

### 外部参考

- Frazier, P. I., *A Tutorial on Bayesian Optimization*, arXiv:1807.02811, 2018 — 各 acquisition の実務比較
- Srinivas et al., "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design", ICML 2010 — UCB の理論保証、$\beta_t$ scheduling
- Wang & Jegelka, "Max-value Entropy Search for Efficient Bayesian Optimization", ICML 2017 — MES 原論文
- Hernández-Lobato et al., "Predictive Entropy Search for Efficient Global Optimization", NeurIPS 2014 — PES 原論文
- BoTorch Documentation — `qLogExpectedImprovement`, `qKnowledgeGradient`, `qMaxValueEntropy`, `qUpperConfidenceBound`

> [!NOTE]
> 次章（第8章）では、GP の限界（高次元・非定常・離散混在）に対処する **代替 surrogate**（Random Forest / BNN / Deep Kernel Learning）を扱い、本章 §7.7 の外挿検知合成基準を BNN posterior と RF disagreement に拡張します。
