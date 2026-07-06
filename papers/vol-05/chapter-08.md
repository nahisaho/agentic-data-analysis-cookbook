# 第8章　Surrogate の代替と拡張 — Random Forest / BNN / Deep Kernel

> [!IMPORTANT]
> **本章の到達目標**
>
> 1. GP surrogate の限界（高次元・離散混在・非定常）を Skill 契約の言葉で特定できる
> 2. Random Forest / Bayesian Neural Network / Deep Kernel Learning の 3 代替 surrogate を判断表から選べる
> 3. **BNN / RF 使用時の外挿検知**（第7章 §7.7 の合成基準を BNN posterior / RF disagreement に拡張）を Skill 契約に書ける
> 4. `surrogate_model_family` enum を各代替に対応させる
>
> **本章で扱わないこと**
>
> - GP kernel（第6章）
> - Acquisition function（第7章）
> - 多目的 / 制約 / 階層 GP（第9-11章）
> - 深層モデルの学習理論（vol-03 の範囲）

---

## 8.1 GP の限界と代替 surrogate の必要性

第6章の GP は少ないデータ、滑らかな関数、低次元入力で強力ですが、次の状況では代替を検討します：

| GP の限界 | 症状 | 推奨代替 |
|---|---|---|
| **高次元入力**（$d \geq 20$）| length scale の推定不安定、$O(N^3)$ の scaling | RF / DKL / BNN |
| **離散/組合せ空間の広大な探索** | kernel design が困難、product/additive でも表現不足 | RF、または第12章 BOCS |
| **非定常性**（領域ごとに滑らかさ違う）| 単一 kernel では表現不能 | BNN / DKL |
| **カテゴリカル変数が多数** | Hamming/Mixed でも計算コスト大 | RF |
| **観測ノイズが入力依存**（heteroskedastic）| 単純 GaussianLikelihood 不適 | BNN（predictive variance が入力依存）|

> [!NOTE]
> 「GP が動かない」ときの第一選択は **RF surrogate**——実装が単純、離散/カテゴリ混在に強く、外挿検知（predictor disagreement）が自然に組み込める。BNN / DKL は「関数形が明らかに非定常」または「深層特徴を BO 入力にしたい」場合の選択。

---

## 8.2 `surrogate_model_family` enum の拡張

第5章 §5.2 と第6章 §6.2 で確立した enum を本章で拡張：

| enum 値 | 対応章 | 用途 |
|---|---|---|
| `single_task_gp` | 第6章 | 単一装置・低〜中次元・滑らか |
| `fixed_noise_gp` | 第6章 | 観測ノイズ既知 |
| `heteroskedastic_gp` | 第6章 | 入力依存ノイズ（`FixedNoiseGaussianLikelihood` + noise model）|
| **`random_forest`** | **本章 §8.3** | 高次元・離散・カテゴリカル |
| **`bayesian_neural_network`** | **本章 §8.4** | 非定常・大規模データ |
| **`deep_kernel_learning`** | **本章 §8.5** | 深層特徴 × GP、非定常 |
| `multi_task_gp` | 第11章 | 複数装置・タスク間 pooling |
| `hierarchical_gp` | 第11章 | 装置差の hierarchical prior |

Skill 契約では：

```yaml
surrogate_model_family: "random_forest"  # 本章で追加された 3 値のいずれか
```

に加えて、代替 surrogate 固有のサブフィールドを **既存 `kernel_spec` の内部**に格納します。

> [!IMPORTANT]
> **`kernel_spec` の意味拡張**（本章で明示的に broadenする）：Ch6 では kernel spec そのもの、本章の RF/BNN/DKL では **surrogate internal specification**（surrogate 固有の内部設定）としても解釈します。field 名は Ch5 §5.2 の 21-field 契約を保つため変更せず、意味を「GP なら kernel spec、非 GP なら surrogate config」と再定義します。将来的な v0.3 では `surrogate_spec.kernel` へリネームを検討。

---

## 8.3 Random Forest surrogate

### 8.3.1 骨格

- **予測**：$N_{\text{tree}}$ 個の decision tree の平均 $\hat{\mu}(x) = \frac{1}{N_{\text{tree}}} \sum_t f_t(x)$
- **不確実性**：tree 間の disagreement $\hat{V}_{\text{disagree}}(x) = \text{Var}_t[f_t(x)]$（**epistemic novelty/disagreement score**）
- **強み**：離散/カテゴリカル/連続混在に強い、外れ値耐性、hyperparameter が少ない
- **弱み**：滑らかさが仮定できない場合の外挿が不安定、非対称な不確実性を扱いにくい

> [!WARNING]
> **RF の tree disagreement は epistemic novelty score であり、calibrated predictive variance ではない**。GP posterior variance のような aleatoric + epistemic の分解を持たず、また NLPD を計算するには explicit な predictive distribution の仮定（Gaussian noise model、conformal interval、OOB residual model 等）が別途必要です。外挿検知（§8.6）には有用でも、CI の解釈は独立に校正すること。

### 8.3.2 実装（scikit-learn）

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,   # sequential_seed_provenance に紐付け
)
model.fit(X, y)

# 予測平均と分散（tree disagreement）
def rf_predict_with_variance(model, X_test):
    per_tree = np.stack([t.predict(X_test) for t in model.estimators_], axis=0)
    mu = per_tree.mean(axis=0)
    var = per_tree.var(axis=0)
    return mu, var
```

### 8.3.3 Skill 契約

```yaml
surrogate_model_family: "random_forest"
kernel_spec:
  # RF は kernel を持たないが、契約構造の一貫性のため kernel_spec 配下に置く
  random_forest:
    n_estimators: 200
    max_depth: null
    min_samples_leaf: 2
    max_features: "sqrt"
    bootstrap: true
  variance_source: "tree_disagreement"  # 外挿検知の分散源（§8.6）
```

---

## 8.4 Bayesian Neural Network surrogate

### 8.4.1 骨格

- **モデル**：ニューラルネット重み $w$ に prior を置き、posterior $p(w | \mathcal{D})$ を推論
- **予測**：$p(y | x, \mathcal{D}) = \int p(y | x, w) p(w | \mathcal{D}) dw$
- **不確実性**：posterior 上のサンプル平均と分散
- **推論手法**：変分推論（VI）、MC Dropout、Deep Ensembles、SGLD
- **強み**：非定常、大規模データ、任意の入力表現（画像・グラフ特徴も可能）
- **弱み**：推論コスト大、posterior calibration の検証が必要、hyperparameter 多数

### 8.4.2 実装スケッチ（MC Dropout）

```python
import torch
import torch.nn as nn

class BNN(nn.Module):
    def __init__(self, d_in, d_hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )
    def forward(self, x):
        return self.net(x)

def bnn_predict_with_variance(model, X_test, n_samples=50):
    was_training = model.training
    model.train()   # dropout を推論時にも有効化
    # 注意: BatchNorm を含むモデルでは、Dropout レイヤーのみ .train()、
    # BatchNorm は .eval() に保つ（training-state 混在の副作用を防ぐ）
    with torch.no_grad():
        preds = torch.stack([model(X_test) for _ in range(n_samples)], dim=0)
    model.train(was_training)  # 元の状態に復帰
    mu = preds.mean(dim=0).squeeze(-1)
    var = preds.var(dim=0).squeeze(-1)
    return mu, var
```

### 8.4.3 Skill 契約

```yaml
surrogate_model_family: "bayesian_neural_network"
kernel_spec:
  bayesian_neural_network:
    inference_method: "mc_dropout"   # mc_dropout | deep_ensembles | variational | sgld
    architecture:
      hidden_layers: [64, 64]
      activation: "relu"
    prior:
      # VI / SGLD の場合
      distribution: "normal"
      params: { mean: 0.0, std: 0.1 }
    posterior_samples: 50           # 推論時のサンプル数
    dropout_rate: 0.1               # mc_dropout の場合
  variance_source: "posterior_samples"  # 外挿検知の分散源（§8.6）
```

> [!WARNING]
> **BNN の posterior calibration** は Skill 契約に検証を含めるべき——`calibration_check` として holdout 上で「予測 95% CI が実測を 95% カバーするか」を測定し、$\pm 10\%$ 以上乖離したら warning。詳細は第15章の失敗パターン。

---

## 8.5 Deep Kernel Learning (DKL)

### 8.5.1 骨格

- **モデル**：$k_{\text{DKL}}(x, x') = k_{\text{base}}(g_\phi(x), g_\phi(x'))$
  - $g_\phi$：深層ニューラルネットで学習された特徴抽出関数
  - $k_{\text{base}}$：Matérn 5/2 などの通常 kernel（第6章）
- **強み**：GP の理論的性質（posterior 計算、不確実性）を保ちつつ、深層表現の柔軟性
- **弱み**：$\phi$ の推定が難しい、少データでは overfit、hyperparameter 最適化が不安定

### 8.5.2 Skill 契約

```yaml
surrogate_model_family: "deep_kernel_learning"
kernel_spec:
  deep_kernel:
    feature_extractor:
      hidden_layers: [64, 32]
      activation: "relu"
      output_dim: 8                # 抽出された特徴空間の次元
    base_kernel:
      type: "matern52"
      ard: true
      length_scale_prior:
        distribution: "gamma"
        params: { concentration: 3.0, rate: 6.0 }
  variance_source: "gp_posterior"  # DKL は GP posterior を持つ
```

DKL の実装は GPyTorch の `gpytorch.models.deep_kernel_learning` が参考、詳細は付録 B。

---

## 8.6 BNN / RF 使用時の外挿検知（第7章 §7.7 の拡張、HRD schema v0.2）

第7章 §7.7 の HRD schema v0.1 は GP 前提（length scale $\ell$ を使う）。本節で **schema v0.2** として、`surrogate_family_adapter` と `scale_source` の 2 サブフィールドを **`hallucinated_recommendation_detection.operational_definition` の内部**に追加します（Ch5 §5.2 の 21-field 契約は変更しない）。

**用語の注意**：以下の $D_{\text{Mah}}$ は共分散行列の非対角要素を無視した **diagonal Mahalanobis analog / standardized nearest-neighbor distance** です。相関の強い連続変数や categorical/one-hot 特徴では別処理が必要（§8.6.4 参照）。

### 8.6.1 対応表

| 判定 | GP | RF | BNN | DKL |
|---|---|---|---|---|
| 予測分散閾値 ($V_{\text{norm}}$) | posterior variance | tree disagreement score | posterior sample variance | GP posterior variance（$g_\phi$ 空間）|
| Mahalanobis analog ($D_{\text{Mah}}$) | GP length scale で正規化 | **学習データ上の per-dim std で正規化**（連続のみ）| **per-dim std で正規化**（連続のみ）| base-kernel length scale in learned $g_\phi(x)$ 空間 |
| Length scale 越え ($L_{\text{ratio}}$) | GP length scale | **per-dim std** | **per-dim std** | base-kernel length scale in learned $g_\phi(x)$ 空間 |

**RF / BNN の場合、length scale の代わりに `training_scale_per_dim`（学習データの標準偏差）を使う** ——スケールを clamp する仕組みは同じ（§7.7.4）。

### 8.6.2 拡張された Skill 契約（HRD schema v0.2）

```yaml
hallucinated_recommendation_detection:
  version: "v0.2"                                   # schema bump: v0.1 → v0.2
  declared: true
  detection_criteria_ref: "vol-05:ch08:hrd_surrogate_adapter_v0_2"
  surrogate_family_adapter: "random_forest"         # v0.2 新設。GP 以外の場合に必須
  operational_definition:
    logic: "OR"
    scale_source:                                   # v0.2 新設
      type: "training_scale_per_dim"                # gp_lengthscale | training_scale_per_dim | dkl_feature_lengthscale
      clamp_min_ratio: 0.05                         # 探索範囲幅の下限比
      clamp_max_ratio: 5.0                          # 探索範囲幅の上限比
    criteria:
      - name: "variance_ratio"
        formula: "variance(x_cand) / median(variance(X_obs))"
        variance_source: "tree_disagreement"        # v0.2 新設: tree_disagreement | posterior_samples | gp_posterior
        threshold: 4.0
      - name: "mahalanobis_distance"                # 実質は diagonal standardized-distance
        formula: "min_i sqrt(sum_j ((x_cand_j - x_ij) / scale_j_eff)^2)"
        threshold: 3.0
      - name: "length_scale_ratio"
        formula: "min_i max_j |x_cand_j - x_ij| / scale_j_eff"
        threshold: 1.5
  action_on_flag: "return_needs_human_review"
  logged_events_field: "hallucination_events"
```

### 8.6.3 RF disagreement の calibration

RF の tree disagreement は GP posterior variance と直接比較できません（**semantic が違う** — §8.3.1）。以下の calibration を Skill 契約に含めます：

- **conformal interval** または **OOB residual model** で explicit な predictive distribution を仮定
- **holdout NLPD**：仮定した predictive distribution 下で $-\log p(y_{\text{true}} | x)$ の平均を測定、閾値超えで warning
- **coverage check**：holdout 95% CI が実測を 95% カバーするか、`coverage_pct` を Group C `surrogate_diagnostics` に記録

### 8.6.4 categorical/相関変数への対応

Diagonal Mahalanobis analog は連続かつ低相関の特徴を前提とします。以下は adapter レベルで扱う：

- **連続かつ相関強**：shrinkage covariance か PCA-whitened distance に切替
- **categorical / one-hot**：**Hamming または Gower 距離**を使う（per-dim std 分数は無意味）
- **zero-variance dim**：clamp か explicit ignore
- **mixed**：連続部分に per-dim std、カテゴリ部分に Hamming、両者の加算（Gower に相当）

`scale_source.type` に将来 `gower_mixed` や `whitened_pca` を追加可能な設計にしておきます。

---

## 8.7 Surrogate 選択判断表（Ch6 §6.8 との統合）

```
[Q1] 入力次元 d は？
  ├─ d ≤ 5  → 第6章 GP（Matérn 5/2 + ARD）
  ├─ 6 ≤ d ≤ 15 → 第6章 GP（additive kernel も検討）or 本章 DKL
  ├─ 16 ≤ d ≤ 50 → 本章 DKL / BNN
  └─ d ≥ 50 → 本章 BNN（RF は疎な特徴で困難）

[Q2] 離散・カテゴリカルが支配的？
  ├─ YES → 本章 RF（第一選択）
  └─ NO  → 次へ

[Q3] 目的関数が明確に非定常（領域で滑らかさが違う）？
  ├─ YES → 本章 BNN / DKL
  └─ NO  → 第6章 GP

[Q4] 観測ノイズが入力依存？
  ├─ YES → 本章 BNN（自然に heteroskedastic）or 第6章 heteroskedastic_gp
  └─ NO  → 通常の GP / RF

[Q5] Human 承認の解釈しやすさを優先？
  ├─ YES → 第6章 GP（不確実性が最も直感的）
  └─ NO  → 本章の代替も可
```

**迷ったら GP → RF → BNN の順**で試すのが実務的。

---

## 8.8 Skill 契約チェックリスト

- [ ] `surrogate_model_family` が §8.2 の enum から明示的に選ばれている
- [ ] `kernel_spec` の内部に対応する詳細フィールド（random_forest / bayesian_neural_network / deep_kernel）が入っている
- [ ] `variance_source` が明示されている（tree_disagreement / posterior_samples / gp_posterior）
- [ ] GP 以外を選んだ場合、`hallucinated_recommendation_detection.surrogate_family_adapter` に対応する adapter 名が入っている
- [ ] `scale_source.type` が surrogate に応じた値（training_scale_per_dim など）になっている
- [ ] BNN / DKL の場合、`calibration_check`（holdout NLPD, coverage）が Group C `surrogate_diagnostics` に記録される仕組みがある
- [ ] `retraining_policy` が代替 surrogate の推論コストと整合している（BNN は every_iteration が高コスト、threshold_triggered を検討）

---

## 8.9 章末演習

**問 1**：入力次元 25、カテゴリカル 5 次元、観測 80 点のケース。§8.7 のフローで surrogate は？

**問 2**：RF surrogate を選んだ場合、第7章 §7.7 の 3 判定はどう修正しますか？

**問 3**：BNN の predictive variance が「観測点上でも 0 に近い」現象が起きた。原因の候補を 2 つ挙げてください。

**問 4**：DKL を使うと外挿検知の $D_{\text{Mah}}$ / $L_{\text{ratio}}$ は $g_\phi(x)$ 空間で計算します。なぜ元の入力空間ではダメか、1 段落で説明してください。

**問 5**：`surrogate_model_family` を Skill 実行中に切り替えるのは第3章 §3.5 のどの逸脱に相当しますか？

---

## 8.10 参考資料

### 本書内

- [第5章](./chapter-05.md) §5.2：`surrogate_model_family` の初出（enum）
- [第6章](./chapter-06.md) §6.2 / §6.8：GP surrogate の kernel_spec 詳細、選択判断表
- [第7章](./chapter-07.md) §7.7：外挿検知 operational 定義（GP 前提）、本章 §8.6 で拡張
- [第11章](./chapter-11.md)（planned）：Multi-task / hierarchical GP
- [第12章](./chapter-12.md)（planned）：BOCS（組合せ空間 BO）と RF の関係
- [第15章](./chapter-15.md)（planned）：surrogate calibration の失敗パターン

### vol-04 との連携

- vol-04 第11章：Kriging surrogate（本章 GP の一括版）

### vol-03 との連携

- vol-03 第7章：深層特徴学習（DKL の $g_\phi$ の基礎）
- vol-03 第9章：BNN の推論手法（VI, MC Dropout, Deep Ensembles）

### 外部参考

- Breiman, L., "Random Forests", Machine Learning 2001 — RF 原論文
- Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016 — MC Dropout
- Wilson et al., "Deep Kernel Learning", AISTATS 2016 — DKL 原論文
- Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", NeurIPS 2017

> [!NOTE]
> 次章（第9章）では、多目的 BO を Skill 化します。qEHVI（expected hypervolume improvement）と Pareto front、「エージェントが勝手に目的関数をスカラー化しない」契約を扱います。
