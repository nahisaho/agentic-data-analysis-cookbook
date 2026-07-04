# 第11章　階層モデル：反復測定・ロット差・測定誤差

> [!IMPORTANT]
> **本章の到達目標**
>
> - **完全プーリング / 完全非プーリング / 部分プーリング**の 3 体制を区別し、材料研究のどの場面でどれを選ぶかを説明できる
> - **反復測定モデル**（試料 × 繰り返し）を PyMC で書ける
> - **ロット差モデル**（試料 × ロット、切片ランダム効果）を書け、`sigma_lot` の事後分布から**「ロット差は無視できるか」**を判断できる
> - **測定誤差モデル**（観測 = 真値 + 誤差）で、**真値を潜在変数**として扱える
> - **中心化 vs 非中心化パラメータ化**の違いを理解し、divergences が出たときに reparameterize できる
> - 階層モデルで **NumPyro/JAX を推奨バックエンドとして使う設定**を組める
> - Advanced Capstone（第13章）に向けた **階層モデル Skill** の provenance テンプレを完成できる

## 本章で扱わないこと

- **完全な混合効果モデル理論**（統計学教科書）：本書はエージェント時代の実装ワークフローに絞る
- **多変量ランダム効果 / LKJ prior**：スカラのランダム効果に絞る（相関構造は付録の参考文献に委ねる）
- **時系列の階層モデル / 状態空間**：本章のスコープ外
- **カテゴリカル反応の階層モデル**：連続反応（回帰系）に絞る
- **収束診断の判断論**：第12章
- **失敗パターン集**：第14章

---

## 11.1 なぜ階層モデルか：材料研究の 3 体制

材料研究のデータは、**ネストされた構造**を持つのが普通です：

- 試料 → その中の繰り返し測定
- 合成バッチ（ロット） → その中の試料
- 装置 → その装置で測った試料
- 研究室 → その研究室の装置

このネスト構造を無視して**「独立同分布」として扱うと**、頻度論では標準誤差の過小推定、Bayesian では過信した CrI が出ます。

### 3 つの選択肢

同じデータに対して、次の 3 つのモデリング体制があります。

| 体制 | モデル | 情報の借用 | 使う場面 |
|---|---|---|---|
| **完全プーリング**（complete pooling） | グループを無視、全データを 1 つの母集団と仮定 | すべて共有 | ロット差・装置差が明確に無視できるとき |
| **完全非プーリング**（no pooling） | グループごとに独立にモデル化 | しない | 各グループのデータが十分多く、共通の母構造がないとき |
| **部分プーリング**（partial pooling） | グループ固有効果 + 母集団分布 | 母集団経由で借用 | データが不均等 or 少ないグループがあるとき（実務のデフォルト） |

**部分プーリングは階層モデルの中核概念**です。データが少ないグループは全体平均に「縮む（shrinkage）」ため、極端な推定を避けられます。

### 材料研究の判断表

| 場面 | 推奨 | 理由 |
|---|---|---|
| 単一装置・単一ロット・n=100+ | 完全プーリング | 階層構造がそもそもない |
| 5 ロット × 20 試料/ロット、ロット差の分散に関心 | **部分プーリング** | `sigma_lot` の事後分布を意思決定に使える |
| 3 装置 × 装置ごとに 200 試料 | 完全非プーリング or 部分プーリング | データが多ければ非プーリング、少ないなら部分プーリング |
| 10 研究室 × 各研究室で 3 試料のみ | **部分プーリング** | データ少 + 情報借用が必要 |

---

## 11.2 Partial pooling の直感

部分プーリングは、**「各グループ固有の値」と「全体の平均」の折衷**を、データ量に応じて自動調整します。

$$
\alpha_g \sim \mathrm{Normal}(\mu_\alpha, \sigma_\alpha), \quad g = 1, \dots, G
$$

- $\mu_\alpha$：全体平均（母集団パラメータ、ハイパーパラメータ）
- $\sigma_\alpha$：グループ間のばらつき（母集団パラメータ）
- $\alpha_g$：グループ $g$ 固有の値

**縮み（shrinkage）の起き方**：

- グループ $g$ のデータが多い → $\alpha_g$ は自身のデータに強く引かれる
- グループ $g$ のデータが少ない → $\alpha_g$ は $\mu_\alpha$ に強く引かれる
- $\sigma_\alpha$ が小さい → 全グループが $\mu_\alpha$ に近い
- $\sigma_\alpha$ が大きい → 各グループが独立に近い

**このメカニズムを手作業でチューニングするのは非現実的**です。階層モデルは Bayesian の枠組みで自動的にこれを行います [[11-1]](#ref-11-1) [[11-2]](#ref-11-2)。

---

## 11.3 反復測定モデル（試料 × 繰り返し）

### 場面

同じ試料に対して、同じ装置で 3〜5 回測定を繰り返す。**繰り返し内のばらつき**（測定精度）と、**試料間のばらつき**（試料の真の分散）を分けたい。

### モデル

$$
y_{s, r} \sim \mathrm{Normal}(\alpha_s, \sigma_{\text{obs}}), \quad \alpha_s \sim \mathrm{Normal}(\mu, \sigma_{\text{sample}})
$$

- $y_{s,r}$：試料 $s$ の $r$ 回目の測定値
- $\alpha_s$：試料 $s$ の真値（潜在）
- $\sigma_{\text{obs}}$：測定装置のノイズ
- $\mu, \sigma_{\text{sample}}$：試料集団の平均と分散

```python
import pymc as pm
import numpy as np

# 合成階層データ（data/synthetic-hierarchy/ から）
# sample_id: 各測定がどの試料か、shape=(N,)
# y_obs:     測定値、shape=(N,)

n_samples = len(np.unique(sample_id))

with pm.Model(coords={"sample": np.arange(n_samples),
                       "obs":    np.arange(len(y_obs))}) as repeated:
    # 母集団パラメータ
    mu           = pm.Normal("mu", mu=0.0, sigma=5.0)
    sigma_sample = pm.HalfNormal("sigma_sample", sigma=2.0)
    sigma_obs    = pm.HalfNormal("sigma_obs",    sigma=1.0)

    # 試料ごとの真値（中心化パラメータ化：後で非中心化に変更）
    alpha = pm.Normal("alpha", mu=mu, sigma=sigma_sample, dims="sample")

    # 尤度
    y = pm.Normal("y", mu=alpha[sample_id], sigma=sigma_obs,
                  observed=y_obs, dims="obs")

    idata = pm.sample(
        draws=1000, tune=1000, chains=4,
        nuts_sampler="numpyro",         # §10.9 の準備を活用
        target_accept=0.95,             # 階層モデルは 0.90 だと divergences が出やすい
        random_seed=42,
    )
```

### 読み方

- `sigma_obs`：装置ノイズ（繰り返し測定 3〜5 回で 1 試料の平均を出す価値の判断材料）
- `sigma_sample`：試料集団の真の分散
- `sigma_obs / sigma_sample` 比：**この比が 1 に近ければ、繰り返し測定がノイズに埋もれるサイン**。試料を増やす方が情報効率が高い

> [!TIP]
> **設計への逆流**：試料 20 個 × 繰り返し 5 回 = 100 測定と、試料 50 個 × 繰り返し 2 回 = 100 測定は、**情報量が異なります**。反復測定モデルを走らせると、どちらの設計が母平均・母分散の推定に効率的かを事後的に評価できます（Advanced Capstone の実例、第13章）。

---

## 11.4 ロット差モデル（試料 × ロット）

### 場面

合成バッチ（ロット）ごとにわずかな差がある可能性がある。**「ロット差は無視できるか？」**を Skill で判定したい。

### モデル

$$
y_{s} \sim \mathrm{Normal}(\alpha_{l(s)}, \sigma_{\text{obs}}), \quad \alpha_l \sim \mathrm{Normal}(\mu, \sigma_{\text{lot}})
$$

- $l(s)$：試料 $s$ が属するロット
- $\sigma_{\text{lot}}$：ロット間ばらつき（**関心量**）

```python
with pm.Model(coords={"lot":    np.arange(n_lots),
                       "sample": np.arange(len(y_obs))}) as lot_effect:
    mu        = pm.Normal("mu", mu=0.0, sigma=5.0)
    sigma_lot = pm.HalfNormal("sigma_lot", sigma=2.0)
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

    # ロットごとの平均
    alpha_lot = pm.Normal("alpha_lot", mu=mu, sigma=sigma_lot, dims="lot")

    y = pm.Normal("y", mu=alpha_lot[lot_id], sigma=sigma_obs,
                  observed=y_obs, dims="sample")

    idata_lot = pm.sample(
        draws=1000, tune=1000, chains=4,
        nuts_sampler="numpyro",
        target_accept=0.95, random_seed=42,
    )
```

### 「ロット差は無視できるか」判定

`sigma_lot` の**事後分布**を見ます：

- **CrI 95% が実務閾値（例：0.05 eV）以下** → ロット差は実務上無視できる（完全プーリングに簡略化可）
- **CrI 95% が閾値を跨ぐ** → 判定保留、追加ロットで検証
- **CrI 95% が閾値以上** → **ロット差は無視できない**、階層モデルを維持

> [!IMPORTANT]
> **`sigma_lot ≈ 0` を「無視できる」と誤読しない**：Bayesian では**事後分布そのもの**を判断材料にします。「点推定が 0.02 だから小さい」は不十分。閾値ベースで **確率的な意思決定**を行い、`P(sigma_lot > threshold | data)` を Skill 出力に含めます：
>
> ```python
> threshold = 0.05  # eV
> prob_significant = (idata_lot.posterior["sigma_lot"] > threshold).mean().item()
> ```
>
> これは第9章 `uncertainty_scheme.posterior_quantity: threshold_probability` の実装。

---

## 11.5 測定誤差モデル（観測 = 真値 + 誤差）

### 場面

装置校正のばらつき・環境ノイズ・オペレーター差などで、**観測値は真値の推定であり、真値自体が不確か**。Errors-in-Variables 型の問題。

### モデル

$$
y_i = x_i^{\text{true}} + \epsilon_i, \quad x_i^{\text{true}} \sim \mathrm{Normal}(\mu, \sigma_x), \quad \epsilon_i \sim \mathrm{Normal}(0, \sigma_{\text{obs}})
$$

観測 $y_i$ とは別に、$x_i^{\text{true}}$ を**潜在変数**として推定します。第9章 §9.3 で挙げた「観測誤差の不確かさ」を実装するのがこの型です。

```python
with pm.Model(coords={"obs": np.arange(len(y_obs))}) as meas_error:
    mu         = pm.Normal("mu", mu=0.0, sigma=5.0)
    sigma_x    = pm.HalfNormal("sigma_x", sigma=2.0)          # 真値の分布
    sigma_obs  = pm.HalfNormal("sigma_obs", sigma=1.0)         # 装置ノイズ

    # 潜在真値：観測数と同じ次元
    x_true = pm.Normal("x_true", mu=mu, sigma=sigma_x, dims="obs")

    # 観測モデル
    y = pm.Normal("y", mu=x_true, sigma=sigma_obs,
                  observed=y_obs, dims="obs")

    idata_me = pm.sample(
        draws=1000, tune=1000, chains=4,
        nuts_sampler="numpyro",
        target_accept=0.95, random_seed=42,
    )
```

### 識別性の注意

**`sigma_x` と `sigma_obs` は識別可能である必要**があります。両方が推定対象で、繰り返し測定などの追加情報がないと、**モデルは 2 つの分散を区別できません**（`sigma_x^2 + sigma_obs^2` は決まるが分解できない）。

| 状況 | 識別可能か | 対応 |
|---|---|---|
| 装置ノイズ `sigma_obs` の外部情報あり（メーカー仕様など） | ✅ | `sigma_obs` を固定 or 強い prior |
| 繰り返し測定がある | ✅ | 反復測定モデルで両分散を分離 |
| 単発測定のみ、外部情報なし | ❌ | 測定誤差モデルは適用不可、完全プーリングに戻す |

> [!WARNING]
> **識別不能な測定誤差モデルは、divergences・低 ESS を大量発生させます**。§11.6 の非中心化パラメータ化で解決すると思い込みやすいですが、根本原因はモデル構造です。まず `sigma_obs` に**強い情報事前分布**を置けるか検討します。

---

## 11.6 中心化 vs 非中心化パラメータ化：divergences への処方箋

### Neal's funnel と divergences

階層モデルで最も頻発する問題が **funnel geometry** です。$\sigma_\alpha$ が小さいとき、$\alpha_g$ の事後分布が漏斗状になり、NUTS は探索に失敗して **divergences** を大量に出します [[11-3]](#ref-11-3) [[11-4]](#ref-11-4)。

### 中心化（centered）パラメータ化

これまで書いてきた形式：

```python
alpha = pm.Normal("alpha", mu=mu, sigma=sigma_sample, dims="sample")
```

$\sigma_\alpha$ が中〜大のときは**この形が効率的**。しかし $\sigma_\alpha$ が小さいと funnel が発生します。

### 非中心化（non-centered）パラメータ化

**標準正規からの再パラメータ化**：

$$
\alpha_g = \mu + \sigma_\alpha \cdot z_g, \quad z_g \sim \mathrm{Normal}(0, 1)
$$

```python
with pm.Model(coords={"sample": np.arange(n_samples),
                       "obs":    np.arange(len(y_obs))}) as repeated_nc:
    mu           = pm.Normal("mu", mu=0.0, sigma=5.0)
    sigma_sample = pm.HalfNormal("sigma_sample", sigma=2.0)
    sigma_obs    = pm.HalfNormal("sigma_obs",    sigma=1.0)

    # 非中心化：標準正規オフセット
    alpha_offset = pm.Normal("alpha_offset", mu=0.0, sigma=1.0, dims="sample")
    alpha        = pm.Deterministic(
        "alpha", mu + sigma_sample * alpha_offset, dims="sample",
    )

    y = pm.Normal("y", mu=alpha[sample_id], sigma=sigma_obs,
                  observed=y_obs, dims="obs")
```

### いつどちらを使うか

| 状況 | 推奨パラメータ化 | 理由 |
|---|---|---|
| $\sigma_\alpha$ の事後が小さい（強い情報借用） | **非中心化** | funnel 回避 |
| $\sigma_\alpha$ の事後が大きい（グループ差大） | 中心化 | posterior が漏斗状にならない |
| データが少ないグループがある | **非中心化** | 情報借用時の探索効率 |
| データが多く、事前に $\sigma_\alpha$ が中〜大とわかる | 中心化 | 実装が読みやすい |

**実務のデフォルト**：まず中心化で書く → divergences が出たら非中心化に切り替える。**両方を Skill バリアントとして持つ**のが安全策です。

> [!IMPORTANT]
> **非中心化にすれば必ず解決するわけではない**：識別性の問題（§11.5）、prior の広すぎ、data の情報不足による本質的な難しさは reparameterization では解決しません。第12章の実務判断論で扱います。

---

## 11.7 NumPyro をバックエンドに使う

### なぜ NumPyro

階層モデルは中〜大規模になりやすく、**JIT コンパイル + XLA による高速化**が効きます。同じモデルで、PyTensor バックエンドの数倍〜数十倍速くなることも珍しくありません。

### 最小設定

第10章 §10.9 で凍結した手順：

```python
import jax
jax.config.update("jax_enable_x64", True)   # 必ず先頭で 1 回

import pymc as pm

with repeated_nc:
    idata = pm.sample(
        draws=1000, tune=1000, chains=4,
        nuts_sampler="numpyro",     # ここだけ変える
        target_accept=0.95,
        random_seed=42,
    )
```

### provenance への追加記録

第10章の `backend_config` に階層モデル特有の情報を追加：

```yaml
backend_config:
  primary:
    backend:                numpyro
    jax_x64:                true
    jax_platform:           cpu             # jax.devices() の結果を要記録
    numpyro_chain_method:   parallel        # or sequential / vectorized
  fallback:
    backend:                pytensor
    fallback_reason_logged: true
  parameterization:
    strategy:               non_centered    # centered / non_centered
    switched_from:          centered        # divergences で切り替えた記録
    switch_reason:          "n_divergences=143 with centered"
```

**`parameterization.switched_from`** を書く理由：Skill 認定後に他者が読んだとき、「なぜ非中心化を選んだか」の履歴がないと、モデル修正時に判断が引き継がれません。

---

## 11.8 応用例：装置間差

### 場面

3〜5 装置で同じ試料を測る。**装置間差**が試料差やロット差より大きいと、Skill の**適用範囲**を装置ごとに切り分ける必要があります。

### モデル（試料 × 装置の 2 階層）

$$
y_{s,i} \sim \mathrm{Normal}(\alpha_s + \gamma_i, \sigma_{\text{obs}}), \quad \alpha_s \sim \mathrm{Normal}(\mu_\alpha, \sigma_{\text{sample}}), \quad \gamma_i \sim \mathrm{Normal}(0, \sigma_{\text{inst}})
$$

- $\gamma_i$：装置 $i$ のオフセット（sum-to-zero 制約 or 弱い prior で識別）
- $\sigma_{\text{inst}}$：装置間ばらつき

```python
with pm.Model(coords={"sample": np.arange(n_samples),
                       "inst":   np.arange(n_instruments),
                       "obs":    np.arange(len(y_obs))}) as inst_effect:
    mu_alpha     = pm.Normal("mu_alpha", mu=0.0, sigma=5.0)
    sigma_sample = pm.HalfNormal("sigma_sample", sigma=2.0)
    sigma_inst   = pm.HalfNormal("sigma_inst",   sigma=1.0)
    sigma_obs    = pm.HalfNormal("sigma_obs",    sigma=1.0)

    # 非中心化
    alpha_offset = pm.Normal("alpha_offset", 0.0, 1.0, dims="sample")
    gamma_offset = pm.Normal("gamma_offset", 0.0, 1.0, dims="inst")
    alpha = pm.Deterministic("alpha",
                             mu_alpha + sigma_sample * alpha_offset, dims="sample")
    gamma = pm.Deterministic("gamma",
                             sigma_inst * gamma_offset, dims="inst")

    y = pm.Normal("y", mu=alpha[sample_id] + gamma[inst_id],
                  sigma=sigma_obs, observed=y_obs, dims="obs")
```

### 装置間差の意思決定

- $\sigma_{\text{inst}}$ の CrI が実務閾値以下 → 装置差は無視、Skill は全装置共通
- 閾値を超える → **装置ごとに別 Skill として認定**、`applicability_domain` に装置ID を明記

これは Skill の**適用範囲仕様**（vol-01 第7章 §7.5）を Bayesian で駆動する典型パターンです。

---

## 11.9 階層モデル Skill の provenance テンプレ

第10章 §10.8 の provenance に、階層モデル特有の情報を追加します。

```yaml
provenance:
  # 第10章のフィールドをすべて継承
  ...

  # 第11章で追加
  hierarchical_structure:
    levels:
      - {name: sample,    n: 20,  data_per_level: 5}
      - {name: lot,       n: 4,   data_per_level: 25}
      - {name: instrument, n: 3,  data_per_level: 33}
    partial_pooling_targets: [alpha, gamma]
    complete_pooling_targets: [mu_alpha]

  parameterization:
    strategy:            non_centered
    switched_from:       centered
    switch_reason:       "n_divergences=143 with centered"
    verified_via:        "prior predictive check + posterior predictive check"

  identifiability_check:
    performed:           true
    method:              "prior predictive check + external calibration data"
    concerns:            []
    reviewed_by:         "PI name"

  shrinkage_summary:
    metric:              "|alpha_group - mu_alpha| / sigma_alpha"
    n_shrunk_groups:     3      # 例：全体平均に強く引かれたグループ数
    groups_with_low_ess: []

  applicability_domain:
    instruments:         [inst-A, inst-B, inst-C]
    lot_range:           [lot-2024-01, lot-2025-12]
    sample_composition:  "半導体単結晶、n=100 まで検証済"
    exclusions:          ["装置 D は sigma_inst の CrI 外、別 Skill 認定必要"]
```

### 合成階層データ

本章の例は **`data/synthetic-hierarchy/`** の合成データを想定しています（第2章 §2.9 で予告）。実データ候補は付録 C。

- **なぜ合成か**：階層構造が制御可能で「真値」が既知、Skill が正しく回収できるかを検証できる
- **提供される階層**：試料 × 繰り返し / 試料 × ロット / 試料 × 装置 / 3 階層フル
- **意図的な難所**：小サンプル群、識別性ギリギリ、divergences を誘発する設定

第13章 Advanced Capstone では、この合成データで **階層モデル Skill の完成を認定**します。

---

## 11.10 章末ワーク

1. **反復測定モデルを 5 分で走らせる**：`data/synthetic-hierarchy/repeated/` から `sample_id` と `y_obs` を読み込み、§11.3 のモデルを NumPyro で回す。`sigma_obs / sigma_sample` 比を計算し、繰り返し数の設計判断を書く
2. **ロット差の意思決定**：§11.4 のモデルで `P(sigma_lot > 0.05)` を計算し、Skill 仕様書の④成功条件に**閾値ベース確率**を書く
3. **識別性チェック**：§11.5 の測定誤差モデルで、`sigma_obs` に**強い prior**（例：`HalfNormal(0.1)`）と**弱い prior**（例：`HalfNormal(5.0)`）で走らせ、`sigma_x` の事後がどう変わるかを比較する
4. **中心化 → 非中心化への切替**：§11.6 の両バリアントで同じデータを走らせ、`n_divergences` を比較。差が出たら `switched_from` を provenance に記録
5. **装置間差の Skill 分割判断**：§11.8 のモデルで `sigma_inst` の 95% CrI を出し、閾値と比較して Skill の `applicability_domain` を決める
6. **NumPyro 高速化の確認**：同じモデルを `pytensor` と `numpyro` で走らせ、`runtime_seconds` と `max_rhat`、`min_ess_bulk` を比較する

---

## 11.11 本章のまとめ

- **3 体制**：完全プーリング / 完全非プーリング / 部分プーリング。実務のデフォルトは部分プーリング
- **反復測定・ロット差・測定誤差**は材料研究の 3 大階層シナリオ。それぞれの `sigma_*` を意思決定に接続する
- **`P(sigma_lot > threshold | data)`** のような**閾値ベース確率**を Skill 出力に含める（`uncertainty_scheme.posterior_quantity: threshold_probability`）
- **測定誤差モデルは識別性が要**。繰り返し測定 or 外部校正情報がないと解けない
- **中心化 vs 非中心化**：divergences が出たら非中心化。両バリアントを持ち、`switched_from` を記録
- **NumPyro を推奨**：`jax_enable_x64` + `nuts_sampler="numpyro"` + `jax.devices()` を凍結
- **装置間差**は Skill の適用範囲決定に直結。`applicability_domain` を Bayesian で駆動
- 階層モデル Skill の provenance には **`hierarchical_structure` / `parameterization` / `identifiability_check` / `shrinkage_summary` / `applicability_domain`** を追加

---

## 参考資料

### 本書内の該当章

- 第2章 §2.9：合成階層データ `data/synthetic-hierarchy/` の位置づけ
- 第4章 §4.3：`task_type: bayesian_inference`
- 第9章：`uncertainty_scheme`（`posterior_quantity: threshold_probability` を本章で活用）
- 第10章 §10.6：非階層の Bayesian（本章はその拡張）
- 第10章 §10.9：JAX/NumPyro 準備（本章のバックエンド）
- 第12章：MCMC 実務判断（divergences の扱い、reparameterization 選択）
- 第13章：Advanced Capstone（本章の階層モデル Skill を統合）
- 付録 A：provenance スキーマ拡張（本章追加フィールドの正式定義）
- 付録 C：階層データの実データ候補

### 外部参考

<a id="ref-11-1">[11-1]</a> Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press. — 階層モデルの実務的教科書
<a id="ref-11-2">[11-2]</a> McElreath, R. (2020). *Statistical Rethinking*, 2nd ed. — partial pooling の直感的解説（Ch 13–14）
<a id="ref-11-3">[11-3]</a> Betancourt, M., & Girolami, M. (2015). Hamiltonian Monte Carlo for Hierarchical Models. In *Current Trends in Bayesian Methodology with Applications*. — funnel geometry と非中心化パラメータ化
<a id="ref-11-4">[11-4]</a> Betancourt, M. (2017). Diagnosing Biased Inference with Divergences. [https://betanalpha.github.io/assets/case_studies/divergences_and_bias.html](https://betanalpha.github.io/assets/case_studies/divergences_and_bias.html) — divergences の実務対処
<a id="ref-11-5">[11-5]</a> Stan Development Team. *Stan User's Guide: Hierarchical Models*. [https://mc-stan.org/docs/stan-users-guide/](https://mc-stan.org/docs/stan-users-guide/) — 対応する Stan 実装（付録 B 参照）
<a id="ref-11-6">[11-6]</a> NumPyro Team. *NumPyro Examples*. [https://num.pyro.ai/en/stable/examples/](https://num.pyro.ai/en/stable/examples/) — JAX バックエンドでの階層モデル実装例
