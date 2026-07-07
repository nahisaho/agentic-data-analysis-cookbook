# 第7章 Normalizing Flow と autoregressive を Skill 化する

> **本章の使い方**
> - **Part II の最後の実装章**です。Ch4 の canonical template（§4.2 骨格、§4.3 13 provenance fields、§4.5 `hallucinatory_composition_detection` 3 態、§4.6 Safety 3-layer、§4.7 禁止フレーズ 10 list、§4.9 9 Skill ID 一覧）を **literal に継承** し、`arim.gen.flow.v0.1` と `arim.gen.autoregressive.v0.1` の 2 Skill を実装します。
> - **教育目的の小規模スクラッチ学習**：ARIM 風 3-4 元系合成データ（数百〜数千サンプル）で単一 GPU 内完結する組成 Flow（RealNVP 骨格）と組成 AR（Transformer decoder 骨格）を、Skill 契約の中で示します。研究本番は本章 scope 外です。
> - **本章の scope**：**Flow の exact likelihood** と **AR の temperature / top-k / top-p sampling** を、Ch4 §4.3.5 `generation_temperature` と Ch6 §6.7 で dict 拡張した `sampling_config` canonical に沿って Skill 化します。生成後の物理制約チェックは Ch8、OOD 検知の完全計算は Ch10、離散候補ランキングは Ch11 に委譲します。
> - **Part II まとめ**：§7.9 で **VAE / Diffusion / Flow / AR の 4 家族選択判断表**（データ量 × 制約表現 × 計算コスト × Skill 契約の複雑さ）を提示します。以降の Ch8-11（スクリーニング／逆設計）は 4 家族いずれかの G ステージ Skill 出力を前提とします。
> - **YAML block は全て `yaml.safe_load` 準拠**：本書 CI で機械検証されます。
>
> **本章の到達目標**
> - Normalizing Flow の骨格（change of variables + coupling layer）を、Ch4 §4.3.1 の `generative_model_family.family = "normalizing_flow"` の canonical field に対応付けて説明できる
> - Autoregressive の骨格（factorized likelihood $p(x) = \prod_i p(x_i \mid x_{<i})$）を、Ch4 §4.3.1 の `generative_model_family.family = "autoregressive"` に対応付け、SMILES-like な組成列生成に落とし込める
> - **Flow が exact likelihood を持つ**利点と、**Diffusion / VAE の ELBO 近似との違い**を説明できる
> - **AR の temperature / top-k / top-p sampling** を、Ch6 §6.7 canonical `sampling_config` の AR 差分として Skill YAML に書ける
> - **`arim.gen.flow.v0.1`** と **`arim.gen.autoregressive.v0.1`** の完全 Skill YAML を、Ch5 §5.7 / Ch6 §6.8 の親骨格を継承して書ける
> - **VAE / Diffusion / Flow / AR** の判断表を使い、ARIM のデータ規模と制約条件から適切な家族を選べる
>
> **本章で扱わないこと**
> - Continuous Normalizing Flow / Flow Matching（本書 scope 外、参考資料に一次情報源を列挙）
> - 大規模言語モデル（LLM）としての AR、材料テキスト表現の完全 taxonomy（SMILES / SELFIES / SLICES 等は付録 C で概説）
> - 生成候補の物理制約チェック（Ch8、`arim.gen.physics_filter.v0.1` F ステージ）
> - OOD 検知の完全計算（Ch10、軸 A/B/C の実装詳細）
> - `arim.gen.candidate_ranking.v0.1` による top-k 選抜（Ch11）
> - Agentic 失敗パターンの taxonomy（Ch14）

---

## 7.1 なぜ Diffusion の後に Flow と AR を扱うか

Ch5（VAE）と Ch6（Diffusion）で **潜在変数モデル 2 系統** を実装したあと、本章で **exact likelihood 系（Flow）** と **明示的な系列生成系（Autoregressive）** を扱う理由は 3 点あります。

### 理由 1：exact likelihood は "分布内判定" の第一原理を与える

VAE の ELBO は $\log p(x)$ の下界、Diffusion の $L_{\text{simple}}$ は VLB の重み付き簡略化であり、いずれも **真の対数尤度に対する近似**です。一方 Normalizing Flow は **change of variables** により

$$
\log p_X(x) = \log p_Z(f(x)) + \log \left| \det \frac{\partial f(x)}{\partial x} \right|
$$

の閉形式で対数尤度を計算できます（$f$ は可逆写像、$p_Z$ は base distribution）。この性質は Ch2 §2.4 で確定した `hallucinatory_composition_detection` の合成規則の **軸 A（Mahalanobis 距離）と軸 B（PCA 再構成誤差）を対数尤度で置換できる**ことを意味します。Ch10 で扱う分布的妥当性判定の理論的基盤として、Flow の exact likelihood は「近似ではなく厳密値で分布内判定できる」参照点を提供します。

### 理由 2：AR は "組成を系列として生成する" 実装 route の代表

材料表現には 2 系統あります：**組成 (continuous simplex on $\mathbb{R}^{n}$)** と **構造 (graph / crystal)**。Ch5-6 で扱った VAE / Diffusion は主に組成 simplex を扱いましたが、**分子表現の SMILES や結晶表現の SLICES は "文字列" として組成・接続を系列化** します。この系列を Transformer decoder で左から右に生成するのが AR family です。§7.4-7.5 では ARIM 風合成データの組成を「元素トークン列 + 化学量論トークン列」として系列化し、教育目的の小規模 AR を実装します。Ch13 の分子逆設計（SMILES 経由）で使う実装 route の骨格を、本章で先取りします。

### 理由 3：4 家族の判断軸を Part II の最後に提示する

Ch5-7 で 4 家族が出揃ったところで、**「自分の ARIM データに対してどの家族を選ぶか」の判断表**を §7.9 で提示します。判断軸は 4 つ：

1. **データ量** — 数百 vs 数千 vs 数万 vs FM 規模
2. **制約表現** — 条件付き生成の柔軟性（CFG のような後付け条件付けが可能か）
3. **計算コスト** — 学習・推論の GPU 時間
4. **Skill 契約の複雑さ** — provenance fields の埋め方、`sampling_config` の canonical レンジ数

この 4 軸で 4 家族を比較することで、**Part II まとめとして Ch8 以降のスクリーニング／逆設計を "どの G ステージ Skill が出力する候補に対して" 適用するか**を明確化します。

### 本章の scope と他章委譲

| 論点 | 本章 §7 | 委譲先 |
|---|---|---|
| Normalizing Flow 骨格（RealNVP 系）と Skill 化 | ✅ | — |
| Autoregressive 骨格（Transformer decoder）と Skill 化 | ✅ | — |
| Flow の exact likelihood と OOD 判定への応用（概説） | ✅ | Ch10 で軸 A/B/C 完全実装 |
| AR の temperature / top-k / top-p の canonical レンジ | ✅ | — |
| VAE / Diffusion / Flow / AR の 4 家族選択判断表 | ✅ (§7.9) | — |
| Continuous Normalizing Flow / Flow Matching | ❌ | 参考資料のみ |
| 大規模 LLM としての AR、SMILES / SELFIES / SLICES 完全 taxonomy | ❌ | 付録 C 概説 |
| 生成候補の物理制約チェック | ❌ | Ch8 |
| 生成候補の surrogate ランキング | ❌ | Ch11 |
| 危険物質 filter list 内容 | ❌ | 付録 C |

---

## 7.2 Normalizing Flow の骨格 — change of variables と coupling layers

### 7.2.1 change of variables

Flow は可逆写像 $f: \mathcal{X} \to \mathcal{Z}$ を使い、複雑な分布 $p_X$ を単純な base distribution $p_Z$（通常は標準正規）に射影します。可逆性から

$$
p_X(x) = p_Z(f(x)) \cdot \left| \det \frac{\partial f(x)}{\partial x} \right|
$$

が得られ、対数を取ると §7.1 冒頭の式になります。学習は **負の対数尤度** を直接最小化します：

$$
\mathcal{L}(\theta) = - \mathbb{E}_{x \sim p_{\text{data}}} \left[ \log p_Z(f_\theta(x)) + \log \left| \det \frac{\partial f_\theta(x)}{\partial x} \right| \right]
$$

**サンプリング** は $z \sim p_Z$、$x = f^{-1}(z)$ の 2 段で完結します（NFE = 1）。Diffusion の DDPM が 1000 ステップ、DDIM が 20-100 ステップだったのに対し、Flow は **サンプリング速度で常に最速**です。

### 7.2.2 coupling layer（RealNVP）

Flow の可逆性とヤコビアン計算可能性を両立させる標準実装が **coupling layer**（Dinh et al., 2017）です。入力 $x \in \mathbb{R}^{n}$ を 2 分割 $x = (x_a, x_b)$ し、

$$
y_a = x_a, \quad y_b = x_b \odot \exp(s_\theta(x_a)) + t_\theta(x_a)
$$

とします。$s_\theta$ / $t_\theta$ はニューラルネット（任意）、$y_a = x_a$ なので逆写像も閉形式：

$$
x_a = y_a, \quad x_b = (y_b - t_\theta(y_a)) \odot \exp(-s_\theta(y_a))
$$

ヤコビアン $\frac{\partial y}{\partial x}$ は下三角ブロック構造となり、$\log \det = \sum_i s_\theta(x_a)_i$ で計算コスト $O(n)$。**coupling layer を交互に $x_a / x_b$ の役割を入れ替えて重ねる**ことで、任意の可逆関数を近似します。RealNVP / Glow / NICE 系はこの延長です。

### 7.2.3 材料組成への適用と Skill 契約の含意

組成 $c \in \mathbb{R}^{n_{\text{elem}}}$ は $\sum_i c_i = 1, c_i \geq 0$ の simplex 上にあります。Flow の base distribution は $\mathbb{R}^{n}$ 上の Gaussian なので、**simplex constraint を直接扱えません**。実装 route は 2 つ：

1. **Route Flow-A**：logit 変換で simplex $\to \mathbb{R}^{n-1}$ に写像してから Flow を学習、サンプリング時に softmax で simplex に戻す
2. **Route Flow-B**：simplex 内在的 Flow（Dirichlet flow, categorical normalizing flow）を使う

本章は教育目的で **Route Flow-A（logit + RealNVP）** に絞ります。§7.3 の実装で具体化します。**Skill 契約の含意**：logit → softmax 復元時に **temperature scaling** を挿入するため、Ch6 §6.7 の `temperature_scaling`（softmax 温度）を Flow でも継承します。

### 7.2.4 Flow は CFG を持たない

Ch6 §6.6 で扱った classifier-free guidance は **noise prediction network $\epsilon_\theta$ に対する加算的な条件強化**でした。Flow では

- 学習時：$\log p_X(x \mid c) = \log p_Z(f_\theta(x; c)) + \log |\det \partial f_\theta / \partial x|$ として条件 $c$ をネットに入力
- サンプリング時：$z \sim p_Z$、$x = f^{-1}(z; c)$ で条件付き生成

となり、**CFG に相当する事後的な "条件強調" 機構は canonical には存在しません**。Skill YAML では `guidance_scale: null` が Flow の canonical 値です（VAE と同型）。

---

## 7.3 組成 Flow の教育的実装

### 7.3.1 データ表現と logit 変換

Ch5 §5.3 と Ch6 §6.4.1 で使った 4 元系組成データ（$n_{\text{elem}} = 4$）を **additive log-ratio (ALR) 変換で $\mathbb{R}^3$ に写像**します：

$$
c_i \mapsto \ell_i = \log \frac{c_i}{c_{n_{\text{elem}}}}, \quad i = 1, \ldots, n_{\text{elem}} - 1
$$

$c_{n_{\text{elem}}}$（reference 元素）を分母に固定することで simplex から $\mathbb{R}^{n_{\text{elem}}-1}$ への **厳密な双射（可逆写像）**が得られます。逆変換は softmax（$\ell$ に $\ell_{n_{\text{elem}}}=0$ を padding してから）で厳密に $c$ を復元します。$c_i = 0$ の候補は学習前に前処理で除外（Ch5 §5.3 継承）：$c_i \geq c_{\min} = 10^{-4}$ を必要条件として **§7.7 Skill YAML の `preprocessing_provenance.composition_min_fraction` に pin し、`training_data_provenance.sha256` が指すデータに対して前処理済みであることを保証**します。ALR の reference 元素選択規則（`alr_reference_element_policy: "last_element_by_sorted_symbol"`）も同 provenance に pin されます。

**Skill 契約の含意**：本章の Flow は **ALR-logit 空間 $\mathbb{R}^{n_{\text{elem}}-1}$ 上の生成モデル**として定義されます。組成空間側の対数尤度が必要な場合（Ch10 での OOD 判定の軸 A に使う場合など）、Jacobian 補正項

$$
\log p_c(c) = \log p_\ell(\ell) - \sum_{i=1}^{n_{\text{elem}}} \log c_i
$$

を加算します（ALR 変換の Jacobian）。§7.7 の `mode: "score"` が返す `log_prob_out` は **ALR-logit 空間の対数尤度** が canonical 出力です（Jacobian 補正は Ch10 側で明示的に付与）。この責務分離により、Skill 側は変換に依存しない exact likelihood を返し、下流の判定 Skill が用途に応じて Jacobian を管理します。

### 7.3.2 Coupling network

**コードスニペット 1 — Affine coupling layer**

```python
import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    """
    RealNVP-style affine coupling layer for a 3-dim logit space.
    x = (x_a, x_b), x_a: mask=1, x_b: mask=0.
    参照: Dinh et al., 2017 (RealNVP)
    """
    def __init__(self, d: int, d_hidden: int, mask: torch.Tensor):
        super().__init__()
        # 対称性回避: mask は 0/1 の binary tensor（shape=(d,)）
        assert mask.shape == (d,), f"mask must have shape ({d},)"
        assert set(mask.unique().tolist()).issubset({0.0, 1.0}), "mask must be binary"
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(
            nn.Linear(d, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, 2 * d),
        )

    def forward(self, x: torch.Tensor):
        """
        Returns y (transformed), log_det (log |det ∂y/∂x|).
        """
        x_a = x * self.mask
        h = self.net(x_a)
        s, t = h.chunk(2, dim=-1)
        # 数値安定化: s を [-5, 5] に tanh で bound（Glow 系の常套手段）
        s = torch.tanh(s) * 5.0
        # mask=0 の部分だけ変換
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        y = x_a + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor):
        y_a = y * self.mask
        h = self.net(y_a)
        s, t = h.chunk(2, dim=-1)
        s = torch.tanh(s) * 5.0
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        x = y_a + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return x
```

`s = torch.tanh(s) * 5.0` は Glow 系（Kingma & Dhariwal, 2018）の数値安定化テクニックです。$\log \det$ が発散すると学習が破綻するため、**5.0 の bound は canonical な既定値**として §7.7 の Skill YAML で `s_clamp` field に固定します。

### 7.3.3 RealNVP スタックと学習ループ

**コードスニペット 2 — RealNVP stack と train_flow**

```python
class CompositionFlow(nn.Module):
    """
    ALR-logit($n_{elem}-1$ 次元) 上の RealNVP。coupling layer を n_flows 段 stack。
    base distribution は N(0, I_d)。条件変数 c_cond は net に concat して条件付け。
    """
    def __init__(self, d: int = 3, d_hidden: int = 128, n_flows: int = 8, d_cond: int = 0):
        super().__init__()
        self.d = d
        self.d_cond = d_cond
        # mask を交互に反転（RealNVP 標準、任意次元 d に対応）
        masks = []
        base_mask = (torch.arange(d) % 2).float()   # e.g. d=3 → [0,1,0], d=4 → [0,1,0,1]
        for i in range(n_flows):
            masks.append(base_mask if i % 2 == 0 else 1.0 - base_mask)
        d_eff = d + d_cond  # condition を concat した場合
        self.flows = nn.ModuleList([
            AffineCoupling(d_eff, d_hidden, torch.cat([m, torch.ones(d_cond)])) for m in masks
        ])
        # base distribution
        self.register_buffer("base_mean", torch.zeros(d))
        self.register_buffer("base_std", torch.ones(d))

    def log_prob(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """
        exact log p(x | c) を返す。x: (B, 3), c: (B, d_cond) or None.
        """
        if c is not None:
            assert c.shape[-1] == self.d_cond, f"c dim mismatch: {c.shape[-1]} vs {self.d_cond}"
            h = torch.cat([x, c], dim=-1)
        else:
            assert self.d_cond == 0, "c is None but d_cond > 0"
            h = x
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        for flow in self.flows:
            h, ld = flow(h)
            log_det_sum = log_det_sum + ld
        # base に戻った x_part（先頭 d 次元）だけで base log_prob
        z = h[..., :self.d]
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.d * torch.log(torch.tensor(2 * torch.pi))
        return log_pz + log_det_sum

    def sample(self, n: int, c: torch.Tensor | None = None, temperature: float = 1.0) -> torch.Tensor:
        """
        z ~ N(0, temperature^2 I) → x = f^{-1}(z; c).
        temperature は base distribution の分散スケーリング（Glow, Kingma & Dhariwal 2018）。
        """
        device = self.base_mean.device
        z = torch.randn(n, self.d, device=device) * temperature
        if c is not None:
            assert c.shape[0] == n and c.shape[-1] == self.d_cond
            h = torch.cat([z, c], dim=-1)
        else:
            assert self.d_cond == 0
            h = z
        # inverse は逆順
        for flow in reversed(self.flows):
            h = flow.inverse(h)
        return h[..., :self.d]  # logit を返す（softmax は呼び出し側で）


def train_flow(model: CompositionFlow, data_loader, optimizer, max_epochs: int) -> dict:
    """
    負の対数尤度最小化。§7.7 Skill YAML の training_metrics.final_nll と一致。
    """
    model.train()
    metrics = {"final_nll": None, "final_epoch": None}
    for epoch in range(max_epochs):
        for x_logit, c in data_loader:  # x_logit は §7.3.1 で変換済み
            optimizer.zero_grad()
            log_p = model.log_prob(x_logit, c if c.numel() > 0 else None)
            loss = -log_p.mean()
            loss.backward()
            optimizer.step()
        metrics["final_nll"] = loss.item()
        metrics["final_epoch"] = epoch + 1
    return metrics
```

### 7.3.4 共通ヘルパ（条件パック / トークン ID）

Flow / AR のサンプリング・学習コード（コードスニペット 3–6）は、以下 2 種の共通ヘルパに依存します。canonical 実装として本節でまとめて定義します。

**コードスニペット 2.5 — 共通ヘルパ**

```python
# --- 条件変数パック（Flow / AR 共通）--------------------------------------
def _pack_conditions(
    conditions_values: dict | None, d_cond: int,
    n_samples: int = 1, device: torch.device | None = None,
) -> torch.Tensor | None:
    """
    Ch5 §5.4 正規化済み条件 dict を (n_samples, d_cond) tensor に pack。
    dict が None の場合は None を返す（無条件生成）。
    key の順序は Ch5 §5.4 canonical 順序（sorted by key）に強制。
    """
    if conditions_values is None:
        return None
    device = device or torch.device("cpu")
    keys = sorted(conditions_values.keys())
    vec = torch.tensor([conditions_values[k] for k in keys],
                       dtype=torch.float32, device=device)
    assert vec.numel() == d_cond, f"d_cond mismatch: got {vec.numel()}, expected {d_cond}"
    return vec.unsqueeze(0).expand(n_samples, -1).contiguous()

# --- Token ID 定数（AR のみ、build_vocab 呼び出し後に module-level に束縛）--
# 使用例:
#   vocab = build_vocab(element_set)
#   PAD_TOKEN_ID = vocab["token_to_id"]["<PAD>"]
#   BOS_TOKEN_ID = vocab["token_to_id"]["<BOS>"]
#   EOS_TOKEN_ID = vocab["token_to_id"]["<EOS>"]
#   UNK_TOKEN_ID = vocab["token_to_id"]["<UNK>"]
# canonical: SPECIAL_TOKENS は先頭固定順序（§7.5.1）なので ID = 0,1,2,3 で決定的。
PAD_TOKEN_ID = 0  # SPECIAL_TOKENS[0]
BOS_TOKEN_ID = 1  # SPECIAL_TOKENS[1]
EOS_TOKEN_ID = 2  # SPECIAL_TOKENS[2]
UNK_TOKEN_ID = 3  # SPECIAL_TOKENS[3]
```

> [!NOTE]
> Token ID の 0–3 決定性は `SPECIAL_TOKENS` を先頭固定した §7.5.1 の canonical に依存します。エージェントが `build_vocab` を書き換えて順序変更した場合、Skill YAML の `vocab_provenance.element_vocab_sha256` sha256 が変化するため、Ch10 の provenance ゲートで検出されます。

### 7.3.5 サンプリングと softmax 復元

**コードスニペット 3 — sample_flow with softmax 復元**

```python
def sample_flow(
    model: CompositionFlow,
    n_samples: int,
    conditions_values: dict | None,
    sampling_config: dict,
) -> torch.Tensor:
    """
    Route Flow-A（logit + softmax 復元）canonical 実装。
    sampling_config の canonical 検証は呼び出し前に validate_flow_sampling_config で実施。
    参照: §7.6.2 (canonical レンジ), Ch4 §4.3.5 (generation_temperature).
    """
    validate_flow_sampling_config(**sampling_config)
    model.eval()
    with torch.no_grad():
        c = _pack_conditions(conditions_values, model.d_cond,
                             n_samples=n_samples,
                             device=next(model.parameters()).device)  # (n_samples, d_cond) or None
        logits = model.sample(
            n_samples,
            c=c,
            temperature=sampling_config["base_temperature"],
        )
        # reference 元素の logit を 0 として復元（§7.3.1 逆変換）
        n_elem = logits.shape[-1] + 1
        full_logits = torch.cat([logits, torch.zeros(n_samples, 1, device=logits.device)], dim=-1)
        # softmax 温度復元
        compositions = torch.softmax(full_logits / sampling_config["temperature_scaling"], dim=-1)
    return compositions
```

Flow の `sampling_config` は Diffusion の 5 field から次のように差分します（§7.6 で正式化）：

- `sampler_type`：Flow では **enum なし**（可逆写像 1 段のみ）
- `n_sampling_steps`：Flow では NFE=1 なので固定（field なし）
- `eta`：Flow では stochasticity は base distribution 側なので `base_temperature` に統合
- `guidance_scale`：Flow では canonical なし（null）
- `temperature_scaling`：Flow でも softmax 復元時の温度として継承（Ch6 §6.7 canonical）
- **`base_temperature`（Flow 独自）**：base distribution $N(0, \tau^2 I)$ の $\tau$。Glow 論文で導入された「サンプリング多様性の制御」

### 7.3.6 Flow の exact likelihood 副産物：OOD スコア

学習後の `model.log_prob(x_logit)` は **ALR-logit 空間上の exact な対数尤度** を返します（§7.3.1）。テスト時候補 $x^*$ の $\log p_\ell(\ell^*)$ を **OOD スコアの軸 A としてそのまま使えます**。組成空間側の対数尤度が必要な場合、§7.3.1 の Jacobian 補正を Ch10 側で明示的に付与します（責務分離）。Ch2 §2.4 の合成規則の軸 A（Mahalanobis）と軸 B（PCA 再構成）の代替として、Flow を使う場合は **exact log-likelihood を単一軸で使える**のが実装的な利点です。ただし Ch10 で議論する通り、exact likelihood にも failure mode（Nalisnick et al., 2019：OOD data に高尤度を割り当てる病理）があり、単独ではなく Ch4 §4.5 の 3 態合成が canonical です。

---

## 7.4 Autoregressive の骨格 — 系列生成と factorized likelihood

### 7.4.1 factorized likelihood

AR は入力を系列 $x = (x_1, x_2, \ldots, x_L)$ とし、

$$
p_\theta(x) = \prod_{i=1}^{L} p_\theta(x_i \mid x_{<i})
$$

に因数分解します。$L$ は可変長。各条件付き分布 $p_\theta(x_i \mid x_{<i})$ を Transformer decoder（causal attention）で実装するのが 2020 年代の主流です。**学習は teacher-forcing の cross entropy**：

$$
\mathcal{L}(\theta) = - \mathbb{E}_{x \sim p_{\text{data}}} \sum_{i=1}^{L} \log p_\theta(x_i \mid x_{<i})
$$

これは exact log-likelihood 最大化と一致します（Flow と同様、ELBO 近似ではない）。

### 7.4.2 材料組成の系列化

組成 $c$ を系列にする方法は複数あります。本章は教育目的で **元素・化学量論のインターリーブ**を採用：

```
["<BOS>", "Fe", "0.3", "O", "0.4", "Ti", "0.2", "Al", "0.1", "<EOS>"]
```

- **語彙 (vocabulary)**：`<BOS>` / `<EOS>` / `<PAD>` / 元素記号（Ch2 §2.1 で扱う ARIM 合成データの元素集合、$\sim 80$ 種） + 化学量論トークン（0.05 刻みで discretize、$\sim 21$ 種）
- **系列長**：4 元系なら $L = 2 + 4 \times 2 = 10$（BOS/EOS + 元素×4 + 量論×4）

**Skill 契約の含意**：語彙は canonical に固定します（`element_vocab_sha256` と `stoichiometry_vocab_sha256` を Skill YAML に記録）。エージェントが語彙を勝手に拡張する（未知元素を追加する等）と学習分布外の生成につながるため、§7.8 の `disallowed_operations` で `extend_vocabulary_at_inference` を hard_reject します。

### 7.4.3 Transformer decoder の骨格

**コードスニペット 4 — Minimal Transformer decoder**

```python
class CompositionAR(nn.Module):
    """
    教育目的の最小 Transformer decoder。causal mask のみで conditional bias は
    条件変数を prefix embedding として prepend する方式（Ch5 §5.4 の canonical 条件付け）。
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_cond: int = 0,
        max_len: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.d_cond = d_cond
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        if d_cond > 0:
            self.cond_proj = nn.Linear(d_cond, d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """
        tokens: (B, L) LongTensor. Returns (B, L, vocab_size) logits.
        """
        B, L = tokens.shape
        assert L <= self.max_len, f"L={L} exceeds max_len={self.max_len}"
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(tokens) + self.pos_emb(pos)
        if c is not None:
            assert c.shape[-1] == self.d_cond, f"c dim mismatch: {c.shape[-1]} vs {self.d_cond}"
            # 条件を最初のトークン位置に加算（prefix bias 方式）
            h = h + self.cond_proj(c).unsqueeze(1)
        # causal mask
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=tokens.device), diagonal=1
        )
        h = self.blocks(h, mask=causal, is_causal=True)
        return self.head(h)
```

### 7.4.4 train_ar と temperature / top-k / top-p sampling

**コードスニペット 5 — train_ar と canonical sampling**

```python
def train_ar(model: CompositionAR, data_loader, optimizer, max_epochs: int) -> dict:
    """
    Teacher-forcing cross entropy 最小化。§7.8 Skill YAML の
    training_metrics.{final_ce, final_epoch, final_perplexity} を返す。
    """
    import math
    model.train()
    ce_loss = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    metrics = {"final_ce": None, "final_epoch": None, "final_perplexity": None}
    for epoch in range(max_epochs):
        for tokens, c in data_loader:
            # 入力: tokens[:, :-1], 予測: tokens[:, 1:]
            logits = model(tokens[:, :-1], c if c.numel() > 0 else None)
            loss = ce_loss(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics["final_ce"] = loss.item()
        metrics["final_epoch"] = epoch + 1
        metrics["final_perplexity"] = math.exp(loss.item())
    return metrics


def sample_ar(
    model: CompositionAR,
    n_samples: int,
    conditions_values: dict | None,
    sampling_config: dict,
) -> list[list[int]]:
    """
    canonical sampling: temperature + top-k + top-p の 3 stage 組み合わせ。
    参照: §7.6.3 (canonical レンジ), Ch6 §6.7 (Diffusion 側 canonical との差分).
    """
    validate_ar_sampling_config(**sampling_config)
    model.eval()
    device = next(model.parameters()).device
    c = _pack_conditions(conditions_values, model.d_cond,
                         n_samples=1, device=device)  # AR は 1-sample ずつループ生成
    T = sampling_config["temperature"]
    k = sampling_config["top_k"]
    p = sampling_config["top_p"]
    max_len = sampling_config["max_new_tokens"]

    sequences: list[list[int]] = []
    with torch.no_grad():
        for _ in range(n_samples):
            seq = [BOS_TOKEN_ID]
            for step in range(max_len):
                tokens = torch.tensor([seq], device=device)
                logits = model(tokens, c[:1] if c is not None else None)[:, -1, :]
                # 温度 scaling
                logits = logits / T
                # top-k フィルタ
                if k > 0:
                    top_k_vals, _ = logits.topk(k, dim=-1)
                    kth = top_k_vals[..., -1:]
                    logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
                # top-p (nucleus) フィルタ
                if p < 1.0:
                    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumprobs = probs.cumsum(dim=-1)
                    remove = cumprobs > p
                    # 先頭 1 トークンは必ず残す
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
                next_id = torch.distributions.Categorical(logits=logits).sample().item()
                seq.append(next_id)
                if next_id == EOS_TOKEN_ID:
                    break
            sequences.append(seq)
    return sequences
```

**temperature / top-k / top-p の Ch4 §4.3.5 対応**：

- `temperature` は Ch4 §4.3.5 `generation_temperature` の **AR 側 canonical 実装** です（Ch5 の VAE `generation_temperature: 1.0` scalar と同型）。ただし AR は softmax の shape を強く変えるため、Diffusion の `temperature_scaling`（softmax 組成復元のみ）とは **別 field** として `sampling_config.temperature` を持ちます。
- `top_k` / `top_p` は Ch6 §6.7 canonical `sampling_config` の **AR 拡張**（`sampler_type` / `n_sampling_steps` / `eta` は Diffusion 固有、Flow の `base_temperature` は Flow 固有と並列）。付録 A で back-port 予定です。

---

## 7.5 組成 AR の教育的実装 — 語彙定義とデータローダ

### 7.5.1 語彙定義

**コードスニペット 6 — Vocabulary construction**

```python
import hashlib
import json

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
STOICHIOMETRY_GRID = [round(x, 2) for x in [i * 0.05 for i in range(21)]]  # 0.00, 0.05, ..., 1.00

def build_vocab(element_set: list[str]) -> dict:
    """
    canonical 語彙構築。element_set は Ch2 §2.1 の ARIM 合成データ元素集合（sorted）。
    STOICHIOMETRY_GRID は 0.05 刻み固定。
    """
    element_tokens = sorted(element_set)
    stoichiometry_tokens = [f"stoich_{v:.2f}" for v in STOICHIOMETRY_GRID]
    tokens = list(SPECIAL_TOKENS) + element_tokens + stoichiometry_tokens
    token_to_id = {t: i for i, t in enumerate(tokens)}
    # 語彙の 2 subdomain ごとに独立 sha256 を計算（Skill YAML の
    # element_vocab_sha256 / stoichiometry_vocab_sha256 に個別 pin する canonical）
    def _hash(list_of_str: list[str]) -> str:
        payload = json.dumps(list_of_str, sort_keys=False, ensure_ascii=False).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()
    vocab = {
        "tokens": tokens,
        "token_to_id": token_to_id,
        "size": len(tokens),
        "element_vocab_sha256": _hash(element_tokens),
        "stoichiometry_vocab_sha256": _hash(stoichiometry_tokens),
    }
    return vocab
```

### 7.5.2 系列変換

**コードスニペット 7 — Composition → sequence**

```python
def composition_to_sequence(
    composition: dict[str, float], vocab: dict, tol: float = 1e-3,
) -> list[int]:
    """
    e.g. {"Fe": 0.3, "O": 0.4, "Ti": 0.2, "Al": 0.1} → 
         [BOS, Fe, stoich_0.30, O, stoich_0.40, Ti, stoich_0.20, Al, stoich_0.10, EOS]
    Skill 契約: 元素は Ch2 §2.1 canonical 順序（sorted）で並べる（エージェント自由な並び替え禁止）。
    """
    total = sum(composition.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"composition must sum to 1.0 (tol={tol}), got {total}")
    t2i = vocab["token_to_id"]
    seq = [t2i["<BOS>"]]
    for elem in sorted(composition.keys()):
        if elem not in t2i:
            raise KeyError(f"unknown element {elem} (Ch2 §2.1 element_set 外)")
        seq.append(t2i[elem])
        # 0.05 刻みに丸める
        v = round(composition[elem] * 20) / 20
        seq.append(t2i[f"stoich_{v:.2f}"])
    seq.append(t2i["<EOS>"])
    return seq


def sequence_to_composition(seq: list[int], vocab: dict) -> dict[str, float]:
    """
    逆変換。以下いずれかを満たさない sequence は ValueError（§7.8
    disallowed_operations.emit_unparseable_sequence で hard_reject される）:
      - 先頭が <BOS>
      - <EOS> で終端（<EOS> 以降は PAD のみ許容）
      - 元素 → stoich → 元素 の順序を厳守
      - 元素の重複なし
      - 元素順序が composition_to_sequence と同じ sorted 順
    Skill 契約: parse 失敗した候補は candidates_out から除外
    （不完全 candidate の "自信あり" 報告禁止、Ch4 §4.7 の safety 原理に接続する Ch7 拡張）。
    """
    tokens = vocab["tokens"]
    t2i = vocab["token_to_id"]
    if len(seq) < 2 or seq[0] != t2i["<BOS>"]:
        raise ValueError("sequence must start with <BOS>")
    out: dict[str, float] = {}
    seen_elements: list[str] = []
    i = 1
    saw_eos = False
    while i < len(seq):
        tok_id = seq[i]
        if tok_id == t2i["<EOS>"]:
            saw_eos = True
            # <EOS> 以降は PAD のみ許容
            for j in range(i + 1, len(seq)):
                if seq[j] != t2i["<PAD>"]:
                    raise ValueError(f"non-PAD token after <EOS> at pos {j}")
            break
        elem = tokens[tok_id]
        if not elem.isalpha() or elem in SPECIAL_TOKENS:
            raise ValueError(f"expected element at pos {i}, got {elem}")
        if elem in out:
            raise ValueError(f"duplicate element {elem} at pos {i}")
        if i + 1 >= len(seq):
            raise ValueError(f"missing stoichiometry after {elem}")
        stoich_tok = tokens[seq[i + 1]]
        if not stoich_tok.startswith("stoich_"):
            raise ValueError(f"expected stoich token at pos {i+1}, got {stoich_tok}")
        out[elem] = float(stoich_tok.split("_")[1])
        seen_elements.append(elem)
        i += 2
    if not saw_eos:
        raise ValueError("sequence did not terminate with <EOS>")
    if seen_elements != sorted(seen_elements):
        raise ValueError(f"elements must be in sorted order, got {seen_elements}")
    total = sum(out.values())
    if total < 1e-6:
        raise ValueError("empty composition")
    # 正規化（0.05 刻みなので合計が 1.0 に厳密には一致しない場合がある）
    return {k: v / total for k, v in out.items()}
```

### 7.5.3 学習と推論を通じた Skill 契約

- **学習分布外の系列生成防止**：`sample_ar` 出力を `sequence_to_composition` で parse し、parse 失敗（元素 → stoich 順序違反、未知元素等）は candidates_out から除外。§7.8 の `disallowed_operations.emit_unparseable_sequence` が該当。
- **語彙固定契約**：`element_vocab_sha256` と `stoichiometry_vocab_sha256` を Skill YAML の provenance に記録。推論時に語彙 sha256 が学習時と一致しない場合 hard_reject。
- **系列長制約**：$L \leq$ `max_len` を assertion。ARIM 合成データが 3-4 元系なら $L \leq 10$ 想定、`max_new_tokens: 20` を canonical 既定値とします。

---

## 7.6 Skill 契約：Flow / AR の canonical `sampling_config`

Ch6 §6.7 で確定した `sampling_config` の canonical dict 構造を、Flow / AR がどう継承／差分するかを本節で明示します。以降の Ch12（capstone）と Ch13（分子/結晶）が本表を参照します。

### 7.6.1 推奨レンジ表（YAML block 1）

**YAML block 1 — Flow / AR canonical `sampling_config`**

```yaml
# =============================================================
# Ch7 §7.6 canonical: Flow / AR サンプリング推奨レンジ表
# 参照: Ch4 §4.3.5 generation_temperature を Ch6 で dict 拡張した sampling_config を、
#       Ch7 で Flow / AR 固有 field を追加して canonical 化。付録 A で back-port 予定。
# =============================================================
recommended_ranges:
  flow:
    base_temperature:
      allowed_range: [0.5, 1.0]
      default: 0.7
      rationale: >
        Glow (Kingma & Dhariwal, 2018) 準拠。1.0 は base distribution そのまま、
        0.7 は多様性を若干抑えて分布中心を強調。<0.5 は mode collapse リスク。
    temperature_scaling:
      allowed_range: [0.8, 1.2]
      default: 1.0
      rationale: >
        softmax 復元時の温度。Ch6 §6.7 canonical と同一（家族横断で共有）。
        逸脱時は組成が one-hot 化（<1.0）または uniform 化（>1.0）。
    guidance_scale:
      allowed_value: null                # Flow は CFG を持たない（§7.2.4）
      rationale: "Flow に CFG 相当機構は canonical には存在しない"
  autoregressive:
    temperature:
      allowed_range: [0.5, 1.5]
      default: 1.0
      rationale: >
        LLM 系論文（Holtzman et al., 2020）で議論された既定値。
        <0.5 は greedy 化（多様性喪失）、>1.5 は無意味系列生成リスク。
    top_k:
      allowed_range: [0, 100]            # 0 は無効化（top_p のみ使用）
      default: 40
      rationale: >
        Holtzman et al., 2020 の nucleus 対比。0（無効）または [10, 100]。
        材料語彙は $\sim 100$ 元素 + $21$ stoich = $\sim 120$ 程度なので、
        top_k > 100 は事実上無効化と同義。
    top_p:
      allowed_range: [0.7, 1.0]
      default: 0.9
      rationale: >
        nucleus sampling (Holtzman et al., 2020) 準拠。0.9 が LLM 系の教科書的既定。
        <0.7 は greedy 化過剰、1.0 は無効化。
    max_new_tokens:
      allowed_range: [10, 30]
      default: 20
      rationale: >
        4-6 元系組成に相当。$L = 2 + n_{\text{elem}} \times 2$ なので、
        6 元系まで対応する 20 が canonical 既定。
```

### 7.6.2 Flow 用 validate 関数

**コードスニペット 8 — canonical validate for Flow**

```python
def validate_flow_sampling_config(
    base_temperature: float,
    temperature_scaling: float,
    guidance_scale=None,  # Flow では null 固定
) -> None:
    """
    Ch7 §7.6 canonical 検証。逸脱時は Ch7 独自 disallowed_operations
    （§7.7 flow_base_temperature_out_of_range 等）として hard_reject。
    Ch4 §4.7 の安全性原理を Flow 実装層に延伸した Ch7 拡張。
    """
    if not (0.5 <= base_temperature <= 1.0):
        raise ValueError(
            f"base_temperature out of recommended range: got {base_temperature}, "
            f"allowed [0.5, 1.0] (Ch7 §7.6, Kingma & Dhariwal 2018)"
        )
    if not (0.8 <= temperature_scaling <= 1.2):
        raise ValueError(f"temperature_scaling out of range: {temperature_scaling}")
    if guidance_scale is not None:
        raise ValueError(
            f"Flow does not support guidance_scale (§7.2.4). "
            f"got {guidance_scale}, expected null."
        )
```

### 7.6.3 AR 用 validate 関数

**コードスニペット 9 — canonical validate for AR**

```python
def validate_ar_sampling_config(
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
) -> None:
    """
    Ch7 §7.6 canonical 検証。逸脱時は Ch7 独自 disallowed_operations
    （§7.8 ar_temperature_out_of_range 等）として hard_reject。
    Ch4 §4.7 の安全性原理を AR 実装層に延伸した Ch7 拡張。
    """
    if not (0.5 <= temperature <= 1.5):
        raise ValueError(
            f"temperature out of recommended range: got {temperature}, "
            f"allowed [0.5, 1.5] (Ch7 §7.6, Holtzman et al., 2020)"
        )
    if not (0 <= top_k <= 100):
        raise ValueError(f"top_k out of range: {top_k}, allowed [0, 100]")
    if not (0.7 <= top_p <= 1.0):
        raise ValueError(
            f"top_p out of recommended range: got {top_p}, "
            f"allowed [0.7, 1.0] (Ch7 §7.6, Holtzman et al., 2020)"
        )
    if not (10 <= max_new_tokens <= 30):
        raise ValueError(f"max_new_tokens out of range: {max_new_tokens}, allowed [10, 30]")
```

### 7.6.4 Ch4 §4.3 との対応マップ

| Ch7 §7.6 field | 家族 | Ch4 §4.3 canonical | Skill YAML パス |
|---|---|---|---|
| `base_temperature` | Flow | §4.3.5 を Ch7 §7.6 で dict 拡張 | `sampling_config.base_temperature` |
| `temperature_scaling` | Flow / Diffusion 共有 | §4.3.5 を Ch6 §6.7 で dict 拡張 | `sampling_config.temperature_scaling` |
| `temperature` | AR | §4.3.5 を Ch7 §7.6 で dict 拡張 | `sampling_config.temperature` |
| `top_k` | AR | §4.3.5 を Ch7 §7.6 で dict 拡張 | `sampling_config.top_k` |
| `top_p` | AR | §4.3.5 を Ch7 §7.6 で dict 拡張 | `sampling_config.top_p` |
| `max_new_tokens` | AR | §4.3.5 を Ch7 §7.6 で dict 拡張 | `sampling_config.max_new_tokens` |
| `guidance_scale` | Flow: null / AR: null | §4.3.5（Diffusion のみ non-null） | top-level `guidance_scale` |

---

## 7.7 `arim.gen.flow.v0.1` Skill YAML 完全形

Ch4 §4.2 template と Ch5 §5.7 / Ch6 §6.8 の親骨格を継承した完全形です。

**YAML block 2 — `arim.gen.flow.v0.1` full Skill 契約書**

```yaml
# =============================================================
# arim.gen.flow.v0.1 — 組成 Flow Skill（Ch7 canonical 実装、Route Flow-A）
# 継承: Ch4 §4.2 template ⑧ を literal に継承、Ch5 §5.7 / Ch6 §6.8 と一貫
# =============================================================
skill:
  id: "arim.gen.flow.v0.1"
  version: "v0.1"
  description: "組成 Normalizing Flow (RealNVP, ALR-logit + softmax 復元) による分布内サンプリングと exact likelihood 計算（ALR-logit 空間）。Ch7 canonical 実装。"

pipeline_position:
  stage: "G"
  upstream_stages: ["PRE"]
  downstream_stages: ["SF", "F", "H"]

input_schema:
  mode:
    type: "enum"
    values: ["train", "generate", "score"]     # score: exact log_prob(x) を返す
  training_data:
    type: "dict"
    required_keys: ["dataset_id", "version", "sha256", "license"]
    condition: "required if mode=train"
  hyperparameters:
    type: "dict"
    schema:
      d: {type: "int", range: [2, 32]}          # logit 次元 = n_elem - 1
      d_hidden: {type: "int", range: [32, 512]}
      n_flows: {type: "int", range: [4, 16]}
      s_clamp: {type: "float", allowed_value: 5.0}   # §7.3.2 canonical
      d_cond: {type: "int", range: [0, 8]}
      max_epochs: {type: "int", range: [1, 500]}
      batch_size: {type: "int"}
      lr: {type: "float"}
  n_samples:
    type: "int"
    range: [1, 1000]
    condition: "required if mode=generate"
  conditions_values:
    type: "dict"
    schema_ref: "condition_variables_declared"
    condition: "required if condition_variables_declared is non-empty"
  sampling_config:
    type: "dict"
    schema_ref: "sampling_config"                 # Ch7 §7.6 Flow 版
    condition: "required if mode=generate"
  x_to_score:
    type: "tensor"
    shape_hint: "(N, d)"
    condition: "required if mode=score"

output_schema:
  trained_model_artifact_id:
    type: "string"
    condition: "emitted if mode=train"
  training_metrics:
    type: "dict"
    keys: ["final_nll", "final_epoch"]
  candidates_out:
    type: "list_of_dict"
    required_keys: ["candidate_id", "generated_x"]
    max_length_field: "n_samples"
    condition: "emitted if mode=generate"
  log_prob_out:
    type: "tensor"
    shape_hint: "(N,)"
    description: "ALR-logit 空間の exact log-likelihood（§7.3.1、§7.3.6）。組成空間側が必要な場合は Ch10 で Jacobian 補正を付与"
    condition: "emitted if mode=score"
  detection_result:
    type: "list_of_dict"
    schema_ref: "hallucinatory_composition_detection.result"

generative_model_family:
  family: "normalizing_flow"                      # Ch4 §4.3.1 canonical enum
  id: "arim.ch7.composition_flow_v0.1"
  version: "v0.1.0"
  hyperparameters:
    d: 3                                          # 4 元系 → logit 3 次元
    d_hidden: 128
    n_flows: 8
    s_clamp: 5.0
    d_cond: 2                                     # target_band_gap + target_formation_energy
latent_dim: 3                                     # Ch4 §4.3.2、Flow の base distribution 次元 = d（ALR-logit 空間の次元、$n_{elem} - 1$）
training_data_provenance:                         # Ch4 §4.3.3
  dataset_id: "arim.synthetic-generative.compositions.v1"
  version: "v1.0.0"
  sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  license: "arim_educational_v1"
preprocessing_provenance:                         # Ch7 §7.3.1 canonical 拡張（付録 A で back-port 予定）
  # Flow は ALR 変換を挟むため、変換のメタデータを provenance に固定する
  composition_min_fraction: 1.0e-4                # §7.3.1 の c_min（未満は学習前に除外）
  alr_reference_element_policy: "last_element_by_sorted_symbol"  # ALR の分母元素の選択規則（Ch2 §2.1 sorted 順の末尾）
  alr_dim: 3                                      # $n_{elem} - 1$、latent_dim と一致
condition_variables_declared:                     # Ch4 §4.3.4、Ch5 §5.4 継承
  - name: "target_band_gap_eV"
    range: [0.5, 6.0]
    type: "float"
    unit: "eV"
    normalization:
      mean: 2.1
      std: 1.4
      applied_at: "input"
  - name: "target_formation_energy_meV_per_atom"
    range: [-3500, 500]
    type: "float"
    unit: "meV/atom"
    normalization:
      mean: -1500.0
      std: 800.0
      applied_at: "input"
condition_extensibility:                          # Ch5 §5.4 canonical
  allow_agent_to_add_conditions: false
  additional_conditions_require: "human_approval"
generation_temperature: 1.0                       # Ch4 §4.3.5 scalar canonical（Ch5 と同型）
generation_temperature_range_allowed: [0.8, 1.2]  # temperature_scaling と一致
guidance_scale: null                              # Flow は CFG なし（§7.2.4）
guidance_scale_range_allowed: null
sampling_config:                                  # Ch7 §7.6 Flow 版 canonical
  base_temperature: 0.7
  temperature_scaling: 1.0
  base_temperature_range_allowed: [0.5, 1.0]
  temperature_scaling_range_allowed: [0.8, 1.2]
foundation_model_reference: null                  # Route A のみ（Flow は本章では B なし）
filter_rules_applied: []                          # Ch4 §4.3.6、F ステージは Ch8 に委譲
filter_rules_delegated_to:
  skill: "arim.gen.physics_filter.v0.1"
  skill_version: "v0.1"
  invocation_stage: "F"
screening_model_family: null
top_k_returned: null                              # Ch11 に委譲
inverse_design_authorization: "autonomous"
synthesis_launch_authorization: null
hallucinatory_composition_detection_delegated_to:
  skill: "arim.gen.ood_detection.v0.1"
  skill_version: "v0.1"
  invocation_stage: "H"
safety_screening_passed: false
dual_use_review_completed: false

authorization_gates:
  L1_static_filter_pass:
    required_for: ["safety_screening_passed_becomes_true"]
    approver_id: "agent:arim.gen.safety_screening.v0.1"
    authorization_id: "l1_auth_20260707_143512_iter1"
    status: "pending"
    approver_id_regex: "^agent:"
    parent_authorization_id: null
  L2_dual_use_review:
    required_for: ["dual_use_review_completed_becomes_true"]
    reviewer_role: "organizational_dual_use_committee"
    approver_id: "human:<staff_id>"
    authorization_id: "l2_auth_20260707_143512_iter1"
    status: "pending"
    approver_id_regex: "^human:"
    parent_authorization_id: "vol-06:L1_static_filter_pass:l1_auth_20260707_143512_iter1"
  L3_facility_policy_approval:
    required_for: ["facility_policy_id_pinned"]
    reviewer_role: "facility_safety_officer"
    facility_policy_id: "arim.facility.policy.v1"
    approver_id: "human:<staff_id>"
    authorization_id: "l3_auth_20260707_143512_iter1"
    status: "pending"
    approver_id_regex: "^human:"
    parent_authorization_id: "vol-06:L2_dual_use_review:l2_auth_20260707_143512_iter1"
  L4_synthesis_launch:
    required_for: ["synthesis_launch_authorization"]
    reviewer_role: "facility_synthesis_lead"
    approver_id: "human:<staff_id>"
    authorization_id: "l4_auth_20260707_143512_iter1"
    status: "pending"
    approver_id_regex: "^human:"
    parent_authorization_id: "vol-06:L3_facility_policy_approval:l3_auth_20260707_143512_iter1"

outputs_disallowed_natural_language:              # Ch4 §4.7 canonical 10 literal 日本語 phrase（Ch5/6 と一字一句同じ）
  - "この候補は安全です"
  - "合成可能です"
  - "危険ではありません"
  - "dual-use ではありません"
  - "safety_screening を通過したので合成可能です"
  - "推奨候補です"
  - "実験を進めてください"
  - "この候補で問題ありません"
  - "合成推奨です"
  - "安全性は確認されています"

disallowed_operations:                            # Ch7 独自: Flow サンプリング逸脱（Ch4 §4.7 の安全性原理に接続）
  - name: "flow_base_temperature_out_of_range"
    description: "§7.6.1 canonical [0.5, 1.0] からの逸脱は base distribution の分散スケーリングを崩す"
    enforcement: "hard_reject"
  - name: "temperature_scaling_out_of_range"
    description: "§7.6.1 canonical [0.8, 1.2] からの逸脱は softmax 復元段で組成分布を破壊"
    enforcement: "hard_reject"
  - name: "add_guidance_scale_at_inference"
    description: "Flow は CFG を持たない（§7.2.4）。guidance_scale を non-null にする操作は禁止"
    enforcement: "hard_reject"
  - name: "add_condition_not_in_declared"
    description: "Ch5 §5.4 継承。allow_agent_to_add_conditions=false"
    enforcement: "hard_reject"
  - name: "condition_scale_interpolated"
    description: "Ch5 §5.4 継承。正規化 mean/std の変更禁止"
    enforcement: "hard_reject_and_audit_log"
  - name: "modify_s_clamp_at_inference"
    description: "学習時に固定した s_clamp=5.0（§7.3.2）の推論時変更禁止"
    enforcement: "hard_reject"
  - name: "swap_reference_element_at_inference"
    description: "logit 変換の reference 元素（§7.3.1）を推論時に変更すると可逆性が破綻"
    enforcement: "hard_reject"

provenance:
  input_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  skill_version: "v0.1"
  run_datetime_utc: "2026-07-07T00:00:00Z"
  package_versions:
    torch: "2.4.1"
    numpy: "2.0.2"
  random_seed: 20260707
  event_hash: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
```

> [!NOTE]
> **`arim.gen.flow.v0.1` の観察点**（Ch6 §6.8 との差分中心）：
> - `generative_model_family.family: "normalizing_flow"` と `hyperparameters` が Flow 固有（`d` / `n_flows` / `s_clamp`）
> - `guidance_scale: null` — Flow は CFG を持たない canonical（§7.2.4）
> - `sampling_config` は Ch7 §7.6 Flow 版：`sampler_type` / `n_sampling_steps` / `eta` は non-existent、代わりに `base_temperature` を持つ
> - `mode: "score"` が canonical 追加：Flow の exact likelihood を OOD スコア軸 A として返す用途（§7.3.6、Ch10 で完全実装）
> - `foundation_model_reference: null` — 本章では Route A のみ扱う
> - `disallowed_operations` の Flow 固有項目（`flow_base_temperature_out_of_range` / `modify_s_clamp_at_inference` / `swap_reference_element_at_inference`）は Ch4 §4.7 の safety 原理を Flow 実装層に延伸した Ch7 拡張

---

## 7.8 `arim.gen.autoregressive.v0.1` Skill YAML 完全形

**YAML block 3 — `arim.gen.autoregressive.v0.1` full Skill 契約書**

```yaml
# =============================================================
# arim.gen.autoregressive.v0.1 — 組成 Autoregressive Skill（Ch7 canonical 実装）
# 継承: Ch4 §4.2 template ⑧ を literal に継承、Ch5 §5.7 / Ch6 §6.8 と一貫
# =============================================================
skill:
  id: "arim.gen.autoregressive.v0.1"
  version: "v0.1"
  description: "組成 Autoregressive (Transformer decoder, 元素・化学量論トークン列) による分布内サンプリング。Ch7 canonical 実装。"

pipeline_position:
  stage: "G"
  upstream_stages: ["PRE"]
  downstream_stages: ["SF", "F", "H"]

input_schema:
  mode:
    type: "enum"
    values: ["train", "generate"]
  training_data:
    type: "dict"
    required_keys: ["dataset_id", "version", "sha256", "license"]
    condition: "required if mode=train"
  vocab:
    type: "dict"
    required_keys: ["element_vocab_sha256", "stoichiometry_vocab_sha256", "size"]
    condition: "required if mode=train or mode=generate"
  hyperparameters:
    type: "dict"
    schema:
      d_model: {type: "int", range: [64, 512]}
      n_heads: {type: "int", range: [2, 16]}
      n_layers: {type: "int", range: [2, 12]}
      max_len: {type: "int", range: [10, 32]}
      d_cond: {type: "int", range: [0, 8]}
      max_epochs: {type: "int", range: [1, 500]}
      batch_size: {type: "int"}
      lr: {type: "float"}
  n_samples:
    type: "int"
    range: [1, 1000]
    condition: "required if mode=generate"
  conditions_values:
    type: "dict"
    schema_ref: "condition_variables_declared"
    condition: "required if condition_variables_declared is non-empty"
  sampling_config:
    type: "dict"
    schema_ref: "sampling_config"                 # Ch7 §7.6 AR 版
    condition: "required if mode=generate"

output_schema:
  trained_model_artifact_id:
    type: "string"
    condition: "emitted if mode=train"
  training_metrics:
    type: "dict"
    keys: ["final_ce", "final_epoch", "final_perplexity"]
  candidates_out:
    type: "list_of_dict"
    required_keys: ["candidate_id", "generated_x", "generated_sequence"]
    max_length_field: "n_samples"
  detection_result:
    type: "list_of_dict"
    schema_ref: "hallucinatory_composition_detection.result"

generative_model_family:
  family: "autoregressive"                        # Ch4 §4.3.1 canonical enum
  id: "arim.ch7.composition_autoregressive_v0.1"
  version: "v0.1.0"
  hyperparameters:
    d_model: 128
    n_heads: 4
    n_layers: 4
    max_len: 20
    d_cond: 2
latent_dim: 128                                   # Ch4 §4.3.2、AR の hidden dim = d_model（Ch4 §4.3.2 の "AR の hidden dim" 該当）
training_data_provenance:
  dataset_id: "arim.synthetic-generative.compositions.v1"
  version: "v1.0.0"
  sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  license: "arim_educational_v1"
condition_variables_declared:                     # Ch5 §5.4 継承
  - name: "target_band_gap_eV"
    range: [0.5, 6.0]
    type: "float"
    unit: "eV"
    normalization:
      mean: 2.1
      std: 1.4
      applied_at: "prefix_embedding"              # AR は prefix bias 方式（§7.4.3）
  - name: "target_formation_energy_meV_per_atom"
    range: [-3500, 500]
    type: "float"
    unit: "meV/atom"
    normalization:
      mean: -1500.0
      std: 800.0
      applied_at: "prefix_embedding"
condition_extensibility:                          # Ch5 §5.4 canonical
  allow_agent_to_add_conditions: false
  additional_conditions_require: "human_approval"
generation_temperature: 1.0                       # Ch4 §4.3.5 scalar canonical（sampling_config.temperature と一致）
generation_temperature_range_allowed: [0.5, 1.5]
guidance_scale: null                              # AR は canonical CFG なし（本章 scope）
guidance_scale_range_allowed: null
sampling_config:                                  # Ch7 §7.6 AR 版 canonical
  temperature: 1.0
  top_k: 40
  top_p: 0.9
  max_new_tokens: 20
  temperature_range_allowed: [0.5, 1.5]
  top_k_range_allowed: [0, 100]
  top_p_range_allowed: [0.7, 1.0]
  max_new_tokens_range_allowed: [10, 30]
vocab_provenance:                                 # Ch7 §7.5.1 canonical 拡張（付録 A で back-port 予定）
  element_vocab_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  stoichiometry_vocab_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  vocab_size: 105                                 # SPECIAL_TOKENS(4) + elements(~80) + stoichiometry(21) の例
foundation_model_reference: null                  # Route A のみ
filter_rules_applied: []
filter_rules_delegated_to:
  skill: "arim.gen.physics_filter.v0.1"
  skill_version: "v0.1"
  invocation_stage: "F"
screening_model_family: null
top_k_returned: null                              # Ch4 §4.3.8、Ch11 に委譲（AR の top_k は sampling filter、canonical field とは別）
inverse_design_authorization: "autonomous"
synthesis_launch_authorization: null
hallucinatory_composition_detection_delegated_to:
  skill: "arim.gen.ood_detection.v0.1"
  skill_version: "v0.1"
  invocation_stage: "H"
safety_screening_passed: false
dual_use_review_completed: false

authorization_gates:
  L1_static_filter_pass:
    required_for: ["safety_screening_passed_becomes_true"]
    approver_id: "agent:arim.gen.safety_screening.v0.1"
    authorization_id: "l1_auth_20260707_143512_iter1"
    status: "pending"
    approver_id_regex: "^agent:"
    parent_authorization_id: null
  L2_dual_use_review:
    required_for: ["dual_use_review_completed_becomes_true"]
    reviewer_role: "organizational_dual_use_committee"
    approver_id: "human:<staff_id>"
    authorization_id: "l2_auth_20260707_143512_iter1"
    status: "pending"
    approver_id_regex: "^human:"
    parent_authorization_id: "vol-06:L1_static_filter_pass:l1_auth_20260707_143512_iter1"
  L3_facility_policy_approval:
    required_for: ["facility_policy_id_pinned"]
    reviewer_role: "facility_safety_officer"
    facility_policy_id: "arim.facility.policy.v1"
    approver_id: "human:<staff_id>"
    authorization_id: "l3_auth_20260707_143512_iter1"
    status: "pending"
    approver_id_regex: "^human:"
    parent_authorization_id: "vol-06:L2_dual_use_review:l2_auth_20260707_143512_iter1"
  L4_synthesis_launch:
    required_for: ["synthesis_launch_authorization"]
    reviewer_role: "facility_synthesis_lead"
    approver_id: "human:<staff_id>"
    authorization_id: "l4_auth_20260707_143512_iter1"
    status: "pending"
    approver_id_regex: "^human:"
    parent_authorization_id: "vol-06:L3_facility_policy_approval:l3_auth_20260707_143512_iter1"

outputs_disallowed_natural_language:              # Ch4 §4.7 canonical 10 literal 日本語 phrase
  - "この候補は安全です"
  - "合成可能です"
  - "危険ではありません"
  - "dual-use ではありません"
  - "safety_screening を通過したので合成可能です"
  - "推奨候補です"
  - "実験を進めてください"
  - "この候補で問題ありません"
  - "合成推奨です"
  - "安全性は確認されています"

disallowed_operations:                            # Ch7 独自: AR サンプリング逸脱（Ch4 §4.7 の safety 原理を AR 実装層に延伸）
  - name: "ar_temperature_out_of_range"
    description: "§7.6.1 canonical [0.5, 1.5] からの逸脱は greedy 過剰または無意味系列生成を招く"
    enforcement: "hard_reject"
  - name: "ar_top_k_out_of_range"
    description: "§7.6.1 canonical [0, 100] からの逸脱は語彙サイズと不整合"
    enforcement: "hard_reject"
  - name: "ar_top_p_out_of_range"
    description: "§7.6.1 canonical [0.7, 1.0] からの逸脱は nucleus sampling の教科書的推奨から逸脱"
    enforcement: "hard_reject"
  - name: "max_new_tokens_out_of_range"
    description: "§7.6.1 canonical [10, 30] からの逸脱は語彙・組成系列の想定を破壊"
    enforcement: "hard_reject"
  - name: "extend_vocabulary_at_inference"
    description: "推論時の語彙拡張禁止。学習時の element_vocab_sha256 / stoichiometry_vocab_sha256 と一致しない場合 reject（§7.5.3）"
    enforcement: "hard_reject_and_audit_log"
  - name: "emit_unparseable_sequence"
    description: "sequence_to_composition が失敗した候補を candidates_out に含めない（§7.5.3）"
    enforcement: "hard_reject"
  - name: "add_condition_not_in_declared"
    description: "Ch5 §5.4 継承。allow_agent_to_add_conditions=false"
    enforcement: "hard_reject"
  - name: "condition_scale_interpolated"
    description: "Ch5 §5.4 継承。正規化 mean/std の変更禁止"
    enforcement: "hard_reject_and_audit_log"

provenance:
  input_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  skill_version: "v0.1"
  run_datetime_utc: "2026-07-07T00:00:00Z"
  package_versions:
    torch: "2.4.1"
    numpy: "2.0.2"
  random_seed: 20260707
  event_hash: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
```

> [!NOTE]
> **`arim.gen.autoregressive.v0.1` の観察点**（Ch6 §6.8 / Ch7 §7.7 との差分中心）：
> - `generative_model_family.family: "autoregressive"` と `hyperparameters` が AR 固有（`d_model` / `n_heads` / `n_layers` / `max_len`）
> - `sampling_config` は Ch7 §7.6 AR 版：4 field（`temperature` / `top_k` / `top_p` / `max_new_tokens`）
> - `vocab_provenance` が **AR 固有 canonical 拡張**（付録 A で back-port 予定）：`element_vocab_sha256` / `stoichiometry_vocab_sha256` を pin
> - `condition_variables_declared[*].normalization.applied_at: "prefix_embedding"` — AR は prefix bias 方式（§7.4.3、Ch5/6 の `"input"` とは異なる canonical 値）
> - `guidance_scale: null` — 本章 scope の AR は CFG を持たない
> - `disallowed_operations` の AR 固有項目（`ar_temperature_out_of_range` / `extend_vocabulary_at_inference` / `emit_unparseable_sequence`）は Ch4 §4.7 の safety 原理を AR 実装層に延伸した Ch7 拡張

---

## 7.9 VAE / Diffusion / Flow / AR の 4 家族選択判断表（Part II まとめ）

Ch5-7 で 4 家族の Skill 化が出揃いました。ARIM 施設の実務家が **「自分のデータに対してどの家族を使うか」** を判断するための表を、Part II の締めくくりとして提示します。

### 7.9.1 判断軸の定義

- **軸 1：データ量** — 学習に必要なサンプル数の目安（教育目的の小規模スクラッチ実装の場合）
- **軸 2：制約表現** — 条件付き生成の柔軟性、CFG のような後付け条件強化の可否
- **軸 3：計算コスト** — 学習の GPU 時間（数千サンプル、単一 GPU 想定）とサンプリングの NFE（Number of Function Evaluations）
- **軸 4：Skill 契約の複雑さ** — `sampling_config` の canonical field 数、`disallowed_operations` の Ch4 §4.7 延伸項目数

### 7.9.2 判断表

| 軸 | VAE (Ch5) | Diffusion (Ch6) | Flow (Ch7) | AR (Ch7) |
|---|---|---|---|---|
| **データ量下限** | 数百〜 | 数千〜（既存 FM 使えばゼロ） | 数百〜 | 数千〜（可変長系列で表現力を活かすため） |
| **潜在空間の明示** | ✅ 明示（`latent_dim` 直接） | ❌ 明示なし | △ base distribution が明示（$x$ 自身が latent） | ❌ 明示なし（hidden state に埋め込み） |
| **条件付き生成** | cVAE（学習時条件付け） | CFG で推論時に強度調整可（§6.6） | 条件をネット入力（強度調整なし） | prefix embedding（強度調整なし） |
| **exact likelihood** | ❌ ELBO は下界 | ❌ $L_{\text{simple}}$ は近似 | ✅ change of variables で厳密 | ✅ factorized で厳密 |
| **サンプリング NFE** | 1（decode 1 回） | DDIM 20-100 / DDPM 1000 | 1（inverse 1 段） | max_new_tokens 回（10-30） |
| **学習時 GPU 時間** | 数分〜数十分 | 数十分〜数時間 | 数分〜数十分 | 数十分〜数時間（Transformer） |
| **`sampling_config` field 数** | 0（`generation_temperature` scalar のみ） | 4（`sampler_type` / `n_sampling_steps` / `eta` / `temperature_scaling`） + top-level `guidance_scale` | 2（`base_temperature` / `temperature_scaling`） | 4（`temperature` / `top_k` / `top_p` / `max_new_tokens`） |
| **Ch4 §4.7 延伸 `disallowed_operations`** | 3-4 項目 | 7 項目 | 7 項目 | 8 項目 |
| **OOD 判定への直接利用** | Mahalanobis / PCA / 物理制約の合成規則（Ch2 §2.4） | 同上 | exact log-likelihood 単軸で可能（§7.3.6、ただし Nalisnick 2019 の failure mode に注意） | exact log-likelihood 系列積で可能（ただし系列長依存） |
| **既存 FM の利用可能性** | 少ない | 豊富（MatterGen / DiffCSP / CDVAE 等、§6.5） | 中規模（Glow 系の材料応用は限定的） | LLM 系（材料 GPT 系は研究途上、vol-03 §11 参照） |
| **本書での実装 route** | Route A のみ | Route A / Route B 両方 | Route A のみ | Route A のみ |

### 7.9.3 選択フローチャート（決定木）

```
質問 1: 既存 Foundation Model を使うか？
  YES → Diffusion (Route B、MatterGen / DiffCSP / CDVAE、§6.5)
  NO  → 質問 2 へ

質問 2: 表現形式は系列（分子 SMILES / 結晶 SLICES 等）か連続組成か？
  系列 → AR (Ch7 §7.4、Ch13 §13a の分子逆設計 route)
  連続 → 質問 3 へ

質問 3: 学習データは数千サンプル以上あるか？
  YES → 質問 4 へ
  NO  (数百-千程度) → VAE (Ch5) または Flow (Ch7)
                     → 質問 5 へ

質問 4: 条件付き生成の強度調整（CFG のような）が必要か？
  YES → Diffusion (Route A、§6.6)
  NO  → VAE (cVAE、Ch5 §5.4) または Flow

質問 5: exact likelihood での OOD 判定が主目的か？
  YES → Flow (Ch7 §7.3.6、§7.7 mode=score)
  NO  → VAE (Ch5、実装が最も単純)
```

### 7.9.4 Ch8 以降との接続

Ch8（物理制約フィルタ）以降は、**上表のどの家族の G ステージ Skill が出力する候補に対しても** 適用可能です。以降の章では 4 家族を横断する canonical field（`generative_model_family.family` / `candidates_out` / `sampling_config`）に依拠して記述されます。特に：

- **Ch8 §8.x** の `filter_rules_applied` は 4 家族横断で同一 canonical
- **Ch10 §10.x** の分布的妥当性判定は Flow の exact log-likelihood を軸 A の代替として利用可能
- **Ch11 §11.x** の離散候補ランキングは 4 家族どれの `candidates_out` も同一入力として扱う
- **Ch12 capstone** で cVAE / cDiffusion を主軸に、Flow の likelihood 補助を組み合わせる

---

## 7.10 よくある失敗と対処

### 7.10.1 Flow の学習発散（Ch14 §14.1 で正式に taxonomize）

**症状**：negative log-likelihood が epoch 数ステップで NaN、または $\log \det$ が発散。

**原因と対処**：`s = torch.tanh(s) * 5.0` の bound が外れている、coupling layer 段数が不足で表現力不足、あるいは前処理で $c_i < c_{\min} = 10^{-4}$ の候補が除外されず ALR 変換で $\log 0$ 発生。**canonical 対処**：`s_clamp: 5.0` を §7.7 で固定、$c_{\min} = 10^{-4}$ と ALR reference 元素選択規則を §7.7 Skill YAML の `preprocessing_provenance.composition_min_fraction` / `preprocessing_provenance.alr_reference_element_policy` に pin（`training_data_provenance.sha256` はデータ artifact 同定のみを担う）、$n_{\text{flows}}$ の下限を 4 に固定。エージェントが `s_clamp` を勝手に変更する操作は **§7.7 の `modify_s_clamp_at_inference` で hard_reject**（Ch4 §4.7 の安全性原理を Flow 実装層に延伸した Ch7 拡張、Ch4 §4.7 の canonical 禁止フレーズとは別軸で並存）。

### 7.10.2 AR の系列破綻（無意味な元素-元素連続等）

**症状**：`sample_ar` が `["Fe", "Ti", "0.4", "O", ...]` のように「元素 → stoich」順序を守らない系列を出力。

**原因と対処**：学習不足、または temperature 過大。**canonical 対処**：§7.5.2 の `sequence_to_composition` で parse 失敗した候補を candidates_out から除外（§7.8 の `emit_unparseable_sequence` が hard_reject、Ch4 §4.7 の safety 原理を AR 実装層に延伸した Ch7 拡張）。**parse 成功率**を training_metrics に加えて監視することも検討値です（v0.2 で正式化予定）。

### 7.10.3 Flow の "exact likelihood ≠ OOD detector" 誤解

**症状**：Flow の $\log p(x^*)$ が高いのに、$x^*$ が学習データ分布外の候補。

**原因**：Nalisnick et al., 2019 が指摘した **generative models can assign higher likelihood to OOD samples than in-distribution samples** の failure mode。特に画像領域では CIFAR-10 で学習した Flow が SVHN に高尤度を割り当てる例が有名。**対処**：exact likelihood 単軸に頼らず、Ch4 §4.5 の 3 態合成（likelihood + Mahalanobis + PCA）を Ch10 で完全実装する（本章 §7.3.6 の注記を参照）。

### 7.10.4 AR の温度暴走（Ch14 §14.1 で正式に taxonomize）

**症状**：エージェントが「多様性を上げるため」に `temperature` を 2.0 に上げ、無意味な組成系列が生成される。

**原因**：温度上限 1.5 の教科書的既定（Holtzman et al., 2020）の逸脱。**対処**：`validate_ar_sampling_config` が [0.5, 1.5] の範囲外を **§7.8 の `ar_temperature_out_of_range` で hard_reject**（Ch4 §4.7 の safety 原理を AR 実装層に延伸した Ch7 拡張、Ch4 §4.7 canonical 禁止フレーズ 10 list とは別軸で並存）。

### 7.10.5 AR の語彙拡張暴走

**症状**：エージェントが「新しい元素 X を追加すれば有望候補になる」と判断し、学習時語彙にない元素トークンを推論時に追加。

**原因**：`element_vocab_sha256` の provenance 契約違反。**対処**：§7.8 の `extend_vocabulary_at_inference` で **hard_reject_and_audit_log**。推論時に `element_vocab_sha256` / `stoichiometry_vocab_sha256` を学習時 pin 値と個別に照合し、いずれかが不一致なら reject。Ch4 §4.7 の safety 原理を AR 実装層に延伸した Ch7 拡張。

### 7.10.6 条件変数のスケール逸脱（Ch5 §5.4 継承）

**症状**：エージェントが「band_gap の正規化 mean を微調整すれば条件遵守が改善する」と判断し、`condition_variables_declared[*].normalization.mean` を推論時に変更。

**原因**：Ch5 §5.4 canonical `condition_extensibility` 違反。**対処**：Flow / AR ともに §7.7 / §7.8 の `condition_scale_interpolated` で hard_reject_and_audit_log。Ch5 §5.7 → Ch6 §6.8 → Ch7 §7.7 / §7.8 で家族横断に継承される canonical 契約。

---

## 7.11 章末チェックリスト

**理論レベル**

- [ ] Flow の change of variables 式 $\log p_X(x) = \log p_Z(f(x)) + \log |\det \partial f / \partial x|$ が書けるか
- [ ] coupling layer が **可逆性** と **ヤコビアン計算可能性** を両立する仕組みを説明できるか（RealNVP、Dinh et al., 2017）
- [ ] AR の factorized likelihood $p_\theta(x) = \prod_i p_\theta(x_i \mid x_{<i})$ が書けるか
- [ ] Flow / AR の exact likelihood が VAE の ELBO 近似 と Diffusion の $L_{\text{simple}}$ 近似 とどう本質的に異なるかを説明できるか
- [ ] temperature / top-k / top-p の 3 段フィルタが AR sampling でどう順番に適用されるか（temperature 除算 → top-k mask → top-p mask → sampling）を説明できるか

**Skill 契約レベル**

- [ ] Ch4 §4.7 canonical 10 phrase list は自然言語 output に関する規約（10 literal 日本語 phrase）。本章 Flow 固有（base_temperature 逸脱、s_clamp 変更、reference 元素 swap）と AR 固有（temperature 逸脱、語彙拡張、unparseable 出力）の Skill 契約違反は **Ch4 §4.7 の phrase 番号ではなく、Ch7 §7.7 / §7.8 `disallowed_operations` の Ch7 拡張項目として扱う**（Ch4 §4.7 と Ch7 `disallowed_operations` は安全性原理を共有）ことを説明できるか
- [ ] Flow の `guidance_scale: null` が canonical で、`add_guidance_scale_at_inference` が hard_reject なのはなぜかを §7.2.4 と接続できるか
- [ ] AR の `vocab_provenance.element_vocab_sha256` を Skill YAML に pin する必然性を §7.5.3 で説明できるか
- [ ] Ch14 §14.1（生成一般失敗）と §14.2（Agentic 特有失敗）のうち、本章 §7.10 でカバーしたのが §14.1 系（一般失敗）であり、Agentic 側は Ch12 capstone と Ch14 で扱うことを認識しているか

**Part II まとめレベル**

- [ ] §7.9 の 4 家族判断表を使い、ARIM の 500 サンプル × 4 元系 × band_gap 条件付き の実データに対してどの家族を選ぶか説明できるか
- [ ] Ch8 以降のスクリーニング／逆設計が **4 家族どれの G ステージ Skill 出力にも適用可能**な canonical field（`candidates_out` / `generative_model_family.family` / `sampling_config`）に依拠している理由を説明できるか

---

## 7.12 参考資料

**理論・原著論文（Flow）**

- **NICE（Dinh et al., 2015）**：https://arxiv.org/abs/1410.8516 —— coupling layer の原型
- **RealNVP（Dinh et al., 2017）**：https://arxiv.org/abs/1605.08803 —— §7.2.2 と §7.3.2 の骨格
- **Glow（Kingma & Dhariwal, 2018）**：https://arxiv.org/abs/1807.03039 —— §7.3.2 の `s_clamp` テクニック、§7.3.5 の `base_temperature`
- **Normalizing Flows: An Introduction and Review（Kobyzev et al., 2020）**：https://arxiv.org/abs/1908.09257 —— サーベイ

**理論・原著論文（AR）**

- **Attention Is All You Need（Vaswani et al., 2017）**：https://arxiv.org/abs/1706.03762 —— Transformer decoder の原著
- **The Curious Case of Neural Text Degeneration（Holtzman et al., 2020）**：https://arxiv.org/abs/1904.09751 —— §7.4.4 の nucleus (top-p) sampling、§7.6 canonical 既定値の一次情報源

**failure mode**

- **Do Deep Generative Models Know What They Don't Know?（Nalisnick et al., 2019）**：https://arxiv.org/abs/1810.09136 —— §7.10.3 の exact-likelihood-OOD paradox の一次情報源

**分子・材料表現**

- **SMILES（Weininger, 1988）**：J. Chem. Inf. Comput. Sci. 28(1), 31-36 —— 分子表現の原著（Ch13 §13a で詳述）
- **SELFIES（Krenn et al., 2020）**：https://arxiv.org/abs/1905.13741 —— 生成モデル向け分子表現（付録 C 概説）
- **SLICES（Xiao et al., 2023）**：https://arxiv.org/abs/2303.02150 —— 結晶構造の文字列表現（Ch13 §13b で言及）

**Continuous Normalizing Flow / Flow Matching（本章 scope 外の一次情報源）**

- **FFJORD（Grathwohl et al., 2019）**：https://arxiv.org/abs/1810.01367 —— Continuous NF
- **Flow Matching for Generative Modeling（Lipman et al., 2023）**：https://arxiv.org/abs/2210.02747 —— Flow Matching

**ライブラリ**

- **PyTorch nn.Transformer**：https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html —— §7.4.3
- **nflows**：https://github.com/bayesiains/nflows —— Flow の canonical Python 実装
- **normflows**：https://github.com/VincentStimper/normalizing-flows —— 別の教育向け実装

**本書内参照**

- **Ch4 §4.2 template**：本章 §7.7 / §7.8 の親骨格
- **Ch4 §4.3.1-4.3.9**：`generative_model_family` / `latent_dim` / `training_data_provenance` / `condition_variables_declared` / `generation_temperature` / `filter_rules_applied` / `screening_model_family` / `top_k_returned` / `inverse_design_authorization` の canonical 定義
- **Ch5 §5.4**：条件変数の正規化契約と `condition_extensibility`（本章 §7.7 / §7.8 で継承、AR は `applied_at: "prefix_embedding"` に差分）
- **Ch5 §5.7**：親 Skill YAML（本章 §7.7 / §7.8 で継承）
- **Ch6 §6.7**：`sampling_config` canonical dict 化（本章 §7.6 で Flow / AR 拡張）
- **Ch6 §6.8**：`arim.gen.diffusion.v0.1` full Skill YAML（本章 §7.7 / §7.8 の直接の兄弟）
- **Ch4 §4.5**：`hallucinatory_composition_detection` 3 態（本章 §7.7 / §7.8 では delegated 態を実装、Ch5 §5.7 / Ch6 §6.8 と同型）
- **Ch4 §4.6**：Safety 3-layer（本章 §7.6.2 / §7.6.3 の validate 関数で Layer 1 を実装）
- **Ch4 §4.7**：禁止フレーズ 10 list は canonical 10 literal 日本語 phrase として §7.7 / §7.8 に literal 化。Flow / AR 固有の Skill 契約違反は Ch7 §7.7 / §7.8 `disallowed_operations` の Ch7 拡張項目として扱う（Ch4 §4.7 の safety 原理に接続）
- **Ch4 §4.9**：Skill ID 一覧（本章は `arim.gen.flow.v0.1` と `arim.gen.autoregressive.v0.1`）
- **Ch7 で先行導入**：`vocab_provenance`（AR 固有、Ch4 §4.3.3 拡張）と `sampling_config.base_temperature` / `sampling_config.top_k` / `sampling_config.top_p` / `sampling_config.max_new_tokens`（Ch6 §6.7 の家族横断 canonical 拡張）。いずれも付録 A で back-port 予定

---

**次章予告**（第8章）：**物理制約フィルタと合成可能性スクリーニングを Skill 化する**。本章までで 4 家族すべての G ステージ Skill が揃いました。Ch8 では **生成候補集合に対する rule-based / model-based スクリーニング**を扱います：Pymatgen による電荷中性・酸化数・化学量論チェック、matminer による組成 → 物性特徴量、分子側の SC-Score / SA-Score / RA-Score、結晶側の熱力学的 hull distance と ICSD historical existence。Ch5-7 で扱った **"分布内検知"**（生成器内 OOD）と、Ch8 の **"生成後の物理・合成可能性チェック"** の責務分離を明確化し、`arim.gen.physics_filter.v0.1` F ステージ Skill として実装します。
