# 第6章 Diffusion Model を Skill 化する — 材料生成の主戦場

> **本章の使い方**
> - **Part II の中核章**です。Ch4 の canonical template（§4.2 骨格、§4.3 13+1 provenance fields、§4.5 `hallucinatory_composition_detection` 3 態、§4.6 Safety 3-layer、§4.7 禁止フレーズ 10 list、§4.9 9 Skill ID 一覧）と **Ch5 §5.7 の親 Skill YAML** を継承し、`arim.gen.diffusion.v0.1` として実装します。
> - **2 route で扱います**：
>   - **Route A（教育目的スクラッチ）**：ARIM 風 3-4 元系合成データ（数百〜数千サンプル）で **組成 Diffusion** を単一 GPU 内でスクラッチ学習。DDPM / DDIM の骨格を Skill 契約と一対一で対応付けて説明します。
>   - **Route B（既存材料 FM の fetch）**：**MatterGen（HF Hub `microsoft/mattergen`）**、**DiffCSP（Google Drive）**、**CDVAE（GitHub README 経由）** の 3 経路 fetch を、Ch3 §3.8 の library_allowlist / fm_fetch canonical と、**本章 §6.5 で先行導入する `foundation_model_reference` field**（Ch4 §4.3.6 は `filter_rules_applied` が canonical、`foundation_model_reference` は付録 A で back-port 予定）に沿って Skill 化します。研究本番は Route B が実践 route です。
> - **本章の scope**：**「Diffusion 生成器の学習」「条件付き生成 (classifier-free guidance)」「サンプリング温度と guidance scale の Skill 契約」** に責務を絞ります。生成後の物理制約チェックは Ch8、OOD 検知の完全計算（軸 A/B/C）は Ch10、Flow / AR は Ch7、surrogate ランキングは Ch11 に委譲します。
> - **YAML block は全て `yaml.safe_load` 準拠**：本書 CI で機械検証されます。
>
> **本章の到達目標**
> - DDPM の forward / reverse process と simplified loss $L_{\text{simple}}$ を、Ch4 §4.3.1 の `generative_model_family.family = "diffusion"` の canonical field に一対一で対応付けて説明できる
> - **DDIM** の deterministic sampling と NFE（Number of Function Evaluations）削減が、Ch4 §4.3.5 `generation_temperature` を **本章 §6.7 で dict 拡張した `sampling_config.sampler_type`** サブフィールドで宣言されることを理解する
> - **組成 Diffusion** を 3-4 元系合成データで単一 GPU 内スクラッチ学習し、`training_data_provenance` の 4-tuple に沿って provenance を残せる
> - **MatterGen / DiffCSP / CDVAE** の 3 経路 fetch を、Ch3 §3.8 の library_allowlist（HF Hub / GitHub / Google Drive）と checksum 検証を通じて Skill 化できる
> - **Classifier-free guidance (CFG)** の guidance scale $w$ を Ch4 §4.3.4 `condition_variables_declared` と組み合わせ、**§6.7 の canonical 推奨レンジ表**を用いて「エージェントは guidance scale を勝手に上げない」契約を書ける
> - **`arim.gen.diffusion.v0.1`** の完全 Skill YAML を、Ch4 §4.2 template を全フィールド埋めた形で書ける（Route A / Route B 両モード宣言つき）
>
> **本章で扱わないこと**
> - VAE / Flow / AR の骨格と Skill 実装（Ch5 / Ch7）
> - 生成候補の物理制約チェック（Ch8、`arim.gen.physics_filter.v0.1` F ステージ）
> - OOD 検知の完全計算（Ch10、軸 A/B/C の実装詳細）
> - `arim.gen.candidate_ranking.v0.1` による top-k 選抜（Ch11）
> - 分子・SMILES 生成 diffusion（Ch13 §13a、SELFIES / MolDiff の議論）
> - 危険物質 filter list 内容（付録 C）と MCP handler 実装（付録 B）
> - Agentic 失敗パターンの taxonomy（Ch14）

---

## 6.1 なぜ Diffusion が材料生成の主戦場か

前章 Ch5 で扱った VAE と比較して、Diffusion Model を **Part II の中核** に据える理由は次の 4 点です。

**理由 1：条件付き生成の柔軟性 (classifier-free guidance)**

VAE の cVAE は、学習時に $p(x \mid c)$ の条件付き分布を直接モデル化します。しかし、**推論時に新しい条件で強度を調整することは困難**です。Diffusion Model の **classifier-free guidance (CFG)** は、学習時に無条件モデル $\epsilon_\theta(x_t, t)$ と条件付きモデル $\epsilon_\theta(x_t, t, c)$ を同時に学習し、推論時に

$$
\tilde\epsilon = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
$$

の形で **guidance scale $w$ を動的に調整**できます（$\emptyset$ は無条件、$w=1$ で純粋な条件付き生成、$w>1$ で条件方向を強化）。材料逆設計では「目標物性への条件付けをどれだけ強くするか」を実行時に決められるため、cVAE より柔軟です（§6.6）。

ただしこの柔軟性は、Ch6 独自の安全境界と衝突します：**エージェントが $w$ を勝手に上げると学習分布外の候補を出すリスク**が増大します。Ch6 §6.8 の `disallowed_operations`（Ch4 §4.7 の安全性原理を Diffusion 実装層に延伸したもの）で $w$ の動的上書きを禁止し、§6.7 の推奨レンジ表と §6.10 の失敗パターンで補強します。

**理由 2：既存材料 FM の主流実装が Diffusion**

2023 年以降、材料生成の Foundation Model は Diffusion ベースが主流です：

| 材料 FM | 手法 | 対象 | fetch 経路 | ライセンス |
|---|---|---|---|---|
| **MatterGen**（Microsoft, 2024） | Score-based diffusion on crystal graph | 結晶構造 | HF Hub `microsoft/mattergen` | MIT |
| **DiffCSP**（Jiao et al., 2023） | Equivariant diffusion for crystal structure prediction | 結晶構造 | Google Drive（論文 repo 経由） | Non-commercial（要確認） |
| **CDVAE**（Xie et al., 2022） | VAE + diffusion on lattice / atom coords | 結晶構造 | GitHub `txie-93/cdvae` | MIT |

**vol-03 §11 の材料 FM 経由での実践 route** の主戦場が Diffusion であるため、本章で Skill 化しておくことが実務上の requirement です。fetch 経路が 3 種類あることは Ch3 §3.8 の `library_allowlist` / `fm_fetch` canonical で扱う理由でもあります（§6.5）。

**理由 3：組成 Diffusion のスクラッチ学習が教育的**

VAE のスクラッチ学習（Ch5 §5.3）は encoder / decoder / ELBO の 3 要素で完結しますが、**Diffusion のスクラッチ学習は 6 要素**を扱います：

1. Forward noising schedule（$\beta_t$）
2. Reverse denoising network（U-Net or MLP、本章は組成なので MLP）
3. Simplified loss $L_{\text{simple}}$
4. DDPM ancestral sampling（stochastic）
5. DDIM deterministic sampling（NFE 削減）
6. Classifier-free guidance の joint training（10-20% の確率で条件を drop）

これらを **Skill 契約と一対一で対応させる** ことで、Ch4 canonical template の各 provenance field がなぜ必要かを実装レベルで理解できます。

**理由 4：Diffusion 特有の provenance が Ch4 §4.3 の canonical に組み込まれている**

Ch4 §4.3.5 `generation_temperature` を **本章 §6.7 で dict 拡張** し、Diffusion 固有のサブフィールドとして次を宣言します（Ch6 §6.8 では top-level `sampling_config` field として実装）：

- `sampler_type`: DDPM / DDIM / DPM-Solver（本章は DDPM と DDIM のみ扱う）
- `n_sampling_steps`: 逆過程のステップ数（DDPM は $T$、DDIM は $\ll T$）
- `guidance_scale`: CFG の $w$
- `temperature_scaling`: ancestral sampling 時の noise 乗数（既定 1.0）

これらは §6.7 の推奨レンジ表と §6.8 の Skill YAML で埋めます。

**Ch5 との対比まとめ**

| 論点 | Ch5 VAE | Ch6 Diffusion |
|---|---|---|
| 学習 | encoder + decoder + ELBO | noising schedule + denoiser + simplified loss |
| 条件付け | cVAE（学習時固定） | CFG（推論時 $w$ 可変） |
| サンプリング | prior からの 1 shot | 反復（DDPM: $T$ 回、DDIM: $\ll T$ 回） |
| Skill 契約の焦点 | `condition_variables_declared` | `guidance_scale` + `n_sampling_steps` |
| 主要な暴走リスク | 分布外 latent 操作 | guidance scale の暴走・NFE の勝手な変更 |

---

## 6.2 DDPM の骨格 — forward / reverse / simplified loss

**Denoising Diffusion Probabilistic Model (DDPM)**（Ho et al., 2020）の骨格を、Skill 契約と対応付けながら整理します。

### 6.2.1 Forward process（noising）

$T$ ステップで元データ $x_0$ にガウスノイズを加え、$x_T \approx \mathcal{N}(0, I)$ に到達させます。

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I)
$$

**閉形式**：$\alpha_t = 1 - \beta_t$、$\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$ とおくと、

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}\, x_0, (1-\bar\alpha_t) I)
$$

任意の $t$ で $x_0$ から直接 $x_t$ をサンプルできるため、学習が効率的です。

**Skill 契約との対応**：$\beta_t$ の schedule（linear / cosine）は **Ch4 §4.3.1 `generative_model_family.hyperparameters.beta_schedule`** で宣言します。

### 6.2.2 Reverse process（denoising）

逆過程 $p_\theta(x_{t-1} \mid x_t)$ をパラメトリック分布で近似します。DDPM は次の重要な reparameterization を用います：

- 真の逆過程は $q(x_{t-1} \mid x_t, x_0)$ のガウス分布（$x_0$ で条件付ければ閉形式）
- $x_0$ をノイズ $\epsilon$ で書き換え、**denoiser $\epsilon_\theta(x_t, t)$ でノイズを予測**する形にする

### 6.2.3 Simplified loss $L_{\text{simple}}$

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[ \|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon,\ t)\|^2 \right]
$$

**実装は 30 行以内で書ける**（§6.4）ため、教育的に扱いやすい loss です。

**Skill 契約との対応**：$L_{\text{simple}}$ 系列（$L_{\text{simple}}$ / $L_{\text{vlb}}$ / hybrid）の選択は **Ch4 §4.3.1 `generative_model_family.hyperparameters.loss_variant`** で宣言します。本章は $L_{\text{simple}}$ のみ扱います。

### 6.2.4 DDPM ancestral sampling

推論時は $x_T \sim \mathcal{N}(0, I)$ からスタートし、$t = T, T-1, \dots, 1$ まで **$T$ 回の denoiser 呼び出し** で $x_0$ を得ます：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

**NFE = $T$**（通常 $T \in [500, 1000]$）で、これが Diffusion の推論コストの主因です。§6.3 の DDIM で削減します。

**Skill 契約との対応**：$T$ は **`sampling_config.n_sampling_steps`**（Ch4 §4.3.5 を本章 §6.7 で拡張）、stochastic vs deterministic は **`sampler_type`** で宣言します。

---

## 6.3 DDIM — deterministic sampling と NFE 削減

**Denoising Diffusion Implicit Model (DDIM)**（Song et al., 2021）は、DDPM と同じ学習済みモデル $\epsilon_\theta$ を用いつつ、**推論を deterministic** にし、かつ **NFE を大幅削減** します。

### 6.3.1 DDIM の更新式

**部分列 $\{t_s\}_{s=0}^{S}$**（$t_0=0 < t_1 < \dots < t_S = T$）を選び、隣接ステップ $t_s \to t_{s-1}$ の更新を次で定義します（$t \equiv t_s$、$t' \equiv t_{s-1}$、$t' < t$、DDPM の隣接 $t-1$ ではないことに注意）：

$$
x_{t'} = \sqrt{\bar\alpha_{t'}} \underbrace{\frac{x_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}}_{\text{予測 } \hat{x}_0} + \sqrt{1-\bar\alpha_{t'} - \sigma_t^2}\, \epsilon_\theta(x_t, t) + \sigma_t z
$$

$\sigma_t = 0$ に固定すると **deterministic**（同じ $x_T$ からは同じ $x_0$）になります。$S = T$ とすれば DDPM の隣接更新（$t' = t-1$）に一致します。

### 6.3.2 NFE 削減

$S \ll T$（例：$S = 50$、$T = 1000$）を選ぶことで、**NFE が $T$ から $S$ へ 10-20 倍削減** されます。画像領域の初期研究（Song et al., 2021）では **$S = 50$ で DDPM $T = 1000$ に近い FID/IS 品質が得られる** と報告されており、本章は教育目的で $S = 50$ を既定とします。ただし材料生成領域では対象タスク・foundation model ごとに再検証が必要で、絶対的な保証ではありません（§6.7 の推奨レンジ表と §6.10.4 参照）。

**Skill 契約との対応**：DDIM を使う場合、本章 §6.7 で導入する `sampling_config`（Ch4 §4.3.5 `generation_temperature` の dict 拡張）は次のように埋めます：

```yaml
sampling_config:
  sampler_type: "ddim"                    # canonical enum: ddpm | ddim（dpm_solver は v0.2 で追加予定、§6.3.3）
  n_sampling_steps: 50                    # DDPM なら 1000、DDIM なら 20-100 が既定
  eta: 0.0                                # DDIM stochasticity（0.0 = deterministic）
  temperature_scaling: 1.0                # ancestral sampling 時の softmax 温度
# guidance_scale は top-level（Ch5 §5.7 と同型、§6.6 CFG 参照）
guidance_scale: 3.0
```

**エージェントの権限**：**`sampler_type` と `n_sampling_steps` は Skill が pin する**（`allowed_agent_actions` に含めない）。§6.7 で canonical に扱います。

### 6.3.3 DPM-Solver 等の高速化手法（本章 scope 外）

**DPM-Solver / DPM-Solver++ / UniPC** など、NFE をさらに 10-20 まで削減する手法があります。本章は教育目的で DDPM / DDIM に絞りますが、本章 §6.7 で導入する `sampling_config.sampler_type` の canonical enum は将来これらを追加できるよう **未定義値を含まない** 設計です。使いたい場合は次のバージョン `arim.gen.diffusion.v0.2` として追加してください（Ch4 §4.9 の Skill ID 命名規則）。

---

## 6.4 組成 Diffusion の教育的実装

**Route A（教育目的スクラッチ）** として、ARIM 風 3-4 元系合成データ（Ch2 §2.3 で紹介した `arim.synthetic-generative.compositions.v1` を継承）で組成 Diffusion を単一 GPU 内で学習します。

### 6.4.1 データ表現

組成 $x \in \mathbb{R}^{d}$（$d$ = 元素数、例：Fe-Ni-Cr-Mn の 4 元系なら $d=4$）を、softmax で正規化した確率単体上のベクトルとして扱います。**単純化のため**、simplex 上の diffusion（Dirichlet forward）ではなく、$\mathbb{R}^d$ 上で標準の Gaussian diffusion を行い、出力時に **softmax + 温度スケーリング** で組成に戻します。

> [!NOTE]
> Simplex 上の diffusion（Simplex Diffusion, Richemond et al., 2022）は組成データに理論的に整合しますが、本章の教育目的の scope 外です。研究本番は Route B（MatterGen 等）で結晶グラフ diffusion を使うため、Route A では最小骨格に留めます。

### 6.4.2 Denoiser network（MLP）

組成データは低次元（$d \le 10$）のため、U-Net は不要です。**時刻埋め込み付き MLP** で十分です：

**コードスニペット 1 — 組成 Diffusion Denoiser（PyTorch）**

```python
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding (Ch4 §4.3.1 hyperparameters.time_embedding_dim)."""
    def __init__(self, dim: int):
        super().__init__()
        # sinusoidal embedding は half=dim//2 で sin/cos を concat し 2*half 次元を返す。
        # dim が奇数だと最終次元が dim-1 となり CompositionDenoiser の Linear と shape mismatch する。
        # Skill 契約 (§6.8 input_schema.hyperparameters.d_time.multiple_of: 2) により保証。
        if dim % 2 != 0:
            raise ValueError(f"TimeEmbedding.dim は偶数である必要があります: {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = torch.exp(
            torch.arange(half, device=t.device) * (-torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class CompositionDenoiser(nn.Module):
    """
    組成 Diffusion 用 MLP denoiser.
    Skill 契約: `generative_model_family.family = "diffusion"`,
              `generative_model_family.hyperparameters.arch = "mlp_denoiser"`

    CFG null-condition の実装：条件変数 c と別に「null flag（is_null）」を
    concatenate し、is_null=True のとき c を 0 にすると同時に flag を 1 にする。
    これにより「c=0」と「本当に unconditional」を model が区別できるようになる
    （c を zero にするだけでは「平均条件」と衝突する。Ho & Salimans, 2022 の
    learned null token の等価実装）。
    """
    def __init__(self, d_comp: int, d_hidden: int = 128, d_time: int = 32,
                 d_cond: int | None = None):
        super().__init__()
        self.time_embed = TimeEmbedding(d_time)
        self.d_cond = d_cond
        # 条件がある場合、c (d_cond) + null_flag (1 dim) を concat
        d_cond_eff = (d_cond + 1) if d_cond else 0
        d_in = d_comp + d_time + d_cond_eff
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_comp),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                c: torch.Tensor | None = None,
                is_null: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
          c: [B, d_cond] 条件変数（cVAE 相当）。None の場合は条件なしモデル（d_cond=None 時）。
          is_null: [B] bool tensor。True の要素は unconditional 予測として扱う。
                   c is not None かつ is_null=None のときは全て False（条件付き）とみなす。
        """
        parts = [x_t, self.time_embed(t)]
        if self.d_cond:
            assert c is not None, "d_cond>0 のモデルには c が必須（unconditional は is_null=True で表現）"
            if is_null is None:
                is_null = torch.zeros(c.size(0), dtype=torch.bool, device=c.device)
            null_flag = is_null.float().unsqueeze(-1)                # [B, 1]
            c_effective = c * (1.0 - null_flag)                      # is_null=True の要素は 0 化
            parts += [c_effective, null_flag]
        h = torch.cat(parts, dim=-1)
        return self.net(h)
```

### 6.4.3 Forward noising schedule と学習ループ

**コードスニペット 2 — DDPM 学習ループ（`arim.gen.diffusion.v0.1` Route A）**

```python
import torch
import torch.nn.functional as F

def make_beta_schedule(T: int = 1000, schedule: str = "linear") -> torch.Tensor:
    """Skill 契約: `hyperparameters.beta_schedule` (§6.2.1)."""
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, T)
    elif schedule == "cosine":
        # Nichol & Dhariwal, 2021 (§6.2.1)
        s = 0.008
        steps = torch.arange(T + 1) / T
        f = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = f / f[0]
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return beta.clamp(1e-5, 0.999)
    else:
        raise ValueError(f"unknown beta_schedule: {schedule}")


def train_diffusion(model, dataloader, T=1000, epochs=100, lr=1e-3,
                    cfg_drop_prob=0.1, device="cuda"):
    """
    Route A 学習ループ。
    Skill 契約:
      - loss_variant = "l_simple" (§6.2.3)
      - cfg_drop_prob = 0.1 (§6.6, joint training で条件を drop する確率)
    """
    betas = make_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for x0, c in dataloader:                          # c: 条件（cVAE 相当）
            x0, c = x0.to(device), c.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            eps = torch.randn_like(x0)
            x_t = (
                alpha_bars[t].sqrt().unsqueeze(-1) * x0
                + (1 - alpha_bars[t]).sqrt().unsqueeze(-1) * eps
            )
            # Classifier-free guidance の joint training:
            # cfg_drop_prob の確率で「unconditional」として学習（is_null=True で null flag を立てる、
            # CompositionDenoiser 内で c は 0 化されて null_flag が 1 になる）
            is_null = torch.rand(x0.size(0), device=device) < cfg_drop_prob
            eps_pred = model(x_t, t, c, is_null=is_null)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()
        # provenance に記録すべき: final_loss, final_epoch (Ch4 §4.3 training_metrics)
    return model
```

### 6.4.4 サンプリング（DDPM ancestral + DDIM deterministic 両モード）

**コードスニペット 3 — DDPM / DDIM 統合サンプラ**

```python
@torch.no_grad()
def sample_diffusion(model, n_samples: int, d_comp: int, T: int = 1000,
                     sampler_type: str = "ddim",       # Ch4 §4.3.5 拡張（§6.7）
                     n_sampling_steps: int = 50,       # DDIM の場合の S
                     guidance_scale: float = 3.0,      # §6.6 CFG w
                     temperature_scaling: float = 1.0, # §6.4.1 softmax の温度（§6.7 canonical）
                     c: torch.Tensor | None = None,
                     eta: float = 0.0, device: str = "cuda") -> torch.Tensor:
    """
    Skill 契約:
      - sampler_type in {"ddpm", "ddim"} (§6.3)
      - guidance_scale range: §6.7 推奨レンジ表 [1.0, 7.5]
      - eta = 0.0 => DDIM deterministic
      - temperature_scaling: softmax 温度、§6.7 canonical レンジ [0.8, 1.2]
    """
    if temperature_scaling <= 0:
        raise ValueError(f"temperature_scaling must be positive: {temperature_scaling}")
    # §6.7.2 canonical 検証（Ch4 §6.6 Layer 1 必要条件）
    # validate_sampling_config は本章 §6.7.2 のコードスニペット 7 で定義。
    # 実装統合時は import 順に注意（本コード実行前に validate_sampling_config が
    # 読み込まれている必要あり。教科書コードでは §6.7.2 → §6.4.4 の順で読み込むこと）。
    validate_sampling_config(
        sampler_type=sampler_type,
        n_sampling_steps=n_sampling_steps,
        eta=eta,
        guidance_scale=guidance_scale,
        temperature_scaling=temperature_scaling,
    )
    # DDPM は ancestral sampling で full T ステップ実行が理論的既定。
    # n_sampling_steps は provenance 表明値として T と一致していなければならない
    # （NFE 契約と実実行の乖離を防ぐ、§6.7.1 rationale）。
    if sampler_type == "ddpm" and n_sampling_steps != T:
        raise ValueError(
            f"DDPM は full T ステップ実行が必要。n_sampling_steps={n_sampling_steps} "
            f"は T={T} と一致していなければならない（§6.7.1）。"
            f"NFE を削減したい場合は sampler_type='ddim' を使用してください（§6.3）。"
        )
    betas = make_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    if sampler_type == "ddim":
        steps = torch.linspace(0, T - 1, n_sampling_steps).long().to(device)
        steps = steps.flip(0)  # T-1 -> 0
    elif sampler_type == "ddpm":
        steps = torch.arange(T - 1, -1, -1, device=device)
    else:
        raise ValueError(f"unknown sampler_type: {sampler_type}")

    x = torch.randn(n_samples, d_comp, device=device)
    for i, t in enumerate(steps):
        t_batch = t.expand(n_samples)
        # CFG: 条件付き予測と無条件予測の線形結合（is_null flag で「本当に unconditional」を表現、
        # CompositionDenoiser の learned null 実装。§6.4.2 参照）
        if c is not None:
            is_null_true = torch.ones(n_samples, dtype=torch.bool, device=device)
            is_null_false = torch.zeros(n_samples, dtype=torch.bool, device=device)
            eps_uncond = model(x, t_batch, c, is_null=is_null_true)
            eps_cond = model(x, t_batch, c, is_null=is_null_false)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            # d_cond=None のモデル（unconditional 専用）
            eps = model(x, t_batch, None)

        alpha_bar_t = alpha_bars[t]
        alpha_bar_prev = alpha_bars[steps[i + 1]] if i < len(steps) - 1 else torch.tensor(1.0, device=device)
        x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()

        if sampler_type == "ddim":
            sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t)).sqrt() * (1 - alpha_bar_t / alpha_bar_prev).sqrt()
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = alpha_bar_prev.sqrt() * x0_pred + (1 - alpha_bar_prev - sigma ** 2).sqrt() * eps + sigma * noise
        else:  # ddpm
            sigma = betas[t].sqrt()
            noise = torch.randn_like(x) if t > 0 else 0.0
            x = (1 / alphas[t].sqrt()) * (x - (betas[t] / (1 - alpha_bar_t).sqrt()) * eps) + sigma * noise

    # 組成に戻す: softmax（温度スケーリング temperature_scaling は §6.7 canonical レンジで validate 済み）
    return torch.softmax(x / temperature_scaling, dim=-1)
```

**学習と推論の両方が単一 GPU（8-16 GB VRAM）で数分〜数十分で完結**するため、Ch5 と同様、教育目的スクラッチとして機能します。

---

## 6.5 結晶構造 Diffusion — MatterGen / DiffCSP / CDVAE の fetch

**Route B（既存材料 FM）** では、3 経路の fetch を Ch3 §3.8 の `library_allowlist` / `fm_fetch` canonical に沿って Skill 化します。

### 6.5.1 fetch 経路の 3 分類

| 経路 | 対象 FM | 主要な検証項目 | Skill 契約への影響 |
|---|---|---|---|
| **HF Hub** | MatterGen | commit_sha_hash、model card ライセンス、safetensors 形式 | Ch6 §6.5 で先行導入する `foundation_model_reference.fetch_channel = "hf_hub"`（Ch4 §4.3.6 は `filter_rules_applied` が canonical） |
| **GitHub** | CDVAE、（一部 DiffCSP repo） | git tag、リリースアセット sha256 | `fetch_channel = "github_release"` |
| **Google Drive** | DiffCSP（一次公開） | ファイル ID、公開状態、sha256（**要手動計算**） | `fetch_channel = "google_drive_manual"` |

**Google Drive は最も脆弱な経路**：公開状態の変更・アカウント停止でファイルが消える可能性があります。Ch4 §4.6 の Safety 3-layer の Layer 1（Skill 静的 filter）で、**Google Drive fetch を含む Skill は必ず sha256 lock を要求** します。

### 6.5.2 HF Hub fetch（MatterGen）

**コードスニペット 4 — MatterGen fetch（HF Hub, `arim.gen.diffusion.v0.1` Route B）**

```python
from huggingface_hub import snapshot_download
import hashlib
import json
from pathlib import Path

def fetch_mattergen(revision: str, sha256_lock: dict[str, str],
                    cache_dir: Path) -> Path:
    """
    Skill 契約 (Ch6 §6.5 canonical, Ch4 §4.3.6 は filter_rules_applied):
      foundation_model_reference:
        name: "microsoft/mattergen"
        fetch_channel: "hf_hub"
        revision_commit_hash: <revision>
        weights_sha256: <sha256_lock の canonical value>
    """
    local_dir = snapshot_download(
        repo_id="microsoft/mattergen",
        revision=revision,             # commit SHA を lock（tag/branch は禁止）
        cache_dir=str(cache_dir),
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "config/*"],
    )
    # sha256 検証（Ch4 §4.6 Layer 1: filter passing の必要条件）
    # 契約: sha256_lock の value は "sha256:" prefix 付き（Skill YAML の weights_sha256 と同型）
    for fname, expected_sha in sha256_lock.items():
        actual = "sha256:" + hashlib.sha256(Path(local_dir, fname).read_bytes()).hexdigest()
        if actual != expected_sha:
            raise RuntimeError(
                f"[fetch failure] sha256 mismatch: {fname}\n"
                f"  expected: {expected_sha}\n"
                f"  actual:   {actual}"
            )
    return Path(local_dir)
```

### 6.5.3 Google Drive fetch（DiffCSP）

**コードスニペット 5 — DiffCSP fetch（Google Drive, 手動 sha256 lock 必須）**

```python
import gdown
import hashlib
from pathlib import Path

def fetch_diffcsp(file_id: str, expected_sha256: str, dest: Path) -> Path:
    """
    Skill 契約 (Ch6 §6.5 canonical, Ch4 §4.3.6 は filter_rules_applied):
      foundation_model_reference:
        name: "diffcsp_perov_5"
        fetch_channel: "google_drive_manual"
        google_drive_file_id: <file_id>
        weights_sha256: <expected_sha256>  # Layer 1 で必須
    警告 (§6.10):
      Google Drive は公開状態変更でファイルが消える可能性がある。
      sha256 mismatch 時は fetch failure として **候補生成を拒否**（Ch6 §6.8 `disallowed_operations` の Ch6 拡張項目、Ch4 §4.7 の安全性原理を共有）。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dest), quiet=False)
    # 契約: expected_sha256 は "sha256:" prefix 付き（Skill YAML の weights_sha256 と同型）
    actual = "sha256:" + hashlib.sha256(dest.read_bytes()).hexdigest()
    if actual != expected_sha256:
        dest.unlink(missing_ok=True)     # 失敗時は artifact を残さない
        raise RuntimeError(
            f"[fetch failure] DiffCSP sha256 mismatch (Google Drive)\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}\n"
            f"  action:   check https://github.com/jiaor17/DiffCSP for updated file_id"
        )
    return dest
```

### 6.5.4 GitHub Release fetch（CDVAE）

**コードスニペット 6 — CDVAE fetch（GitHub Release, git tag lock）**

```python
import subprocess
import hashlib
from pathlib import Path

def fetch_cdvae(git_tag: str, expected_sha256: str, dest_dir: Path) -> Path:
    """
    Skill 契約 (Ch6 §6.5 canonical, Ch4 §4.3.6 は filter_rules_applied):
      foundation_model_reference:
        name: "cdvae"
        fetch_channel: "github_release"
        git_tag: <git_tag>       # v1.0.0 等の immutable tag（branch は禁止）
        weights_sha256: <expected_sha256>
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", git_tag,
         "https://github.com/txie-93/cdvae", str(dest_dir / "cdvae")],
        check=True,
    )
    ckpt_path = dest_dir / "cdvae" / "prop_models" / "carbon.ckpt"  # 例
    # 契約: expected_sha256 は "sha256:" prefix 付き（Skill YAML の weights_sha256 と同型）
    actual = "sha256:" + hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"[fetch failure] CDVAE sha256 mismatch\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}"
        )
    return ckpt_path
```

### 6.5.5 fetch 失敗時の Skill 挙動

Ch4 §4.7 の安全性原理（canonical field vs 自然言語判定の飛躍禁止）に接続：**fetch 失敗時、Skill は候補を返さない**（**沈黙する**）。「fetch できなかったので代わりに ○○ を使いました」は **Ch6 独自 `disallowed_operations.fetch_failed_substitute`（§6.8 で literal 化）** として禁止。エージェントは Ch4 §4.6 Layer 2（組織レイヤ）にエスカレートします。

---

## 6.6 条件付き生成 — Classifier-Free Guidance (CFG)

### 6.6.1 CFG の原理

学習時に **条件を確率 $p_{\text{drop}}$（既定 0.1）で null に drop** することで、同一モデル $\epsilon_\theta$ が

- $\epsilon_\theta(x_t, t, c)$：条件付き予測
- $\epsilon_\theta(x_t, t, \emptyset)$：無条件予測

の両方を担います（§6.4.3 の `cfg_drop_prob`）。推論時に

$$
\tilde\epsilon = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
$$

の形で **guidance scale $w$** を可変にできます。

- $w = 1.0$：純粋な条件付き生成（CFG なし）
- $w > 1.0$：条件方向を **強化**（多様性↓、条件遵守↑）
- $w \gg 1.0$：**学習分布外への外挿リスク急増**（§6.7 の推奨レンジ表）

### 6.6.2 CFG の Skill 契約への影響

Ch4 §4.3.4 `condition_variables_declared` と組み合わせ、次の宣言を Skill YAML に埋めます（§6.8）：

- `condition_variables_declared`: 学習時に登録された条件変数（**推論時に追加禁止**、Ch5 §5.4 `condition_extensibility.allow_agent_to_add_conditions=false` 契約を継承、Ch4 §4.7 の安全性原理に接続）
- `guidance_scale`（top-level、Ch5 §5.7 と同型）: 推奨レンジ内（§6.7）
- `disallowed_operations`:
  - `"guidance_scale_out_of_recommended_range"`
  - `"add_condition_not_in_declared"`

### 6.6.3 条件変数の正規化契約（Ch5 §5.4 継承）

Ch5 §5.4 の canonical 契約を継承します：**cVAE / cDiffusion の条件変数は、学習時と推論時で同じ正規化 (mean / std) を使う**。Skill YAML の `condition_variables_declared[].normalization` に `mean` / `std` / `applied_at` を宣言し、推論時の逸脱は **fetch failure と同等のハード停止**（「条件変数のスケールが違いますが近い値で補完しました」は **Ch6 独自 `disallowed_operations.condition_scale_interpolated`（§6.8）** として禁止、Ch4 §4.7 の安全性原理に接続）。

---

## 6.7 Skill 契約：サンプリング温度と guidance scale の canonical 推奨レンジ表

本節は **Ch4 §4.3.5 `generation_temperature` の Ch6 拡張版（dict 化した `sampling_config`）の canonical 実装**として、以降の Ch7-15 が参照する **推奨レンジ表** を定めます。

### 6.7.1 推奨レンジ表（教科書的既定値、YAML block 1）

**YAML block 1 — Diffusion サンプリング推奨レンジ（`arim.gen.diffusion.v0.1` canonical）**

```yaml
# =============================================================
# Ch6 §6.7 canonical: Diffusion サンプリング推奨レンジ表
# 参照: Ch4 §4.3.5 generation_temperature を Ch6 で dict 拡張した sampling_config の allowed_range
# 用途: 以降 Ch7 (Flow/AR)・Ch12 (capstone)・Ch13 (分子/結晶) が本表を参照
# =============================================================
recommended_ranges:
  sampler_type:
    allowed_values: ["ddpm", "ddim"]     # 本章 scope。dpm_solver は v0.2 で追加予定
    default: "ddim"
  n_sampling_steps:
    ddpm:
      allowed_range: [500, 1000]
      default: 1000
      rationale: >
        DDPM は ancestral sampling で full T ステップ実行が理論的既定。
        n_sampling_steps は provenance 表明値として training-time T と一致する必要がある
        （sample_diffusion 内で assert）。NFE 削減は sampler_type='ddim' を使う。
        推奨は 1000（既定）、資源制約下で 500 まで T を下げて再学習は可能。
    ddim:
      allowed_range: [20, 100]
      default: 50
      rationale: >
        画像領域の初期研究（Song et al., 2021）では 50 ステップで DDPM 1000 ステップ相当の
        FID 品質が報告された。材料生成領域では対象 FM ごとに再検証が必要（§6.3.2, §6.10.4）。
  eta:                                    # DDIM stochasticity
    allowed_range: [0.0, 1.0]
    default: 0.0
    rationale: "0.0 = deterministic（再現性確保）、1.0 = DDPM 相当"
  guidance_scale:                         # CFG w
    allowed_range: [1.0, 7.5]
    default: 3.0
    rationale: >
      w=1.0 は条件遵守が弱い。w>7.5 は学習分布外への外挿リスク急増（§6.10 失敗パターン）。
      教科書的推奨は 2.0-5.0（Ho & Salimans, 2022）。
  temperature_scaling:                    # softmax 組成復元時の温度（§6.4.1）
    allowed_range: [0.8, 1.2]
    default: 1.0
    rationale: >
      x を softmax(x / temperature_scaling) で組成に射影する際の温度。1.0 が理論的既定。
      逸脱時は組成が one-hot 化（<1.0）または uniform 化（>1.0）し、学習分布の性質が崩れる（§6.10）。
```

### 6.7.2 「エージェントが勝手に上げない」契約

**canonical 契約**：Ch4 §4.6 Safety 3-layer Layer 1 の filter 実装として、Skill は **推論呼び出し時に以下を検証**します。

**コードスニペット 7 — canonical 温度・guidance 検証関数**

```python
def validate_sampling_config(
    sampler_type: str, n_sampling_steps: int, eta: float,
    guidance_scale: float, temperature_scaling: float,
) -> None:
    """
    Ch6 §6.7 canonical 検証。逸脱時は Ch6 独自 `disallowed_operations.guidance_scale_out_of_recommended_range`
    （§6.8）として hard_reject。Ch4 §4.7 の安全性原理（canonical field vs 自然言語判定の飛躍禁止）
    に接続する Ch6 拡張：「guidance scale を少し上げれば条件を満たします」→ 禁止。
    Ch4 §4.6 Layer 1 の必要条件（十分条件は Layer 2/3 の governance）。
    """
    if sampler_type not in ("ddpm", "ddim"):
        raise ValueError(f"disallowed sampler_type: {sampler_type} (Ch6 §6.7)")

    ranges = {
        "ddpm": (500, 1000),
        "ddim": (20, 100),
    }
    lo, hi = ranges[sampler_type]
    if not (lo <= n_sampling_steps <= hi):
        raise ValueError(
            f"n_sampling_steps out of recommended range for {sampler_type}: "
            f"got {n_sampling_steps}, allowed [{lo}, {hi}]"
        )

    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta out of range: {eta}")

    if not (1.0 <= guidance_scale <= 7.5):
        raise ValueError(
            f"guidance_scale out of recommended range: got {guidance_scale}, "
            f"allowed [1.0, 7.5] (§6.7, Ho & Salimans, 2022)"
        )

    if not (0.8 <= temperature_scaling <= 1.2):
        raise ValueError(f"temperature_scaling out of range: {temperature_scaling}")
```

### 6.7.3 Ch4 §4.3 との対応マップ

| Ch6 §6.7 field | Ch4 §4.3 canonical | Skill YAML パス（§6.8 と一致） |
|---|---|---|
| `sampler_type` | §4.3.5 を Ch6 §6.7 で dict 拡張（付録 A で back-port 予定） | `sampling_config.sampler_type` |
| `n_sampling_steps` | 同上 | `sampling_config.n_sampling_steps` |
| `eta` | 同上 | `sampling_config.eta` |
| `guidance_scale` | §4.3.5（Ch5 §5.7 と同型で top-level） | `guidance_scale` |
| `temperature_scaling` | §4.3.5 を Ch6 §6.7 で dict 拡張 | `sampling_config.temperature_scaling` |

---

## 6.8 `arim.gen.diffusion.v0.1` Skill YAML 完全形

Ch4 §4.2 template と Ch5 §5.7 の親 Skill YAML を継承した、**Route A / Route B 両モード宣言つきの完全形**です。

**YAML block 2 — `arim.gen.diffusion.v0.1` full Skill 契約書**

```yaml
# =============================================================
# arim.gen.diffusion.v0.1 — 組成/結晶 Diffusion Skill（Ch6 canonical 実装）
# 継承: Ch4 §4.2 template ⑧ を literal に継承、Ch5 §5.7 と一貫
# =============================================================
skill:
  id: "arim.gen.diffusion.v0.1"
  version: "v0.1"
  description: "組成 Diffusion (Route A) / 結晶 Diffusion (Route B) による分布内サンプリング。Ch6 canonical 実装。"

# --- Ch2 §2.7 canonical pipeline stage anchor ---
pipeline_position:
  stage: "G"
  upstream_stages: ["PRE"]                # library_allowlist / fm_fetch
  downstream_stages: ["SF", "F", "H"]

# --- 入力仕様 ---
input_schema:
  mode:
    type: "enum"
    values: ["train", "generate"]
  route:
    type: "enum"
    values: ["A_scratch", "B_pretrained"]
  # Route A (train) 入力
  training_data:
    type: "dict"
    required_keys: ["dataset_id", "version", "sha256", "license"]
    condition: "required if route=A_scratch and mode=train"
  hyperparameters:
    type: "dict"
    schema:
      T: {type: "int", range: [500, 1000]}   # §6.7 canonical DDPM range と一致（DDIM は T=1000 で学習して n_sampling_steps=50 で推論）
      beta_schedule: {type: "enum", values: ["linear", "cosine"]}
      loss_variant: {type: "enum", values: ["l_simple"]}
      arch: {type: "enum", values: ["mlp_denoiser"]}     # Route A のみ
      d_hidden: {type: "int", range: [32, 512]}
      d_time: {type: "int", range: [16, 128], multiple_of: 2}  # TimeEmbedding が偶数次元を要求（§6.4.2）
      cfg_drop_prob: {type: "float", range: [0.05, 0.2]}
      max_epochs: {type: "int", range: [1, 500]}
      batch_size: {type: "int"}
      lr: {type: "float"}
  # Route B 入力
  foundation_model_reference:
    type: "dict"
    schema_ref: "foundation_model_reference"  # Ch6 §6.5 canonical 拡張（Ch4 §4.3.6 は filter_rules_applied）
    condition: "required if route=B_pretrained"
  # generate mode 入力
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
    schema_ref: "sampling_config"                 # Ch6 §6.7 canonical 拡張（Ch4 §4.3.5 の dict 化）
    condition: "required if mode=generate"

# --- 出力仕様 ---
output_schema:
  trained_model_artifact_id:
    type: "string"
    condition: "emitted if mode=train"
  training_metrics:
    type: "dict"
    keys: ["final_l_simple", "final_epoch", "cfg_drop_prob_effective"]
  candidates_out:
    type: "list_of_dict"
    required_keys: ["candidate_id", "generated_x", "sampling_trajectory_id"]
    max_length_field: "n_samples"
  detection_result:
    type: "list_of_dict"
    schema_ref: "hallucinatory_composition_detection.result"

# --- 生成 × 逆設計 13+1 provenance fields（Ch4 §4.3 完全定義、Ch5 §5.7 継承） ---
generative_model_family:
  family: "diffusion"                     # Ch4 §4.3.1 canonical enum
  id: "arim.ch6.composition_diffusion_v0.1"  # Route A の場合
  version: "v0.1.0"
  hyperparameters:                        # Ch4 §4.3.1 hyperparameters 拡張
    T: 1000
    beta_schedule: "cosine"
    loss_variant: "l_simple"
    arch: "mlp_denoiser"
    cfg_drop_prob: 0.1
latent_dim: null                          # Ch4 §4.3.2、Diffusion は latent なし（null 許容）
training_data_provenance:                 # Ch4 §4.3.3、Route A の場合のみ有効
  dataset_id: "arim.synthetic-generative.compositions.v1"
  version: "v1.0.0"
  sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  license: "arim_educational_v1"
condition_variables_declared:             # Ch4 §4.3.4、CFG で使用（Ch5 §5.4 継承）
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
condition_extensibility:                  # Ch5 §5.4 canonical（Ch4 未定義、付録 A で back-port 予定）
  allow_agent_to_add_conditions: false
  additional_conditions_require: "human_approval"
generation_temperature: 1.0               # Ch4 §4.3.5 scalar canonical（Ch5 §5.7 と同型）
generation_temperature_range_allowed: [0.8, 1.2]
guidance_scale: 3.0                       # Ch4 §4.3.5 canonical（VAE では null、Diffusion は §6.7 レンジ）
guidance_scale_range_allowed: [1.0, 7.5]  # §6.7 canonical レンジ
sampling_config:                          # Ch6 §6.7 で先行導入、付録 A で back-port 予定（Ch4 §4.3.5 拡張）
  sampler_type: "ddim"                    # canonical enum: ddpm | ddim
  n_sampling_steps: 50
  eta: 0.0                                # DDIM stochasticity（0.0 = deterministic）
  temperature_scaling: 1.0                # softmax の温度（sample_diffusion の実装引数と一致、§6.4.4）
  n_sampling_steps_range_allowed:         # §6.7 canonical レンジ（sampler_type ごと）
    ddpm: [500, 1000]
    ddim: [20, 100]
  eta_range_allowed: [0.0, 1.0]
  temperature_scaling_range_allowed: [0.8, 1.2]
foundation_model_reference:               # Ch6 §6.5 で先行導入（Ch4 §4.3.6 は filter_rules_applied、付録 A で back-port 予定）
  # 本 field は Route B の場合のみ non-null（Route A では null）
  # fetch_channel の canonical enum: hf_hub | github_release | google_drive_manual
  # 以下は MatterGen の場合の例（DiffCSP / CDVAE の例は本 YAML 直後の Note 参照）
  name: "microsoft/mattergen"
  fetch_channel: "hf_hub"
  revision_commit_hash: "0000000000000000000000000000000000000000"
  weights_sha256:
    "config.json": "sha256:aaaa000000000000000000000000000000000000000000000000000000000000"
    "model.safetensors": "sha256:bbbb000000000000000000000000000000000000000000000000000000000000"
  license: "MIT"
  fetched_at_utc: "2026-07-07T00:00:00Z"
filter_rules_applied: []                  # Ch4 §4.3.6、本 Skill は F ステージを実装しない（Ch8 に委譲）
filter_rules_delegated_to:                # Ch3 §3.6 `_delegated_to` pattern（Ch5 §5.7 と同型）
  skill: "arim.gen.physics_filter.v0.1"
  skill_version: "v0.1"
  invocation_stage: "F"
screening_model_family: null              # Ch4 §4.3.7、本 Skill は screening を実装しない
top_k_returned: null                      # Ch4 §4.3.8、本 Skill は ranking を実装しない（Ch11 に委譲）
inverse_design_authorization: "autonomous"  # Ch4 §4.3.9 enum、G ステージなので autonomous
synthesis_launch_authorization: null      # Ch4 §4.4、A ステージのみ発火、本 Skill では null
hallucinatory_composition_detection_delegated_to:
  # H ステージは arim.gen.ood_detection.v0.1 に委譲（Ch4 §4.5 delegated 態 canonical、Ch3 §3.6 pattern、Ch5 §5.7 と同型）
  skill: "arim.gen.ood_detection.v0.1"
  skill_version: "v0.1"
  invocation_stage: "H"
safety_screening_passed: false            # Ch4 §4.6 Layer 1 は SF ステージ、本 Skill では未通過
dual_use_review_completed: false          # Ch4 §4.6 Layer 2 は D ステージ

# --- 権限ゲート（Ch4 §4.2 canonical、Ch4 §4.8 の vol-06 独自 gate 名 space、Ch5 §5.7 と同型） ---
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

# --- 自然言語出力禁止契約（Ch4 §4.7 canonical 10 phrase と完全同期、Ch5 §5.7 と同一 literal） ---
outputs_disallowed_natural_language:
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

# --- Ch6 独自: Diffusion サンプリング逸脱の禁止契約（§6.7 参照、Ch5 §5.7 disallowed_operations と同型） ---
disallowed_operations:
  - name: "guidance_scale_out_of_recommended_range"
    description: "§6.7 canonical [1.0, 7.5] からの逸脱は学習分布外への外挿を招く（Ch4 §4.7 の安全性原理に接続する Ch6 拡張）"
    enforcement: "hard_reject"
  - name: "n_sampling_steps_out_of_recommended_range"
    description: "§6.7 canonical DDIM=[20,100] / DDPM=[500,1000] からの逸脱は品質保証範囲外"
    enforcement: "hard_reject"
  - name: "sampler_type_out_of_allowed"
    description: "allowed_values=[ddpm, ddim] 外の sampler を自律で選択（dpm_solver 等は v0.2 で追加）"
    enforcement: "hard_reject"
  - name: "add_condition_not_in_declared"
    description: "Ch5 §5.4 `condition_extensibility` 継承。allow_agent_to_add_conditions=false（Ch4 §4.7 の安全性原理に接続）"
    enforcement: "hard_reject"
  - name: "condition_scale_interpolated"
    description: "Ch5 §5.4 継承。正規化 mean/std の変更禁止（Ch4 §4.7 の安全性原理に接続する Ch6 拡張）"
    enforcement: "hard_reject_and_audit_log"
  - name: "modify_beta_schedule_at_inference"
    description: "学習時に固定した beta_schedule / T の推論時変更禁止"
    enforcement: "hard_reject"
  - name: "fetch_failed_substitute"
    description: "§6.5.5 継承。fetch 失敗時の代替 FM 継続禁止（沈黙する、Ch4 §4.7 の安全性原理に接続する Ch6 拡張）"
    enforcement: "hard_reject_and_audit_log"

# --- provenance 5 必須（vol-01 継承、Ch5 §5.7 と同型） ---
provenance:
  input_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  skill_version: "v0.1"
  run_datetime_utc: "2026-07-07T00:00:00Z"
  package_versions:
    torch: "2.4.1"
    numpy: "2.0.2"
    huggingface_hub: "0.24.7"             # Route B 用
    gdown: "5.2.0"                        # Route B 用（DiffCSP）
  random_seed: 20260707
  event_hash: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
```

> [!NOTE]
> **本 Skill YAML の観察点**（Ch5 §5.7 との差分中心）：
> - `generative_model_family.family: "diffusion"`（Ch5 は `"vae"`）と `hyperparameters` の中身が Diffusion 固有（`T` / `beta_schedule` / `loss_variant` / `cfg_drop_prob`）
> - `guidance_scale: 3.0`（Ch5 は `null`）— Diffusion では non-null が canonical、`guidance_scale_range_allowed` を併記
> - `sampling_config` は **本章 §6.7 で先行導入した canonical 拡張**（Ch4 §4.3.5 に `sampler_type` / `n_sampling_steps` / `eta` / `temperature_scaling` を dict 化して追加。付録 A で back-port 予定）
> - `foundation_model_reference` は **本章 §6.5 で先行導入**（Ch4 §4.3.6 は `filter_rules_applied` のみ canonical、`foundation_model_reference` は付録 A で back-port 予定）。上例は MatterGen だが、DiffCSP / CDVAE の場合の literal は次段落を参照
> - `authorization_gates` は Ch5 §5.7 と同一の 4 gate dict（L1-L4 canonical）、gate 名 space も `vol-06:` prefix を継承
> - `outputs_disallowed_natural_language` は Ch4 §4.7 の 10 literal 日本語 phrase を Ch5 §5.7 と一字一句同じに記載。Diffusion 固有の Skill 契約違反（guidance scale / fetch / condition scale / n_sampling_steps / beta_schedule）は Ch6 §6.8 `disallowed_operations` の Ch6 拡張項目として並存（Ch4 §4.7 の safety 原理を共有）
> - `hallucinatory_composition_detection_delegated_to` により H ステージを `arim.gen.ood_detection.v0.1` に委譲（G ステージ Skill は Ch5 §5.7 と同じ delegated 態 canonical）

**`foundation_model_reference` の Route B 3 経路 literal 例**（`fetch_channel` ごとに required key が異なる）：

```yaml
# --- 例 1: HF Hub（MatterGen） --- ※ 上記 Skill YAML 本体に記載
foundation_model_reference:
  name: "microsoft/mattergen"
  fetch_channel: "hf_hub"
  revision_commit_hash: "0000000000000000000000000000000000000000"
  weights_sha256:
    "config.json": "sha256:aaaa000000000000000000000000000000000000000000000000000000000000"
    "model.safetensors": "sha256:bbbb000000000000000000000000000000000000000000000000000000000000"
  license: "MIT"
  fetched_at_utc: "2026-07-07T00:00:00Z"

# --- 例 2: Google Drive（DiffCSP） ---
foundation_model_reference:
  name: "diffcsp_perov_5"
  fetch_channel: "google_drive_manual"
  google_drive_file_id: "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456"
  weights_sha256:
    "checkpoint.ckpt": "sha256:cccc000000000000000000000000000000000000000000000000000000000000"
  license: "non_commercial_academic"      # 論文 repo の LICENSE を要確認
  fetched_at_utc: "2026-07-07T00:00:00Z"

# --- 例 3: GitHub Release（CDVAE） ---
foundation_model_reference:
  name: "cdvae"
  fetch_channel: "github_release"
  github_repo: "txie-93/cdvae"
  git_tag: "v1.0.0"                       # immutable tag（branch は禁止、Ch4 §4.6 Layer 1 の必要条件）
  release_asset: "prop_models/carbon.ckpt"
  weights_sha256:
    "prop_models/carbon.ckpt": "sha256:dddd000000000000000000000000000000000000000000000000000000000000"
  license: "MIT"
  fetched_at_utc: "2026-07-07T00:00:00Z"
```

---

## 6.9 Optional 節：結晶グラフ拡張（`dgl` / `torch-geometric`）

Route A で結晶構造レベルの生成に踏み込みたい場合、原子座標と格子を扱うため graph neural network 層が必要になります。**教育目的では単一 GPU に収まる範囲**（最大 32 原子 unit cell）に限定します。

### 6.9.1 追加ライブラリと制約

- **`dgl`**: DGL Diffusion baseline に対応。CPU-only または CUDA 11.8。
- **`torch-geometric`**: PyG での equivariant network 実装（DimeNet++ / SchNet 系）。

**Ch4 §4.3.6 は `filter_rules_applied` が canonical**。本章 §6.9 の追加 package は Ch6 §6.5 で先行導入した `foundation_model_reference.package_versions` に宣言します（付録 A で back-port 予定）：

```yaml
package_versions:
  dgl: "2.1.0"                            # Optional, Ch6 §6.9
  torch-geometric: "2.5.3"                # Optional, Ch6 §6.9
```

### 6.9.2 単一 GPU 制約の Skill 契約

**32 原子上限**を Skill 側で強制：

```yaml
input_schema:
  hyperparameters:
    max_atoms_per_unit_cell:
      type: "int"
      range: [1, 32]                      # §6.9 教育目的 scope
      default: 20
```

大規模 unit cell は Route B（MatterGen 等の既存 FM）へ委譲。

### 6.9.3 本節の位置付け

結晶グラフ拡張の詳細な理論と実装は **Ch13 §13b（結晶構造逆設計）** で扱います。本節は「Route A の scope 内で結晶グラフに触れる場合の Skill 契約」に留めます。

---

## 6.10 よくある失敗と対処

### 6.10.1 サンプリング温度の暴走（Ch14 §14.1 で正式に taxonomize）

**症状**：`guidance_scale = 15.0` などで実行すると、条件遵守は表面上高いが、**学習分布外の候補**が量産される。

**Skill レベルの対処**：§6.7 の canonical レンジで validate（§6.7.2 のコード）し、逸脱時は Ch6 独自 `disallowed_operations.guidance_scale_out_of_recommended_range`（§6.8）として **禁止**（Ch4 §4.7 の canonical 10 phrase 自体は自然言語 output に関する規約、本項は Skill sampling 契約の Ch6 拡張。安全性原理は共有）。

**Agentic レベルの対処**（Ch14 §14.2 preview）：エージェントが「試行錯誤で $w$ を大きくして条件を通そうとする」場合、Skill YAML の `disallowed_operations` の `guidance_scale_out_of_recommended_range`（§6.8、`enforcement: hard_reject`）と、`authorization_gates.L1_static_filter_pass` で **Skill 呼び出し時点で拒否**。`sampling_config.n_sampling_steps_range_allowed` / `guidance_scale_range_allowed` を Skill が pin し、Agent の delta は認めません（§6.7.2 の validate 関数が Layer 1 の必要条件）。

### 6.10.2 fetch 経路の脆弱性（Google Drive）

**症状**：DiffCSP の Google Drive ファイルが公開停止 / URL 変更 → fetch 失敗。

**Skill レベルの対処**：§6.5.3 のとおり、sha256 mismatch 時は **候補を返さない**（沈黙）。エージェントは Ch4 §4.6 Layer 2 にエスカレート。**Ch6 独自 `disallowed_operations.fetch_failed_substitute`（§6.8）** として「fetch できなかったので代わりに ○○ を使いました」を禁止（Ch4 §4.7 の安全性原理に接続する Ch6 拡張）。

### 6.10.3 Mode collapse（生成の多様性喪失）

**症状**：学習後、多くの候補が同じ組成に収束する。

**Diffusion 特有の症状**：VAE の mode collapse とは異なる形で現れます。**特定の条件ベクトル $c$ に対して、CFG で強い guidance をかけると出力分散が消失** します。

**対処**：

1. §6.7 の `guidance_scale` を推奨レンジの下側（1.5-3.0）へ
2. `cfg_drop_prob` を大きく（0.15-0.20）して無条件成分を鍛える
3. Ch10 の distributional coverage 診断（軸 A）で候補分散を測る

### 6.10.4 NFE を勝手に減らす暴走

**症状**：エージェントが「時間短縮のため DDIM のステップ数を 5 まで下げる」判断をする。

**対処**：§6.7 の `n_sampling_steps` レンジ [20, 100] を **allowed_agent_actions に含めない**（Skill が pin）。逸脱時は Ch6 独自 `disallowed_operations.n_sampling_steps_out_of_recommended_range`（§6.8）として hard_reject。

### 6.10.5 条件変数のスケール逸脱（Ch5 §5.4 継承）

**症状**：学習時に mean=2.1, std=1.4 で正規化した `target_band_gap_eV` に、推論時に生値 (unnormalized) を渡す。

**対処**：Skill が入力バリデーションで **必ず正規化 mean/std を再適用**。逸脱時は Ch6 独自 `disallowed_operations.condition_scale_interpolated`（§6.8）として **hard_reject_and_audit_log**（Ch4 §4.7 の安全性原理に接続する Ch6 拡張、Ch5 §5.4 継承）。

### 6.10.6 学習時の cfg_drop_prob 不足

**症状**：`cfg_drop_prob = 0.02` など小さすぎる値で学習 → 無条件モデルが弱く、CFG の効果が薄い。

**対処**：Skill YAML の `hyperparameters.cfg_drop_prob` の allowed_range を [0.05, 0.2] に固定（§6.8 の YAML block 2）。

---

## 6.11 章末チェックリスト

**Skill 実装レベル**

- [ ] `arim.gen.diffusion.v0.1` の `generative_model_family.family = "diffusion"` が Ch4 §4.3.1 canonical と一致するか
- [ ] Route A / Route B の 2 モードが `input_schema.route` で宣言され、`training_data_provenance` と `foundation_model_reference` の条件付き必須性が正しく書けているか
- [ ] `sampling_config` の 4 field（`sampler_type` / `n_sampling_steps` / `eta` / `temperature_scaling`）と top-level `guidance_scale` が §6.7 canonical レンジ内か（Ch4 §4.3.5 の Ch6 拡張）
- [ ] Route B の場合、fetch 経路（HF Hub / GitHub / Google Drive）が Ch3 §3.8 の `library_allowlist` にあるか、sha256 lock が Ch4 §4.6 Layer 1 の必要条件を満たすか
- [ ] `condition_variables_declared[].normalization` の mean / std / applied_at が Ch5 §5.4 継承のフォーマットで書けているか
- [ ] `disallowed_operations` に `guidance_scale_out_of_recommended_range` と `n_sampling_steps_out_of_recommended_range` が含まれているか

**理論レベル**

- [ ] DDPM の forward $q(x_t \mid x_0)$ の閉形式が書けるか
- [ ] $L_{\text{simple}}$ の形と、これが $L_{\text{vlb}}$ の重み付き簡略化であることを説明できるか
- [ ] DDIM が同じ学習済みモデルで NFE を削減できる原理（$\sigma_t = 0$ の deterministic path）を説明できるか
- [ ] CFG の $\tilde\epsilon = \epsilon_\text{uncond} + w (\epsilon_\text{cond} - \epsilon_\text{uncond})$ を書け、$w$ の生成品質への影響を §6.7 の推奨レンジと接続できるか

**Skill 契約レベル**

- [ ] Ch4 §4.7 canonical 10 phrase list は自然言語 output に関する規約（10 literal 日本語 phrase）。本章 Diffusion 固有の Skill 契約違反（guidance scale 暴走、fetch 代替、条件スケール補完、n_sampling_steps 逸脱、beta_schedule 変更）は **Ch4 §4.7 の phrase 番号ではなく、Ch6 §6.8 `disallowed_operations` の Ch6 拡張項目として扱う**（Ch4 §4.7 と Ch6 `disallowed_operations` は安全性原理を共有）ことを説明できるか
- [ ] fetch 失敗時に「代替 FM で継続する」判断を Skill が拒否する契約が書けているか（§6.5.5）
- [ ] Ch14 §14.1（生成一般失敗）と §14.2（Agentic 特有失敗）のうち、本章 §6.10 でカバーしたのが §14.1 系（一般失敗）であり、Agentic 側は Ch12 capstone と Ch14 で扱うことを認識しているか

**次章接続**

- [ ] Ch7（Flow / AR）が §6.7 の推奨レンジ表を **どう継承／差分するか**を予測できるか（Flow は $L_{\text{simple}}$ ではなく exact likelihood、AR は Diffusion と異なる temperature 概念を持つ）

---

## 6.12 参考資料

**理論・原著論文**

- **DDPM（Ho et al., 2020）**：https://arxiv.org/abs/2006.11239 —— §6.2 の骨格
- **DDIM（Song et al., 2021）**：https://arxiv.org/abs/2010.02502 —— §6.3 の deterministic sampling
- **Classifier-Free Guidance（Ho & Salimans, 2022）**：https://arxiv.org/abs/2207.12598 —— §6.6 CFG と §6.7 推奨レンジの一次情報源
- **Improved DDPM（Nichol & Dhariwal, 2021）**：https://arxiv.org/abs/2102.09672 —— cosine beta schedule（§6.2.1 コードスニペット 2）

**材料 Foundation Model**

- **MatterGen（Zeni et al., Microsoft, 2024）**：https://arxiv.org/abs/2312.03687 —— §6.5.2 HF Hub fetch
- **DiffCSP（Jiao et al., 2023）**：https://arxiv.org/abs/2309.04475 —— §6.5.3 Google Drive fetch
- **CDVAE（Xie et al., 2022）**：https://arxiv.org/abs/2110.06197 —— §6.5.4 GitHub fetch

**ライブラリと実装**

- **Hugging Face Hub**：https://huggingface.co/docs/huggingface_hub —— §6.5.2 の `snapshot_download` API
- **gdown**：https://github.com/wkentaro/gdown —— §6.5.3 の Google Drive fetch
- **Diffusers（Hugging Face）**：https://huggingface.co/docs/diffusers —— 本章 scope 外だが実践 route の主流
- **DGL**：https://www.dgl.ai/ —— §6.9 Optional
- **PyTorch Geometric**：https://pytorch-geometric.readthedocs.io —— §6.9 Optional

**本書内参照**

- **Ch4 §4.2 template**：本章 §6.8 の親骨格
- **Ch4 §4.3.1-4.3.9**：`generative_model_family` / `latent_dim` / `training_data_provenance` / `condition_variables_declared` / `generation_temperature` / `filter_rules_applied` / `screening_model_family` / `top_k_returned` / `inverse_design_authorization` の canonical 定義
- **Ch6 §6.5 / §6.7 で先行導入**：`foundation_model_reference`（Ch4 §4.3.6 は `filter_rules_applied` が canonical、Ch6 で先行導入）と `sampling_config`（Ch4 §4.3.5 `generation_temperature` の dict 拡張）。いずれも付録 A で back-port 予定
- **Ch4 §4.5**：`hallucinatory_composition_detection` の 3 態（本章 §6.8 では delegated 態を実装、Ch5 §5.6 で config/result 態、Ch5 §5.7 と同型）
- **Ch4 §4.6**：Safety 3-layer（本章 §6.5.1 と §6.10.2 で Layer 1 filter を実装）
- **Ch4 §4.7**：禁止フレーズ 10 list は canonical 10 literal 日本語 phrase として §6.8 に literal 化。Diffusion 固有の Skill 契約違反は Ch6 §6.8 `disallowed_operations` の Ch6 拡張項目として扱う（Ch4 §4.7 の safety 原理に接続）
- **Ch4 §4.9**：Skill ID 一覧（本章は `arim.gen.diffusion.v0.1`）
- **Ch5 §5.4**：条件変数の正規化契約（本章 §6.6.3 で継承）
- **Ch5 §5.7**：親 Skill YAML（本章 §6.8 で継承）

---

**次章予告**（第7章）：**Normalizing Flow と autoregressive を Skill 化する**。本章 §6.7 の推奨レンジ表を Flow / AR がどう継承／差分するかを最初に整理し、Flow の exact likelihood と AR の temperature scaling を、Ch4 §4.3.5 `generation_temperature` と本章 §6.7 で拡張した `sampling_config` canonical に沿って Skill 化します。§7.4 では **VAE / Diffusion / Flow / AR の 4 家族選択判断表**（データ量 × 制約表現 × 計算コスト × Skill 契約の複雑さ）を提示し、Part II の総合まとめとします。
