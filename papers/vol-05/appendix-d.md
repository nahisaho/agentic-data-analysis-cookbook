# 付録 D: Capstone 発展スクリプト集

> [!IMPORTANT]
> 本付録は、第 14 章 14b の §14.3 で宣言された契約（`objectives_spec` / `constraints_spec` / `batch_diversity_policy` / `pending_experiments` / `parallel_facility_policy` / `acquisition_spec` / `stop_condition v0.3`）を **実装レベル** に展開する。全スクリプトは illustrative reference であり、実運用では付録 A の `arim.bo.*.v0.2` Skill を **付録 B の MCP handler** 経由で走らせること。付録 D のコードを直接運用系に貼付することは Ch16 §16.9 の facility_bo_skill_registry 契約違反にあたる。
>
> **canonical schema 準拠ルール**（付録 A / B / C と共通、vol-05 全体で不変）:
>
> - Skill ID（dot 形）: `arim.bo.multi_objective_qehvi.v0.2` / `arim.bo.constrained_qehvi.v0.2` / `arim.bo.batch_qlognei.v0.2` / `arim.bo.hierarchical_gp.v0.2`
> - Skill identity（slash 形）: `^skill:arim.bo.hierarchical_gp/v0.2` 等（Ch16 §16.1.1 H-R2-1）
> - `authorization_id`: `elauth_YYYYMMDD_HHMMSS_iter<n>`（Ch16 §16.1.1 H-R2-1、canonical）
> - `status` enum: `pending | approved | denied | revoked`（Ch16 §16.1.1 H7）
> - `event_hash`: `"sha256:" + hexdigest(RFC 8785 JCS(payload))`（Ch10 §10.5.2、canonical anchor: `vol-05:ch10:event_canonical_serialization`）
> - 21 フィールド provenance: Group A (11) + Group B (5) + Group C (2) + Group D (3)（付録 A §A.2 canonical outputs schema）
> - Ch14 §14.2.1 canonical 6-var search space: `x_temperature` / `x_pressure` / `x_hold_time` / `x_mass_frac_A` / `x_mass_frac_B` / `x_atmosphere`
> - Linear constraint: `x_mass_frac_A + x_mass_frac_B <= 0.85`（Ch14 §14.2.1）
> - `x_atmosphere`: ordinal encoding `{Ar: 0, N2: 1, Air: 2}` + hamming kernel（Ch6 §6.4 / Ch14 §14.2.2）
> - `task_map`: `{"A": 0, "B": 1, "C": 2}`（Ch11 §11.5 / Ch14 §14.3.1）
> - HITL bypass writer identity（MCP-internal のみ）: `^service:mcp_gate_service`（付録 B §B.3）
>
> **canonical 依存**:
>
> - Ch14 §14.3.3 統合契約 YAML（本付録の全 iteration の Ref 元）
> - Ch14 §14.3.5 `stop_condition v0.3` 6 clauses
> - Ch11 §11.4.2 Hierarchical GP skeleton（本付録 D.2 で executable-adjacent に展開）
> - Ch12 §12.3 `pending_experiments.tensor_encoding_contract.derivation_deterministic: true`
> - 付録 B §B.3 FLAT envelope（`run_id` / `skill_execution_id` / `created_at` / `created_by` が business fields の siblings）
> - 付録 C §C.2 C-BO-05 dataset（本付録の iteration 6-10 が消費する演習データ）

---

## D.1 概観と依存ライブラリ

### D.1.1 スクリプト範囲と Skill 対応表

第 14 章 §14.3.7 で「後半 5 iteration のスクリプトは付録 D で扱う」と宣言されている。本付録が実装対象とするのは以下の 5 iteration（iteration 6-10）と、それを支える横断コンポーネント：

| ID | 対応 §（本付録） | Ch14 契約 | Skill（付録 A）              | 主な目的                                                     |
|:---|:--|:--|:--|:--|
| iter 6  | D.4.1 | §14.3.3       | `arim.bo.hierarchical_gp.v0.2` + `arim.bo.multi_objective_qehvi.v0.2` + `arim.bo.batch_qlognei.v0.2` + `arim.bo.constrained_qehvi.v0.2` | 通常進行（3 装置 batch=4、Pareto 前進）                                            |
| iter 7  | D.4.2 | §14.3.3, §14.2.4 | 同上                                                                                                                                    | `budget_remaining.experiments_available_for_bo` を跨ぐ提案が発生し fatal で reject |
| iter 8  | D.4.3 | §14.3.5       | 同上                                                                                                                                    | `constraint_stall` clause 発火 → `hierarchical_model_degeneracy_alert` に類する Human エスカレーション |
| iter 9  | D.4.4 | §14.3.3, §14.3.6 | 同上                                                                                                                                    | Human から `constraints_spec` 緩和が承認され recovery                                          |
| iter 10 | D.4.5 | §14.3.5, §14.3.7 | 同上                                                                                                                                    | `iterations` clause で正常停止、deliverables を出力                                            |

### D.1.2 依存ライブラリ（provenance に必ず記録する）

`bo_library_stack_metadata.library_version` に固定する。付録 A §A.2 (c) の canonical outputs schema で **immutable pin** に指定されているフィールド。

```yaml
library_versions:
  version: "v0.2"
  runtime:
    python: ">=3.11,<3.13"
    torch: "==2.4.0"
    botorch: "==0.11.3"
    gpytorch: "==1.12"
    numpy: "==1.26.4"
    pandas: "==2.2.2"
    pyyaml: "==6.0.1"
  optional:
    ax_platform: "==0.4.0"       # Ax 経由で generation_strategy を pin する場合のみ
    linear_operator: "==0.5.3"   # gpytorch dependency（explicit pin）
  reproducibility:
    lockfile_sha256: "sha256:3f1c0000abcd1234ef56789012345678"
    resolved_at: "2026-12-11T09:00:00Z"
```

> [!NOTE]
> BoTorch 0.11 系は `qLogNoisyExpectedHypervolumeImprovement` を `botorch.acquisition.multi_objective.logei` からエクスポートする。BoTorch 0.10 以前ではモジュールパスが異なり、また `MultiTaskGP` の `output_tasks` 引数の semantics が微妙に変わっている。**API drift note**: 本付録のコードは 0.11.x を想定。0.12 以降で API 変更があった場合は付録 B §B.3 の `bo_library_stack_metadata` を更新した上で Skill テンプレートを再検証すること。

### D.1.3 コード規約

- Python は 3.11+、type hints を厳格に付与。docstring は英語、行内コメントは日本語（本書全体の慣行）。
- Skill 側から呼び出す関数は `def propose_*` / `def evaluate_*` の名前で、副作用は明示的に return（append-only event 追記は MCP handler 側の責務、付録 B）。
- Skill が MCP handler を経由せず直接 event stream に書き込むのは fatal（付録 B §B.3 (e) `Ch15.event_stream_direct_write` 相当）。
- 乱数 seed は `sequential_seed_provenance` で pin。BoTorch の `optimize_acqf` は seed を受け取らないため、`torch.manual_seed()` + `numpy.random.default_rng()` の 2 段で pin する。

---

## D.2 Hierarchical GP のフル実装

### D.2.1 kernel 定義（k_global + per-task k_dev）

Ch11 §11.4.2 の HierGP skeleton は「実行不可の pseudo-code」だと明示されていた。本付録では、`apply_mask: block_diagonal_by_task` を実装することで **executable-adjacent な reference 実装** に展開する。数学的な covariance は：

$$
k\!\bigl((x_i, t_i), (x_j, t_j)\bigr) = k_{\text{global}}(x_i, x_j) + \mathbb{1}[t_i = t_j] \cdot k_{\text{dev}}^{(t_i)}(x_i, x_j)
$$

- $k_{\text{global}}$: 全装置共通の Matérn 5/2 kernel（ARD、`x_atmosphere` のみ Hamming）
- $k_{\text{dev}}^{(t)}$: 装置 $t$ 固有の deviation kernel（同じく Matérn 5/2 + Hamming 混合、length_scale prior は tighter に）
- 混合カーネル契約は Ch6 §6.4 / Ch14 §14.2.2 の `mixed_kernel` に整合

```python
# ---- hierarchical_gp.py ----
"""Hierarchical GP reference for Ch14b iteration 6-10. Not production.

canonical: Ch11 §11.4.2 (kernel), Ch14 §14.2.2 (matern52+hamming),
Ch14 §14.3.1 (task_map {A:0, B:1, C:2}).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import gpytorch
import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

# Ch14 §14.2.1: 5 continuous + 1 categorical ordinal + 1 task index
_CONTINUOUS_DIMS = [0, 1, 2, 3, 4]   # temp, pressure, hold_time, mass_A, mass_B
_CATEGORICAL_DIM = 5                 # x_atmosphere (Ar=0, N2=1, Air=2)
_TASK_DIM = 6                        # facility_id (A=0, B=1, C=2)


def _build_mixed_kernel(ard_dims, ls_prior, os_prior):
    """k(x, x') = k_matern52(x_cont) + k_hamming(x_cat), wrapped in ScaleKernel."""
    matern = MaternKernel(nu=2.5, ard_num_dims=ard_dims,
                          active_dims=_CONTINUOUS_DIMS,
                          lengthscale_prior=GammaPrior(*ls_prior))
    # CategoricalKernel: exp(-|x-x'|/lengthscale) → Hamming-like for ordinal categorical (Ch6 §6.4)
    hamming = CategoricalKernel(active_dims=[_CATEGORICAL_DIM])
    return ScaleKernel(matern + hamming, outputscale_prior=GammaPrior(*os_prior))


class HierarchicalGP(gpytorch.models.ExactGP):
    """Executable ref for Ch11 §11.4.2:
        cov((x,t),(x',t')) = k_global(x,x') + [t==t'] * k_dev_t(x,x')
    train_X columns: 0..4 continuous (unit cube), 5 x_atmosphere ord, 6 task idx.
    """

    def __init__(self, train_x, train_y, likelihood, num_tasks: int = 3):
        super().__init__(train_x, train_y, likelihood)
        self._num_tasks = num_tasks
        self.mean_module = gpytorch.means.ConstantMean()
        # k_global: shared, looser ls prior
        self.global_kernel = _build_mixed_kernel(
            ard_dims=len(_CONTINUOUS_DIMS), ls_prior=(3.0, 6.0), os_prior=(2.0, 0.15))
        # k_dev_t: per-task, tighter ls prior (device-specific detail)
        self.deviation_kernels = torch.nn.ModuleList([
            _build_mixed_kernel(ard_dims=len(_CONTINUOUS_DIMS),
                                ls_prior=(5.0, 20.0), os_prior=(2.0, 1.0))
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        # cov = k_global + Σ_t M_t ⊙ k_dev_t where M_t = 1[task==t] ⊗ 1[task==t]
        x_input = x[..., :-1]
        task_idx = x[..., -1].long()
        mean = self.mean_module(x_input)
        cov_global = self.global_kernel(x_input).evaluate()
        cov_dev = torch.zeros_like(cov_global)
        for t in range(self._num_tasks):
            m = (task_idx == t).to(cov_global.dtype)
            block_mask = m.unsqueeze(-1) * m.unsqueeze(-2)
            cov_dev = cov_dev + block_mask * self.deviation_kernels[t](x_input).evaluate()
        cov = cov_global + cov_dev + 1e-6 * torch.eye(
            cov_global.shape[-1], dtype=cov_global.dtype, device=cov_global.device)
        return gpytorch.distributions.MultivariateNormal(mean, cov)
```

> [!WARNING]
> **API drift note**: `CategoricalKernel` は BoTorch 0.11+ で提供される。`exp(-|x-x'|/lengthscale)` の形で Hamming-like な距離を実装しており、ordinal encoding された `x_atmosphere`（Ar=0, N2=1, Air=2）に対して canonical に振る舞う。厳密な Hamming 距離を one-hot 展開された categorical に適用したい場合は `x_atmosphere` を 3 one-hot dim に展開し、`HammingIMQKernel(vocab_size=3)` を使う；本付録では簡潔さを優先し ordinal encoding + `CategoricalKernel` を採用している。ライブラリを更新する場合は付録 A §A.6 の Skill テンプレート単体テスト（付録 C C-BO-04 dataset で inter-task correlation の recovery を確認）を再実行し、`bo_library_stack_metadata.library_version` を pin し直すこと。

### D.2.2 MTGP wrapper（BoTorch との bridge）

Ch11 §11.2 の判断表で **HierGP を採用したが acquisition 側は BoTorch の `qLogNEHVI` を再利用したい**。BoTorch の acquisition は `botorch.models.model.Model` インタフェースを要求するので、`gpytorch.models.ExactGP` を BoTorch model に wrap する thin adapter が必要。

```python
from botorch.models.gpytorch import GPyTorchModel


class HierarchicalGPBoTorchModel(HierarchicalGP, GPyTorchModel):
    """BoTorch-compatible wrapper for HierarchicalGP.

    Provides `posterior(X, output_indices=None, observation_noise=False)`
    through GPyTorchModel mixin. `_num_outputs = 1` (single-output GP over
    hardness OR density; use one instance per objective).
    """

    _num_outputs = 1

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        num_tasks: int = 3,
        noise_variance: Optional[torch.Tensor] = None,
    ) -> None:
        if noise_variance is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        else:
            # fixed_noise_gp per Ch14 §14.2.2 canonical
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=noise_variance
            )
        HierarchicalGP.__init__(self, train_x, train_y, likelihood, num_tasks=num_tasks)
```

> [!NOTE]
> Ch9 で multi-objective の場合、目的ごとに **独立に** HierGP を fit する（`ModelListGP`）。目的間の相関を surrogate に載せるのは Ch11 で扱う `MultiTaskGP` 系の別トピック。本 capstone は「目的ごとに HierGP」× 「装置 3 拠点」＝ 6 個の GP を fit する構成。

### D.2.3 fit 手順、length_scale prior、task_correlation_matrix 抽出

```python
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP
from gpytorch.mlls import SumMarginalLogLikelihood


@dataclass(frozen=True)
class FitDiagnostics:
    """Post-fit diagnostics for Ch11 §11.7 / Ch14 §14.3.5 task_diagnostic clause."""
    log_marginal_likelihood_per_objective: dict[str, float]
    global_lengthscales_per_objective: dict[str, list[float]]
    deviation_lengthscales_per_task_per_objective: dict[str, dict[int, list[float]]]
    task_correlation_matrix_per_objective: dict[str, list[list[float]]]
    diagnostic_flag_per_objective: dict[str, str]   # canonical enum


def fit_hierarchical_models(train_x, train_y_multi, noise_variance_multi, num_tasks=3):
    """Fit one HierarchicalGP per objective, return ModelListGP + diagnostics."""
    models = []
    for obj_name in train_y_multi:
        m = HierarchicalGPBoTorchModel(
            train_x=train_x, train_y=train_y_multi[obj_name].unsqueeze(-1),
            num_tasks=num_tasks, noise_variance=noise_variance_multi[obj_name])
        models.append(m)
    model_list = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
    fit_gpytorch_mll(mll)
    return model_list, _extract_diagnostics(model_list, list(train_y_multi.keys()), num_tasks)


def _extract_diagnostics(model_list, objective_names, num_tasks) -> FitDiagnostics:
    """Extract length_scales + proxy task_correlation_matrix from outputscales."""
    lml, global_ls, dev_ls, task_corr, diag_flag = {}, {}, {}, {}, {}
    for obj_name, m in zip(objective_names, model_list.models):
        m.eval()
        with torch.no_grad():
            out = m(m.train_inputs[0])
            lml[obj_name] = float(out.log_prob(m.train_targets).item())
        # additive base kernel: kernels[0] is Matérn (continuous)
        g_matern = m.global_kernel.base_kernel.kernels[0]
        global_ls[obj_name] = g_matern.lengthscale.detach().cpu().flatten().tolist()
        dev_ls[obj_name] = {}
        for t in range(num_tasks):
            d_matern = m.deviation_kernels[t].base_kernel.kernels[0]
            dev_ls[obj_name][t] = d_matern.lengthscale.detach().cpu().flatten().tolist()
        # Proxy task correlation: for true MTGP this is normalized B (Ch11 §11.2.1).
        # For HierGP we approximate via outputscale ratio.
        os_g = float(m.global_kernel.outputscale.detach().cpu().item())
        os_dev = [float(m.deviation_kernels[t].outputscale.detach().cpu().item())
                  for t in range(num_tasks)]
        mat = [[1.0] * num_tasks for _ in range(num_tasks)]
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    denom = os_g + (os_dev[i] * os_dev[j]) ** 0.5
                    mat[i][j] = float(os_g / denom) if denom > 0.0 else 0.0
        task_corr[obj_name] = mat
        # Ch14 §14.3.5 canonical thresholds
        offdiag = [abs(mat[i][j]) for i in range(num_tasks) for j in range(num_tasks) if i != j]
        if min(offdiag) >= 0.95:
            diag_flag[obj_name] = "all_ones_degenerate"
        elif max(offdiag) <= 0.10:
            diag_flag[obj_name] = "identity_degenerate"
        else:
            diag_flag[obj_name] = "healthy_partial_pooling"
    return FitDiagnostics(lml, global_ls, dev_ls, task_corr, diag_flag)
```

> [!NOTE]
> `_extract_diagnostics` の `task_correlation_matrix` は **HierGP 用の proxy** である。Ch11 §11.2.1 で本来の $\tilde{B}$ 行列は MTGP（ICM）の低ランク task covariance から計算される。HierGP では outputscale の比で近似しているため、Ch14 §14.3.5 の閾値（`all_ones_offdiag_min: 0.95` / `identity_offdiag_max_abs: 0.10`）を **HierGP 版として運用** することを付録 A §A.6 の Skill 契約で明示している。

---

## D.3 統合 acquisition (qLogNEHVI + 制約 callable + X_pending)

### D.3.1 制約 callable の構築（feasibility model → probability_feasible）

Ch10 §10.6 の hard_constraints_ceramic_v0_2 と Ch14 §14.3.3 の `constraints_spec` に従い、`y_yield >= 60` を feasibility surrogate（別の `SingleTaskGP`）で推論する。BoTorch の `qLogNEHVI` は `constraints: list[Callable[[Tensor], Tensor]]` を受け取り、**負の値 = feasible / 正の値 = infeasible** の規約で処理する（`sigmoid(-c(y)) ≈ P[feasible]`）。

```python
# ---- constraint_callables.py ----
"""Constraint callables for qLogNEHVI in Ch14 §14.3.3.

canonical: Ch10 §10.3 (layers 1/2/3), Ch14 §14.3.3 (feasibility_probability_threshold: 0.85).
"""
from __future__ import annotations
from typing import Callable
import torch
from botorch.models import ModelListGP


def build_yield_constraint_callable(feasibility_model: ModelListGP,
                                    yield_min: float = 60.0,
                                    yield_scale: float = 100.0) -> Callable:
    """Return c(Y) such that c(Y) <= 0 means feasible.
    Contract: last column of Y_and_yield_sample is y_yield.
    In production, Skill bridges via botorch compute_feasibility_indicator with X-preserving hook.
    """
    def _c(Y_and_yield_sample: torch.Tensor) -> torch.Tensor:
        return (yield_min - Y_and_yield_sample[..., -1]) / yield_scale
    return _c


def build_temperature_constraint_callable(temp_max: float = 800.0,
                                          temp_scale: float = 100.0) -> Callable:
    """Input-space constraint x_temperature <= temp_max.
    Prefer optimize_acqf's inequality_constraints (linear) or bounds; callable is fallback.
    """
    def _c(X: torch.Tensor) -> torch.Tensor:
        return (X[..., 0] * (800.0 - 200.0) + 200.0 - temp_max) / temp_scale
    return _c
```

> [!WARNING]
> **API drift note**: BoTorch 0.11 系では `qLogNoisyExpectedHypervolumeImprovement` の `constraints` 引数は「Y sample を受け取る callable のリスト」である。0.10 以前と feasibility indicator の合成方法（sigmoid スカラー化 vs. multiply）が異なる。上記コードは **contract 定義に近い擬似実装** で、実運用の Skill（付録 A §A.4 `arim.bo.constrained_qehvi.v0.2`）は BoTorch の internal helper (`compute_feasibility_indicator`) を使う。

### D.3.2 X_pending tensor 構築（task_index 含む）

Ch12 §12.3 の `pending_experiments.tensor_encoding_contract.derivation_deterministic: true` に従い、`X_pending` を human-readable な entries から機械的に導出する。

```python
# ---- pending_tensor.py ----
"""Build X_pending from pending_experiments.entries[*] (Ch12 §12.3, Ch14 §14.3.1)."""
from __future__ import annotations
import torch

_TASK_MAP = {"A": 0, "B": 1, "C": 2}
_ATM_MAP = {"Ar": 0, "N2": 1, "Air": 2}
_BOUNDS = {
    "x_temperature": (200.0, 800.0), "x_pressure": (0.1, 5.0),
    "x_hold_time": (30.0, 180.0), "x_mass_frac_A": (0.1, 0.6),
    "x_mass_frac_B": (0.1, 0.5),
}


def _normalize(row: dict) -> list[float]:
    return [(row[k] - lo) / (hi - lo) for k, (lo, hi) in _BOUNDS.items()]


def build_x_pending_tensor(pending_entries: list[dict],
                           dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return X_pending of shape (num_pending, 7). Columns 0..4 unit cube,
    5 x_atmosphere ordinal, 6 task index. Non-canonical facility_id / atmosphere
    or A+B > 0.85 raises ValueError (defense in depth, Layer-3 also checks).
    """
    rows = []
    for e in pending_entries:
        if e["facility_id"] not in _TASK_MAP:
            raise ValueError(f"non-canonical facility_id: {e['facility_id']!r}")
        x = e["x"]
        if x["x_atmosphere"] not in _ATM_MAP:
            raise ValueError(f"non-canonical x_atmosphere: {x['x_atmosphere']!r}")
        if x["x_mass_frac_A"] + x["x_mass_frac_B"] > 0.85:
            raise ValueError(f"A+B={x['x_mass_frac_A']+x['x_mass_frac_B']} > 0.85")
        rows.append(_normalize(x) + [float(_ATM_MAP[x["x_atmosphere"]]),
                                     float(_TASK_MAP[e["facility_id"]])])
    return torch.tensor(rows, dtype=dtype)
```

### D.3.3 optimize_acqf(sequential=True, fixed_features) で装置固定 batch

`experiment_launch_authorization.scope.per_facility_batch_size: {A: 2, B: 1, C: 1}`（Ch14 §14.3.3）を尊重するため、装置ごとに `fixed_features={6: task_id}` を pin して sequential greedy で候補を選ぶ。

```python
# ---- propose_batch.py ----
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
import torch


def propose_next_batch(model_list, ref_point, X_baseline, X_pending,
                      per_facility_batch_size, bounds_normalized,
                      constraints_callables, seed):
    """Sequential greedy per-facility fixed_features batch proposal.
    Returns list of tensors, one per facility, each shape (q_facility, 7).
    """
    torch.manual_seed(seed)
    task_map = {"A": 0, "B": 1, "C": 2}
    candidates_all = []
    for facility_id, q_fac in per_facility_batch_size.items():
        if q_fac <= 0:
            continue
        task_idx = task_map[facility_id]
        # cross_facility fantasize: append previously-selected candidates to X_pending
        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model_list, ref_point=ref_point, X_baseline=X_baseline,
            X_pending=torch.cat([X_pending] + candidates_all, dim=0) if candidates_all else X_pending,
            prune_baseline=True, constraints=constraints_callables)
        cand, _ = optimize_acqf(
            acq_function=acqf, bounds=bounds_normalized, q=q_fac,
            num_restarts=20, raw_samples=512,
            sequential=True,                       # Ch12 §12.4.2
            # shape contract: X arrives (..., q, 7) with task_index (dim 6) pinned by fixed_features before posterior() call
            fixed_features={6: float(task_idx)})   # pin task dim to this facility
        candidates_all.append(cand)
    return candidates_all
```

> [!IMPORTANT]
> `X_pending` の中に「前 facility ですでに選ばれた候補」を append しながら次 facility の acquisition を回すことで、**cross_facility fantasize**（Ch11 §11.5 `pending_experiments_visibility: cross_facility`）を実現している。この順序を破ると Ch12 §12.7 の `duplicate_candidate_events` が発火し得る。

---

## D.4 Iteration 6-10 スクリプト（メインループ）

### D.4.1 Iteration 6: 通常進行

前提: iteration 1-5 が正常完了。iteration 5 で `task_correlation_matrix.diagnostic = "healthy_partial_pooling"`（§14.3.6 のエスカレーションは発生しなかった別トラジェクトリ）。C-BO-05 dataset（付録 C §C.2）から 20 点の初期 + iter 1-5 の 5 iter × batch_size 4 = 20 点、合計 40 点が observed。

```yaml
# iteration 6 の canonical contract snapshot（Ch14 §14.3.3 に整合）
iteration_index: 6
run_id: "run-2026-1210-04"
skill_execution_id: "skill-exec-20261216-0930-iter6"
created_at: "2026-12-16T09:30:00Z"
created_by: "^skill:arim.bo.hierarchical_gp/v0.2"

# ---- Ch11/Ch14 §14.3.2 surrogate ----
surrogate:
  version: "v0.2"
  model_family: "hierarchical_gp"
  kernel_spec: {global_kernel: "matern52+hamming (mixed, ARD)", deviation_kernels_per_task: "matern52+hamming (tighter ls prior)", canonical_ref: "vol-05:appendix-d:D.2.1"}
  fit_diagnostics:
    log_marginal_likelihood_per_objective: {y_hardness: -18.9, y_density: -22.4}
    task_correlation_matrix_per_objective:
      y_hardness: [[1.00, 0.71, 0.66], [0.71, 1.00, 0.60], [0.66, 0.60, 1.00]]
      y_density:  [[1.00, 0.68, 0.63], [0.68, 1.00, 0.59], [0.63, 0.59, 1.00]]
    diagnostic_flag_per_objective: {y_hardness: "healthy_partial_pooling", y_density: "healthy_partial_pooling"}

# ---- Ch14 §14.3.3 統合契約 ----
objectives_spec:  {ref: "vol-05:ch14:objectives_spec_v0_2"}
constraints_spec: {ref: "vol-05:ch14:constraints_spec_v0_2"}
batch_diversity_policy:
  version: "v0.2"
  batch_size: 4
  strategy: "joint_acquisition"
  minimum_pairwise_distance: {metric: "standardized_euclidean", threshold: 0.3, threshold_source: "human_declared", enforcement: "post_hoc_skill_validation"}
  fallback_on_violation: "sequential_greedy_replan"

acquisition_spec:
  name: "qLogNEHVI"
  ref_point_source: "objectives_spec.reference.reference_point"
  constraints: "constraints_spec.hard_constraints (via callables)"

experiment_launch_authorization:
  authorization_id: "elauth_20261216_093000_iter6"
  run_id: "run-2026-1210-04"
  skill_execution_id: "skill-exec-20261216-0930-iter6"
  created_at: "2026-12-16T09:35:00Z"
  created_by: "^skill:arim.bo.constrained_qehvi/v0.2"
  status: "approved"
  approved_by: "^human:project_lead_XXXX"
  scope:
    batch_size: 4
    per_facility_batch_size: {A: 2, B: 1, C: 1}
    authorized_candidate_ids: ["cand-iter6-A1", "cand-iter6-A2", "cand-iter6-B1", "cand-iter6-C1"]
    candidate_facility_map:
      cand-iter6-A1: A
      cand-iter6-A2: A
      cand-iter6-B1: B
      cand-iter6-C1: C
    max_experiments_authorized: 4
  approved_at: "2026-12-16T09:35:00Z"
  event_hash: "sha256:aaaa0000iter6auth"
  previous_event_hash: "sha256:zzzz9999iter5tail"

candidates:
  - {candidate_id: "cand-iter6-A1", facility_id: "A", x: {x_temperature: 705.2, x_pressure: 3.10, x_hold_time: 98.0, x_mass_frac_A: 0.38, x_mass_frac_B: 0.29, x_atmosphere: "Ar"}, acquisition_value: 1.62}
  - {candidate_id: "cand-iter6-A2", facility_id: "A", x: {x_temperature: 745.0, x_pressure: 3.45, x_hold_time: 112.0, x_mass_frac_A: 0.41, x_mass_frac_B: 0.30, x_atmosphere: "Ar"}, acquisition_value: 1.48}
  - {candidate_id: "cand-iter6-B1", facility_id: "B", x: {x_temperature: 690.0, x_pressure: 2.90, x_hold_time: 105.0, x_mass_frac_A: 0.36, x_mass_frac_B: 0.31, x_atmosphere: "Ar"}, acquisition_value: 1.55}
  - {candidate_id: "cand-iter6-C1", facility_id: "C", x: {x_temperature: 720.0, x_pressure: 3.20, x_hold_time: 100.0, x_mass_frac_A: 0.40, x_mass_frac_B: 0.28, x_atmosphere: "N2"}, acquisition_value: 1.40}

batch_diversity_metrics:
  min_pairwise_distance: 0.44
  max_pairwise_distance: 1.72
  post_hoc_validation_passed: true

budget_remaining:
  version: "v0.2"
  total_experiments: 60
  experiments_completed: 40         # 20 init + iter 1-5 × 4 = 40
  experiments_remaining: 20
  pending_experiments: 0
  reserved_for_confirmation_experiments: 4
  experiments_available_for_bo: 16
  batch_size_ceiling: 16
```

以下、iteration 6 の実行スクリプト（メインループの 1 反復分）：

```python
# ---- capstone_iteration.py ----
"""Reference driver for Ch14b iteration 6. Not for production; use appendix B §B.3."""
from __future__ import annotations
import torch
from hierarchical_gp import fit_hierarchical_models
from pending_tensor import build_x_pending_tensor
from constraint_callables import build_yield_constraint_callable
from propose_batch import propose_next_batch


def run_iteration_6(state: dict) -> dict:
    """Execute iteration 6 of Ch14b Capstone (illustrative only)."""
    model_list, diag = fit_hierarchical_models(
        state["train_x"], state["train_y_multi"],
        state["noise_variance_multi"], num_tasks=3)
    # task_diagnostic clause pre-check (Ch14 §14.3.5)
    for obj_name, flag in diag.diagnostic_flag_per_objective.items():
        if flag in ("all_ones_degenerate", "identity_degenerate"):
            raise StopIterationSignal(reason="task_diagnostic",
                                      obj_name=obj_name, diagnostic=flag)
    x_pending = build_x_pending_tensor(state["pending_entries"])
    yield_c = build_yield_constraint_callable(state["feasibility_model"], yield_min=60.0)
    cand_per_fac = propose_next_batch(
        model_list=model_list,
        ref_point=torch.tensor([7.0, 5.0], dtype=torch.float64),   # Ch14 §14.3.3
        X_baseline=state["train_x"], X_pending=x_pending,
        per_facility_batch_size={"A": 2, "B": 1, "C": 1},
        bounds_normalized=state["bounds_normalized"],
        constraints_callables=[yield_c],
        seed=state["sequential_seed_provenance"]["iter6"])
    return {"iteration_index": 6, "candidates": cand_per_fac,
            "fit_diagnostics": diag,
            "provenance": _emit_provenance(state, cand_per_fac, diag)}


class StopIterationSignal(RuntimeError):
    """Raised when a stop_condition clause is met; Skill must halt candidate generation."""
    def __init__(self, reason: str, **kwargs) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = kwargs
```

### D.4.2 Iteration 7: budget_remaining チェック発火例

iteration 6 の 4 実験のうち 2 実験（A の 2 件）が完了 → `experiments_completed = 40 + 2 = 42`, `experiments_remaining = 18`, pending には iter 6 の B, C 各 1 が持ち越しで 2。従って `experiments_available_for_bo = 18 - 2 - 4 = 12`。iteration 7 の候補生成前に、Skill が `batch_size = 4` を要求しても pass するが、`batch_size_ceiling = experiments_available_for_bo = 12` の範囲でしか承認されない。

```yaml
iteration_index: 7
budget_remaining:
  version: "v0.2"
  experiments_completed: 42
  experiments_remaining: 18
  pending_experiments: 2                # iter 6 のうち B, C 各 1 が持ち越し
  reserved_for_confirmation_experiments: 4
  experiments_available_for_bo: 12
  batch_size_ceiling: 12
  # ★ この iteration では batch_size=4 を維持可能。しかし Skill が「もう少し攻めよう」と
  #   誤って batch_size=13 を提案した場合、overrun_policies により fatal.

overrun_precheck_events:                 # canonical, append-only (Ch14 §14.2.4 overrun_policies)
  - event_id: "evt-overrun-precheck-iter7-001"
    event_hash: "sha256:beef00007overrun"
    previous_event_hash: "sha256:0000...genesis_overrun_precheck"    # first entry in stream → genesis hash
    run_id: "run-2026-1210-04"
    skill_execution_id: "skill-exec-20261218-1000-iter7"
    created_at: "2026-12-18T10:00:00Z"
    created_by: "^service:mcp_gate_service"
    proposed_batch_size: 13
    proposed_pending_after: 15           # 2 + 13
    experiments_available_for_bo: 12
    verdict: "fatal"
    matched_policy: "on_proposed_batch_over_available"
    action: "reject_proposal"
```

対応コード：

```python
def enforce_budget_ceiling(
    proposed_batch_size: int,
    budget_remaining: dict,
) -> None:
    """Ch14 §14.2.4 overrun_policies enforcement (fatal on breach)."""
    avail = budget_remaining["experiments_available_for_bo"]
    pending = budget_remaining["pending_experiments"]
    if proposed_batch_size > avail:
        raise BudgetOverrunFatal(
            policy="on_proposed_batch_over_available",
            proposed=proposed_batch_size,
            available=avail,
        )
    if proposed_batch_size + pending > budget_remaining["experiments_remaining"] - budget_remaining["reserved_for_confirmation_experiments"]:
        raise BudgetOverrunFatal(
            policy="on_pending_plus_proposed_over_budget",
            proposed=proposed_batch_size,
            pending=pending,
        )


class BudgetOverrunFatal(RuntimeError):
    """Skill が予算超過を提案した場合の fatal。MCP handler が reject_proposal を append。"""

    def __init__(self, policy: str, **details) -> None:
        super().__init__(policy)
        self.policy = policy
        self.details = details
```

Skill 側の応答: `batch_size=13` の提案は `BudgetOverrunFatal` で reject され、Skill は自動で `batch_size=4` に retry する。Skill 自身が clause を緩めるのは fatal（`stop_condition.approval_to_relax: fatal`, §14.2.4）だが、**batch_size を自動で縮める** ことは緩和ではなく制約遵守なので可能。

### D.4.3 Iteration 8: constraint_stall 発火・Human エスカレーション

iteration 7 で 4 実験実行 → 内 3 実験の `y_yield` が 55-58 に停滞（60 未満）。iteration 8 の fit 後、feasibility_probability_average が 0.52 に低下。§14.3.5 の `constraint_stall` clause（`feasibility_probability_average < 0.60 for 2 iter`）が発火する条件を満たすので、iteration 8 では次候補生成前に停止。

```yaml
iteration_index: 8

budget_remaining:
  version: "v0.2"
  total_experiments_authorized: 60
  experiments_completed: 48         # iter 7 start=42 + iter 6 backlog (2) + iter 7 batch (4) all done
  pending_experiments: 0            # iter 8 has not proposed yet due to escalation
  reserved_for_confirmation_experiments: 4
  experiments_remaining: 12         # 60 - 48 = 12
  experiments_available_for_bo: 8   # 12 - 0 - 4 = 8

feasibility_probability_history:            # Group C, append-only. Full entries per D.6.0 schema
                                            # (computed_at / computed_by / event_hash / previous_event_hash) elided for brevity
  - {iteration_index: 6, average: 0.86}
  - {iteration_index: 7, average: 0.58}    # 1 iter 低下（stall 未成立）
  - {iteration_index: 8, average: 0.52}    # 2 iter 連続で 0.60 未満 → clause 発火

stop_condition_evaluation:
  version: "v0.3"
  evaluated_at: "2026-12-22T09:00:00Z"
  clauses_evaluated:
    - {name: "iterations",         satisfied: false, reason: "iteration_index=8 < 10"}
    - {name: "budget",             satisfied: false, reason: "experiments_remaining=12 > 5"}
    - {name: "hypervolume",        satisfied: false, reason: "HV improvement 0.045 > 0.02 * scale=1.5"}
    - {name: "constraint_stall",   satisfied: true,  reason: "feasibility_probability_average < 0.60 for 2 iter (0.58, 0.52)"}
    - {name: "task_diagnostic",    satisfied: false, reason: "diagnostic_flag_per_objective all healthy_partial_pooling"}
    - {name: "human_stop",         satisfied: false, reason: "no signal"}
  combination_rule: "any"
  first_satisfied_clause: "constraint_stall"
  action: "halt_candidate_generation_and_escalate"

# hierarchical_model_degeneracy_alert canonical schema (populated only when task_diagnostic degenerate)
# 本 iter では null（診断は healthy_partial_pooling）。canonical form:
#   alert_type: enum {"all_ones_degenerate", "identity_degenerate"}
#   correlation_matrix_snapshot: list[list[float]]     # per-objective ~B matrix at trigger
#   recommended_actions: list[str]                     # human-readable remediation hints
hierarchical_model_degeneracy_alert: null

human_escalation_event:                     # append-only, Ch5 §5.6 canonical
  event_id: "evt-escalation-iter8-001"
  event_hash: "sha256:cccc0000iter8escalate"
  # Cross-stream causal chain (documented in D.7.2 provenance bundle metadata);
  # per-stream verify_chain_for_run uses previous_event_hash within same stream.
  previous_event_hash: "sha256:0000...genesis_human_escalation"    # first entry in stream → genesis hash. Cross-stream causal chain is documented in provenance bundle, not verified by verify_chain_for_run.
  run_id: "run-2026-1210-04"
  skill_execution_id: "skill-exec-20261222-0900-iter8"
  created_at: "2026-12-22T09:05:00Z"
  created_by: "^skill:arim.bo.constrained_qehvi/v0.2"
  escalation_type: "constraint_stall"
  clause: "constraint_stall"
  diagnostic_summary: "feasibility_probability_average < 0.60 for 2 iter (0.58, 0.52); recent 3 yields = [55, 58, 57]; recommend constraint relaxation or search space narrowing"
  recommended_actions:
    - "human_declared_constraint_relaxation: yield_min 60 → 55 for iterations 9-10 only"
    - "search_space_narrowing: exclude x_atmosphere=Air"
    - "additional_exploration: high x_pressure region (>= 3.5 MPa)"
  approver_required: "^human:project_lead_XXXX"
  status: "pending"
```

Skill 側は **自律的に constraint を緩めるのは fatal**（Ch14 §14.2.4 `approval_to_relax: fatal`）。Skill は Human 応答が届くまで次候補生成を停止し、`stop_condition` を毎 iteration 開始時に再評価する。実装は D.6.1 `evaluate()` を参照（本節で個別コードを再掲する代わりに、evaluator を単一実装に集約する）。

### D.4.4 Iteration 9: recovery

Human 承認: `constraints_spec_relaxation_approval` により、iteration 9-10 のみ `yield_min` を 60 → 55 に緩和。この緩和は **contract に対する Human override** であり、Skill 側は緩和後の `constraints_spec` を新 hash で pin し、以降の provenance に埋め込む。

```yaml
iteration_index: 9

constraints_spec_relaxation_approval:       # canonical, append-only
  approval_id: "approval:HITL-20261224-CONSREL-01"
  run_id: "run-2026-1210-04"
  skill_execution_id: "skill-exec-20261224-0900-iter9"
  created_at: "2026-12-24T09:00:00Z"
  created_by: "^skill:arim.bo.constrained_qehvi/v0.2"
  approved_by: "^human:project_lead_XXXX"
  approved_at: "2026-12-24T09:00:00Z"
  scope:
    apply_from_iteration: 9
    apply_until_iteration: 10                # 2 iteration のみ
    changes:
      - {constraint: "yield_min", old_expr: "y_yield >= 60", new_expr: "y_yield >= 55", rationale: "constraint_stall recovery per iter 8 escalation"}
  constraints_spec_hash_before: "sha256:cccc0000constraints60"
  constraints_spec_hash_after:  "sha256:dddd0000constraints55"
  auto_revert_after_iteration_10: true
  event_hash: "sha256:eeee0000consrel"
  previous_event_hash: "sha256:cccc0000iter8escalate"

# ---- iteration 9 の contract は緩和後の hash で再 pin ----
constraints_spec:
  version: "v0.2-relaxed"
  hash: "sha256:dddd0000constraints55"
  hard_constraints:
    - {name: "yield_min", expr: "y_yield >= 55", enforcement_layer: [1, 2, 3]}
    - {name: "temp_max",  expr: "x_temperature <= 800", enforcement_layer: [1, 2, 3]}
  soft_constraints:
    - {name: "yield_target", expr: "y_yield >= 75", violation_penalty: "cEI_multiplier"}
  feasibility_model:
    y_yield: {model_family: "single_task_gp", output_type: "regression"}
  feasibility_probability_threshold: 0.85    # 閾値自体は不変

feasibility_probability_history:             # 緩和後の recompute 値を append
  - {iteration_index: 8, average: 0.52,  constraint_version: "v0.2"}
  - {iteration_index: 9, average: 0.81,  constraint_version: "v0.2-relaxed"}

candidates:
  - {candidate_id: "cand-iter9-A1", facility_id: "A", x: {x_temperature: 730, x_pressure: 3.30, x_hold_time: 105, x_mass_frac_A: 0.40, x_mass_frac_B: 0.29, x_atmosphere: "Ar"}, acquisition_value: 1.55}
  # ... (batch=4 の残り 3 件、B/C も含む)
```

### D.4.5 Iteration 10: 収束確認と deliverables 出力

iteration 10 は本 capstone の最終 iteration。`stop_condition.clauses.iterations` が満たされ、正常停止となる。deliverables 出力（Pareto front / provenance bundle / 確認実験計画）を D.7 で扱う。

```yaml
iteration_index: 10

stop_condition_evaluation:
  version: "v0.3"
  evaluated_at: "2026-12-28T18:00:00Z"
  clauses_evaluated:
    - {name: "iterations",       satisfied: true,  reason: "iteration_index=10 >= 10 (14b scope)"}
    - {name: "budget",           satisfied: true,  reason: "experiments_remaining=4 <= 5"}
    - {name: "hypervolume",      satisfied: true,  reason: "delta_HV=0.028 < 0.02 * scale=1.5 (threshold=0.030)"}  # HV converged at iter 10 but iterations clause fires first (combination_rule: any)
    - {name: "constraint_stall", satisfied: false, reason: "fp_avg last 2 iter = [0.81, 0.83]"}
    - {name: "task_diagnostic",  satisfied: false, reason: "all objectives healthy_partial_pooling"}
    - {name: "human_stop",       satisfied: false, reason: "no signal"}
  first_satisfied_clause: "iterations"
  action: "halt_candidate_generation_and_emit_deliverables"

final_state:
  total_experiments_executed: 56          # 20 init + 9 iter × 4
  reserved_for_confirmation_experiments: 4  # 予算 60 - 56 = 4（うち 1 は buffer）
  pareto_front_size: 7                     # 詳細は D.7.1
  hypervolume_final: 3.42
  hypervolume_scale_value: 1.50            # ref_point_bounding_box
```

---

## D.5 budget_remaining operational コード

### D.5.1 budget_remaining schema

Ch14 §14.2.4 の `budget_remaining v0.2` を **Python dataclass** として実装。derive フィールド（`experiments_available_for_bo` / `batch_size_ceiling`）は必ず計算で導出し、Skill が直接書き込むことを禁止する。

```python
# ---- budget.py ----
"""Ch14 §14.2.4 budget_remaining operational reference."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class BudgetRemaining:
    """Immutable snapshot of budget_remaining per Ch14 §14.2.4."""

    version: str
    total_experiments: int
    experiments_completed: int
    pending_experiments: int
    wall_clock_total_hours: float
    wall_clock_used_hours: float
    reserved_for_confirmation_experiments: int
    unit_cost_hours: float
    fatal_on_overrun: bool = True

    @property
    def experiments_remaining(self) -> int:
        return self.total_experiments - self.experiments_completed

    @property
    def wall_clock_remaining_hours(self) -> float:
        return self.wall_clock_total_hours - self.wall_clock_used_hours

    @property
    def experiments_available_for_bo(self) -> int:
        return (
            self.experiments_remaining
            - self.pending_experiments
            - self.reserved_for_confirmation_experiments
        )

    @property
    def batch_size_ceiling(self) -> int:
        return max(0, self.experiments_available_for_bo)

    def to_provenance_dict(self) -> dict:
        base = asdict(self)
        base.update({
            "experiments_remaining": self.experiments_remaining,
            "wall_clock_remaining_hours": self.wall_clock_remaining_hours,
            "experiments_available_for_bo": self.experiments_available_for_bo,
            "batch_size_ceiling": self.batch_size_ceiling,
            "overrun_policies": {
                "on_proposed_batch_over_available": "fatal",
                "on_pending_plus_proposed_over_budget": "fatal",
                "on_confirmation_reserve_encroachment": "fatal",
            },
        })
        return base
```

### D.5.2 iteration 前後の update 手順

```python
def update_budget_at_iteration_start(
    prev: BudgetRemaining,
    newly_completed: int,
    newly_submitted_pending: int,
    wall_clock_delta_hours: float,
) -> BudgetRemaining:
    """iteration N の候補生成前に呼ぶ。前 iter の実験完了/pending 追加を反映。"""
    return BudgetRemaining(
        version=prev.version,
        total_experiments=prev.total_experiments,
        experiments_completed=prev.experiments_completed + newly_completed,
        pending_experiments=(
            prev.pending_experiments + newly_submitted_pending - newly_completed
        ),  # pending had submitted, completed drains them
        wall_clock_total_hours=prev.wall_clock_total_hours,
        wall_clock_used_hours=prev.wall_clock_used_hours + wall_clock_delta_hours,
        reserved_for_confirmation_experiments=prev.reserved_for_confirmation_experiments,
        unit_cost_hours=prev.unit_cost_hours,
    )
```

### D.5.3 stop_condition との連動

`stop_condition.clauses.budget` の条件式 `experiments_remaining <= 5`（Ch14 §14.3.5 canonical）は `BudgetRemaining.experiments_remaining` を直接参照する。これにより `stop_condition` 評価と `overrun_policies` 判定の間で数値が食い違うのを防ぐ。

> [!NOTE]
> **canonical stop_condition は `experiments_remaining` を使う**。`experiments_available_for_bo` は overrun_policies（Ch14 §14.2.4）の pre-launch batch sizing で用いる operational な派生フィールドであり、stop_condition の evaluation には登場しない。両者は同一 dataclass のプロパティとして計算されるが、参照する clause / policy が明確に分離されている。

---

## D.6 stop_condition operational コード

### D.6.0 Group C event streams: feasibility_probability_history / task_diagnostic_history

本付録で導入する 2 系列は App-D-canonical な Group C event streams（append-only, per-iteration）である。Ch14 §14.3.5 の `stop_condition` clauses `constraint_stall`（`feasibility_probability_history` を参照）/ `task_diagnostic`（`task_diagnostic_history` を参照）が canonical に評価し、chain 整合は付録 B §B.6 の `verify_chain_for_run(stream_name, run_id)` で iteration 10 完了時に per-stream 独立に確認される（D.7.2）。

```yaml
feasibility_probability_history_schema:
  version: "v0.2"
  stream_type: "Group C"                  # append-only, per-iteration
  entry_schema:
    iteration_index:            {type: "int",       required: true}
    average:                    {type: "float",     required: true, note: "canonical value field: average P[feasible] across observed pending set (Ch10 §10.3 feasibility average)"}
    computed_at:                {type: "timestamp", required: true, format: "RFC 3339 UTC"}
    computed_by:                {type: "identity",  required: true, allowed: ["^service:mcp_gate_service", "^skill:arim.bo.constrained_qehvi/v0.2"]}
    event_hash:                 {type: "sha256",    required: true, note: "RFC 8785 JCS + SHA-256 of the entry"}
    previous_event_hash:        {type: "sha256",    required: true, note: "tail of previous entry in same stream; head entry uses run genesis hash"}
    constraint_version:         {type: "str",       required: false, note: "constraints_spec version active when computed"}

task_diagnostic_history_schema:
  version: "v0.2"
  stream_type: "Group C"
  entry_schema:
    iteration_index:            {type: "int",       required: true}
    diagnostic_flag_per_objective: {type: "map[str,str]", required: true, note: "objective_name → flag ∈ {healthy_partial_pooling, all_ones_degenerate, identity_degenerate, insufficient_observations}"}
    computed_at:                {type: "timestamp", required: true, format: "RFC 3339 UTC"}
    computed_by:                {type: "identity",  required: true, allowed: ["^service:mcp_gate_service", "^skill:arim.bo.hierarchical_gp/v0.2"]}
    event_hash:                 {type: "sha256",    required: true}
    previous_event_hash:        {type: "sha256",    required: true, note: "tail of previous entry in same stream; head entry uses run genesis hash"}

writer_identity_rules:
  - "Automated recompute (gate polling before iteration N) uses ^service:mcp_gate_service; Skill-emitted entries use ^skill:arim.bo.<name>/v0.2; ^human:* writer is fatal (Ch15 §15.2)."
```

### D.6.1 clause evaluator（v0.3 6 clauses）

D.4.3 の記述で「Skill は Human 応答が届くまで次候補生成を停止し、`stop_condition` を毎 iteration 開始時に再評価する」とした evaluator の完全版を、helper 関数と一緒に整理する。

```python
# ---- stop_condition.py ----
"""Ch14 §14.3.5 stop_condition v0.3 evaluator (6 clauses)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StopConditionResult:
    version: str                              # canonical "v0.3"
    clauses_evaluated: list[dict]
    first_satisfied_clause: Optional[str]
    action: str


def evaluate(state: dict) -> StopConditionResult:
    """Evaluate v0.3 clauses in canonical order (Ch14 §14.3.5):
    iterations, budget, hypervolume, constraint_stall, task_diagnostic, human_stop.
    First-match wins (canonical enum combination_rule: any).
    """
    order = [("iterations", _c_iterations), ("budget", _c_budget),
             ("hypervolume", _c_hypervolume), ("constraint_stall", _c_constraint_stall),
             ("task_diagnostic", _c_task_diagnostic), ("human_stop", _c_human_stop)]
    result = []
    first = None
    for name, fn in order:
        satisfied, reason = fn(state)
        result.append({"name": name, "satisfied": satisfied, "reason": reason})
        if satisfied and first is None:
            first = name
    if first == "iterations":
        action = "halt_candidate_generation_and_emit_deliverables"
    elif first is not None:
        action = "halt_candidate_generation_and_escalate"
    else:
        action = "proceed"
    return StopConditionResult(version="v0.3", clauses_evaluated=result,
                               first_satisfied_clause=first, action=action)


def _c_iterations(s): return s["iteration_index"] >= 10, f"iter={s['iteration_index']}"
def _c_budget(s):
    # Ch14 §14.3.5 canonical: budget clause references experiments_remaining (not the derived
    # experiments_available_for_bo, which is used by overrun_policies for batch sizing).
    r = s["budget_remaining"]["experiments_remaining"]
    return r <= 5, f"experiments_remaining={r} vs threshold=5"


def _c_hypervolume(s):
    hv = s.get("hypervolume_history", [])[-3:]
    if len(hv) < 3:
        return False, "insufficient history (<3 iter)"
    # canonical net improvement over 3-iter window; monotonic-under-qEHVI in noise-free case.
    # Use hv[-1] - hv[-3] rather than max-min so a transient dip does not mask a real gain.
    impr = hv[-1] - hv[-3]
    threshold = 0.02 * s["hypervolume_scale_value"]
    return impr < threshold, (
        f"delta_HV={impr:.4f} vs threshold={threshold:.4f} → "
        f"{'satisfied' if impr < threshold else 'not satisfied (still improving)'}"
    )


def _c_constraint_stall(s):
    fp = s.get("feasibility_probability_history", [])[-2:]
    if len(fp) < 2:
        return False, "insufficient history"
    return all(h["average"] < 0.60 for h in fp), f"fp_last2={fp}"


def _c_task_diagnostic(s):
    diag = s.get("task_diagnostic_history", [])[-2:]
    if len(diag) < 2:
        return False, "insufficient history"
    degen = ("all_ones_degenerate", "identity_degenerate")
    # canonical field: diagnostic_flag_per_objective (map objective_name → flag).
    ok = all(
        any(v in degen for v in d["diagnostic_flag_per_objective"].values())
        for d in diag
    )
    return ok, f"diag_last2={diag}"


def _c_human_stop(s):
    ok = bool(s.get("human_stop_signal", False))
    return ok, ("human override" if ok else "no signal")
```

### D.6.2 hypervolume_improvement 計算

```python
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import DominatedPartitioning
import torch


def compute_hypervolume(
    pareto_Y: torch.Tensor,          # (P, m) objective values (maximization convention)
    ref_point: torch.Tensor,         # (m,)
) -> float:
    """BoTorch DominatedPartitioning.compute_hypervolume() wrapper."""
    if pareto_Y.numel() == 0:
        return 0.0
    partitioning = DominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
    return float(partitioning.compute_hypervolume().item())


def hypervolume_scale_value(ref_point: torch.Tensor, y_bounds_upper: torch.Tensor) -> float:
    """Ch14 §14.3.5 hypervolume_scale = ref_point_bounding_box."""
    return float(torch.prod(torch.abs(y_bounds_upper - ref_point)).item())
```

### D.6.3 task_correlation_matrix diagnostic 判定

D.2.3 の `_extract_diagnostics()` が返す `diagnostic_flag_per_objective` を、Ch14 §14.3.5 の閾値と `require_two_consecutive_iters: true` で判定する。

> [!NOTE]
> **canonical semantics**: `task_diagnostic` clause は「**≥ 1 objective が degenerate correlation matrix を持つ**」ことが **直近 2 iteration の EACH で** 成り立つ場合に発火する（any-per-iter, then all-across-iters）。これは Ch14 §14.3.5 の wording に一致する。D.6.1 の `_c_task_diagnostic` の `all(any(v in degen for v in d.values()) for d in diag)` が正確にこの意味論を表現している。

判定コアは以下：

```python
def classify_task_correlation(
    matrix: list[list[float]],
    all_ones_offdiag_min: float = 0.95,
    identity_offdiag_max_abs: float = 0.10,
) -> str:
    """Ch14 §14.3.5 task_diagnostic_thresholds classifier."""
    n = len(matrix)
    offdiag_vals = [
        matrix[i][j] for i in range(n) for j in range(n) if i != j
    ]
    if not offdiag_vals:
        return "insufficient"
    if min(offdiag_vals) >= all_ones_offdiag_min:
        return "all_ones_degenerate"
    if max(abs(v) for v in offdiag_vals) <= identity_offdiag_max_abs:
        return "identity_degenerate"
    return "healthy_partial_pooling"
```

`min_observations_per_task: 5` の precheck は fit 前に走らせる（fit 後の $\tilde{B}$ は少データだと noisy になり誤診断を招く）：

```python
def precheck_min_observations_per_task(
    task_indices: torch.Tensor,
    min_obs: int = 5,
) -> bool:
    """Return True if all tasks have >= min_obs observations."""
    unique, counts = torch.unique(task_indices.long(), return_counts=True)
    return bool((counts >= min_obs).all().item())
```

### D.6.4 human_stop_signal の polling / event 読取

`stop_condition.clauses.human_stop` は Human から独立に発信可能（Ch14 §14.2.4）。Skill は毎 iteration 開始時に `authorization_events_stream` から `human_stop_signal_event` を polling する：

```python
def poll_human_stop_signal(
    events_stream_reader,     # 付録 B §B.6 の read-only reader
    run_id: str,
    since_event_hash: str,
) -> bool:
    """Poll authorization_events_stream for canonical human_stop_signal_event.

    canonical event type: 'human_stop_signal'
    Writer identity must match `^human:*`; ^skill:* / ^service:* writers are ignored (fatal at handler).
    """
    for evt in events_stream_reader.iter_events_after(run_id, since_event_hash):
        if evt["event_type"] != "human_stop_signal":
            continue
        if not evt["created_by"].startswith("^human:"):
            # Ch15 §15.2 fatal: skill impersonating human
            raise RuntimeError(
                f"non-human writer for human_stop_signal: {evt['created_by']}"
            )
        return True
    return False
```

---

## D.7 全 10 iteration 完了時 deliverables

### D.7.1 Pareto front 出力（JSON）

iteration 10 終了時、observed points の中で **feasible かつ non-dominated** な点を Pareto front として抽出する。

```python
def emit_pareto_front(
    obs_Y_hardness: torch.Tensor,
    obs_Y_density: torch.Tensor,
    obs_yield: torch.Tensor,
    obs_X_original: list[dict],          # human-readable x values
    feasibility_threshold: float = 55.0,   # iter 9-10 の緩和後
) -> list[dict]:
    """Emit Pareto front records for deliverable bundle."""
    feasible_mask = (obs_yield >= feasibility_threshold)
    if not feasible_mask.any():
        return []
    # BoTorch convention: maximization. Flip density sign.
    Y = torch.stack(
        [obs_Y_hardness[feasible_mask], -obs_Y_density[feasible_mask]],
        dim=-1,
    )
    nd_mask = is_non_dominated(Y)
    front_indices = torch.nonzero(feasible_mask).flatten()[nd_mask]

    records = []
    for i in front_indices.tolist():
        records.append({
            "point_id": f"pareto-{i:03d}",
            "x": obs_X_original[i],
            "y_hardness": float(obs_Y_hardness[i].item()),
            "y_density":  float(obs_Y_density[i].item()),
            "y_yield":    float(obs_yield[i].item()),
        })
    return records
```

Deliverable JSON schema:

```yaml
pareto_front_deliverable:
  version: "v0.2"
  run_id: "run-2026-1210-04"
  emitted_at: "2026-12-28T18:15:00Z"
  emitted_by: "^skill:arim.bo.multi_objective_qehvi/v0.2"
  final_iteration_index: 10
  ref_point: [7.0, 5.0]
  hypervolume_final: 3.42
  feasibility_threshold_used: 55.0
  constraint_version_active: "v0.2-relaxed"
  points:
    - {point_id: "pareto-007", x: {x_temperature: 730, x_pressure: 3.30, x_hold_time: 105, x_mass_frac_A: 0.40, x_mass_frac_B: 0.29, x_atmosphere: "Ar"}, y_hardness: 9.72, y_density: 4.35, y_yield: 63.1, facility_id: "A"}
    - {point_id: "pareto-012", x: {x_temperature: 705, x_pressure: 3.10, x_hold_time: 98,  x_mass_frac_A: 0.38, x_mass_frac_B: 0.29, x_atmosphere: "Ar"}, y_hardness: 9.51, y_density: 4.28, y_yield: 68.4, facility_id: "A"}
    # ... 7 点合計
```

### D.7.2 provenance bundle（`verify_chain_for_run` 済み snapshot）

付録 B §B.6 の `verify_chain_for_run(stream_name, run_id)` を各 event stream に対して独立に実行し、pass した snapshot を bundle する。**各 stream は independently に検証される**（App-B §B.6 canonical: per-stream chaining）。cross-stream 因果関係（例: iter 7 overrun_precheck → iter 8 human_escalation）は本 bundle の metadata にのみ記録し、`verify_chain_for_run` 自体は cross-stream chain を検証しない。

```python
def emit_provenance_bundle(
    run_id: str,
    verify_chain_for_run,                    # from appendix B §B.6: (stream_name, run_id) → result
    event_streams: dict[str, list[dict]],    # {stream_name: [events...]}  # for head/tail hash lookup
) -> dict:
    """Bundle provenance across all event streams with per-stream chain verification.

    Each stream is verified independently per App-B §B.6 (verify_chain_for_run).
    Cross-stream causal chains are documented here as metadata only.
    """
    streams_to_verify = [
        "authorization_events_stream",
        "overrun_precheck_events",
        "human_escalation_event_stream",
        "feasibility_probability_history",
        "task_diagnostic_history",
        "constraint_violation_events",
        "duplicate_candidate_events",
        "stale_pending_events",
        "batch_drift_events",
    ]
    chain_verifications = {
        s: verify_chain_for_run(stream_name=s, run_id=run_id)
        for s in streams_to_verify
    }
    verification = {}
    for stream_name, result in chain_verifications.items():
        events = event_streams.get(stream_name, [])
        verification[stream_name] = {
            "num_events": len(events),
            "chain_ok": result["ok"],
            "head_event_hash": events[0]["event_hash"] if events else None,
            "tail_event_hash": events[-1]["event_hash"] if events else None,
            "break_at_index": result.get("break_at_index"),
        }
    return {
        "version": "v0.2",
        "run_id": run_id,
        "emitted_at": "2026-12-28T18:20:00Z",
        "streams": verification,
        "canonical_serialization": "RFC 8785 JCS + SHA-256",
        "chain_verification_semantics": "per-stream (App-B §B.6 verify_chain_for_run); cross-stream chains are metadata only",
    }
```

bundle 例:

```yaml
provenance_bundle:
  version: "v0.2"
  run_id: "run-2026-1210-04"
  emitted_at: "2026-12-28T18:20:00Z"
  streams:
    authorization_events_stream:
      num_events: 34
      chain_ok: true
      head_event_hash: "sha256:0000...init"
      tail_event_hash: "sha256:ffff...iter10"
    constraint_violation_events:
      num_events: 0
      chain_ok: true
      head_event_hash: null
      tail_event_hash: null
    experiment_status_transition_events:
      num_events: 56
      chain_ok: true
      head_event_hash: "sha256:1111...submit"
      tail_event_hash: "sha256:2222...complete"
    overrun_precheck_events:
      num_events: 1                                    # iter 7 の budget overrun 一件
      chain_ok: true
    human_escalation_event_stream:
      num_events: 2                                    # iter 8 escalation + iter 9 recovery approval
      chain_ok: true                                   # このストリームは escalation request と対応する
                                                       # constraints_spec_relaxation_approval（応答）の両方を運ぶ
  canonical_serialization: "RFC 8785 JCS + SHA-256"
```

### D.7.3 確認実験計画（top-k Pareto candidates の再測定案）

Pareto front から **top-3 候補** を選び、`reserved_for_confirmation_experiments` の枠内で再測定する計画を作る。

```yaml
confirmation_experiment_plan:
  version: "v0.2"
  emitted_at: "2026-12-28T18:25:00Z"
  emitted_by: "^skill:arim.bo.multi_objective_qehvi/v0.2"
  parent_run_id: "run-2026-1210-04"
  selection_policy:
    method: "top_k_hypervolume_contribution"
    k: 3
    tie_breaker: "highest_y_yield"
  reserved_budget:
    experiments: 4                    # 60 - 56 = 4
    hours: 16                         # 4 × 4h unit
  confirmations:
    - {point_id: "pareto-007", facility_id: "A", replicates: 2, purpose: "highest hypervolume contributor + Δ hardness confirmation"}
    - {point_id: "pareto-012", facility_id: "A", replicates: 1, purpose: "high yield (68.4) reproducibility"}
    - {point_id: "pareto-021", facility_id: "B", replicates: 1, purpose: "cross-facility validation of Pareto point"}
  requires_new_experiment_launch_authorization: true    # iter 10 完了後の新規承認
  parent_authorization_id: "elauth_20261228_180000_iter10"
  approver_required: "^human:project_lead_XXXX"
```

> [!TIP]
> 確認実験は **新たな iteration ではなく別 run** として起こす（`run_id` を変える）ことを推奨。理由：surrogate refit のトリガや `stop_condition` の再評価を、探索 run と確認 run で切り分けたほうが監査ログが読みやすくなる（Ch16 §16.5）。

---

## D.8 チェックリスト（本付録の全スクリプトが章 §14.3 契約と一致するか）

```yaml
appendix_d_conformance_checklist:
  version: "v0.2"
  items:
    - {id: "AD-01", item: "全 iteration の contract が Ch14 §14.3.3 統合 YAML に整合しているか", status: "ok"}
    - {id: "AD-02", item: "task_map が {A:0, B:1, C:2} に固定されているか（Ch11 §11.5 / Ch14 §14.3.1）", status: "ok"}
    - {id: "AD-03", item: "search_space_bounds が Ch14 §14.2.1 canonical 6-var に一致しているか", status: "ok"}
    - {id: "AD-04", item: "linear constraint A+B<=0.85 が Skill 側でも pre-check されているか", status: "ok"}
    - {id: "AD-05", item: "x_atmosphere の ordinal encoding {Ar:0, N2:1, Air:2} が Hamming kernel と整合しているか", status: "ok"}
    - {id: "AD-06", item: "acquisition_spec.name が全 iter で qLogNEHVI に固定されているか（Ch9 §9.4 canonical）", status: "ok"}
    - {id: "AD-07", item: "per_facility_batch_size が experiment_launch_authorization.scope に必ず現れるか", status: "ok"}
    - {id: "AD-08", item: "pending_experiments.tensor_encoding_contract.derivation_deterministic: true が保持されているか", status: "ok"}
    - {id: "AD-09", item: "stop_condition.version が v0.3 で全 6 clauses が canonical order で並んでいるか", status: "ok"}
    - {id: "AD-10", item: "iter 7 の budget overrun が Skill 側 retry で吸収されているか", status: "ok"}
    - {id: "AD-11", item: "iter 8 の constraint_stall が Human エスカレーション経由でのみ緩和されているか", status: "ok"}
    - {id: "AD-12", item: "iter 9 の constraints_spec_relaxation_approval に auto_revert_after_iteration_10 が付与されているか", status: "ok"}
    - {id: "AD-13", item: "iter 10 で iterations clause が first_satisfied_clause となり deliverables に遷移するか", status: "ok"}
    - {id: "AD-14", item: "Pareto front deliverable の feasibility_threshold が constraint_version_active と一致しているか", status: "ok"}
    - {id: "AD-15", item: "provenance bundle の全 stream について verify_chain_for_run(stream_name, run_id) が chain_ok: true を返しているか（per-stream 独立、App-B §B.6）", status: "ok"}
    - {id: "AD-16", item: "確認実験計画が別 run_id で起票され、新規 experiment_launch_authorization を要求しているか", status: "ok"}
    - {id: "AD-17", item: "Skill が constraints_spec を直接編集していないか（緩和は Human approval 経由のみ）", status: "ok"}
    - {id: "AD-18", item: "Skill が pending_experiments を Skill 権限でキャンセルしていないか（cancellation_policy: requires_human_approval）", status: "ok"}
    - {id: "AD-19", item: "event_hash が RFC 8785 JCS + SHA-256 で計算されているか（付録 B §B.6 参照）", status: "ok"}
    - {id: "AD-20", item: "全 event で FLAT envelope（run_id / skill_execution_id / created_at / created_by が business fields の sibling）を satisfy しているか", status: "ok"}
    - {id: "AD-21", item: "authorization_id が elauth_YYYYMMDD_HHMMSS_iter<n> の canonical form か", status: "ok"}
    - {id: "AD-22", item: "task_correlation_matrix の diagnostic 閾値（0.95 / 0.10）が Ch14 §14.3.5 と一致しているか", status: "ok"}
    - {id: "AD-23", item: "min_observations_per_task: 5 の precheck が fit 前に走っているか", status: "ok"}
    - {id: "AD-24", item: "hypervolume_scale = ref_point_bounding_box が全 iter で同じ値で計算されているか", status: "ok"}
    - {id: "AD-25", item: "human_stop_signal の writer が ^human:* に限定され、^skill:* / ^service:* が reject されるか", status: "ok"}
```

---

## D.9 まとめ

本付録 D は Ch14 §14.3.7 で planned とされていた **後半 5 iteration のスクリプト** と、それを支える横断コンポーネント（Hierarchical GP 実装 / 統合 acquisition / `budget_remaining` / `stop_condition v0.3` / deliverables）を、illustrative reference として **実装レベル** に展開した。要点：

1. **HierGP は BoTorch native ではないが GPyTorch カスタム + `GPyTorchModel` mixin で BoTorch acquisition と接続可能**（D.2）。task_correlation_matrix は HierGP 用の proxy で計算し、Ch14 §14.3.5 の閾値で分類する。
2. **統合 acquisition は `qLogNoisyExpectedHypervolumeImprovement` を軸に、constraint callables と `X_pending` を組み合わせ、`optimize_acqf(sequential=True, fixed_features={task_dim: t})` で装置固定 batch を作る**（D.3）。cross_facility fantasize は Skill 側で明示的に前の facility の候補を `X_pending` に append する。
3. **iteration 6-10 の中で起こり得る 4 パターンの契約発火**（通常進行 / budget overrun / constraint_stall / recovery）を canonical YAML で示した（D.4）。Skill は自律的に constraint を緩めず、必ず Human approval を経由する。
4. **budget_remaining と stop_condition は 1 個の Python dataclass / evaluator に集約**することで、数値の食い違いを構造的に防ぐ（D.5 / D.6）。
5. **Deliverables は Pareto front / provenance bundle / 確認実験計画の 3 点セット**（D.7）。確認実験は必ず新 run_id + 新規 experiment_launch_authorization で起票する。

**次章への橋渡し**: 本付録の全スクリプトが Ch14 §14.3 契約に整合していることを D.8 のチェックリストで確認した。実運用では、これらの reference をコピー & 貼付するのではなく、**付録 A §A.6 (`arim.bo.hierarchical_gp.v0.2`) / §A.3 (`arim.bo.multi_objective_qehvi.v0.2`) / §A.4 (`arim.bo.constrained_qehvi.v0.2`) / §A.7 (`arim.bo.batch_qlognei.v0.2`) の Skill テンプレート** を facility_bo_skill_registry（Ch16 §16.9）経由で bind し、**付録 B §B.3 の MCP handler** 経由で起動すること。付録 D のコードは Skill 実装者が「契約フィールドが実装のどこに落ちるか」を追いかけるための **契約 ↔ コード対応表** として使うのが正しい読み方である。

> [!IMPORTANT]
> 本付録の全 YAML は `yaml.safe_load` でパース可能である（付録 C §C.8.3 と同じ規約：`-` と `+` の sentinel を混在させない、`{}` / `[]` は explicit null / empty で表現）。付録 D 収録の Python コードは illustrative であり、**executable-adjacent** な密度を目指しているが production ready ではない。Skill 起票者は本付録を読んだ後、必ず付録 A の Skill テンプレートに戻り、そこから MCP 経由で実行するフロー（付録 B）に接続すること。
