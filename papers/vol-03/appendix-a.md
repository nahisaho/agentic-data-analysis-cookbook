# 付録A　深層 × Agentic Skill テンプレート集

> **本付録の目的**：第4〜15章で作った深層 × Agentic Skill の**再利用可能な雛形**を、`SKILL.md` + `provenance schema` + `minimal example` の 3 点セットで提供します。vol-02 の provenance を **GPU / 深層 / Agentic 拡張**した完全スキーマも本付録末尾に置きます。

> [!IMPORTANT]
> テンプレートは**そのままコピーして使うためのもの**ではなく、**「何を書き、何を検証すべきか」を思い出すためのチェックリスト**です。実際のデータ・目的に合わせて調整してください。深層学習では学習コスト・GPU リソースが大きいため、テンプレの `④ 成功条件` `⑤ 禁止事項` を最初にレビューしてから走らせる運用を強く推奨します。

---

## A.1 本付録の使い方

| 目的 | 参照するテンプレート | 該当章 |
|---|---|---|
| CNN / ViT で画像分類 Skill を作る | A.2（画像分類） | 第6章 |
| 画像→物性値の回帰 Skill を作る | A.3（画像回帰） | 第6章・第7章 |
| 1D CNN / 1D Transformer でスペクトル分類 Skill を作る | A.4（スペクトル分類） | 第5章 |
| TabNet / FT-Transformer で表形式回帰 Skill を作る | A.5（表形式回帰） | 第5章・第12章 |
| Foundation Model 転移学習 Skill を作る | A.6（転移学習） | 第7章・第11章 |
| SSL（SimCLR / MAE 等）事前学習 Skill を作る | A.7（自己教師あり） | 第7章 |
| BNN / MC-Dropout / Deep Ensemble Skill を作る | A.8（不確かさ） | 第8章・第9章 |
| provenance schema の全項目を確認する | A.9（完全スキーマ） | 第4・11・13章 |

**テンプレの共通要素**（6 要素）は vol-01 §A.2.2 に準拠：①目的 / ②入力条件 / ③出力形式 / ④成功条件 / ⑤禁止事項（severity 付き）/ ⑥再現性条件。vol-03 では **⑦ Agentic 契約** を追加します（agent tier / 承認ゲート / 不確かさ停止条件）。

### A.1.1 Skill / リポジトリ配置規約（vol-01 A.2.1・vol-02 A.1.1 の継承 + vol-03 追加要素）

Skill ディレクトリ構造は **vol-01 §A.2.1・vol-02 §A.1.1 と同一**です。配置先も同じ 3 系統：

- **リポジトリ配布**：`.github/skills/<name>/`（Copilot CLI 自動検出、実運用の推奨）
- **個人利用**：`~/.copilot/skills/<name>/`
- **下書き**：`skills/<name>/`（本書の説明・演習用。完成後にどちらかへ移す）

vol-03 で新規に追加されるパスは **Skill ディレクトリの内側** の 2 ディレクトリです。

**リポジトリルート（Skill 横断で共有される資産）**

```text
<vol-03 repo root>/
├── data/
│   ├── synthetic-hierarchy/          ← vol-02 継承（第13章 capstone で再利用）
│   └── synthetic-deep/               ← ★vol-03 追加：深層向け合成データ（付録C）
│       ├── images/                   ← 画像サンプル（SEM 風、Raman 風）
│       ├── spectra/                  ← 1D スペクトル
│       └── metadata.parquet          ← ラベル・階層情報
├── scripts/
│   ├── generate_synthetic_hierarchy.py   ← vol-02 継承
│   └── generate_synthetic_deep.py        ← ★vol-03 追加
├── weights/                          ← ★vol-03 追加：ローカル weight cache
│   ├── foundation/                       ← FM の hash-verified copy（第11章）
│   └── fine_tuned/                       ← 派生モデル
└── .github/skills/                   ← 各 Skill はここに配置（vol-01 A.2.1）
    └── <skill-name>/
        └── ...
```

**Skill ディレクトリの内側（vol-03 で追加される 2 ディレクトリ）**

vol-02 の `artifacts/` / `diagnostics/` に加え、深層学習の中間物と GPU 実行環境情報の置き場を明示します。

```text
.github/skills/<skill-name>/
├── SKILL.md                          ← vol-01 §A.2.2（同一）
├── references/                       ← vol-01 §A.2.3〜（同一）
├── scripts/                          ← vol-01 §A.2（同一）
├── tests/                            ← vol-01 §A.2（同一）
├── examples/                         ← vol-01 §A.2（同一）
├── artifacts/                        ← vol-02 §A.1.1（同一、深層章では拡張）
│   ├── model_v<SemVer>.safetensors        ← 深層モデルは safetensors 推奨
│   ├── model_state_v<SemVer>.pt           ← state_dict のみ保存する場合
│   └── ...
├── diagnostics/                      ← vol-02 §A.1.1（同一）
├── checkpoints/                      ← ★vol-03 追加：学習中間 checkpoint
│   ├── epoch_<NNN>.safetensors            ← 各 epoch or best のみ保存
│   ├── checkpoint_registry.jsonl          ← append-only、Ch4 の overwrite policy
│   └── README.md                          ← 保持ポリシー・削除規則
└── wandb_run/                        ← ★vol-03 追加（optional、第15章 §15.10）
    ├── run_id.txt                         ← MLflow / W&B run_id の pin
    └── metrics_hash.txt                   ← tracking との改ざん検知
```

> [!TIP]
> `checkpoints/` は **append-only**（Ch4 `checkpoint_overwrite_policy: append_only`）が原則。ディスク節約が必要な場合は最終 signed checkpoint と best のみを残し、中間は削除ログを `checkpoint_registry.jsonl` に記録する。**silent な上書きは AG-03（第14章）に該当**。

### A.1.2 テンプレの読み方

各テンプレは以下 3 セクションで構成：

1. **`SKILL.md` 雛形** — 6 + 1 要素をコピペ可能な形で提示
2. **最小コード骨格** — PyTorch / JAX / Hugging Face の実装スケルトン
3. **チェックリスト（PR 前）** — 提出前に埋めるべき項目

**書き方の共通ルール**：

- 目的行は 1 文で書く（`- 目的: ...`）
- 入力・出力条件は data contract（Ch4）を参照 (`- input_schema_ref: data-contract-<name>.yaml`)
- Agentic 契約は Ch4 §4.6-4.9 の tier / 承認 / 停止契約を Skill 側で pin
- Severity は `CRITICAL` / `HIGH` / `MEDIUM` / `LOW` の 4 段（Ch14 §14.3 準拠）

---

## A.2 画像分類 Skill テンプレート（第6章）

### A.2.1 `SKILL.md` 雛形

```yaml
skill_name: "image_classification_skill"
version: "0.1.0"
agent_tier_required: "L2"                    # 承認範囲内での fine-tune 可
purpose: >
  ARIM 画像データ（SEM / 光学顕微鏡 / TEM 等）から材料相・結晶構造・
  欠陥種別を分類する深層 Skill。ImageNet 事前学習 or 材料 FM 転移。

inputs:
  data_contract_ref: "data-contract-image-<domain>.yaml"
  minimum_samples_per_class: 30              # 少ないと augmentation 依存が過大
  class_balance_ratio_max: 20                # 不均衡 20:1 超は要 stratify + weighted loss

outputs:
  predicted_class: "int"
  class_probabilities: "list[float]"
  uncertainty_score: "float (Ch8 継承)"
  gradcam_reference: "URI (Ch10 継承)"

success_criteria:
  macro_f1_min: 0.80
  per_class_recall_min: 0.60                 # 全クラスで
  uncertainty_calibration_ece_max: 0.05      # Ch8

forbidden:
  - action: "test_time_augmentation_beyond_declared_policy"
    severity: "HIGH"
  - action: "class_rebalance_by_dropping_minority_at_inference"
    severity: "CRITICAL"
  - action: "leakage_via_augmentation_over_split_boundary"
    severity: "CRITICAL"
  - action: "silent_checkpoint_overwrite"                        # AG-03
    severity: "HIGH"

reproducibility:
  gpu_backend_declared: true
  cudnn_deterministic: true
  random_seed_per_worker: "int"
  augmentation_policy_hash: "sha256"
  weights_sha256: "sha256"

agentic_contract:
  agent_tier: "L2"
  approved_hp_range:
    lr: [1e-5, 1e-3]
    batch_size: [8, 128]
    epochs_max: 100
  training_job_approval_required: true       # Ch4
  fm_update_gate_required_if_using_fm: true  # Ch11
  uncertainty_stop_threshold_ref: "Ch8 skill"
  human_review_ref: "gate1 approval id"
```

### A.2.2 最小コード骨格（PyTorch + timm）

```python
# scripts/train_image_classifier.py
import hashlib
import json
from pathlib import Path
import torch
import timm
from torch.utils.data import DataLoader

def train(cfg: dict, data: dict, approval_record: dict) -> dict:
    """
    cfg: SKILL.md の approved_hp_range 内であることを事前検証済み
    approval_record: training_job_approval provenance
    """
    # 1. 決定論設定
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = cfg["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = False

    # 2. モデル構築（timm から）
    model = timm.create_model(
        cfg["model_name"],
        pretrained=True,
        num_classes=cfg["num_classes"],
    )

    # 3. checkpoint dir（append-only）
    ckpt_dir = Path(cfg["output_dir"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    registry_path = ckpt_dir / "checkpoint_registry.jsonl"

    # 4. 学習ループ（省略）
    for epoch in range(cfg["epochs"]):
        # ... train + val
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.safetensors"
        _save_safetensors(model, ckpt_path)
        _append_registry(registry_path, ckpt_path, epoch, val_metrics)

    # 5. provenance 出力
    return {
        "weights_uri": str(ckpt_path),
        "weights_sha256": _sha256(ckpt_path),
        "approval_id": approval_record["approval_id"],
        "gpu_backend": torch.version.cuda,
        "cudnn_deterministic": True,
        "seed": cfg["seed"],
    }
```

### A.2.3 チェックリスト（PR 前）

- [ ] `approved_hp_range` 内のハイパーパラメータで学習
- [ ] `training_job_approval` provenance が存在
- [ ] `checkpoints/checkpoint_registry.jsonl` が append-only
- [ ] `augmentation_policy_hash` が学習・推論で一致
- [ ] Gate1 承認済み（Ch13）
- [ ] macro F1・per-class recall・ECE すべて閾値クリア
- [ ] Grad-CAM で少なくとも 3 サンプル可視化（Ch10）

---

## A.3 画像回帰 Skill テンプレート（第6章・第7章）

### A.3.1 `SKILL.md` 雛形（差分のみ）

```yaml
skill_name: "image_regression_skill"
version: "0.1.0"
agent_tier_required: "L2"
purpose: >
  画像から連続値（粒径・膜厚・組成比・応力値等）を回帰する Skill。
  vol-02 の校正曲線と直列に接続可（校正済み画像特徴 → PyMC 回帰）。

outputs:
  predicted_value: "float"
  prediction_sigma: "float"                  # 予測分散、Ch8/9 継承
  uncertainty_source: "enum: aleatoric|epistemic|both"

success_criteria:
  r2_min: 0.75
  rmse_max_relative: 0.15                    # 目標値の 15% 以内
  calibration_of_prediction_interval:
    coverage_90pct_min: 0.85
    coverage_90pct_max: 0.95                 # 過大 / 過小どちらも fail

forbidden:
  - action: "report_point_estimate_without_sigma"
    severity: "CRITICAL"
  - action: "collapse_sigma_to_zero_for_pretty_plots"
    severity: "CRITICAL"
```

### A.3.2 最小コード骨格

```python
# 損失は Gaussian NLL で aleatoric uncertainty を捕捉
import torch.nn.functional as F

def gaussian_nll_loss(pred_mu, pred_log_sigma, target):
    sigma = torch.exp(pred_log_sigma).clamp(min=1e-6)
    return 0.5 * (torch.log(2 * torch.pi * sigma**2) + (target - pred_mu)**2 / sigma**2).mean()
```

### A.3.3 チェックリスト（PR 前）

- [ ] `prediction_sigma` を常に出力（点推定のみは禁止）
- [ ] Coverage が 85-95% の範囲
- [ ] Aleatoric / Epistemic の分解を記録（Ch8）
- [ ] 校正曲線と直列接続時は Ch13 chain slot に載せる

---

## A.4 スペクトル分類 Skill テンプレート（第5章）

### A.4.1 `SKILL.md` 雛形（差分のみ）

```yaml
skill_name: "spectrum_classification_skill"
version: "0.1.0"
agent_tier_required: "L2"
purpose: >
  Raman / IR / XPS 等の 1D スペクトルから物質種別を分類する Skill。
  1D CNN or 1D Transformer、Raman FM 転移も選択可（第7章）。

inputs:
  data_contract_ref: "data-contract-spectrum-<technique>.yaml"
  wavelength_alignment_required: true        # vol-01 継承
  baseline_correction_applied_in_provenance: true

success_criteria:
  macro_f1_min: 0.85                         # スペクトルは通常画像より F1 高い
  per_instrument_f1_min: 0.70                # 装置間汎化（Ch5 継承）

forbidden:
  - action: "train_test_split_by_random_shuffle_ignoring_instrument"
    severity: "CRITICAL"
  - action: "augmentation_that_shifts_peak_positions"           # 化学的意味破壊
    severity: "CRITICAL"
  - action: "resample_to_uniform_wavenumber_after_augmentation"  # DG-06 系
    severity: "HIGH"

agentic_contract:
  agent_tier: "L2"
  approved_augmentation_ops:
    - "gaussian_noise (sigma <= 0.05 * intensity_range)"
    - "intensity_scale (factor in [0.9, 1.1])"
    - "baseline_shift (bounded, provenance-recorded)"
  forbidden_augmentation_ops:
    - "wavenumber_shift"                     # 化学的に別物質
    - "peak_deletion"
```

### A.4.2 最小コード骨格

```python
# 1D CNN の最小構造
import torch.nn as nn

class Spectrum1DCNN(nn.Module):
    def __init__(self, in_length, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))
```

### A.4.3 チェックリスト（PR 前）

- [ ] Split が **装置ごと** に stratify されている（instrument leak 禁止）
- [ ] Augmentation で peak position を移動していない
- [ ] Wavelength alignment が学習・推論で同一
- [ ] Baseline correction hash が provenance に記録

---

## A.5 表形式回帰 Skill テンプレート（第5章・第12章）

### A.5.1 `SKILL.md` 雛形（差分のみ）

```yaml
skill_name: "tabular_regression_skill"
version: "0.1.0"
agent_tier_required: "L2"
purpose: >
  実験条件・組成・プロセスパラメータ等の表形式データから
  物性値を回帰する Skill。TabNet / FT-Transformer / GBM(vol-02) の比較を含む。

inputs:
  data_contract_ref: "data-contract-tabular-<domain>.yaml"
  categorical_encoding: "target_encoding|onehot|embedding"
  categorical_encoding_fit_on_train_only: true            # data leak 禁止

success_criteria:
  r2_min: 0.70
  vs_gbm_baseline_improvement_pct_min: 5                   # GBM 比 5% 以上改善
  # 改善が 5% 未満なら深層を選ばず GBM を採用（第12章）

forbidden:
  - action: "target_encoding_fitted_on_full_dataset"
    severity: "CRITICAL"
  - action: "adopt_deep_when_gbm_baseline_within_5pct"     # unnecessary complexity
    severity: "MEDIUM"
```

### A.5.2 チェックリスト（PR 前）

- [ ] GBM ベースラインと比較（vol-02 継承）、5% 未満改善なら深層を採用しない
- [ ] Target encoding が train のみで fit
- [ ] Feature importance を SHAP / permutation で記録（Ch10）

---

## A.6 転移学習 Skill テンプレート（第7章・第11章）

### A.6.1 `SKILL.md` 雛形

```yaml
skill_name: "foundation_model_transfer_skill"
version: "0.1.0"
agent_tier_required: "L2"                              # L3 は施設承認委員会経由
purpose: >
  Materials FM（例：MatBERT, MolFormer, MatSciBERT, SEM-FM）を凍結 or
  部分 fine-tune して下流タスクに転移する Skill。

inputs:
  foundation_model_ref: "hf_hub_repo_or_local_uri"
  revision_commit_hash: "40hex sha"                    # Ch11、必ず pin
  fm_update_gate_provenance: "id"

success_criteria:
  transfer_improvement_over_scratch_pct_min: 20        # scratch より 20% 以上改善
  no_forgetting_on_original_domain: "if fine-tune, verify"

forbidden:
  - action: "load_fm_without_hash_verify"              # AG-08
    severity: "CRITICAL"
  - action: "change_default_fm_version_without_gate"   # AG-02
    severity: "CRITICAL"
  - action: "self_sign_as_fm_update_gate_approver"     # AG-06
    severity: "CRITICAL"

reproducibility:
  foundation_model_provenance:
    hub_repo: "org/repo"
    revision_commit_hash: "40hex"
    fetched_at_utc: "ISO8601"
    hash_verified: true
    license: "e.g., cc-by-nc-4.0"
    upstream_license_compatibility: true

agentic_contract:
  agent_tier: "L2"
  fine_tune_scope_declared:
    layers_frozen: []                                  # e.g., ["backbone.stages.0-2"]
    layers_trainable: []
    parameter_count_trainable: "int"
  scope_change_requires_new_gate1: true                # Ch13
```

### A.6.2 最小コード骨格（Hugging Face）

```python
from transformers import AutoModel, AutoConfig
import hashlib

def load_fm_with_verify(repo_id: str, revision: str, expected_sha: str):
    # 1. 明示的 revision pin（40hex）で fetch
    model = AutoModel.from_pretrained(repo_id, revision=revision)
    # 2. hash 検証（Ch11 の fm_fetch_and_verify）
    weight_files = list(model.state_dict().values())
    computed = _compute_sha256_of_state_dict(weight_files)
    if computed != expected_sha:
        raise RuntimeError(f"FM hash mismatch: {computed} vs {expected_sha}")
    return model
```

### A.6.3 チェックリスト（PR 前）

- [ ] `revision_commit_hash` が 40-hex（省略・短縮禁止）
- [ ] hash verification が成功
- [ ] `fm_update_gate_provenance` が存在、approver が non-agent
- [ ] Fine-tune scope（frozen / trainable 層）を provenance に記録
- [ ] Scratch ベースラインと比較、20% 以上改善
- [ ] Upstream license と下流 license が互換

---

## A.7 自己教師あり事前学習 Skill テンプレート（第7章）

### A.7.1 `SKILL.md` 雛形（差分のみ）

```yaml
skill_name: "ssl_pretraining_skill"
version: "0.1.0"
agent_tier_required: "L3"                              # 長時間学習、承認委員会
purpose: >
  ラベルなし材料データで SimCLR / MoCo / MAE 等の SSL を行い、
  下流 fine-tune 用の表現を作る Skill。

inputs:
  unlabeled_data_size_min: 10000                       # SSL は大量データ前提
  labeled_downstream_data_ref: "for probe evaluation"

success_criteria:
  linear_probe_accuracy_min: 0.60                      # freeze + linear で評価
  downstream_finetune_beats_scratch_by_pct: 15

forbidden:
  - action: "leakage_of_downstream_labels_into_ssl_augmentation"
    severity: "CRITICAL"
  - action: "reuse_ssl_representation_for_new_downstream_without_probe_eval"
    severity: "HIGH"

agentic_contract:
  agent_tier: "L3"
  gpu_hours_estimated: "int"                           # 事前見積り、施設承認
  early_stop_on_probe_plateau: true                    # 探索の萎縮回避
  ssl_augmentation_policy_hash: "sha256"
```

### A.7.2 チェックリスト（PR 前）

- [ ] 事前学習と下流評価で **データが完全分離**
- [ ] Linear probe で表現品質を評価
- [ ] GPU quota snapshot が承認済み
- [ ] SSL augmentation が「下流に必要な変動」を保持している

---

## A.8 不確かさ推定 Skill テンプレート（第8章・第9章）

### A.8.1 `SKILL.md` 雛形

```yaml
skill_name: "uncertainty_estimation_skill"
version: "0.1.0"
agent_tier_required: "L2"
purpose: >
  Deep Ensemble / MC-Dropout / BNN (variational) で予測不確かさを
  推定し、Aleatoric / Epistemic に分解する Skill。

variant:
  choose_one:
    - "deep_ensemble (N >= 5)"
    - "mc_dropout (T >= 20)"
    - "variational_bnn"

outputs:
  prediction_mu: "float"
  prediction_sigma_aleatoric: "float"
  prediction_sigma_epistemic: "float"
  prediction_sigma_total: "float"

success_criteria:
  ece_calibration_max: 0.05                            # expected calibration error
  reliability_diagram_recorded: true
  coverage_at_90pct_within: [0.85, 0.95]

forbidden:
  - action: "report_variance_as_sigma_to_pm_normal"    # DG-08、単位混同
    severity: "CRITICAL"
  - action: "hide_epistemic_when_ood_detected"         # 「知らない」を隠す
    severity: "CRITICAL"
  - action: "collapse_ensemble_to_mean_only"
    severity: "HIGH"

agentic_contract:
  agent_tier: "L2"
  autonomous_stop_threshold_defined_by: "human_analyst"    # 閾値は人が決める
  autonomous_stop_action: "route_to_human_gate2"       # Ch13
  threshold_change_requires_gate3: true                # Ch13
```

### A.8.2 最小コード骨格（Deep Ensemble）

```python
import torch.nn as nn

class DeepEnsemble:
    def __init__(self, model_class, n_members=5, **kwargs):
        self.members = [model_class(**kwargs) for _ in range(n_members)]

    def predict(self, x):
        preds = torch.stack([m(x) for m in self.members])
        mu = preds.mean(dim=0)
        sigma_epistemic = preds.std(dim=0)             # 分散
        return mu, sigma_epistemic
```

### A.8.3 チェックリスト（PR 前）

- [ ] `sigma` は **標準偏差**であり分散ではない（DG-08 対策）
- [ ] ECE が 0.05 以下
- [ ] Reliability diagram を artifact に保存
- [ ] Aleatoric / Epistemic が別々に出力
- [ ] 停止閾値が Human 定義、agent が書き換えない

---

## A.9 深層 × Agentic 拡張 provenance 完全スキーマ

以下は vol-01 §A.4・vol-02 §A.7 の provenance schema を **深層 × Agentic 拡張**した完全版です。すべての深層 Skill はこのスキーマの subset を生成することが期待されます。

```yaml
# vol-03 deep_agentic_provenance_schema.yaml
schema_version: "vol-03/1.0"
extends: "vol-02/A.7"

# --- Reproducibility (deep-specific) ---
gpu_backend:
  vendor: "nvidia | amd | apple | cpu_only"
  driver_version: "string"
  cuda_or_rocm_version: "string"
  device_model: "e.g., A100-80GB"
  device_count: "int"
cudnn_deterministic: "bool"
cudnn_benchmark: "bool"                                # false 推奨
random_seed: "int"
random_seed_per_worker: "int (torch DataLoader workers)"
numpy_seed: "int"
python_hash_seed: "int"                                # PYTHONHASHSEED
framework_versions:
  pytorch: "string or null"
  jax: "string or null"
  transformers: "string or null"
  timm: "string or null"

# --- Data & augmentation ---
input_shape: "list[int]"
augmentation_provenance:
  policy_name: "string"
  policy_hash: "sha256"                                # 学習・推論で一致
  augmentation_applied_only_on_train: true             # Ch5
  test_time_augmentation:
    enabled: "bool"
    declared_ops: "list"
    forbidden_ops_verified: true

# --- Model & weights ---
model_architecture:
  family: "cnn | vit | transformer | tabnet | ft_transformer | 1d_cnn | bnn | ensemble"
  name: "e.g., resnet50, vit_base_patch16_224"
  parameter_count: "int"
  parameter_count_trainable: "int"
weights_uri: "content-addressable URI"
weights_sha256: "sha256"
weights_format: "safetensors | pt_state_dict"
foundation_model_provenance:
  used: "bool"
  hub_repo: "string or null"
  revision_commit_hash: "40hex or null"
  fetched_at_utc: "ISO8601 or null"
  hash_verified: "bool"
  license: "string or null"
  upstream_license_compatibility: "bool"

# --- Training config ---
finetune_config:
  layers_frozen: "list of paths"
  layers_trainable: "list of paths"
  lr_schedule: "string"
  batch_size: "int"
  epochs: "int"
  optimizer: "e.g., adamw"
  loss: "string"
  mixed_precision: "bool"
  gradient_clipping: "float or null"

# --- Uncertainty ---
uncertainty:
  method: "deep_ensemble | mc_dropout | vi_bnn | none"
  ensemble_size: "int or null"
  dropout_samples: "int or null"
  aleatoric_captured: "bool"
  epistemic_captured: "bool"
  ece_measured: "float"
  reliability_diagram_uri: "URI or null"

# --- Agentic contract (vol-03 core) ---
agent_authorization:
  agent_tier: "L1 | L2 | L3"
  approved_hp_range_ref: "id"
  hp_range_exceedance_detected: "bool"
  on_exceedance_action_taken: "string"

training_job_approval:
  approval_id: "sha256"
  approver_identity: "opaque_id_or_pubkey"
  approver_role: "e.g., pi, ml_lead, facility_committee"
  approver_is_non_agent: "bool"                        # true 必須
  approved_at_utc: "ISO8601"
  approval_scope: "individual | batched"
  approved_hp_range_snapshot: "dict"

checkpoint_overwrite_policy:
  mode: "append_only | signed_versioning"              # Ch4
  registry_uri: "path to checkpoint_registry.jsonl"
  silent_overwrite_detected: false

fm_update_gate:
  applied: "bool"
  gate_id: "sha256 or null"
  approver_role: "e.g., project_lead + facility"
  approver_is_non_agent: "bool"

uncertainty_stop_threshold:
  threshold_value: "float"
  threshold_defined_by: "role"
  threshold_provenance_hash: "sha256"                  # 実行時変更検知
  autonomous_stop_triggered: "bool"
  stop_routed_to_human_gate2: "bool"

human_review_ref:
  gate1_approval_id: "sha256 or null"                  # fine-tune 起動
  gate2_disposition_id: "sha256 or null"               # 停止後処置
  gate3_exception_id: "sha256 or null"                 # 診断緩和

# --- Audit ---
audit_result_provenance:
  audit_id: "sha256"                                   # Ch14 §14.8
  registry_hash: "sha256"                              # Ch14 registry version
  registry_version: "e.g., v2.0"
  decision: "pass | warn | block"
  non_agent_verifier_signatures: "list (quorum >= 2 for block exceptions)"

# --- Distribution (Ch15) ---
distribution_bundle_ref:
  bundle_id: "sha256 or null"
  usage_restrictions_declared: "bool"
  originating_org_trust_bundle_included: "bool"
  hard_expiry_date: "ISO8601 or null"

# --- Ledger ---
responsibility_ledger_entry_id: "sha256"               # Ch15 canonical_projection
```

### A.9.1 Skill 種別 × 必須項目マトリクス

| 項目 | 画像分類 A.2 | 画像回帰 A.3 | スペクトル A.4 | 表形式 A.5 | 転移学習 A.6 | SSL A.7 | 不確かさ A.8 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `gpu_backend` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `cudnn_deterministic` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `random_seed_per_worker` | ✅ | ✅ | ✅ | – | ✅ | ✅ | ✅ |
| `augmentation_provenance` | ✅ | ✅ | ✅ | – | ✅ | ✅ | ✅ |
| `weights_sha256` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `foundation_model_provenance` | 任意 | 任意 | 任意 | – | ✅ | – | 任意 |
| `finetune_config` | 任意 | 任意 | 任意 | – | ✅ | ✅ | 任意 |
| `uncertainty` | 任意 | ✅ | 任意 | 任意 | 任意 | – | ✅ |
| `training_job_approval` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `checkpoint_overwrite_policy` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `fm_update_gate` | – | – | – | – | ✅ | – | – |
| `uncertainty_stop_threshold` | 任意 | ✅ | 任意 | 任意 | 任意 | – | ✅ |
| `audit_result_provenance` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## A.10 テンプレを新規 Skill に適用する手順

1. **目的 1 文で書く** — テンプレの `purpose` を書き換える
2. **agent tier を決める** — L1/L2/L3、迷ったら低い方（L2）から
3. **data_contract_ref を作る or 参照** — vol-01/02 の contract に準拠
4. **成功条件を数値で書く** — 「良さそう」ではなく閾値で
5. **禁止事項を severity 付きで列挙** — CRITICAL は必ず具体的に
6. **Agentic 契約を pin** — approval / gate / stop threshold の参照 ID
7. **provenance schema の subset を必須項目マトリクスで確認**
8. **PR 前チェックリストを埋める**
9. **Ch14 audit を pre-merge で 1 回走らせる**（`agentic_deep_failure_audit`）

## A.11 まとめ

- 深層 × Agentic Skill は「モデルコード」よりも「契約・provenance・承認・停止」の言語化が本体
- テンプレは 6 + 1（Agentic 契約）要素で構成
- `augmentation_provenance` `foundation_model_provenance` `training_job_approval` `uncertainty` `audit_result_provenance` は事実上ほぼ全 Skill で必須
- Ch14 audit をテンプレ完成後に必ず 1 回走らせる

## 参考資料

### 本書内の該当章

- 第4章：Skill 契約 6 要素、agent tier、approved_hp_range、training_job_approval
- 第5章：Augmentation contract、augmentation_provenance
- 第6-7章：CNN/ViT/1D CNN 実装、SSL、転移学習
- 第8-9章：Deep Ensemble / MC-Dropout / BNN、不確かさ停止
- 第10章：Grad-CAM、feature attribution、diagnostics
- 第11章：Foundation Model provenance、fm_update_gate
- 第12章：checkpoint registry、GBM vs 深層の比較
- 第13章：Gate1/2/3、integrated_provenance_chain
- 第14章：failure_pattern_registry、agentic_deep_failure_audit、audit_id
- 第15章：organization_gpu_resource_policy、model_distribution_contract、responsibility_ledger

### 外部参考

- Paszke et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
- Wightman. timm: PyTorch Image Models. https://github.com/rwightman/pytorch-image-models
- Wolf et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP.
- Lakshminarayanan et al. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS.
- Gal & Ghahramani (2016). Dropout as a Bayesian Approximation. ICML.
- Blundell et al. (2015). Weight Uncertainty in Neural Networks. ICML.
- He et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR.
- Chen et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.
- Arik & Pfister (2021). TabNet: Attentive Interpretable Tabular Learning. AAAI.
- Sigstore project: https://www.sigstore.dev/（provenance signing 参考）
