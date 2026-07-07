# 第8章 物理制約フィルタと合成可能性スクリーニングを Skill 化する

> **本章の使い方**
> - **Part III の第 1 章**です。Ch5-7 で実装した G ステージ Skill（`arim.gen.vae.v0.1` / `arim.gen.diffusion.v0.1` / `arim.gen.flow.v0.1` / `arim.gen.autoregressive.v0.1`）の `candidates_out` を入力として、**Ch2 §2.7 canonical pipeline の F ステージ（物理制約フィルタ）と S1 ステージ（合成可能性 proxy）を Skill 化**します。
> - **本章の scope**：**Skill 契約としての多段フィルタ**（Filter A → B → C の順序と provenance）、**rule-based 物理チェック**（Pymatgen 依存、電荷中性・化学量論・酸化数）、**model-based 特徴量チェック**（matminer 依存）、**合成可能性 proxy**（分子：SC/SA/RA-Score、結晶：hull distance + ICSD historical existence）を、Ch4 §4.3.6 `filter_rules_applied` と §4.3.7 `screening_model_family` の canonical に沿って実装します。
> - **Ch5-7 との責務分離**：本章は **F ステージ（Filter A/B の hard reject）と S1 ステージ（Filter C の soft score）** を扱います。**H ステージの OOD detection（`arim.gen.ood_detection.v0.1`、Ch5 §5.4 で config/result 態を導入・Ch10 で完全計算）とは pipeline stage も判定基準も異なります**（§8.8 で分離マップを提示）。
> - **YAML block は全て `yaml.safe_load` 準拠**：本書 CI で機械検証されます。
> - **教育目的の合成可能性 proxy**：SC/SA/RA-Score は代表的 heuristic 実装に依拠し、hull distance は Materials Project の pre-computed energy を fetch する Route A で扱います。本番運用の precision/recall は §8.5 で議論します。
>
> **本章の到達目標**
> - **Skill 契約の多段フィルタ**（Filter A → B → C）の順序を Ch4 §4.3.6 `filter_rules_applied` の list 順として provenance に固定できる
> - **Ch4 §4.3.6 canonical 3 rules**（`charge_neutrality` / `stoichiometry_integer_multiplier` / `oxidation_state_bounds`）を Pymatgen で実装し、Ch8 拡張 rule（`lattice_stability_precheck` 等）との切り分けを説明できる
> - **matminer 特徴量ベースの outlier フィルタ**を、学習データ分布からの **per-feature z-score / IQR しきい値**で構築し、H ステージ `arim.gen.ood_detection.v0.1` の Mahalanobis / PCA 合成スコアと責務分離できる
> - **合成可能性 proxy** の複数手法（分子：SC/SA/RA、結晶：hull distance / ICSD）を **Ch2 §2.5 canonical に沿った "proxy であり GO/NO-GO ではない" 契約** として Skill YAML に落とせる
> - **`arim.gen.physics_filter.v0.1`** と **`arim.gen.synthesizability_proxy.v0.1`** の完全 Skill YAML を、Ch4 §4.6 Layer 1/2/3 hand-off 宣言と Ch5-7 の親骨格を継承して書ける
> - **Ch5 OOD detection と Ch8 physics/synth の責務分離マップ**（§8.8）を説明でき、両者を混同する失敗パターン（§8.9）を回避できる
>
> **本章で扱わないこと**
> - DFT proxy による物性予測（Ch9、`arim.gen.dft_proxy.v0.1` S1 併走 Skill）
> - 分布的妥当性判定の完全実装（Ch10、`arim.gen.ood_detection.v0.1` H ステージの完全計算）
> - 離散候補ランキング（Ch11、`arim.gen.candidate_ranking.v0.1` T ステージ）
> - 合成試行の GO/NO-GO 判定（Ch4 §4.4 `synthesis_launch_authorization`、A ステージ Human 承認）
> - Layer 1 危険物質 filter list の実装詳細（Ch4 §4.6 `arim.gen.safety_screening.v0.1` SF ステージ、付録 C）
> - Agentic 失敗パターンの taxonomy（Ch14）

---

## 8.1 なぜ生成後スクリーニングが必要か — 3 つの動機

Ch5-7 の G ステージ Skill は、いずれも学習分布内の候補を生成する契約を持ちます。しかし、**分布内サンプリングそのものは物理妥当性を保証しない**という事実が、本章の存在理由です。

### 8.1.1 動機 1：分布内 ≠ 物理的に整合

学習データが 4 元系合成組成（例：`{Fe, Ti, O, Al}` を含む）で、各元素は canonical な酸化数（Fe: +2/+3、Ti: +3/+4、O: -2、Al: +3）を取り、電荷中性を満たしています。しかし、VAE / Diffusion / Flow / AR いずれも **stoichiometry の連続緩和**（Ch5 §5.4 の softmax 正規化 / Ch6 §6.6 / Ch7 §7.3.1 の ALR-logit）を通じて生成します。この連続緩和は、**学習分布の周辺から近距離のサンプルとして「電荷不整合な組成」を出力しうる**——例えば `{Fe: 0.55, Al: 0.15, O: 0.30}`（Fe⁺³ + Al⁺³ + O⁻² で計算した net charge が 0 にならない組成）を生成しても、生成器の loss は「学習データからの Mahalanobis 距離」で低い可能性があります。

**この失敗モードは、H ステージ OOD detection の統計軸（Mahalanobis 距離・reconstruction 誤差）のみでは検出できません**——学習分布内であっても、離散化した結果（例えば整数比への丸め後）が電荷中性を破ることは頻繁に起こります。OOD detection の canonical 合成スコアは Ch5 §5.4 / Ch10 で `0.4 * z_M + 0.4 * z_R + 0.2 * v` と pin されており、**この `v`（physical_violation 軸）の実体は F ステージ physics_filter が算出する rule violation 数を H が受け取って正規化した値です**（Ch10 で具体化）。

> [!NOTE]
> **F hard-gate 下で `v` 軸の役割は「invariant / audit」です**：canonical pipeline では F が hard gate（violation あり候補は下流に流れない）なので、F を pass して H に届く候補は原則 `v = 0` になります。それでも Ch5 / Ch10 canonical で `v` 軸を H composite score に**残す理由**は 3 つあります：(1) config drift（F skip や tolerance 緩和）が起きた場合の **fail-safe な audit 軸**として `v > 0` が現れると即座に anomaly detection できる、(2) F の tolerance ぎりぎりで pass した候補（例：CHARGE_NEUTRALITY_TOL 直下）を H が **境界付近の soft-score として下降補正**できる（境界候補の 0/1 二値化を避ける）、(3) 拡張 rule 追加時に F/H の責務境界を再定義せずに新規 violation 軸を H に注入できる。したがって「canonical 通常運用では v=0 でも、H スコアリング枠組みから v 軸を除去してはならない」（Ch10 で具体化）。

つまり **H は F の物理判定 provenance を委譲入力として合成 OOD スコアを組み立てる**契約であり、F を skip した H 単体では `v = 0` 固定となり本節が指摘する物理不整合を捕捉できません（audit 軸としての機能も失われる）。「学習分布内か」の統計判定と「物理法則に整合するか」の決定論的判定は直交軸であり、canonical H スコアは両者を明示的に線形結合する設計です（§8.8 で軸分離の詳細）。

### 8.1.2 動機 2：分布内 ≠ 合成可能

学習データが「過去に実際に合成された組成」で構成されていても、生成器は **未合成の組合せ**を分布内候補として出力します（これが逆設計の目的です）。しかし、未合成候補には 2 種類あります：

- **合成困難な候補**：熱力学的に準安定ですらない（例：hull energy が +200 meV/atom 以上）
- **合成前例が近い候補**：ICSD historical に類似組成が存在し、合成 route が既知

生成器はこの 2 種を区別しません。**Ch8 は proxy スコアリングで両者を separate する契約層**を提供します。

### 8.1.3 動機 3：Skill 契約層としての hand-off point

Ch4 §4.6 の Safety 3-layer 契約では、**Layer 1 静的 filter** の後に **Layer 2 dual-use review** と **Layer 3 facility policy review** が hand-off されます。Ch8 の physics_filter と synthesizability_proxy は Layer 1 と並列に走る **物理・合成可能性の必要条件チェック**を担い、**Layer 2/3 の hand-off を宣言する義務**を負います（§8.6 / §8.7 の Skill YAML `handoff_to` field）。この宣言があるからこそ、Ch14 の失敗パターン「Skill が Filter A を通しただけで A ステージ Human 承認を skip」が検出可能になります。

> [!IMPORTANT]
> **F/S1 ステージは "必要条件" レイヤ**：本章の 2 Skill を通過した候補は「合成試行の GO 候補」ではなく「A ステージ Human 承認への提案候補」です。Ch4 §4.7 canonical 禁止フレーズ list（"合成可能です" / "推奨候補です" 等）は本章 Skill でも literal に継承します（§8.6 / §8.7）。

---

## 8.2 多段フィルタの canonical 順序 — Filter A → B → C

Ch4 §4.3.6 の `filter_rules_applied` は **list 型**で、**要素の順序が provenance として意味を持つ** canonical です。本章では、以下 3 段の順序を **Ch2 §2.7 pipeline の教育的概念順序**として説明します。ただし **A/B は F ステージ内で連続実行し、C は H ステージ通過後の S1 ステージで独立実行**する 2 Skill 設計です（§8.2.2 の 2 関数 pin、§8.8 責務マトリクスで詳細）：

- Filter **A**（Physics rule-based）: **F ステージ内**（`arim.gen.physics_filter.v0.1`）
- Filter **B**（Feature outlier）: **F ステージ内**（同上 Skill、A の直後）
- H ステージ `arim.gen.ood_detection.v0.1`（Ch10）: **必須通過、Ch8 の範囲外**
- Filter **C**（Synthesizability proxy）: **S1 ステージ**（`arim.gen.synthesizability_proxy.v0.1`、H 通過候補のみ）

したがって「A→B→C」の記法は **pipeline 全体を通した概念順序**であり、単一 Skill 内で連続実行される契約ではありません（§8.2.2 で違反 anti-pattern を明示）。

### 8.2.1 canonical 順序と reject 判定

| 段 | 名称 | 目的 | Reject 判定 | 依存ライブラリ |
|---|---|---|---|---|
| **A** | Physics rule-based | 電荷不整合・化学量論違反・酸化数逸脱の hard reject | **hard reject**（filter 未通過候補は次段に渡さない） | Pymatgen（`Composition`, `Element`, `Species`） |
| **B** | Feature-based outlier | 学習分布特徴量の統計 outlier | **hard reject**（統計しきい値超過候補は次段に渡さない） | matminer（`ElementProperty`, `Stoichiometry` featurizer） |
| **C** | Synthesizability proxy | 合成可能性 proxy score（GO/NO-GO ではない） | **soft score**（reject せず、S1 ステージのランキング入力として score を付与） | Pymatgen + Materials Project API（結晶）、RDKit + SC/SA/RA implementations（分子） |

> [!IMPORTANT]
> **A/B は hard reject、C は soft score**：この非対称性が Ch8 の設計 canonical です。C 段でスコアが低い候補も次段（S1 / S2 / T）に流し、**Ch11 のランキング Skill で top-k から除外される**設計にします。C 段の proxy には偽陰性（合成可能な候補を "合成困難" と誤判定）が本質的に含まれるため、C 段で reject すると **逆設計の主目的（未合成候補の発見）を毀損**します。

### 8.2.2 candidate_id と provenance の伝播

各段は入力候補の `candidate_id` を保持し、reject/pass の記録を `filter_rules_applied` list に追記します。段間で `candidate_id` を新規発行することは**禁止**（元 G ステージ Skill の `candidate_id` と 1:1 対応を保つ）。

**コードスニペット 1 — Ch8 の 2 Skill: F ステージと S1 ステージの pipeline 配置**

Filter A/B（F ステージ）と Filter C（S1 ステージ）は **pipeline stage が異なる別 Skill** です。両者の間には **H ステージ `arim.gen.ood_detection.v0.1`（Ch10）が必ず走る** canonical で、以下 2 関数として独立に定義します：

```python
def apply_physics_filter(
    candidates: list[dict],       # G ステージ出力（各 dict に "candidate_id" と生成データ）
    stage_a_config: dict,
    stage_b_config: dict,
) -> dict:
    """
    Ch8 F ステージ canonical: Filter A → Filter B の 2 段 hard reject。
    provenance には reject/pass の記録を list として保持。
    F ステージ後の候補は H ステージ arim.gen.ood_detection.v0.1（Ch10）に渡り、
    そこでも hard gate を通過した候補のみ S1 ステージ（apply_synthesizability_proxy）に到達する。
    """
    provenance_log: list[dict] = []
    survivors_a = []
    for cand in candidates:
        passed, rule_hits = _stage_a_physics(cand, stage_a_config)
        # _stage_a_physics は rule_hits[0] に preprocessing entry を返す設計。
        # ここで preprocessing_log と canonical/extension rules を separate field に整理する。
        preprocessing_log = rule_hits[0]                       # preprocessing_id を持つ dict
        rules_evaluated = rule_hits[1:]                        # [0:3] canonical, [3:] extension
        provenance_log.append({
            "candidate_id": cand["candidate_id"],
            "stage": "A",
            "passed": passed,
            "preprocessing_log": preprocessing_log,            # filter_rules_applied 対象外
            "rules_evaluated": rules_evaluated,                # [0:3]==YAML filter_rules_applied[0:3]
        })
        if passed:
            survivors_a.append(cand)

    survivors_b = []
    for cand in survivors_a:
        passed, rule_hits = _stage_b_featurizer(cand, stage_b_config)
        provenance_log.append({
            "candidate_id": cand["candidate_id"],
            "stage": "B",
            "passed": passed,
            "rules_evaluated": rule_hits,
        })
        if passed:
            survivors_b.append(cand)

    return {
        "survivors": survivors_b,               # YAML output_schema.survivors（H に渡す）
        "provenance_log": provenance_log,
        "counts": {
            "n_input": len(candidates),
            "n_after_a": len(survivors_a),
            "n_after_b": len(survivors_b),
        },
    }

def apply_synthesizability_proxy(
    candidates_after_fh: list[dict],       # F と H を pass 済み候補のみ
    stage_c_config: dict,
) -> dict:
    """
    Ch8 S1 ステージ canonical: Filter C soft score のみ、reject しない。
    入力候補は必ず F ステージ（apply_physics_filter の `survivors` フィールド）と
    H ステージ（arim.gen.ood_detection.v0.1、Ch10）を pass 済みであること。
    YAML input_schema.validation_rule と disallowed_operations.invoke_without_verified_f_h_pass_provenance
    に従い、F/H pass provenance が付いていない候補は本関数で fail-fast する（silent proceed 禁止）。
    低スコア候補も drop せず全て S2/T ステージに流す（Ch11 T ステージが top-k 選抜を所有）。
    """
    # F/H pass provenance の enforcement（YAML input_schema.validation_rule と一致）
    for cand in candidates_after_fh:
        if not cand.get("physics_filter_run_id"):
            raise ValueError(
                f"candidate {cand.get('candidate_id')} lacks physics_filter_run_id; "
                "S1 requires concrete F stage execution id (flag alone is insufficient, §8.7)"
            )
        if not cand.get("physics_filter_passed", False):
            raise ValueError(
                f"candidate {cand.get('candidate_id')} lacks physics_filter_passed=True; "
                "S1 Skill invocation without verified F pass is disallowed (§8.7)"
            )
        if not cand.get("ood_detection_run_id"):
            raise ValueError(
                f"candidate {cand.get('candidate_id')} lacks ood_detection_run_id; "
                "S1 requires concrete H stage execution id (flag alone is insufficient, §8.7)"
            )
        if not cand.get("ood_detection_passed", False):
            raise ValueError(
                f"candidate {cand.get('candidate_id')} lacks ood_detection_passed=True; "
                "S1 Skill invocation without verified H pass is disallowed (§8.7)"
            )
        trace = set(cand.get("upstream_stage_trace", []))
        if not {"F", "H"} <= trace:
            raise ValueError(
                f"candidate {cand.get('candidate_id')} upstream_stage_trace missing F/H: got {trace}"
            )

    # method_id と candidate_domain の互換性 enforcement（YAML validation_rule と一致、§8.5 canonical）
    _MOLECULE_METHODS = {"sc_score", "sa_score", "ra_score"}
    _CRYSTAL_METHODS = {"hull_distance", "icsd_historical"}
    method_id = stage_c_config["method_id"]
    domain = stage_c_config.get("candidate_domain")
    if method_id in _MOLECULE_METHODS:
        if domain != "molecule":
            raise ValueError(f"{method_id} requires candidate_domain='molecule', got {domain}")
        for cand in candidates_after_fh:
            if "smiles" not in cand:
                raise ValueError(f"{method_id} requires candidate.smiles for {cand.get('candidate_id')}")
    elif method_id in _CRYSTAL_METHODS:
        if domain != "crystal":
            raise ValueError(f"{method_id} requires candidate_domain='crystal', got {domain}")
        for cand in candidates_after_fh:
            if "generated_x" not in cand and "structure" not in cand:
                raise ValueError(
                    f"{method_id} requires candidate.generated_x or structure for {cand.get('candidate_id')}"
                )
    else:
        raise ValueError(f"unknown method_id: {method_id}")

    provenance_log: list[dict] = []
    scored = []
    scores = []
    for cand in candidates_after_fh:
        score = _stage_c_synthesizability_score(cand, stage_c_config)
        cand["synthesizability_proxy_score"] = score   # float [0, 1]、高いほど合成前例に近い
        scored.append(cand)
        scores.append(score)
        provenance_log.append({
            "candidate_id": cand["candidate_id"],
            "stage": "C",
            "passed": True,           # C 段は必ず True（reject 禁止 canonical、§8.2.1）
            "proxy_score": score,
            "proxy_method": stage_c_config["method_id"],
        })
    score_summary = {
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "mean": (sum(scores) / len(scores)) if scores else None,
    }
    return {
        "scored_candidates": scored,       # 全入力候補（reject なし）
        "proxy_provenance": {              # YAML output_schema.proxy_provenance
            "method_id": stage_c_config["method_id"],
            "candidate_domain": stage_c_config.get("candidate_domain", "crystal"),
            "n_scored": len(scored),
            "score_summary": score_summary,
            "per_candidate_log": provenance_log,
        },
    }
```

> [!IMPORTANT]
> **F と S1 の間には H が必ず走る**：本節では H ステージのコードは示しません（Ch10 で完全実装）。**F ステージ出力を直接 S1 ステージに渡す pipeline 配線は Ch14 §14.2 taxonomy 対象**（F/H の hard gate 契約違反）です。上の 2 関数を pipeline 配線するときは、必ず `apply_physics_filter → ood_detection → apply_synthesizability_proxy` の順で組みます。

---

## 8.3 Filter A — 物理制約 rule-based（Ch4 §4.3.6 canonical 3 rules）

### 8.3.1 canonical 3 rules と拡張 rule の切り分け

Ch4 §4.3.6 は以下 3 rule を **Part I からの canonical set** として固定しました：

| rule_id | 意味 | 実装依存 |
|---|---|---|
| `charge_neutrality` | Σ(oxidation_state × mole_fraction) = 0（許容誤差 1e-6） | Pymatgen `Composition.oxi_state_guesses()` |
| `stoichiometry_integer_multiplier` | 全 mole_fraction を最小整数比に換算可能（最大分母 K=20 canonical） | 分数近似（`fractions.Fraction`） |
| `oxidation_state_bounds` | 各元素の推定酸化数が Pymatgen `Element.common_oxidation_states` の範囲内 | Pymatgen `Element` |

本章では以下 3 rule を **Ch8 拡張**として追加し、**Ch2 §2.7 canonical pipeline の F ステージで並列に走る**契約とします（付録 A で back-port 予定）：

| rule_id | 意味 | 実装依存 | 拡張理由 |
|---|---|---|---|
| `lattice_stability_precheck` | 結晶候補の対称性が Pymatgen `SpacegroupAnalyzer` で有効な空間群を返す | Pymatgen `SpacegroupAnalyzer`（tolerance canonical: 0.01 Å） | 結晶構造 Diffusion（Ch6 optional 節）出力の pre-DFT sanity |
| `element_availability_check` | 使用元素が付録 C canonical `allowed_element_set` に含まれる（危険物質 fast reject） | 付録 C list との集合演算 | Layer 1 危険物質 filter と並列の冗長チェック（fail-safe） |
| `formula_max_atoms` | 単位胞あたりの total atoms が上限値以下（canonical: 200） | structure がある場合は `Structure.num_sites`、無い場合は canonicalize 後整数比合計を fallback | 過大 unit cell の hard reject（Ch9 DFT proxy の compute budget 保護） |

> [!NOTE]
> **Canonical 3 rules は必ず走る、拡張 3 rules は opt-in**：Skill YAML の `filter_rules_applied` list には **canonical 3 rules の rule_id が必ず先頭 3 要素として存在**する契約です（§8.6 で pin）。拡張 rule は Skill 呼び出し時に `hyperparameters.extension_rules_enabled` で選択します。

### 8.3.2 Pymatgen による canonical 3 rules の実装

Filter A の 3 rules は **入力 composition に対する扱いが 2 通りに分かれる** canonical です：

| rule_id | 入力 composition | 理由 |
|---|---|---|
| `stoichiometry_integer_multiplier` | **raw**（生成器出力そのまま） | この rule 自体が「K=20 で整数比化可能か」を検査する canonical 検査項目のため、事前に丸めてしまうと tautology になる |
| `charge_neutrality` | **canonicalize 後**（整数 dict） | Pymatgen `oxi_state_guesses()` が整数値組成を要求するため（非整数だと ValueError） |
| `oxidation_state_bounds` | **canonicalize 後**（整数 dict） | 同上 |
| `element_availability_check`（拡張） | **raw**（生成器出力そのまま） | canonicalize 済み整数 dict は GCD 還元で trace 元素が silent drop されるため、危険物質 fail-safe を bypass しかねない |
| `formula_max_atoms`（拡張） | **canonicalize 後**（整数 dict）＋ structure（あれば優先） | Ch9 DFT proxy 予算保護は本来 unit-cell 保護のため、`structure.num_sites` があればそれを優先し、composition-only 候補では整数比合計を fallback（provenance に basis を記録） |
| `lattice_stability_precheck`（拡張） | Structure（composition 非使用） | 空間群判定は Structure が必要 |

§8.9.1 で議論する false negative の主原因は fractional 表現の丸め誤差なので、以下 `canonicalize_composition_to_integer_ratio` を **charge/oxidation 2 rules の前段で走る preprocessing** として本節で定義し、`_stage_a_physics`（§8.3.4）で 2 rules に共通入力として渡します。ただし `stoichiometry_integer_multiplier` rule には **raw composition をそのまま渡し**、rule 自体が独立に K/tolerance 検査を再実行します（§8.3.4 のコードで pin）。この preprocessing は `filter_rules_applied` list に load される canonical rule ではなく、Filter A 内部の shared canonical preprocessing です。

**コードスニペット 2 — Filter A canonical preprocessing + 3 rules 実装**

```python
from pymatgen.core import Composition, Element
from fractions import Fraction
from math import gcd
from functools import reduce

CHARGE_NEUTRALITY_TOL = 1e-6              # canonical 許容誤差
STOICHIOMETRY_MAX_DENOMINATOR = 20         # canonical 最大分母 K
STOICHIOMETRY_MATCH_TOL = 1e-3             # 整数比マッチング許容誤差

def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)

def canonicalize_composition_to_integer_ratio(
    composition: dict[str, float],
) -> tuple[dict[str, int] | None, dict]:
    """
    Ch8 Filter A canonical preprocessing: 全 rule の前に必ず走る shared step。
    fractional mole fraction 表現を **共通分母 LCM で整数化**し、GCD で最小整数比に還元。
    pymatgen `Composition.oxi_state_guesses(target_charge=0)` は整数値組成を要求する
    （非整数だと ValueError: Charge balance analysis requires integer values in Composition）
    ため、canonical rules に渡す前に本 step で **必ず整数 dict に変換**する。
    max_delta > STOICHIOMETRY_MATCH_TOL なら preprocessing 自体を fail として返す。
    戻り値: (整数比 composition dict[str, int] or None, 正規化 log)。
    """
    fractions_list = [
        Fraction(v).limit_denominator(STOICHIOMETRY_MAX_DENOMINATOR)
        for v in composition.values()
    ]
    max_delta = max(
        abs(orig - float(frac))
        for orig, frac in zip(composition.values(), fractions_list)
    )
    log = {
        "preprocessing_id": "canonicalize_composition_to_integer_ratio",
        "max_delta": max_delta,
        "max_denominator": STOICHIOMETRY_MAX_DENOMINATOR,
        "stoichiometry_match_tol": STOICHIOMETRY_MATCH_TOL,
    }
    if max_delta > STOICHIOMETRY_MATCH_TOL:
        log["passed"] = False
        log["reason"] = "max_delta_exceeds_stoichiometry_match_tol"
        return None, log

    # 共通分母 LCM で整数化 → GCD で最小整数比に還元
    denoms = [f.denominator for f in fractions_list]
    common_denom = reduce(_lcm, denoms)
    numerators = [f.numerator * (common_denom // f.denominator) for f in fractions_list]
    if all(n == 0 for n in numerators):
        log["passed"] = False
        log["reason"] = "all_numerators_zero_after_lcm"
        return None, log
    g = reduce(gcd, [n for n in numerators if n != 0])
    reduced = [n // g for n in numerators]

    canonical: dict[str, int] = {
        elem: n for elem, n in zip(composition.keys(), reduced) if n > 0
    }
    log["passed"] = True
    log["integer_ratio"] = canonical
    return canonical, log

def rule_charge_neutrality(composition: dict[str, int]) -> tuple[bool, dict]:
    """
    Σ(oxidation_state × mole_fraction) = 0（許容誤差 CHARGE_NEUTRALITY_TOL）
    Pymatgen の oxi_state_guesses は組成に対して整合する酸化数割当の list を返す。
    1 つでも整合割当が存在すれば pass。
    入力は canonicalize_composition_to_integer_ratio 済みの **整数 dict**（pymatgen
    oxi_state_guesses は整数値組成を要求する — 非整数だと ValueError）。
    """
    comp = Composition(composition)
    guesses = comp.oxi_state_guesses(target_charge=0)
    # guesses は list of dict (dict は element -> avg_oxi_state)、空 list なら整合割当なし
    if not guesses:
        return False, {"rule_id": "charge_neutrality", "guesses": []}
    return True, {
        "rule_id": "charge_neutrality",
        "guesses": [dict(g) for g in guesses[:3]],   # 上位 3 割当のみ記録
    }

def rule_stoichiometry_integer_multiplier(
    composition: dict[str, float],
) -> tuple[bool, dict]:
    """
    全 mole_fraction を最小整数比に換算可能か（最大分母 K=STOICHIOMETRY_MAX_DENOMINATOR）
    STOICHIOMETRY_MATCH_TOL 以内で分数近似できることを要求。
    入力は **raw composition**（生成器出力そのまま）を渡す。canonicalize 済み値を渡すと
    tautology（常に True）になり Ch4 canonical purpose を破るため、`_stage_a_physics`
    （§8.3.4）では明示的に raw を渡す（§8.3.2 表参照）。
    """
    fractions_list = [
        Fraction(v).limit_denominator(STOICHIOMETRY_MAX_DENOMINATOR)
        for v in composition.values()
    ]
    # 近似誤差の確認
    for orig, frac in zip(composition.values(), fractions_list):
        if abs(orig - float(frac)) > STOICHIOMETRY_MATCH_TOL:
            return False, {
                "rule_id": "stoichiometry_integer_multiplier",
                "max_error": max(
                    abs(o - float(f)) for o, f in zip(composition.values(), fractions_list)
                ),
                "max_denominator": STOICHIOMETRY_MAX_DENOMINATOR,
            }
    return True, {
        "rule_id": "stoichiometry_integer_multiplier",
        "integer_ratios": [str(f) for f in fractions_list],
    }

def rule_oxidation_state_bounds(
    composition: dict[str, int],
    integer_tolerance: float = 0.15,
) -> tuple[bool, dict]:
    """
    各元素の推定酸化数が Pymatgen Element.common_oxidation_states の範囲内。
    canonical: oxi_state_guesses が返す全割当 list を評価し、
      **いずれか 1 割当**が全元素で common_oxidation_states に整合すれば pass。
    整数性判定は round ではなく |float - nearest_int| <= integer_tolerance を採用（mixed-valence 対応）。
    integer_tolerance canonical: 0.15（1/3 mixed-valence 0.33 は integer と見なさない範囲）。
    入力は canonicalize_composition_to_integer_ratio 済みの整数 dict。
    """
    comp = Composition(composition)
    guesses = comp.oxi_state_guesses(target_charge=0)
    if not guesses:
        # charge_neutrality が false の場合、この rule も false
        return False, {"rule_id": "oxidation_state_bounds", "reason": "no_valid_oxidation_guess"}

    # 全割当を評価し、1 つでも「全元素 common に整合」する assignment があれば pass
    per_assignment_status = []
    for idx, assigned in enumerate(guesses):
        violations = []
        for symbol, oxi_state in assigned.items():
            elem = Element(symbol)
            common = list(elem.common_oxidation_states)
            # mixed-valence 対応: 最寄り整数との差が tolerance 以内なら整数扱い
            nearest_int = round(oxi_state)
            if abs(oxi_state - nearest_int) > integer_tolerance:
                # mixed-valence: 上下 2 つの整数が両方 common に含まれれば pass 扱い
                lo, hi = int(oxi_state), int(oxi_state) + 1
                if not (lo in common and hi in common):
                    violations.append({
                        "element": symbol,
                        "assigned_oxi": oxi_state,
                        "type": "mixed_valence_out_of_common",
                        "common_oxidation_states": common,
                    })
            else:
                if nearest_int not in common:
                    violations.append({
                        "element": symbol,
                        "assigned_oxi": oxi_state,
                        "nearest_int": nearest_int,
                        "common_oxidation_states": common,
                    })
        per_assignment_status.append({
            "assignment_index": idx,
            "passed": len(violations) == 0,
            "violations": violations,
        })
        if len(violations) == 0:
            # 1 つでも整合すれば pass（早期 return せず全 log は残さない：上位 3 のみ）
            return True, {
                "rule_id": "oxidation_state_bounds",
                "matched_assignment_index": idx,
                "assigned": dict(assigned),
            }
    # 全割当で violation
    return False, {
        "rule_id": "oxidation_state_bounds",
        "per_assignment_status": per_assignment_status[:3],
    }
```

### 8.3.3 拡張 3 rules の実装スケッチ

**コードスニペット 3 — 拡張 rules 実装**

```python
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

LATTICE_SPACEGROUP_TOL = 0.01              # canonical Å
FORMULA_MAX_ATOMS_DEFAULT = 200            # canonical 単位胞上限

def rule_lattice_stability_precheck(
    structure,                              # Pymatgen Structure
) -> tuple[bool, dict]:
    """
    結晶候補の対称性が有効な空間群を返すか。
    Structure が入力される場合のみ適用（組成のみの候補には not_applicable）。
    """
    if structure is None:
        return True, {"rule_id": "lattice_stability_precheck", "status": "not_applicable"}
    try:
        sga = SpacegroupAnalyzer(structure, symprec=LATTICE_SPACEGROUP_TOL)
        sg_number = sga.get_space_group_number()
        if sg_number is None or sg_number < 1 or sg_number > 230:
            return False, {"rule_id": "lattice_stability_precheck", "sg_number": sg_number}
        return True, {"rule_id": "lattice_stability_precheck", "sg_number": sg_number}
    except Exception as e:                  # noqa: BLE001, Pymatgen が特定 exception を投げるため
        return False, {"rule_id": "lattice_stability_precheck", "error": str(e)}

def rule_element_availability_check(
    composition: dict[str, float],
    allowed_element_set: set[str],          # 付録 C canonical
) -> tuple[bool, dict]:
    """
    使用元素が付録 C canonical allowed_element_set に含まれるか。
    Layer 1 危険物質 filter と冗長だが、fail-safe として本章でも走る。
    **重要**：入力は `_stage_a_physics` で **raw composition**（canonicalize 前）を渡す設計。
    canonicalize 済み整数 dict は trace 元素（numerator が GCD 還元で 0 になる元素）が
    silent drop されるため、危険物質 fail-safe を bypass しかねない（§8.3.4 で pin）。
    """
    used = set(composition.keys())
    disallowed = used - allowed_element_set
    if disallowed:
        return False, {
            "rule_id": "element_availability_check",
            "disallowed_elements": sorted(disallowed),
        }
    return True, {"rule_id": "element_availability_check"}

def rule_formula_max_atoms(
    composition: dict[str, int],
    max_atoms: int = FORMULA_MAX_ATOMS_DEFAULT,
    structure: object | None = None,
) -> tuple[bool, dict]:
    """
    単位胞あたり total atoms が上限以下。
    - structure が与えられれば structure.num_sites を優先（真の unit-cell 保護、Ch9 DFT proxy 予算防衛）
    - structure が無い composition-only 候補では canonicalize 済み整数比の合計を fallback として用いる
      （fallback 側は formula-unit 保護であり unit-cell 保護ではない旨 provenance に明記）
    """
    if structure is not None and hasattr(structure, "num_sites"):
        total_atoms = int(structure.num_sites)
        basis = "unit_cell_num_sites"
    else:
        total_atoms = int(sum(composition.values()))
        basis = "reduced_formula_atom_sum"
    if total_atoms > max_atoms:
        return False, {
            "rule_id": "formula_max_atoms",
            "total_atoms": total_atoms,
            "max_atoms": max_atoms,
            "basis": basis,
        }
    return True, {
        "rule_id": "formula_max_atoms",
        "total_atoms": total_atoms,
        "basis": basis,
    }
```

### 8.3.4 stage_a の統合実装

**コードスニペット 4 — Filter A stage 統合**

```python
CANONICAL_RULES_STAGE_A = [
    "charge_neutrality",
    "stoichiometry_integer_multiplier",
    "oxidation_state_bounds",
]

EXTENSION_RULES_STAGE_A = [
    "lattice_stability_precheck",
    "element_availability_check",
    "formula_max_atoms",
]

def _stage_a_physics(cand: dict, config: dict) -> tuple[bool, list[dict]]:
    """
    Filter A: canonical preprocessing → canonical 3 rules → 拡張 rules。
    canonical preprocessing 失敗（max_delta > STOICHIOMETRY_MATCH_TOL）は即 hard reject
      で、以降の canonical/拡張 rule は None composition のため実行不能：short-circuit する。
    canonical 3 rules いずれか false でも hard reject（早期 return はしない：full log を残す）。

    Provenance ordering canonical (Ch4 §4.3.6 filter_rules_applied 契約):
      Return value is `[preprocessing_entry] + rule_hits`（成功/失敗パス共通）で、
      **fixed-slot invariant** を維持：
        result[0]   == preprocessing entry (preprocessing_id key、filter_rules_applied 対象外)
        result[1:4] == canonical 3 rules (charge_neutrality, stoichiometry_integer_multiplier,
                        oxidation_state_bounds) — YAML `filter_rules_applied[0:3]` と 1:1
        result[4:7] == 拡張 3 rules 固定 3 slot（YAML declared 順、`lattice_stability_precheck`,
                        `element_availability_check`, `formula_max_atoms`）：
                        - `ext_enabled` に含まれる rule は passed: bool + detail
                        - 含まれない rule も `passed: None, detail: {status: "not_enabled"}`
                          で placeholder emit（絶対 index 参照可能な CI/provenance parser のため）
      呼び出し側 `apply_physics_filter` が `preprocessing_log = result[0]`,
      `rules_evaluated = result[1:]`（長さ常に 6）に split する。
    """
    raw_composition = cand["generated_x"]          # dict[str, float]（Ch5-7 canonical shape）
    structure = cand.get("structure")              # 結晶候補のみ
    allowed_set = set(config["allowed_element_set"])
    max_atoms = config.get("max_atoms", FORMULA_MAX_ATOMS_DEFAULT)
    ext_enabled = config.get("extension_rules_enabled", [])
    # config drift 検出：unknown ext rule は preprocessing 前に fail-fast（invariant 保護）
    EXTENSION_RULES_STAGE_A: tuple[str, ...] = (      # §8.3.1 pin と同順（YAML declared 順）
        "lattice_stability_precheck",
        "element_availability_check",
        "formula_max_atoms",
    )
    unknown = set(ext_enabled) - set(EXTENSION_RULES_STAGE_A)
    if unknown:
        raise ValueError(f"unknown extension rules: {unknown}")

    # canonical preprocessing (§8.3.2): 全 rule 前に必ず走る
    composition, preproc_log = canonicalize_composition_to_integer_ratio(raw_composition)
    preprocessing_entry = {
        "preprocessing_id": "canonicalize_composition_to_integer_ratio",
        "passed": preproc_log["passed"], "detail": preproc_log,
    }

    if composition is None:
        # preprocessing 失敗＝分母 K=20 で表現できない fractional 候補：hard reject
        # canonical 3 rules は integer-ratio 前提のため実行不能。"skipped" として記録
        rule_hits: list[dict] = []
        for rule_id in ("charge_neutrality", "stoichiometry_integer_multiplier",
                        "oxidation_state_bounds"):
            rule_hits.append({
                "rule_id": rule_id, "passed": False,
                "detail": {"reason": "skipped_due_to_preprocessing_failure"},
                "kind": "canonical",
            })
        # 拡張 3 slot は常に emit（enabled でない場合は not_enabled、fixed-slot invariant 維持）
        for rule_id in EXTENSION_RULES_STAGE_A:
            if rule_id in ext_enabled:
                rule_hits.append({
                    "rule_id": rule_id, "passed": False,
                    "detail": {"reason": "skipped_due_to_preprocessing_failure"},
                    "kind": "extension",
                })
            else:
                rule_hits.append({
                    "rule_id": rule_id, "passed": None,
                    "detail": {"status": "not_enabled"},
                    "kind": "extension",
                })
        return False, [preprocessing_entry] + rule_hits

    all_passed = True
    rule_hits: list[dict] = []

    # canonical 3 rules（YAML filter_rules_applied[0:3] と同順）
    for rule_id, fn, args in [
        ("charge_neutrality", rule_charge_neutrality, (composition,)),
        ("stoichiometry_integer_multiplier",
         rule_stoichiometry_integer_multiplier, (raw_composition,)),   # raw を渡して canonical purpose 保持
        ("oxidation_state_bounds", rule_oxidation_state_bounds, (composition,)),
    ]:
        passed, log = fn(*args)
        rule_hits.append({"rule_id": rule_id, "passed": passed, "detail": log, "kind": "canonical"})
        if not passed:
            all_passed = False

    # 拡張 3（YAML declared canonical 順で iterate、config["extension_rules_enabled"] に含まれるものだけ実行）
    # element_availability_check は **raw composition** を受け取る（canonicalize で trace 元素が
    # zero-numerator になり silent drop されると、危険物質 fail-safe を bypass するため）。
    # formula_max_atoms は structure がある場合 num_sites を優先（unit-cell 保護）、
    # composition-only 候補では canonicalize 済み整数比の合計を fallback（formula-unit 保護）。
    ext_map = {
        "lattice_stability_precheck": (rule_lattice_stability_precheck, (structure,)),
        "element_availability_check": (rule_element_availability_check, (raw_composition, allowed_set)),
        "formula_max_atoms": (rule_formula_max_atoms, (composition, max_atoms, structure)),
    }
    for rule_id in EXTENSION_RULES_STAGE_A:
        if rule_id in ext_enabled:
            fn, args = ext_map[rule_id]
            passed, log = fn(*args)
            rule_hits.append({"rule_id": rule_id, "passed": passed, "detail": log, "kind": "extension"})
            if not passed:
                all_passed = False
        else:
            # not_enabled でも fixed-slot invariant 維持のため placeholder を emit
            rule_hits.append({
                "rule_id": rule_id, "passed": None,
                "detail": {"status": "not_enabled"},
                "kind": "extension",
            })

    # 成功パスも preprocessing_entry を先頭に置く：呼び出し側が rule_hits[0] を split
    return all_passed, [preprocessing_entry] + rule_hits
```

> [!NOTE]
> **preprocessing と stoichiometry_integer_multiplier の関係**：preprocessing は「fractional 表現 → 整数比丸め」を hard reject 付きで実行し、`stoichiometry_integer_multiplier` rule は **raw composition を入力として max_delta 検査を再実行**します。両者は同じ tolerance を使うため実質的に preprocessing pass ⇔ rule pass ですが、Ch4 §4.3.6 canonical で `filter_rules_applied` list に stoichiometry rule が登録される契約を守るため、rule 自体は独立に呼び出して provenance に記録します。**preprocessing は `preprocessing_log` field に分離**して provenance に載せ（呼び出し側 `apply_physics_filter` で split）、**`filter_rules_applied` list には登録しません**（§8.6 で pin）。したがって `rules_evaluated[0:3]` は常に YAML `filter_rules_applied[0:3]`（canonical 3 rules）と 1:1 対応します。

---

## 8.4 Filter B — matminer 特徴量ベース outlier フィルタ

### 8.4.1 なぜ B 段が必要か（A と C の隙間）

Filter A（rule-based）を通った候補が、以下のパターンで**学習分布から統計的に大きく逸脱**することがあります：

- **原子番号平均が学習分布から大きく外れる**：例えば軽元素中心の学習データに対し、生成器が Ln 系元素を含む組成を出力
- **electronegativity の分散が異常**：似た electronegativity の元素だけで構成された組成（学習データが混合電気陰性度前提の場合の統計逸脱）
- **atomic radius の統計 mismatch**：格子安定性を左右する半径統計が学習データの支持範囲外

これらは Filter A（電荷中性・化学量論）では捕まえられず、Filter C（合成可能性 proxy）にも noise として混入します。H ステージの `arim.gen.ood_detection.v0.1`（Ch5 §5.4 で config 態を導入、Ch10 で完全計算）は Mahalanobis / PCA 合成スコアで多次元相関を含めた OOD 判定を行いますが、Ch8 の Filter B は **`matminer.featurizers.composition` の元素物性統計量に対する per-feature z-score / IQR fence の独立軸**として重ねる設計です（§8.8 で責務分離を明示）。

### 8.4.2 canonical matminer 特徴量セット

以下 6 特徴量を **Ch8 canonical set** として固定します：

| feature_id | matminer 由来（内部 property/stat 名） | 統計しきい値の canonical |
|---|---|---|
| `elem_mean_AtomicNumber` | `ElementProperty(data_source="magpie", features=["Number"], stats=["mean"])`（matminer `MagpieData` の内部 property 名は `Number`） | 学習分布 mean ± 3σ（訓練時に pin） |
| `elem_mean_Electronegativity` | `ElementProperty("magpie", features=["Electronegativity"], stats=["mean"])` | 同上 |
| `elem_std_Electronegativity` | `ElementProperty("magpie", features=["Electronegativity"], stats=["std_dev"])`（matminer `PropertyStats` の stat 名は `std_dev`） | 学習分布 IQR × 1.5 fence |
| `elem_mean_CovalentRadius` | `ElementProperty("magpie", features=["CovalentRadius"], stats=["mean"])` | 学習分布 mean ± 3σ |
| `elem_mean_MeltingT` | `ElementProperty("magpie", features=["MeltingT"], stats=["mean"])` | 学習分布 mean ± 3σ |
| `stoich_2norm` | `Stoichiometry(p_list=[2])` | 学習分布 IQR × 1.5 fence |

> [!IMPORTANT]
> **外部 canonical ID と matminer 内部名の分離**：外部 canonical feature_id は `elem_mean_AtomicNumber`（人間可読）を保持しますが、matminer 内部の `MagpieData` property 名は `Number`（`AtomicNumber` ではない）、`PropertyStats` の stat 名は `std_dev`（`std` ではない）です。§8.4.3 の featurizer 実装ではこのマッピングを明示します。

これら **6 特徴量の統計しきい値は、G ステージ Skill の `training_data_provenance` と 1:1 で pin**されます（Skill YAML §8.6 の `feature_thresholds_provenance` field）。訓練時に fit して固定した mean/std/IQR を推論時に literal 引用する canonical です。

> [!NOTE]
> **統計しきい値の canonical**：Ch4 §4.5 の Mahalanobis 距離とは別軸で、Ch8 Filter B は **1 次元 z-score / IQR fence の 6 独立チェック**です。Mahalanobis は 6 次元同時分布の Cov を使うため相関構造を考慮しますが、Filter B は「1 特徴量あたりの外れ値」を hard reject する fast filter として位置付けます。両者の使い分けは §8.8 の責務分離マップを参照。

### 8.4.3 実装スケッチ

**コードスニペット 5 — Filter B stage 実装**

```python
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from pymatgen.core import Composition
import numpy as np

CANONICAL_FEATURE_SET_B = [
    # (canonical_id, data_source, matminer_property, matminer_stat, threshold_kind)
    ("elem_mean_AtomicNumber", "magpie", "Number", "mean", "3sigma"),
    ("elem_mean_Electronegativity", "magpie", "Electronegativity", "mean", "3sigma"),
    ("elem_std_Electronegativity", "magpie", "Electronegativity", "std_dev", "iqr_1.5"),
    ("elem_mean_CovalentRadius", "magpie", "CovalentRadius", "mean", "3sigma"),
    ("elem_mean_MeltingT", "magpie", "MeltingT", "mean", "3sigma"),
    # stoich_2norm は Stoichiometry(p_list=[2]) で独立に計算
]

def _build_element_property_featurizer() -> tuple[ElementProperty, dict[str, int]]:
    """
    canonical 5 特徴量（AtomicNumber(=Number)/Electronegativity mean, Electronegativity std_dev,
    CovalentRadius mean, MeltingT mean）を matminer 内部名で instantiate。
    feature_labels() の実 label 順を index map に固定して返す。
    """
    ep = ElementProperty(
        data_source="magpie",
        features=["Number", "Electronegativity", "CovalentRadius", "MeltingT"],
        stats=["mean", "std_dev"],
    )
    # 実 label list から必要 5 列の index を lookup（label 形式は matminer 実装依存のため
    # ハードコード禁止：labels() の返り値から動的に構築する）
    labels = ep.feature_labels()
    label_index: dict[str, int] = {}
    for feat_id, _ds, prop, stat, _thr in CANONICAL_FEATURE_SET_B:
        # matminer label は "MagpieData mean Number" 等の形式：substring 一致で探索
        matched = [
            i for i, lab in enumerate(labels)
            if (prop in lab) and (stat in lab.lower().split())
        ]
        if len(matched) != 1:
            raise RuntimeError(
                f"canonical feature {feat_id} (matminer prop={prop}, stat={stat}) "
                f"not uniquely resolvable in labels={labels}; found {len(matched)} matches"
            )
        label_index[feat_id] = matched[0]
    return ep, label_index

# module load 時に 1 度だけ構築（stateless、thread-safe）
_ELEMENT_PROPERTY_EP, _ELEMENT_PROPERTY_INDEX = _build_element_property_featurizer()

def _featurize(composition: dict[str, float]) -> dict[str, float]:
    """
    canonical 6 特徴量を Composition から計算。
    ElementProperty は明示 features/stats で instantiate 済み（_build_... 参照）。
    label→index 対応は起動時に検証済み（RuntimeError で fail-fast）。
    """
    comp = Composition(composition)
    values = _ELEMENT_PROPERTY_EP.featurize(comp)
    feats: dict[str, float] = {
        feat_id: values[idx] for feat_id, idx in _ELEMENT_PROPERTY_INDEX.items()
    }
    st = Stoichiometry(p_list=[2])
    feats["stoich_2norm"] = st.featurize(comp)[0]
    return feats

def _stage_b_featurizer(cand: dict, config: dict) -> tuple[bool, list[dict]]:
    """
    Filter B: canonical 6 特徴量の統計しきい値チェック。
    config["feature_thresholds"] は訓練時に pin した dict:
      {feat_id: {"kind": "3sigma"|"iqr_1.5", "mean"/"std" or "q1"/"q3"}}
    canonical 6 feature の欠落/未計算/未知 threshold key は hard reject（silent pass 禁止）。
    """
    composition = cand["generated_x"]
    thresholds = config["feature_thresholds"]
    feats = _featurize(composition)

    # canonical 6 feature ID の literal 集合（§8.4.2 で pin）
    canonical_ids: tuple[str, ...] = (
        "elem_mean_AtomicNumber", "elem_mean_Electronegativity",
        "elem_std_Electronegativity", "elem_mean_CovalentRadius",
        "elem_mean_MeltingT", "stoich_2norm",
    )                                                    # §8.4.2 pin と同順
    canonical_id_set = set(canonical_ids)
    missing_thr = canonical_id_set - set(thresholds.keys())
    extra_thr = set(thresholds.keys()) - canonical_id_set
    if missing_thr or extra_thr:
        raise ValueError(
            f"feature_thresholds config drift: missing={missing_thr}, extra={extra_thr}"
        )

    rule_hits: list[dict] = []
    all_passed = True
    for feat_id in canonical_ids:                       # canonical 順で必ず 6 全て評価
        thr = thresholds[feat_id]
        value = feats.get(feat_id)
        if value is None:
            # canonical feature の計算失敗は hard reject（silent pass しない）
            rule_hits.append({"rule_id": f"feature_{feat_id}", "passed": False,
                              "detail": {"status": "not_computed"}})
            all_passed = False
            continue
        if thr["kind"] == "3sigma":
            lo = thr["mean"] - 3 * thr["std"]
            hi = thr["mean"] + 3 * thr["std"]
        elif thr["kind"] == "iqr_1.5":
            iqr = thr["q3"] - thr["q1"]
            lo = thr["q1"] - 1.5 * iqr
            hi = thr["q3"] + 1.5 * iqr
        else:
            raise ValueError(f"unknown threshold kind: {thr['kind']}")
        passed = (lo <= value <= hi)
        rule_hits.append({
            "rule_id": f"feature_{feat_id}",
            "passed": passed,
            "detail": {"value": value, "lo": lo, "hi": hi, "kind": thr["kind"]},
        })
        if not passed:
            all_passed = False
    return all_passed, rule_hits
```

---

## 8.5 Filter C — 合成可能性 proxy（分子 SC/SA/RA、結晶 hull + ICSD）

### 8.5.1 5 手法 canonical set と soft score 契約

Filter C は **GO/NO-GO ではなく [0, 1] float の soft score**を付与する契約です（§8.2.1 で強調）。以下 5 手法（分子 3・結晶 2）を Ch8 canonical set とします：

| method_id | 対象 | 出力 | 実装依存 |
|---|---|---|---|
| `sc_score` | 分子 | Synthetic Complexity Score（低いほど合成容易） | Coley et al., 2018 の open impl（`scscore`） |
| `sa_score` | 分子 | Synthetic Accessibility Score（1-10、低いほど容易） | Ertl & Schuffenhauer, 2009（RDKit contrib） |
| `ra_score` | 分子 | Retrosynthesis Accessibility Score（0-1、高いほど容易） | Thakkar et al., 2021（open ML impl） |
| `hull_distance` | 結晶 | Materials Project 参照 hull からの energy 距離（meV/atom、低いほど熱力学的安定） | Pymatgen `PhaseDiagram` + MP API |
| `icsd_historical` | 結晶 | ICSD 類似組成の existence flag + 類似度スコア | 独自 lookup（MP API の `theoretical=false` フィルタ） |

> [!NOTE]
> **method_id は候補タイプに依存**：分子候補（SMILES / SELFIES 形式）には SC/SA/RA、結晶候補（Structure / Composition）には hull_distance / icsd_historical を使い分けます。本章の教育実装では **method は Skill 呼び出しごとに 1 つ指定**し（`hyperparameters.method_id`）、複数手法の重ね合わせは Ch11 のランキング Skill に委譲します。

### 8.5.2 soft score → [0, 1] 正規化 canonical

各 method の raw output を **[0, 1] float に正規化**する canonical 変換を Skill YAML `output_schema.synthesizability_proxy_score` に固定します。

| method_id | raw range | 正規化式 | 高スコア＝ |
|---|---|---|---|
| `sc_score` | [1, 5]（typical） | `1 - (raw - 1) / 4`、clip[0,1] | 合成容易 |
| `sa_score` | [1, 10] | `1 - (raw - 1) / 9`、clip[0,1] | 合成容易 |
| `ra_score` | [0, 1] | そのまま | 合成容易 |
| `hull_distance` | [0, +∞) meV/atom | `exp(-raw / 100)`、clip[0,1] | 熱力学的安定（近い） |
| `icsd_historical` | {0, 1} + similarity | `flag ? similarity : 0.0` | 前例類似 |

**コードスニペット 6 — soft score 正規化**

```python
import math

def normalize_synthesizability_score(raw: float, method_id: str) -> float:
    """
    Ch8 canonical: raw output を [0, 1] に正規化。高スコア = 合成容易 or 前例近い。
    """
    if method_id == "sc_score":
        return max(0.0, min(1.0, 1.0 - (raw - 1.0) / 4.0))
    elif method_id == "sa_score":
        return max(0.0, min(1.0, 1.0 - (raw - 1.0) / 9.0))
    elif method_id == "ra_score":
        return max(0.0, min(1.0, raw))
    elif method_id == "hull_distance":
        # raw は meV/atom、100 meV で e^-1 ≈ 0.37 に
        return max(0.0, min(1.0, math.exp(-max(0.0, raw) / 100.0)))
    elif method_id == "icsd_historical":
        # raw は {flag, similarity} の tuple を float 化した contract-side 実装
        # 本節では raw = flag * similarity として渡される前提
        return max(0.0, min(1.0, raw))
    else:
        raise ValueError(f"unknown method_id: {method_id}")

def _stage_c_synthesizability_score(cand: dict, config: dict) -> float:
    """
    Filter C: method_id に応じた proxy 計算 → [0, 1] 正規化。
    実装本番は各 method の library 呼び出しを持つ（本節では config["proxy_fn"] を注入）。
    """
    method_id = config["method_id"]
    proxy_fn = config["proxy_fn"]              # callable: cand -> raw float
    raw = proxy_fn(cand)
    return normalize_synthesizability_score(raw, method_id)
```

### 8.5.3 precision/recall の教育的注意

SC/SA/RA-Score はいずれも **retrosynthesis の heuristic proxy** であり、precision/recall は原論文で reported されているものの、ARIM の in-house 合成 route と乖離することがあります。hull distance も **DFT convex hull は 0 K の外挿**であり、finite-T 相安定性を保証しません。ICSD historical は **既知組成の類似度**であり、真に新規な組合せへの外挿は本質的に不可能です。

本章の Skill は **これら proxy を数値スコアとして返すが、Skill 契約側で "合成可能" と自然言語で発話することは Ch4 §4.7 禁止フレーズ list で明示禁止**です（§8.7 Skill YAML で literal 継承）。

---

## 8.6 `arim.gen.physics_filter.v0.1` Skill YAML 完全形

Ch4 §4.2 template と Ch5-7 の親骨格を継承した完全形です。

**YAML block 1 — `arim.gen.physics_filter.v0.1` full Skill 契約書**

```yaml
# =============================================================
# arim.gen.physics_filter.v0.1 — 物理制約フィルタ Skill（Ch8 canonical、F ステージ）
# 継承: Ch4 §4.2 template ⑧、Ch5 §5.7 / Ch6 §6.8 / Ch7 §7.7 と一貫
# 責務: Filter A（物理 rule-based）+ Filter B（matminer 特徴量 outlier）の 2 段 hard reject
#       Filter C（合成可能性 proxy）は arim.gen.synthesizability_proxy.v0.1 に委譲
# =============================================================
skill:
  id: "arim.gen.physics_filter.v0.1"
  version: "v0.1"
  description: "Ch5-7 G ステージ出力候補に対する Filter A（Pymatgen 物理 rule）と Filter B（matminer 特徴量 outlier）の 2 段 hard reject。Ch8 canonical 実装。"

pipeline_position:
  stage: "F"
  upstream_stages: ["SF"]                        # immediate: SF → F canonical（Ch4 §4.9）
  downstream_stages: ["H"]                       # immediate: F → H canonical

input_schema:
  candidates_in:
    type: "list_of_dict"
    required_keys: ["candidate_id", "generated_x"]
    optional_keys: ["structure"]                # 結晶候補のみ
    condition: "required"
  hyperparameters:
    type: "dict"
    schema:
      allowed_element_set:
        type: "list_of_str"
        description: "付録 C canonical allowed_element_set の literal 引用"
      max_atoms:
        type: "int"
        allowed_range: [1, 200]
        default: 200
      extension_rules_enabled:
        type: "list_of_str"
        allowed_values:
          - "lattice_stability_precheck"
          - "element_availability_check"
          - "formula_max_atoms"
        default: []
      feature_thresholds:
        type: "dict"
        description: "訓練時に pin した canonical 6 特徴量の統計しきい値。§8.4.2 参照"
        required_keys:
          - "elem_mean_AtomicNumber"
          - "elem_mean_Electronegativity"
          - "elem_std_Electronegativity"
          - "elem_mean_CovalentRadius"
          - "elem_mean_MeltingT"
          - "stoich_2norm"

output_schema:
  survivors:
    type: "list_of_dict"
    required_keys: ["candidate_id", "generated_x"]
    description: "Filter A + B を pass した候補"
  provenance_log:
    type: "list_of_dict"
    required_keys: ["candidate_id", "stage", "passed", "rules_evaluated"]
    optional_keys: ["preprocessing_log"]     # stage=="A" のとき present、`filter_rules_applied` 対象外
    description: |
      全候補の全 rule 判定 log（reject 候補も含む）。stage 別に fixed-slot invariant:
        stage=="A": len(rules_evaluated)==6
          rules_evaluated[0:3] == filter_rules_applied[0:3] (canonical 3 rules 順)
          rules_evaluated[3:6] == filter_rules_applied[3:6] (extension 3 slot、not_enabled は passed:null)
        stage=="B": len(rules_evaluated)==6
          rules_evaluated[0:6] == filter_rules_applied[6:12] (matminer canonical 6 features 順)
      preprocessing_log は stage=="A" 専用 optional field、filter_rules_applied には登録しない。
  counts:
    type: "dict"
    required_keys: ["n_input", "n_after_a", "n_after_b"]

# --- 生成器 field は本 Skill では null（screening Skill）---
generative_model_family: null
latent_dim: null
training_data_provenance: null
condition_variables_declared: []
condition_extensibility:
  allow_agent_to_add_conditions: false
  additional_conditions_require: "human_approval"
generation_temperature: null
generation_temperature_range_allowed: null
guidance_scale: null
guidance_scale_range_allowed: null
sampling_config: null

# --- screening_model_family（Ch4 §4.3.7 canonical 3-tier）---
screening_model_family:
  family: "classifier_chain"                   # rule + featurizer の chain として位置付け
  id: "arim.ch8.physics_filter_v0.1"
  version: "v0.1.0"

# --- filter_rules_applied（Ch4 §4.3.6 canonical list、順序が provenance）---
filter_rules_applied:
  # canonical 3 rules（必ず先頭 3 要素）
  - "charge_neutrality"
  - "stoichiometry_integer_multiplier"
  - "oxidation_state_bounds"
  # Ch8 拡張 rules（extension_rules_enabled で選択、この順で追記）
  - "lattice_stability_precheck"
  - "element_availability_check"
  - "formula_max_atoms"
  # Filter B（matminer 6 特徴量、この順で追記）
  - "feature_elem_mean_AtomicNumber"
  - "feature_elem_mean_Electronegativity"
  - "feature_elem_std_Electronegativity"
  - "feature_elem_mean_CovalentRadius"
  - "feature_elem_mean_MeltingT"
  - "feature_stoich_2norm"

# --- feature_thresholds_provenance（Ch8 canonical 拡張、付録 A で back-port 予定）---
feature_thresholds_provenance:
  # 訓練時 fit の pin。G ステージ Skill の training_data_provenance と 1:1 で紐付く
  fitted_on_dataset_id: "arim.synthetic-generative.compositions.v1"
  fitted_on_dataset_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  fitted_at: "2026-06-15T00:00:00Z"
  matminer_version: "0.9.2"
  pymatgen_version: "2024.10.3"

foundation_model_reference: null

# --- Ch11 委譲 ---
top_k_returned: null

# --- 合成試行承認への hand-off（Ch4 §4.4）---
inverse_design_authorization: "autonomous"     # F ステージは自律実行 canonical
synthesis_launch_authorization: null

# --- OOD detection への委譲（Ch5 §5.4 / Ch10 canonical）---
hallucinatory_composition_detection_delegated_to:
  skill: "arim.gen.ood_detection.v0.1"
  skill_version: "v0.1"
  invocation_stage: "H"

# --- Safety 3-layer hand-off 宣言（Ch4 §4.6 canonical）---
safety_screening_passed: false                 # 上流 SF ステージの結果を継承する契約（本 Skill は判定しない）
dual_use_review_completed: false
handoff_to:                                    # Ch4 §4.6 canonical hand-off
  layer_1:
    stage: "SF"
    skill: "arim.gen.safety_screening.v0.1"
    required_field: "safety_screening_passed"
    relation: "upstream"                       # SF は F の upstream
  layer_2:
    stage: "D"
    reviewer_role: "organizational_dual_use_committee"
    required_field: "dual_use_review_id"
  layer_3:
    stage: "P"
    reviewer_role: "facility_safety_officer"
    required_field: "facility_policy_id"

# --- authorization_gates（Ch4 §4.6 canonical 4-gate dict）---
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

# --- Ch4 §4.7 canonical 10 literal 日本語 phrase（Ch5-7 と一字一句同じ）---
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

# --- Ch8 独自: physics_filter 操作逸脱（Ch4 §4.7 の安全性原理を延伸）---
disallowed_operations:
  - name: "reorder_filter_rules_applied"
    description: "filter_rules_applied の list 順序変更禁止。canonical 3 rules は先頭 3 要素固定（§8.3.1）"
    enforcement: "hard_reject"
  - name: "skip_canonical_3_rules"
    description: "canonical 3 rules（charge_neutrality / stoichiometry_integer_multiplier / oxidation_state_bounds）の skip 禁止"
    enforcement: "hard_reject"
  - name: "modify_feature_thresholds_at_inference"
    description: "訓練時に pin した feature_thresholds の推論時変更禁止。feature_thresholds_provenance に固定"
    enforcement: "hard_reject"
  - name: "loosen_charge_neutrality_tolerance"
    description: "CHARGE_NEUTRALITY_TOL=1e-6 canonical の緩和禁止（§8.3.2）"
    enforcement: "hard_reject"
  - name: "extend_stoichiometry_max_denominator"
    description: "STOICHIOMETRY_MAX_DENOMINATOR=20 canonical の拡大禁止（§8.3.2）"
    enforcement: "hard_reject_and_audit_log"
  - name: "add_candidate_id_at_this_stage"
    description: "candidate_id は G ステージ Skill 発行のもの 1:1 保持。F ステージで新規発行禁止（§8.2.2）"
    enforcement: "hard_reject"
  - name: "return_natural_language_synthesis_recommendation"
    description: "outputs_disallowed_natural_language に反する自然言語 output 禁止（Ch4 §4.7 canonical）"
    enforcement: "hard_reject"

provenance:
  input_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  skill_version: "v0.1"
  run_datetime_utc: "2026-07-07T00:00:00Z"
  package_versions:
    pymatgen: "2024.10.3"
    matminer: "0.9.2"
    numpy: "2.0.2"
  random_seed: 20260707
  event_hash: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
```

> [!NOTE]
> **`arim.gen.physics_filter.v0.1` の観察点**（Ch5-7 G ステージ Skill との差分中心）：
> - `generative_model_family: null` — 本 Skill は screening 側、`screening_model_family` を canonical に持つ
> - `filter_rules_applied` の list は **canonical 3 rules を先頭 3 要素で固定**（§8.3.1）し、拡張 rule と Filter B の 6 特徴量チェックを canonical 順で並べる
> - `feature_thresholds_provenance` は Ch8 canonical 拡張（付録 A back-port 予定）で、G ステージ Skill の `training_data_provenance` と 1:1 で紐付く
> - `handoff_to.layer_1.relation: "upstream"` — 上流 SF ステージの `safety_screening_passed` を継承する（本 Skill は判定しない）
> - `disallowed_operations` の Ch8 固有項目は「rule 順序変更 / canonical rule skip / threshold 変更 / tolerance 緩和」を hard reject し、Ch4 §4.7 安全性原理を screening 実装層に延伸

---

## 8.7 `arim.gen.synthesizability_proxy.v0.1` Skill YAML 完全形

**YAML block 2 — `arim.gen.synthesizability_proxy.v0.1` full Skill 契約書**

```yaml
# =============================================================
# arim.gen.synthesizability_proxy.v0.1 — 合成可能性 proxy Skill（Ch8 canonical、S1 ステージ）
# 継承: Ch4 §4.2 template ⑧、Ch5 §5.7 / Ch6 §6.8 / Ch7 §7.7 と一貫
# 責務: Filter C（5 method canonical set）による [0, 1] soft score 付与
#       reject はせず、S1 → S2 → T の下流に score 付き候補を流す
# =============================================================
skill:
  id: "arim.gen.synthesizability_proxy.v0.1"
  version: "v0.1"
  description: "Ch8 canonical 合成可能性 proxy 5 手法（sc_score / sa_score / ra_score / hull_distance / icsd_historical）による soft score 付与。reject はせず、Ch11 ランキング入力を提供。"

pipeline_position:
  stage: "S1"
  upstream_stages: ["H"]                         # immediate: H → S1 canonical（Ch4 §4.9）
  downstream_stages: ["S2"]                      # immediate: S1 → S2 canonical

input_schema:
  candidates_in:
    type: "list_of_dict"
    required_keys:
      - "candidate_id"
      - "generated_x"
      # F → H → S1 canonical 前提条件を YAML 契約として enforce（Round 15 canonical）
      - "physics_filter_run_id"          # F ステージ arim.gen.physics_filter.v0.1 実行 ID
      - "physics_filter_passed"          # bool、true でなければ本 Skill は fail-fast
      - "ood_detection_run_id"           # H ステージ arim.gen.ood_detection.v0.1 実行 ID
      - "ood_detection_passed"           # bool、true でなければ本 Skill は fail-fast
      - "upstream_stage_trace"           # list[str]、["F", "H"] を含むこと必須
    optional_keys: ["structure", "smiles"]     # 結晶 / 分子候補で使い分け
    condition: "required"
    validation_rule: |
      すべての candidate_in について:
        assert physics_filter_run_id, "F stage run_id missing (flag alone insufficient)"
        assert physics_filter_passed is True, "F stage did not pass"
        assert ood_detection_run_id, "H stage run_id missing (flag alone insufficient)"
        assert ood_detection_passed is True, "H stage did not pass"
        assert set(upstream_stage_trace) >= {"F", "H"}, "F/H trace missing"
      違反時は fail-fast（silent proceed 禁止）
  hyperparameters:
    type: "dict"
    schema:
      method_id:
        type: "enum"
        values: ["sc_score", "sa_score", "ra_score", "hull_distance", "icsd_historical"]
      candidate_domain:
        type: "enum"
        values: ["molecule", "crystal"]
      hull_reference_source:
        type: "string"
        description: "hull_distance method 使用時、Materials Project 参照 hull の source id"
        condition: "required if method_id == hull_distance"
      icsd_similarity_threshold:
        type: "float"
        allowed_range: [0.5, 1.0]
        default: 0.8
        condition: "required if method_id == icsd_historical"
    validation_rule: |
      # method_id と candidate_domain の互換性 canonical（§8.5 表）
      if method_id in ["sc_score", "sa_score", "ra_score"]:
          assert candidate_domain == "molecule", \
              f"{method_id} は molecule 専用（SMILES 必須）"
          assert "smiles" in candidate_in, \
              f"{method_id} は candidate_in.smiles 必須"
      elif method_id in ["hull_distance", "icsd_historical"]:
          assert candidate_domain == "crystal", \
              f"{method_id} は crystal 専用（generated_x composition または structure 必須）"
          assert "generated_x" in candidate_in or "structure" in candidate_in, \
              f"{method_id} は generated_x または structure 必須"
      違反時は fail-fast（silent proceed 禁止）

output_schema:
  scored_candidates:
    type: "list_of_dict"
    required_keys: ["candidate_id", "generated_x", "synthesizability_proxy_score"]
    description: "全入力候補（reject なし）に [0, 1] float の soft score を付与"
  proxy_provenance:
    type: "dict"
    required_keys: ["method_id", "candidate_domain", "n_scored", "score_summary", "per_candidate_log"]

# --- 生成器 field は null ---
generative_model_family: null
latent_dim: null
training_data_provenance: null
condition_variables_declared: []
condition_extensibility:
  allow_agent_to_add_conditions: false
  additional_conditions_require: "human_approval"
generation_temperature: null
generation_temperature_range_allowed: null
guidance_scale: null
guidance_scale_range_allowed: null
sampling_config: null

# --- screening_model_family（Ch4 §4.3.7 canonical 3-tier のみ）---
screening_model_family:
  family: "classifier_chain"                   # method_id によっては ML model chain
  id: "arim.ch8.synthesizability_proxy_v0.1"
  version: "v0.1.0"
# hyperparameters は input_schema.hyperparameters と screening_config に分離（Ch4 §4.3.7 shape 遵守）

screening_config:                              # Ch8 canonical 拡張（付録 A back-port 予定）
  method_id: "hull_distance"                   # 呼び出しごとに上書き
  candidate_domain: "crystal"

# --- proxy_indicators（Ch8 canonical 拡張 dict shape、Ch2 §2.5 の list-of-dict と非互換：
#      Ch2 shape の canonical は `proxy_indicators_ch2: [ {name, range, interpretation}, ... ]`
#      として別 field に持たせるか、Ch2 shape へ back-port する（付録 A で決定）。
#      本 Skill 時点では method_id が呼び出しごとに 1 つに固定される契約のため dict shape とする）---
proxy_indicators:
  method_id: "hull_distance"
  raw_output_range: "[0, +inf)"
  raw_output_unit: "meV/atom"
  normalized_score_range: "[0, 1]"
  normalization_formula: "exp(-max(0, raw) / 100)"
  high_score_semantics: "thermodynamically_stable_or_near_hull"

# --- proxy 参照データ provenance（Ch8 canonical 拡張、付録 A で back-port 予定）---
proxy_reference_provenance:
  # method_id 別に参照 source を pin
  hull_distance:
    source: "materials_project"
    api_version: "2024.9"
    fetched_at: "2026-06-15T00:00:00Z"
    chemical_system: "Fe-Ti-O-Al"
    hull_snapshot_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  icsd_historical:
    source: "materials_project_theoretical_false"
    api_version: "2024.9"
    fetched_at: "2026-06-15T00:00:00Z"
    similarity_metric: "composition_cosine"
  sc_score:
    source: "coley_scscore_v1.0"
    reference_repo: "https://github.com/connorcoley/scscore"
    fetched_at: "2026-06-15T00:00:00Z"
  sa_score:
    source: "rdkit_contrib_sascorer"
    rdkit_version: "2024.03.5"
  ra_score:
    source: "thakkar_rascore_v1.0"
    reference_repo: "https://github.com/reymond-group/RAscore"
    fetched_at: "2026-06-15T00:00:00Z"

foundation_model_reference: null

# --- filter_rules_applied は null（本 Skill は reject しない）---
filter_rules_applied: []

# --- Ch11 委譲 ---
top_k_returned: null

# --- Ch2 §2.5 canonical: 合成 GO/NO-GO は Human 判断 ---
synthesis_decision_owner: "human"              # canonical 固定

# --- Ch4 §4.4 canonical ---
inverse_design_authorization: "autonomous"     # S1 ステージは自律実行
synthesis_launch_authorization: null

hallucinatory_composition_detection_delegated_to:
  skill: "arim.gen.ood_detection.v0.1"
  skill_version: "v0.1"
  invocation_stage: "H"

safety_screening_passed: false
dual_use_review_completed: false
handoff_to:
  layer_1:
    stage: "SF"
    skill: "arim.gen.safety_screening.v0.1"
    required_field: "safety_screening_passed"
    relation: "upstream"
  layer_2:
    stage: "D"
    reviewer_role: "organizational_dual_use_committee"
    required_field: "dual_use_review_id"
  layer_3:
    stage: "P"
    reviewer_role: "facility_safety_officer"
    required_field: "facility_policy_id"

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

# --- Ch8 独自: synthesizability_proxy 操作逸脱 ---
disallowed_operations:
  - name: "reject_candidate_at_filter_c"
    description: "Filter C は soft score のみ、reject 禁止（§8.2.1）。閾値による reject は Ch11 に委譲"
    enforcement: "hard_reject"
  - name: "invoke_without_verified_f_h_pass_provenance"
    description: "F/H stage pass 未検証の候補で本 Skill を呼び出すこと禁止（input_schema.validation_rule 参照）。physics_filter_run_id / physics_filter_passed / ood_detection_run_id / ood_detection_passed のいずれかが欠落、または upstream_stage_trace が {F,H} を含まない候補は fail-fast（flag だけでなく実 run_id 必須）"
    enforcement: "hard_reject_and_audit_log"
  - name: "sum_multiple_proxy_scores_at_this_skill"
    description: "複数 method の重ね合わせは Ch11 のランキング Skill に委譲。本 Skill 内での合成禁止"
    enforcement: "hard_reject"
  - name: "override_synthesis_decision_owner"
    description: "synthesis_decision_owner: human canonical の変更禁止（Ch2 §2.5）"
    enforcement: "hard_reject"
  - name: "modify_normalization_formula_at_inference"
    description: "§8.5.2 canonical 正規化式（method_id 別）の推論時変更禁止"
    enforcement: "hard_reject"
  - name: "swap_hull_reference_source_at_inference"
    description: "訓練時に pin した hull_reference_source（proxy_reference_provenance）の変更禁止"
    enforcement: "hard_reject"
  - name: "return_natural_language_synthesis_recommendation"
    description: "outputs_disallowed_natural_language に反する自然言語 output 禁止（Ch4 §4.7 canonical）"
    enforcement: "hard_reject"
  - name: "loosen_icsd_similarity_threshold_below_default"
    description: "icsd_similarity_threshold: 0.8 canonical default の推論時緩和禁止（Skill 呼び出し時の明示指定のみ許可、agent の autonomous 緩和は禁止）"
    enforcement: "hard_reject_and_audit_log"
  - name: "invert_high_score_semantics"
    description: "high_score = 合成容易 / 前例近い の canonical 意味反転禁止（§8.5.2）"
    enforcement: "hard_reject"

provenance:
  input_sha256: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  skill_version: "v0.1"
  run_datetime_utc: "2026-07-07T00:00:00Z"
  package_versions:
    pymatgen: "2024.10.3"
    rdkit: "2024.03.5"
    numpy: "2.0.2"
  random_seed: 20260707
  event_hash: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
```

> [!NOTE]
> **`arim.gen.synthesizability_proxy.v0.1` の観察点**：
> - `pipeline_position.stage: "S1"` — F/H の下流、S2/T の上流
> - `filter_rules_applied: []` — reject しない Skill、Ch4 §4.3.6 の list は空
> - `synthesis_decision_owner: "human"` — Ch2 §2.5 canonical、agent による autonomous な GO 判断禁止
> - `proxy_indicators` は Ch8 canonical 拡張 dict shape（Ch2 §2.5 の list-of-dict とは非互換：付録 A で back-port 方針を決定）で、`method_id` 別に切り替える動的 field
> - `proxy_reference_provenance` は Ch8 canonical 拡張（付録 A back-port 予定）で、method_id 別に参照 source を pin
> - `disallowed_operations` の Ch8 固有項目は「Filter C reject 禁止 / score 合成禁止 / owner override 禁止」を hard reject し、Ch11 ランキング Skill との責務分離を守る

---

## 8.8 H ステージ OOD detection と Ch8 physics/synth の責務分離マップ

`arim.gen.ood_detection.v0.1`（H ステージ、Ch5 §5.4 で config/result 態を導入、Ch10 で完全計算）と Ch8 の physics_filter / synthesizability_proxy を混同すると、Ch14 の重大な失敗パターン「OOD で reject されたから物理チェック skip」「physics 通ったから OOD 検知不要」に繋がります。以下のマトリクスで責務を明示分離します。

### 8.8.1 責務マトリクス

| 軸 | H ステージ `arim.gen.ood_detection.v0.1` | Ch8 §8.3-4 physics_filter（F ステージ） | Ch8 §8.5 synthesizability_proxy（S1 ステージ） |
|---|---|---|---|
| **目的** | 生成候補が学習分布内か（統計軸 z_M/z_R）＋ 物理不整合スコア軸（v）の canonical 合成 | 生成候補が物理法則に整合するか（決定論的整合性、H の v 軸 provenance を owns） | 生成候補が合成前例に近いか（heuristic 合成容易性） |
| **判定基準** | composite score `0.4*z_M + 0.4*z_R + 0.2*v`（Ch5 §5.4 / Ch10 canonical、v は F provenance の rule violation 正規化） | Pymatgen rule + matminer 1D 統計しきい値 | SC/SA/RA score / hull distance / ICSD 類似度 |
| **入力** | 候補の特徴量ベクトル / 生成器 latent + F ステージ provenance（v 軸算出用） | 候補の組成 dict（+ 結晶なら Structure） | 候補の組成 or SMILES |
| **出力** | scored_candidates (全候補、reject なし) に `ood_score: float ([0,1])` + `ood_flag: bool` + `positivity_violation: bool` を top-level 付与（内部 `hallucinatory_composition_detection` に composite_score / z_M / z_R / v 保持） | survivors + rejected + rule log（H に provenance を渡す） | scored（reject なし） |
| **Reject 判定** | reject 禁止（Ch11 委譲、Ch10 §10.7 pass-through canonical、`ood_score`/`ood_flag`/`positivity_violation` 付与のみ、top-k は Ch11 が **`ood_score` を第 3 軸として** Pareto で判定） | hard reject（Filter A/B） | reject なし（soft score のみ） |
| **依存 library** | scikit-learn (Mahalanobis) + numpy (PCA) + F provenance parser | Pymatgen + matminer | Pymatgen + RDKit + MP API |
| **pipeline stage** | H（Ch2 §2.7、F 直後） | F（Ch2 §2.7、SF 直後） | S1（Ch2 §2.7、H 直後） |
| **canonical Skill ID** | `arim.gen.ood_detection.v0.1` | `arim.gen.physics_filter.v0.1` | `arim.gen.synthesizability_proxy.v0.1` |
| **同時使用の必然性** | 必須（F を skip すると v 軸が算出不能で canonical composite 契約違反） | 必須（H の v 軸入力を produce する upstream） | 必須（S1 併走で `arim.gen.dft_proxy.v0.1` も走る、Ch9） |

### 8.8.2 責務分離が守られなかった場合の failure mode 予告

| failure mode | 誤解 | 該当 Ch14 taxonomy 予告 |
|---|---|---|
| **H で reject されたから physics_filter skip** | H の flag と F の rule reject を同一視 | Ch14 §14.2「H ステージ結果で F ステージを skip」 |
| **physics 通ったから H stage 検知不要** | 物理整合 = 学習分布内という誤解 | Ch14 §14.2「F ステージ結果で H ステージを skip」 |
| **synth score 高いから合成 GO** | soft score を GO/NO-GO と同一視 | Ch14 §14.2「S1 score で A ステージ Human 承認 bypass」 |
| **synth score 低いから候補を drop** | S1 の soft score を hard reject と同一視（Ch11 T ステージの ranking 責務侵食） | Ch14 §14.2「S1 で reject を実装」 |
| **H で Filter B 相当を再実装** | 1D per-feature outlier reject（Filter B の責務）を H の Mahalanobis / PCA 多次元 OOD 判定と混同し、H 側で feature-level しきい値 reject を追加 | Ch14 §14.2「Filter B と H ステージ OOD detection の責務混同」 |
| **synth Skill 内で複数 method 合成** | Ch11 ランキング Skill の責務侵食 | Ch14 §14.2「S1 で ranking を実装」 |

> [!IMPORTANT]
> **F/H は hard gate、S1 は mandatory execute + soft score**：本章 §8.2.1 の非対称性を pipeline レベルでも守ります。
> - **F ステージ（physics_filter）**：hard gate。Filter A/B いずれか fail した候補は **A ステージに到達しない**（下流に流さない）
> - **H ステージ（ood_detection）**：hard gate。flag=true の候補は **A ステージに到達しない**
> - **S1 ステージ（synthesizability_proxy + Ch9 dft_proxy）**：mandatory execute。全候補に score を付与するが reject はしない。低スコア候補も Ch11 T ステージのランキング入力に流し、**top-k 選抜と Human 承認は Ch11 と A ステージが所有**する
>
> 「synth score が低いから S1 で候補を drop」は Ch14 §14.2 taxonomy 対象（Ch11 T ステージ ranking 責務の侵食）。「他 2 軸（F/H）で通ったから S1 skip」も同様に禁止（S1 は execute 自体は mandatory）。

---

## 8.9 よくある失敗と対処

### 8.9.1 Filter A の `charge_neutrality` false negative（Ch14 §14.2 で正式に taxonomize）

**症状**：組成 `{Fe: 0.5, O: 0.5}` を rule_charge_neutrality に直接渡すと **pymatgen が `ValueError: Charge balance analysis requires integer values in Composition!` を raise**、または（実装 version によっては）`guesses = []` で false reject される。

**原因**：Pymatgen `Composition.oxi_state_guesses(target_charge=0)` は **入力組成が整数値であることを前提**とする実装。`{Fe: 0.5, O: 0.5}` のような fractional 表現（生成器出力の mole fraction）を直接 pymatgen Composition に渡すと、pymatgen 側で整数化・GCD 還元が保証されないため上記 exception または空 list を返す。これは Pymatgen bug ではなく仕様。

**対処**：本章 canonical 実装（§8.3.2 `canonicalize_composition_to_integer_ratio` と §8.3.4 `_stage_a_physics`）が **canonical rules の前段で `Fraction.limit_denominator(20) → LCM 共通分母整数化 → GCD 最小整数比還元` の 3 step を必ず走らせる**設計です。`{Fe: 0.5, O: 0.5}` は `{Fe: 1, O: 1}`（整数 dict）に還元されてから pymatgen に渡されます。この preprocessing は `filter_rules_applied` list に登録される canonical rule ではなく、Filter A 内部の shared canonical preprocessing として provenance の `preprocessing_log` field に記録されます。エージェントがこの preprocessing を skip または tolerance を緩めることは §8.6 `disallowed_operations.loosen_charge_neutrality_tolerance` で hard reject 契約済み。

### 8.9.2 Filter B の feature_thresholds drift（Ch14 §14.1 で正式に taxonomize）

**症状**：新しい G ステージ Skill（別 training_data で fit）を接続したところ、Filter B ですべての候補が outlier reject される。

**原因**：`feature_thresholds` は Ch8 canonical で **G ステージ Skill の training_data_provenance と 1:1 pin** される契約です。新 training_data で fit した G ステージ Skill を接続する場合、Filter B の thresholds も **同じ training_data で再 fit した値**を pin し直す必要があります。`feature_thresholds_provenance.fitted_on_dataset_id` が G ステージ Skill の `training_data_provenance.dataset_id` と不一致な場合、Ch10 の provenance ゲートで catch されます。

**対処**：G ステージ Skill を差し替える時は、必ず Ch8 Skill の `feature_thresholds` と `feature_thresholds_provenance` を同時更新する。Skill YAML の CI に 1:1 pin 検査を組む。

### 8.9.3 Filter C の proxy score 単一値信頼

**症状**：`hull_distance` proxy で score = 0.9 の候補を「熱力学的に非常に安定 → 合成 GO」と誤解釈。

**原因**：DFT convex hull は 0 K の外挿であり、finite-T 相安定性を保証しない。§8.5.3 で議論した通り、proxy はいずれも heuristic であり単一値で GO 判定は不可能。

**対処**：Skill 契約側は自然言語で「合成容易」と発話しない（`outputs_disallowed_natural_language` に literal 追加、§8.7 で pin）。Ch11 のランキング Skill で **複数 proxy method の重ね合わせ**と Ch9 の DFT proxy による物性予測を **多目的 Pareto 選抜**として扱う。

### 8.9.4 F/H/S1 の相互 skip（hard gate と mandatory execute の混同）

**症状 A**：エージェントが「H で通ったから physics_filter は不要」と autonomous に F ステージを skip、または逆パターン。
**症状 B**：エージェントが「S1 の synth score が低いから候補を drop」して Ch11 T ステージに流さない。

**原因**：Ch14 §14.2 の失敗パターン。**F/H は hard gate、S1 は mandatory execute + soft score** という非対称性（§8.8 IMPORTANT block）の理解不足。

**対処**：pipeline 実装側で以下 canonical を固定：
- **F と H は hard gate**：いずれか fail した候補を A ステージに流さない（skip も禁止）
- **S1 は mandatory execute**：全 F/H pass 候補に対して score 付与（execute 自体を skip 禁止）、reject は禁止（Ch11 に委譲）

Skill 呼び出し側の autonomous な bypass および S1 での reject は Ch14 §14.2 taxonomy で catch。

### 8.9.5 `synthesis_decision_owner: human` の autonomous 変更

**症状**：エージェントが `synthesis_decision_owner` を `agent` に書き換え、A ステージを skip して合成試行を推薦。

**原因**：Ch2 §2.5 canonical の直接違反。Skill YAML の `disallowed_operations.override_synthesis_decision_owner` で hard reject 契約済み。

**対処**：Skill 呼び出し時の provenance 検証で `synthesis_decision_owner == "human"` を必須にする。違反 event は Ch14 §14.2 taxonomy に登録。

### 8.9.6 filter_rules_applied list の順序破壊

**症状**：エージェントが `filter_rules_applied` list の順序を「効率化」の名目で並べ替え、canonical 3 rules を末尾に。

**原因**：Ch4 §4.3.6 canonical で **list の順序が provenance として意味を持つ**規約の理解不足。

**対処**：Skill YAML の `disallowed_operations.reorder_filter_rules_applied` で hard reject 契約済み。canonical 3 rules は必ず先頭 3 要素、拡張 rule は 4-6 要素、Filter B の 6 特徴量は 7-12 要素の順序を CI で pin 検査。

---

## 8.10 章末チェックリスト

本章の到達目標を、以下 9 項目で自己検証してください。**7 項目以上「はい」で第9章へ進めます**。

- [ ] **§8.1 の 3 動機**（分布内 ≠ 物理的整合 / 分布内 ≠ 合成可能 / hand-off point）を、Ch5-7 G ステージ Skill の限界と対応付けて説明できる
- [ ] **§8.2 の Filter A → B → C canonical 順序**を、A/B が hard reject / C が soft score という非対称性で書ける
- [ ] **§8.3 の canonical 3 rules と拡張 3 rules** を Pymatgen で実装でき、`filter_rules_applied` list の先頭 3 要素固定 canonical を守れる
- [ ] **§8.4 の matminer canonical 6 特徴量セット** と統計しきい値の canonical（3σ / IQR×1.5）を、G ステージ Skill の training_data_provenance と 1:1 で pin できる
- [ ] **§8.5 の 5 method canonical set**（sc_score / sa_score / ra_score / hull_distance / icsd_historical）と `[0, 1]` 正規化式を書ける
- [ ] **§8.6 の `arim.gen.physics_filter.v0.1`** Skill YAML を、Ch4 §4.6 Layer 1/2/3 hand-off 宣言と Ch5-7 の親骨格を継承して書ける
- [ ] **§8.7 の `arim.gen.synthesizability_proxy.v0.1`** Skill YAML を、`synthesis_decision_owner: human` canonical と Ch11 委譲宣言を含めて書ける
- [ ] **§8.8 の責務分離マップ**（OOD / physics / synth の 3 軸独立）を説明でき、5 つの failure mode 予告を Ch14 taxonomy 予定として認識できる
- [ ] **§8.9 の 6 failure modes** に対する Skill 契約側の防御策（`disallowed_operations` field）を説明できる

---

## 8.11 参考資料

### 本書内クロスリファレンス

- **Ch2 §2.5**：`proxy_indicators` canonical、`synthesis_decision_owner: human` 起源
- **Ch2 §2.6**：Safety 3-layer 起源
- **Ch2 §2.7**：canonical pipeline（F / S1 ステージの位置）
- **Ch4 §4.3.6**：`filter_rules_applied` canonical と Ch8 の 3 canonical rules
- **Ch4 §4.3.7**：`screening_model_family` canonical 3-tier
- **Ch4 §4.4**：`synthesis_launch_authorization` A ステージ
- **Ch4 §4.6**：Safety 3-layer 完全契約（Layer 1/2/3 hand-off shape）
- **Ch4 §4.7**：禁止フレーズ 10 list（本章 Skill で literal 継承）
- **Ch4 §4.9**：Skill ID 一覧（`arim.gen.physics_filter.v0.1` = Ch8, `arim.gen.synthesizability_proxy.v0.1` = Ch8）
- **Ch5 §5.4**：`hallucinatory_composition_detection` config 態、`arim.gen.ood_detection.v0.1` の H ステージ Skill 契約導入
- **Ch5 §5.7 / Ch6 §6.8 / Ch7 §7.7 / §7.8**：Skill YAML 親骨格（本章 §8.6 / §8.7 の literal 継承元）
- **Ch7 §7.9**：VAE / Diffusion / Flow / AR の 4 家族選択判断表（本章の入力候補は 4 家族いずれかの `candidates_out`）

### 次章への接続

- **Ch9**：DFT proxy（MEGNet / M3GNet / MACE）による物性予測。S1 ステージで本章 `arim.gen.synthesizability_proxy.v0.1` と併走する `arim.gen.dft_proxy.v0.1` を実装
- **Ch10**：`arim.gen.ood_detection.v0.1` の H ステージ完全実装。§8.8 の責務分離マップの H ステージ側を完成
- **Ch11**：`arim.gen.candidate_ranking.v0.1` T ステージ。本章 Filter C の soft score と Ch9 DFT proxy の物性予測を多目的 Pareto で top-k 選抜

### vol-01〜05 の該当章（本章の前提）

- **vol-03 第2章 / 第3章**：Pymatgen / matminer の基本 API と provenance
- **vol-03 第8-9章**：予測不確かさ管理（本章 Filter B の統計しきい値と接続）
- **vol-04 第4章**：`authorization_gates` shape（vol-06 §4.8 で並存 canonical 化）
- **vol-05 第4-5章**：GP surrogate 起源（Ch11 で本章 S1 出力を Pareto 選抜）

### 外部参考

- **Pymatgen**：https://pymatgen.org —— `Composition.oxi_state_guesses`, `Element.common_oxidation_states`, `SpacegroupAnalyzer`
- **matminer**：https://hackingmaterials.lbl.gov/matminer/ —— `ElementProperty`, `Stoichiometry` featurizers
- **RDKit**：https://www.rdkit.org —— 分子 featurization, SA-Score reference implementation (Contrib 部分)
- **SC-Score**：Coley et al., 2018, "SCScore: Synthetic Complexity Learned from a Reaction Corpus", J. Chem. Inf. Model. —— https://github.com/connorcoley/scscore
- **SA-Score**：Ertl & Schuffenhauer, 2009, "Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions", J. Cheminform.
- **RA-Score**：Thakkar et al., 2021, "Retrosynthetic accessibility score (RAscore)", Chem. Sci. —— https://github.com/reymond-group/RAscore
- **Materials Project API**：https://api.materialsproject.org —— `PhaseDiagram`, hull distance, ICSD historical existence
- **ICSD**：https://icsd.products.fiz-karlsruhe.de —— 結晶構造データベース（Materials Project の `theoretical=false` フィルタで参照）

---
