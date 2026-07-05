# 「AI エージェント時代の生成モデル・材料逆設計入門 — ARIM データで動く Agentic Skill」章構成（v0.1 ドラフト）

> vol-06 は vol-01「AI エージェント時代のデータ分析入門」、vol-02「統計・機械学習分析入門」、vol-03「深層学習分析入門」、vol-04「因果推論・実験計画入門」、vol-05「ベイズ最適化・逐次実験計画入門」の続編。
> vol-01〜05 が **「観測 → 予測 → 因果 → 逐次探索」** ラインを整えたのに対し、
> vol-06 は **「潜在空間で仮想の材料を生成し、望みの物性を持つ候補を提案する」** ラインを Skill 化する——
> ARIM データポータルの実験データ（結晶構造・組成・微細構造画像）を主戦場に、
> **「VAE / Diffusion で候補材料を生成し、Skill で妥当性フィルタを通し、Human-in-the-loop で合成試行を承認する」**
> ループをエージェントの Skill として設計する。

## 版履歴

### v0.1（初版企画ドラフト）
- vol-03 章構成 v0.2 で「vol-05 以降候補」として棚上げされていた **生成モデル** と **材料逆設計** を本巻に統合
- **統合の理由**：ARIM 実データからの生成モデル（vol-03 の材料 Foundation Model と接続）と、生成結果からの逆設計（vol-04 の因果構造で妥当性を判定 + vol-05 の BO で最適化候補を絞り込み）は、Agentic Skill として一体で運用される
- 6 か月・250〜300 ページ規模、本編 15 章 + 付録 3 の骨格を継承
- **エージェントに何を許すか**：潜在空間からの生成・妥当性スクリーニングは自律、**合成試行提案は準自律**、**実験合成の実行判断は必ず Human 承認**（材料合成は不可逆コストが高く、危険物質を生成し得る）

## 前提

- **対象読者**: ARIM データポータル会員のデータ分析者（材料・ナノテク研究者）。Python / Jupyter 経験あり。**vol-01 + vol-02 + vol-03 完読を推奨**（生成モデルの骨格は深層学習に依存）、**vol-04 と vol-05 のいずれか完読を推奨**（逆設計の判定に因果 or BO を使う）
- **vol-01 / vol-02 / vol-03 / vol-04 / vol-05 との関係**:
  - **vol-01 完読推奨**: Skill 設計 6 要素、データ契約、Human-in-the-loop、provenance、6 データ型
  - **vol-02 完読推奨**: 第11章の階層モデル（生成モデルの事後解析）、第9-12章の PyMC
  - **vol-03 完読推奨**: 第7章の転移学習、第9章の不確かさ、第11章の材料 Foundation Model、**第12章の自己教師あり学習**（生成モデルとの共通哲学）、第14章の Agentic 失敗パターン
  - **vol-04 推奨**: 因果構造を生成候補の妥当性判定に使う（第9章の refutation テストを "生成候補フィルタ" に転用）
  - **vol-05 推奨**: 生成候補を BO の初期点として使う（第13章の capstone で組み合わせ）
  - vol-06 の provenance は vol-03 拡張に、**生成 × 逆設計 × Agentic 特有フィールド**を追加：`generative_model_family`（VAE / Diffusion / GAN / normalizing flow / autoregressive）, `latent_dim`, `training_data_provenance`, `condition_variables_declared`, `generation_temperature`, `filter_rules_applied`（物理制約・安定性・合成可能性）, `screening_model_family`（DFT proxy / classifier chain）, `top_k_returned`, `inverse_design_authorization`, **`synthesis_launch_authorization`**, **`hallucinatory_composition_detection`**（学習分布外・非物理的組成の検知）, **`safety_screening_passed`**（毒性・爆発性等のフラグ）
- **最終ゴール（合格ライン）**:
  - **Pillar 1**: **表形式・組成データからの逆設計 Skill**（VAE または Diffusion）を 1 つ以上作れる。生成候補が **物理制約（電荷中性、化学量論、酸化数）を満たすフィルタ**を通り、**エージェントが top-k 候補を提案 → Human 承認**の provenance が残る
  - **Pillar 2**: **画像 / 結晶構造の生成 Skill**（Diffusion または VAE の材料応用）を 1 つ以上作れる。**分布外・非物理的構造の検知**が Skill に組み込まれている
  - **Advanced Capstone**: 上記 2 本を統合した **「Foundation Model 潜在空間で候補生成 → 物理制約フィルタ → 因果的妥当性判定 → BO でランキング → Human 承認」の複合 Skill** を、ARIM 風合成材料データで完成させる
- **分量目安**: 実践書（250〜300 ページ、vol-05 と同等）
- **期限**: 6 か月
- **ハンズオン標準環境**（vol-05 に追加）:
  - vol-03 標準 + `diffusers`（Hugging Face の Diffusion モデル）
  - `pytorch-lightning`（生成モデルの学習管理）
  - `pymatgen`（結晶構造の生成・検証、物理制約チェック）
  - `ase`（原子シミュレーション）
  - `rdkit`（分子構造の生成・検証）
  - `matminer`（組成 → 物性特徴量、生成候補のスクリーニング）
  - `dgl`, `torch-geometric`（グラフ生成モデル）
  - **CI 環境は CPU で完結**を必須要件（vol-03〜05 と同じ）
- **データセット方針**:
  - **主軸**: **ARIM 風合成材料データ**（真の物性関数を持つ、合成可能性フラグ付きの合成材料データ）を本書でリポジトリ配布（`data/synthetic-generative/` 想定）
    - 組成データ：3 元系・4 元系の合成組成 × 真の物性 × 合成可能性フラグ
    - 結晶構造：格子定数と原子位置を持つ小規模結晶構造
    - 微細構造画像：ARIM 風の合成 SEM/TEM 画像
  - **対比・スケール確認**: 公開データセット
    - **Materials Project**（結晶構造 + 物性、逆設計の教科書ベンチマーク）
    - **OQMD**（Open Quantum Materials Database）
    - **JARVIS-DFT**（NIST の材料 DFT データベース）
    - **AFLOW**（結晶構造 + 熱力学的安定性）
    - **QM9 / OQMD**（分子・小結晶）
  - **既存 Foundation Model 活用**: **CDVAE**（結晶構造 VAE）, **DiffCSP**（結晶構造 Diffusion）, **MatterGen**（結晶構造 Diffusion）等を Hugging Face Hub 経由で **既存重み利用**の立場（スクラッチ事前学習は本書スコープ外）
- **参照**:
  - PyTorch Lightning: https://lightning.ai/
  - Diffusers (Hugging Face): https://huggingface.co/docs/diffusers/
  - Pymatgen: https://pymatgen.org/
  - ASE: https://wiki.fysik.dtu.dk/ase/
  - RDKit: https://www.rdkit.org/
  - matminer: https://hackingmaterials.lbl.gov/matminer/
  - CDVAE: https://github.com/txie-93/cdvae (Xie et al. 2022)
  - DiffCSP: https://github.com/jiaor17/DiffCSP (Jiao et al. 2023)
  - MatterGen: https://github.com/microsoft/mattergen (Zeni et al. 2023)
  - Materials Project: https://materialsproject.org/
  - OQMD: https://oqmd.org/
  - JARVIS-DFT: https://jarvis.nist.gov/
  - Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
  - Kingma & Welling "Auto-Encoding Variational Bayes" (ICLR 2014)
  - vol-01 〜 vol-05 リポジトリ（本書の前提）

## 本書で扱わないこと（明示）

vol-06 は **「エージェントが材料候補を生成し、Human が合成試行を承認する Skill」** に scope を絞る。

| トピック | 扱わない理由 | 想定巻 |
|---|---|---|
| **生成モデルのスクラッチ事前学習** | 材料 Foundation Model（CDVAE / DiffCSP / MatterGen 等）は既存重み利用の立場。スクラッチ事前学習は研究室規模を超える | — |
| **強化学習・自律実験ロボティクス** | 「エージェントが合成装置を直接叩く」は本書のスコープ外。合成実行は Human 承認必須 | 別書候補 |
| **理論計算（DFT / MD シミュレーション）の並行実行** | 生成候補のスクリーニングに DFT proxy モデルは使うが、DFT 実行そのものは扱わない | — |
| **生成 AI の LLM 経由での "材料アイデア生成"** | LLM ハルシネーションの妥当性判定が本書の因果的判定と対立するため、本書は数値モデルの生成に集中 | 別書候補 |
| **大規模分散学習・多 GPU での生成モデル学習** | 単一 GPU / 単一ノード規模での fine-tune に絞る（vol-03 と同じ方針） | 別書候補 |
| **生成モデルの数学的完全展開（ELBO の変分展開、SDE 定式化等）** | 既存教科書に譲る。本書は Skill 化と provenance に集中 | — |

## 6 データ型と生成 × 逆設計 Skill の対応

| データ型 | vol-05 で扱えたこと | vol-06 で扱う Agentic Skill |
|---|---|---|
| スペクトル型 | 逐次探索での特徴最適化 | **目標スペクトルからの合成条件逆推定 Skill**、**cVAE で "スペクトル → 組成候補"** |
| クロマトグラム・時系列型 | 反応条件の逐次最適化 | **目標収率プロファイルからの反応条件逆推定** |
| 画像・顕微鏡型 | 粒径・欠陥密度の最適化 | **目標微細構造からの合成条件逆推定 Skill**、**Diffusion で微細構造生成 → 妥当性判定** |
| 回折・散乱パターン型 | 結晶相の最適化 | **目標回折パターンからの結晶構造候補生成 Skill**（CDVAE / DiffCSP 活用） |
| 表形式・プロセス条件型 | 単目的・多目的 BO | **組成 → 物性の逆設計 Skill**（本巻の主戦場）、**cVAE / cDiffusion / autoregressive** の全パターン |
| マルチモーダル統合型 | 多モーダル制約付き BO | **多モーダル条件付き生成**（例：組成 + 目標構造 → プロセス条件） |

**共通する Agentic 観点**：Skill は「候補を生成する」だけでなく、**「候補が物理制約を満たすか、学習分布内か、安全か、実際に合成可能か」を段階的にフィルタし、Human に上位 k 候補と根拠を提示する契約**。合成実行の GO/NO-GO は常に Human。

## 章構成（案）

**分量目安の合計**：本編 250〜300 ページ + 付録 40〜50 ページ

### 第0章 vol-01〜05 の最小復習（目安 15 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第0章 | vol-01〜05 の最小復習 | Skill / provenance / 6 データ型 / 深層 Agentic 権限 / **vol-04 の因果的判定と vol-05 の BO** を 15 ページに圧縮 | vol-06 の前提 |

### 第I部　なぜ「エージェント × 逆設計」なのか（目安 30〜35 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第1章 | 予測 → 因果 → 逐次 → 逆設計 のラダーで vol-01〜05 に何が足りないか | vol-05 の BO は既知の search space を探索するが、逆設計は **新規候補を生成** する。生成モデルの位置づけ、Agentic 特有の hallucination リスク、扱わないことの明示 | 本書の到達点、逆設計の位置付け |
| 第2章 | ARIM データで生成モデルを回すときの Agentic 特有の課題 | **少データ現場での生成の overfitting・mode collapse**、**非物理的候補の生成（電荷不整合・化学量論違反）**、**学習分布外の候補を "自信あり" と誤報告するリスク**、**"合成可能性" の定量化の困難さ**、**エージェントが安全性チェックをスキップする危険**、演習用データ紹介 | 自分のデータの生成モデル適用性判定、演習データ入手 |
| 第3章 | 生成モデルライブラリ地図 — Agentic 使い分け | **Diffusers / PyTorch Lightning / Pymatgen / RDKit / matminer** の位置づけ、**CDVAE / DiffCSP / MatterGen 等の材料 Foundation Model**、**エージェントがどこまで自律的に叩けるか**、vol-03 材料 FM との連携 | ライブラリ使い分けマップ |

### 第II部　生成モデルの Skill 化（目安 60〜70 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第4章 | 生成 × Agentic Skill の設計原則 | vol-03/04/05 第4章の生成 × Agentic 拡張。**生成の provenance**（`generative_model_family`, `latent_dim`, `condition_variables`, `generation_temperature`）、**新設節：合成試行承認ゲート**（`synthesis_launch_authorization`）——**エージェントは候補提示までで、合成実行は Human 承認**。**hallucination 対策**（`hallucinatory_composition_detection`）、**安全性スクリーニング**（`safety_screening_passed`） | 生成 × Agentic Skill 仕様書テンプレート |
| 第5章 | VAE を Skill 化する — 潜在空間で組成候補を生成 | VAE の骨格（encoder / decoder / ELBO）、**組成 VAE**（例：3-4 元系合成データで実装）、**cVAE（条件付き）**、**電荷中性・化学量論の物理制約フィルタ**、**Skill としての "分布外検知" 契約** | 組成 VAE Skill |
| 第6章 | Diffusion Model を Skill 化する — 材料生成の主戦場 | DDPM / DDIM の骨格、**組成 Diffusion**、**結晶構造 Diffusion**（DiffCSP / MatterGen 経由）、**条件付き生成**（classifier-free guidance）、**Skill としての temperature 管理**（"サンプリング温度を勝手に上げない" 契約） | Diffusion Skill、条件付き生成 Skill |
| 第7章 | Normalizing Flow と autoregressive を Skill 化する | Normalizing Flow の骨格、**組成 flow**、**autoregressive（SMILES-like な組成列生成）**、**VAE / Diffusion / Flow / AR の判断表**（データ量 × 制約表現 × 計算コスト） | Flow / AR Skill、生成手法選択判断表 |

### 第III部　スクリーニングと逆設計の Skill 化（目安 55〜65 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第8章 | 物理制約フィルタと合成可能性スクリーニングを Skill 化する | Pymatgen による物理制約チェック（電荷中性、酸化数、化学量論、格子安定性）、matminer による組成 → 物性特徴量、**"合成可能性" の proxy（historical existence / DFT stability / literature mining）**、**Skill としての多段フィルタ契約**（Filter A → Filter B → Filter C の順序と provenance） | 物理制約フィルタ Skill、スクリーニング Skill |
| 第9章 | DFT proxy と surrogate による物性予測を Skill 化する | 生成候補の物性を **DFT proxy モデル**（MEGNet / M3GNet / MACE 等の既存重み）で高速評価、**vol-03 の Foundation Model 特徴を surrogate 入力に**、**Skill としての予測不確かさの管理**（vol-03 第8-9章の不確かさ手法を再利用） | DFT proxy Skill、物性予測 Skill |
| 第10章 | vol-04 の因果的判定を Skill に組み込む — "生成候補は本当に望む物性を出すか" | **vol-04 第9章の refutation テスト**（placebo test、random common cause）を生成候補評価に転用、**"生成候補と学習データの分布差" を causal validity として判定**、**Skill としての "妥当性判定失敗時は候補を返さない" 契約** | 因果的妥当性判定 Skill |
| 第11章 | vol-05 の BO を Skill に組み込む — 生成候補のランキングと絞り込み | 生成候補を BO の初期点として、**Skill としての "生成 → 選抜 → 実験 → 学習" ループ**、**多目的 BO で Pareto 候補を選ぶ**、**"生成候補を BO で再訪問する" フィードバック** | 生成 × BO 統合 Skill |

### 第IV部　総合ハンズオンと運用（目安 45〜55 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第12章 | 総合ハンズオン（Advanced Capstone）— Foundation Model → 候補生成 → 物理フィルタ → 因果的判定 → BO 選抜 → Human 承認 | ARIM 風合成データで、**Phase 1: 目標物性を条件として cVAE / cDiffusion で候補生成** → **Phase 2: Pymatgen で物理制約フィルタ** → **Phase 3: DFT proxy で物性予測** → **Phase 4: 因果的妥当性判定** → **Phase 5: 多目的 BO で top-k 選抜** → **Phase 6: Human 承認**。**エージェントは各 Phase で "根拠と不確かさと外挿警告" を提示** | 複合逆設計 Skill (capstone) |
| 第13章 | 分子・結晶構造の逆設計 — 追加ハンズオン | **RDKit + SMILES での分子 VAE**、**Pymatgen での結晶構造 Diffusion**、**QM9 / Materials Project ベンチマーク**での動作確認、**"実データを持ち込む" 場合のガイド** | 分子 Skill、結晶構造 Skill |
| 第14章 | 生成 × Agentic 特有の失敗パターンと監査 | **セクション 1（生成モデル一般の失敗）**：mode collapse、非物理的候補の生成、分布外候補の "自信あり" 報告、条件付き生成の条件無視、Diffusion の temperature 暴走。**セクション 2（Agentic 特有）**：**エージェントが物理制約フィルタをスキップ、"合成可能性" 判定を勝手に緩める、安全性スクリーニングをスキップ、hallucinated 候補を top-k に含める、Human 承認なしに合成試行を推薦、生成 seed を上書き、条件変数を勝手に変更、危険物質を生成候補として返す** | 生成 × Agentic 失敗チェックリスト |
| 第15章 | 組織展開と終章 — 逆設計の責任分担・合成試行の管理・シリーズ終章 | **逆設計候補の責任分担**（Skill 提案 vs 研究者判断 vs 合成担当）、**危険候補の遮断責任**、**vol-06 の到達点**、**シリーズ全体（vol-01〜06）の到達点と接続図**、**次に読むべき既存教科書・論文リスト**、ARIM 施設としての逆設計運用（複数研究者間での生成モデル共有、合成キュー統合） | 組織運用方針、シリーズ終章 |

### 付録（目安 40〜50 ページ）

| 付録 | タイトル | 内容 |
|---|---|---|
| 付録A | 生成 × Agentic Skill テンプレート集 | 組成 VAE Skill、cVAE Skill、組成 Diffusion Skill、結晶構造 Diffusion Skill、Flow Skill、AR Skill、物理制約フィルタ Skill、DFT proxy Skill、因果的妥当性判定 Skill、生成 × BO 統合 Skill の雛形。**vol-03 の provenance を生成 × 逆設計 × Agentic 向けに拡張**（`generative_model_family`, `latent_dim`, `condition_variables_declared`, `generation_temperature`, `filter_rules_applied`, `screening_model_family`, `top_k_returned`, `synthesis_launch_authorization`, `hallucinatory_composition_detection`, `safety_screening_passed` 等） |
| 付録B | Diffusers / Pymatgen / RDKit / matminer / CDVAE / DiffCSP チートシートと **MCP サーバ実装（逆設計版）** | よく使う API、既存材料 FM の Hugging Face 経由 fetch、**MCP Python SDK で "生成 → フィルタ → 選抜 → 提示" のパイプラインを Skill 化する**ミニマル実装例、**合成試行承認ゲートを MCP レベルで組む** |
| 付録C | 生成モデル演習データ候補と benchmark | **ARIM 風合成材料データの生成スクリプト**（合成可能性フラグと真の物性を持つ）、Materials Project / OQMD / JARVIS-DFT / AFLOW / QM9 のライセンスと入手、**危険物質 filter list**（爆発性・毒性・放射性等）、ARIM 実データを持ち込む際の匿名化ガイドと "生成候補を外部提案する" 際のレビュー手順 |

## 責務分離マップ（重複防止）

### vol-06 内での分離

| 論点 | 予防・設計側 | 実行・事例側 |
|---|---|---|
| 生成モデル選択 | 第5-7章 (VAE / Diffusion / Flow / AR) | 第12章 capstone |
| 物理制約フィルタ | 第8章 | 第14章 (フィルタスキップの失敗) |
| DFT proxy | 第9章 | 第12章 |
| 因果的妥当性判定 | 第10章 | 第12章 |
| 生成 × BO 統合 | 第11章 | 第12章 |
| Agentic 権限 | 第4章 (`synthesis_launch_authorization`) | 第14章・付録B (MCP 実装) |
| 安全性スクリーニング | 第4章 (`safety_screening_passed`) | 第14章・付録C (危険物質 filter list) |

### vol-01〜05 との分離

| 論点 | 既刊側 | vol-06 側 |
|---|---|---|
| 予測 Skill | vol-01/02/03 | 生成候補の物性予測に第9章で活用 |
| 統計モデル | vol-02 第10-12章 | 生成候補の事後解析に活用（生成 seed の事後分布等） |
| 深層学習 | vol-03 全体 | 生成モデルの骨格として第5-7章で活用 |
| 材料 Foundation Model | vol-03 第11章 | **既存 FM を生成モデルに拡張**（CDVAE / DiffCSP / MatterGen 経由）、第6章 |
| 自己教師あり学習 | vol-03 第12章 | 潜在空間の学習として共通哲学、第7章で言及 |
| 因果推論 | vol-04 全体 | **生成候補の妥当性判定**に第10章で活用 |
| DoE | vol-04 第10-11章 | 生成候補の合成試行を DoE で計画（第12章 capstone Phase 6） |
| Bayesian 最適化 | vol-05 全体 | **生成候補のランキングと絞り込み**に第11章で活用 |
| Human-in-the-loop | vol-01 第6章 / vol-05 第4章 | **合成試行の Human 承認**（`synthesis_launch_authorization`、第4章）、**危険候補の遮断責任**（第15章） |
| Agentic 権限 | vol-03/04/05 第4章 | **生成 → フィルタ → 選抜 の 3 段階権限**：生成・フィルタ・スクリーニングは自律 / top-k 選抜は準自律 / **合成実行は必ず Human 承認** |
| 失敗パターン | 各巻 第14章 | 生成モデル一般 + Agentic 特有（第14章 2 セクション構成） |

## 各ハンズオン章の共通構成

vol-03/04/05 と同形式。**第5章で丁寧に説明し、以降は差分中心**。「エージェント役割」節を各章に必ず含める。

- この章で作る Skill の概要
- **エージェント役割**（Skill は何を生成し、何をフィルタし、何を Human に投げるか、**合成試行は誰が判断するか**）
- 入力仕様 / 出力仕様 / 制約条件（vol-05 拡張 + vol-06 生成 × 逆設計 × Agentic 拡張に準拠）
- **生成モデル特有の評価基準**（validity rate / uniqueness / novelty / condition adherence / physical constraint satisfaction / synthesizability proxy）
- 実行例 / 失敗例（**hallucinated 候補と Agentic 失敗を含む**）/ 改善版
- 他生成モデル（VAE ↔ Diffusion ↔ Flow）への転用方法

## 特に注意する重点管理項目

| 項目 | 予防・設計 | 事例・検証 |
|---|---|---|
| **エージェントが物理制約フィルタをスキップ** | 第4章・第8章 `filter_rules_applied` の必須化 | 第14章 |
| **エージェントが hallucinated 候補を top-k に含める** | 第4章 `hallucinatory_composition_detection` | 第14章 |
| **危険物質を生成候補として返す** | 第4章・付録C `safety_screening_passed` + filter list | 第14章 |
| **合成試行を Human 未承認で推薦** | 第4章 `synthesis_launch_authorization` | 第14章・付録B (MCP 実装) |
| **Diffusion の temperature を勝手に上げる** | 第6章 temperature pin | 第14章 |
| **条件付き生成の条件を無視** | 第4章 `condition_variables_declared` | 第14章 |
| **合成可能性判定を勝手に緩める** | 第8章 スクリーニング契約 | 第14章 |
| **生成 seed を上書き** | 第4章 seed provenance | 第14章 |

## シリーズ全体の到達点（第15章での接続図）

```
vol-01: 動く Skill を 1 つ自力で作る
   ↓ 統計/ML の厚みと不確かさを積む
vol-02: 統計/ML Skill + PyMC Skill + 階層モデル capstone
   ↓ 深層と Foundation Model を Agentic に扱う
vol-03: 深層 Skill + FM 呼び出し + PyMC 階層 capstone
   ↓ "なぜ" と "もし" を主張する
vol-04: 因果推論 Skill + DoE Skill + 反実仮想 capstone
   ↓ "次にどの実験" を提案する
vol-05: BO Skill + active learning + 階層 BO capstone
   ↓ "新しい材料を提案する"
vol-06: 生成 Skill + 逆設計 Skill + 生成 × BO 統合 capstone
   ↓
【シリーズ完結】ARIM データポータル会員が
    観測 → 予測 → 因果 → 逐次 → 生成
    の全ラダーを Agentic Skill として運用できる
```

**シリーズ通しての一貫哲学**：**エージェントは Skill として提案する。実験実行・合成試行・介入判断は常に Human が承認する。全 provenance は次巻でも参照可能な形で残す**。
