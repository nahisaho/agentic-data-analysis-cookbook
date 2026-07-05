# 「AI エージェント時代の因果推論・実験計画入門 — ARIM データで動く Agentic Skill」章構成（v0.1 ドラフト）

> vol-04 は vol-01「AI エージェント時代のデータ分析入門」、vol-02「AI エージェント時代の統計・機械学習分析入門」、vol-03「AI エージェント時代の深層学習分析入門」の続編。
> vol-01〜03 が **「観測データから予測を作る」** ラインを整えたのに対し、vol-04 は **「観測データから "なぜ" と "もし" を主張する」** ラインを Skill 化する——
> ARIM データポータルの実験データ（前処理条件・装置設定・オペレータ差）を主戦場に、
> **「どの条件が効いたか」「反実仮想的にこの条件を変えていたらどうなったか」「次にどの介入をすべきか」**
> をエージェントが Skill として提示し、**Human-in-the-loop で介入の因果的妥当性を承認する**までを扱う。

## 版履歴

### v0.1（初版企画ドラフト）
- vol-03 章構成 v0.2 で「vol-04 候補」として棚上げされていた **因果推論** と **実験計画** の 2 テーマを本巻に統合
- **統合の理由**：ARIM 実験データにおいて、「どの条件が効いたか（因果推論）」と「次にどの条件を試すべきか（実験計画）」は同じ研究者が同じ研究テーマの中で連続的に行う判断であり、分冊すると「観測データ ↔ 介入設計」のフィードバックループが切れる
- 6 か月・250〜300 ページ規模、本編 15 章 + 付録 3 の骨格を継承
- **エージェントに何を許すか**：因果推論の DAG 選択・変数選択（confounder/mediator/collider）判断・**介入の実行判断**は Human-in-the-loop 必須（危険な自律化を回避）

## 前提

- **対象読者**: ARIM データポータル会員のデータ分析者（材料・ナノテク研究者）。Python / Jupyter 経験あり。**vol-01 + vol-02 完読を推奨**（vol-03 は必須ではないが Ch7/Ch11 で深層特徴を因果推論に接続する節あり）
- **vol-01 / vol-02 / vol-03 との関係**:
  - **vol-01 完読推奨**: Skill 設計 6 要素、データ契約、**Human-in-the-loop（第6章）**、provenance、6 データ型テンプレート
  - **vol-02 完読推奨**: 第4章（Skill 設計原則）、第7章（CV とデータリーク）、**第10-12章（PyMC / MCMC 診断）**、第11章（階層モデル）、第13章（合成階層データ capstone）
  - **vol-03 推奨**（必須ではない）: 第7章（転移学習）、第11章（Foundation Model 特徴）、第14章（Agentic 失敗パターン）
  - vol-04 の provenance は vol-02 拡張に、**因果 × Agentic 特有フィールド**を追加：`causal_graph_uri`, `causal_graph_sha256`, `treatment_variable`, `outcome_variable`, `confounders_declared`, `identification_strategy`（back-door / front-door / IV / DiD / RDD）, `estimator_family`（PSM / IPW / DR / doubly-robust ML / synthetic control）, `unmeasured_confounder_sensitivity`（E-value / Rosenbaum bounds）, `refutation_tests_passed`（placebo / random common cause / subset）, **`intervention_authorization`**, **`counterfactual_scope_gate`**, **`experimental_design_provenance`**（DoE、randomization seed、blocking factors）, **`sequential_experiment_stop_condition`**
- **最終ゴール（合格ライン）**:
  - **Pillar 1**: **観測データからの因果効果推定 Skill**（材料実験の前処理条件・装置設定・オペレータ差のいずれかについて）を 1 つ以上作れる。DAG が明示され、identification 戦略が provenance に残り、**unmeasured confounder への感度分析**まで含む
  - **Pillar 2**: **古典的実験計画 (DoE) を Skill 化**したもの（要因計画・分割区画・応答曲面・タグチメソッドのいずれか）を 1 つ以上作れる。**randomization と blocking が provenance に残る**
  - **Advanced Capstone**: 上記 2 本を統合した **「観測データで因果構造を仮説化 → DoE で介入検証 → 反実仮想シミュレーションで最適条件を提案」の複合 Skill** を、ARIM 風合成実験データ（vol-02/03 の合成階層データを介入 × 反実仮想の視点で拡張）で完成させる
- **分量目安**: 実践書（250〜300 ページ、vol-03 と同等）
- **期限**: 6 か月
- **ハンズオン標準環境**（vol-02/03 に追加）:
  - vol-02 標準 + `dowhy`, `econml`, `causal-learn`（DAG 探索）, `causalpy`（Bayesian causal impact）
  - `pgmpy`（確率的グラフィカルモデル）
  - `pyDOE2`, `smt`（実験計画）
  - `scikit-uplift`（uplift modeling、材料選抜への転用）
  - `linearmodels`（DiD / IV 推定）
  - PyMC（vol-02 継承、Bayesian causal inference と DoE 事後解析）
  - GraphViz（DAG 可視化）
  - **CI 環境は CPU で完結**を必須要件（vol-03 と同じ）
- **データセット方針**:
  - **主軸**: **ARIM 風合成実験データ**（vol-02/03 の合成階層データを介入 × 反実仮想の視点で拡張）
    - 前処理条件（温度・時間・雰囲気）× 装置差 × オペレータ差の 3 要因合成データ
    - 「どの条件が本当に効いたか」の真値を持つ合成データを本書でリポジトリ配布（`data/synthetic-causal/` 想定）
  - **対比・スケール確認**: 公開データセット
    - **Materials Project の DFT データ**（前処理条件 × 生成エネルギーの関係、観測研究の題材）
    - **AFLOW / NOMAD**（マルチサイト実験の再現性）
    - **ARIM 実データ**（読者現場で持ち込む、匿名化ガイドは付録C）
  - **DoE 演習**: **fractional factorial design** と **応答曲面法**の演習を合成データで実施
- **参照**:
  - DoWhy: https://www.pywhy.org/dowhy/
  - EconML: https://econml.azurewebsites.net/
  - causal-learn: https://causal-learn.readthedocs.io/
  - CausalPy: https://causalpy.readthedocs.io/
  - pgmpy: https://pgmpy.org/
  - pyDOE2: https://github.com/clicumu/pyDOE2
  - SMT: https://smt.readthedocs.io/
  - Judea Pearl "The Book of Why" / Hernán & Robins "Causal Inference: What If"
  - Montgomery "Design and Analysis of Experiments"（DoE 古典）
  - vol-01 / vol-02 / vol-03 リポジトリ（本書の前提）

## 本書で扱わないこと（明示）

vol-04 は **「エージェントが観測データから因果を主張し、次の介入を提案する Skill」** に scope を絞る。以下は将来巻の候補として第1章・第15章で道しるべを示すのみ。

| トピック | 扱わない理由 | 想定巻 |
|---|---|---|
| **ベイズ最適化 / 逐次実験計画（BoTorch / GPflow の active learning）** | 「次にどの試料を測るか」は独立テーマ。DoE の古典的計画は扱うが、GP surrogate に基づく逐次計画は別軸 | vol-05 |
| **生成モデル・材料逆設計（VAE / GAN / Diffusion）** | 「望む物性を持つ材料を生成する」は独立テーマ。反実仮想は扱うが、潜在空間からの生成は扱わない | vol-06 |
| **深層学習ベースの因果推論**（causal forest 単独章、TARNet, CFR 等の深層 CATE） | 深層は道具として ML-based estimator に組み込む扱い（EconML の DR-Learner ベース）、専用章は設けない | — |
| **強化学習・介入の逐次自律実行** | 「エージェントが実験を勝手に実行する」は本書のスコープ外。介入の実行は Human-in-the-loop 必須 | 別書候補 |
| **フォーマル因果グラフの数学的導出（do-calculus 完全版）** | 既存教科書に譲る。本書は Skill 化と provenance に集中 | — |
| **社会科学・疫学に固有の識別戦略**（自然実験のドメイン論、政策評価固有の RDD） | 材料実験に特化。RDD / DiD の骨格は扱うが、ドメイン深堀りはしない | — |

## 6 データ型と因果 × 実験計画 Skill の対応

vol-01〜03 の 6 データ型テンプレートを継承し、**因果 × 実験計画観点**を積む：

| データ型 | vol-02 / vol-03 で扱えたこと | vol-04 で扱う Agentic Skill |
|---|---|---|
| スペクトル型 | 統計/ML/深層による分類・回帰、階層プーリング | **前処理条件（温度・雰囲気）→ スペクトル特徴変化の因果効果推定 Skill**、**装置差を confounder として調整した DR-Learner**、**装置ごとの CATE 推定** |
| クロマトグラム・時系列型 | 時系列 CNN / Transformer、バッチ間再学習 | **反応条件 → 生成物分布の因果推定 Skill**、**DiD で「バッチ変更前後」の効果推定**、**時系列介入の synthetic control** |
| 画像・顕微鏡型 | CNN / ViT、SEM 転移学習 | **前処理条件 → 微細構造（粒径分布・欠陥密度）の因果推定 Skill**、**画像特徴を confounder / mediator として組み込む**、**Grad-CAM 効いた領域と介入変数の対応** |
| 回折・散乱パターン型 | 2D CNN パターン分類、格子定数事後分布 | **合成条件 → 結晶相の因果推定 Skill**、**格子歪みを mediator とした間接効果分解** |
| 表形式・プロセス条件型 | 物性予測、ロット/オペレータ階層 | **表形式実験計画の完全な因果推論 Skill**（DoWhy + EconML）、**要因計画からの効果推定 Skill**、**応答曲面法**、**タグチメソッド（品質工学）** |
| マルチモーダル統合型 | CLIP 系マルチモーダル埋め込み | **多モーダル confounder 調整**（例：スペクトルと画像を両方 confounder に）、**mediator が多モーダル** |

**共通する Agentic 観点**：各データ型で、Skill は「介入の効果を推定する」だけでなく、**「識別戦略が本当に成立するか（backdoor 基準の充足）、unmeasured confounder への感度、反実仮想の外挿範囲」をエージェントに判定させる契約**を持つ。**「介入を実際に実行する」ことは常に Human 承認**。

## 章構成（案）

**分量目安の合計**：本編 250〜300 ページ + 付録 40〜50 ページ

### 第0章 vol-01/02/03 の最小復習（目安 15 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第0章 | vol-01 / vol-02 / vol-03 の最小復習 | Skill / MCP / Human-in-the-loop / データ契約 / provenance / 6 データ型 / 統計 ML 診断 / **PyMC 階層モデル（vol-02 Ch11）** / **深層 Agentic 権限（vol-03 Ch4）** を 15 ページに圧縮 | vol-04 に入る最低限の前提 |

### 第I部　なぜ「エージェント × 因果 × 実験計画」なのか（目安 30〜35 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第1章 | vol-01〜03 の Skill に何が足りないのか — 予測から "なぜ" と "もし" へ | 予測 Skill の限界、**「相関 ≠ 因果」を Agentic 現場で守るには**、ARIM 実データにおける causal question の例、**本書のゴール（2 pillars + capstone）**、扱わないことの明示 | 本書の到達点、"予測"→"介入"→"反実仮想" のラダー |
| 第2章 | ARIM データで因果を主張するときの Agentic 特有の課題 | **観測研究における confounder の可視化 (DAG)**、**装置差・オペレータ差の扱い（confounder or mediator?）**、**少データ現場での識別困難性**、**エージェントが DAG を "勝手に" 書き換える危険性**、演習用データの紹介 | 自分のデータの因果的性質分類、演習データ入手 |
| 第3章 | 因果推論と実験計画のライブラリ地図 — Agentic 使い分け | **DoWhy / EconML / CausalPy / pgmpy / causal-learn** の位置づけ、**pyDOE2 / SMT** の位置づけ、**エージェントがどのライブラリまで自律的に叩けるか**、Bayesian causal 系（CausalPy → PyMC）の接続 | ライブラリ使い分けマップ、エージェント権限マップ |

### 第II部　観測データからの因果推論 Skill（目安 70〜80 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第4章 | 因果 × Agentic Skill の設計原則 | vol-02 第4章 / vol-03 第4章の因果 × Agentic 拡張。**「何を identification 戦略とみなすか」＋「どの DAG で再現するか」＋「エージェントに何を許すか」**。**新設節：介入承認ゲート**（`intervention_authorization`）——**エージェントは効果推定はできるが、実際の介入実行は必ず Human 承認**。反実仮想の外挿範囲、E-value による感度分析の provenance | 因果 × Agentic Skill 仕様書テンプレート |
| 第5章 | DAG と識別戦略 — Skill 化する前に決めること | DAG 記法（confounder / mediator / collider / M-bias / butterfly bias）、**backdoor / frontdoor 基準**、**IV / DiD / RDD の骨格**、**エージェントが DAG を提案する Skill と Human が承認する Skill の分離**、causal-learn による探索型 DAG の使いどころ | DAG Skill、識別戦略の判断表 |
| 第6章 | 介入効果の推定を Skill 化する（前半：Propensity / IPW / DR） | Propensity Score Matching、Inverse Propensity Weighting、Doubly Robust、**ML-based estimator（DR-Learner、DML）**、**装置差を confounder として調整する ARIM 例**、**Skill としての re-fit ポリシー** | ATE / ATT 推定 Skill |
| 第7章 | 介入効果の推定を Skill 化する（後半：DiD / IV / Synthetic Control） | Difference-in-Differences、Instrumental Variables、Synthetic Control（CausalPy 経由）、**「バッチ変更前後」「装置更新前後」などの ARIM 実例**、**エージェントが IV 候補を提案する場合の妥当性判定** | 準実験 Skill、識別戦略比較表 |
| 第8章 | Heterogeneous Treatment Effect と CATE — 「誰に効くか」の Skill 化 | CATE / ITE の推定（EconML の Meta-Learners: S / T / X / DR / R-Learner）、**「装置ごと」「試料組成ごと」の CATE**、**vol-03 の深層特徴を CATE の入力にする発展**、**エージェントが個別介入を推薦する場合の Human 承認ゲート** | CATE Skill、個別介入推薦の判断表 |
| 第9章 | 感度分析と refutation — 因果主張の "検算" を Skill 化 | **E-value、Rosenbaum bounds、Placebo test、Random common cause、Data subset validation**、**unmeasured confounder への強度スケール**、DoWhy の refutation ツール活用、**Skill 側で refutation pass 未達なら結論を出さない契約** | 感度分析 Skill、Refutation checklist |

### 第III部　実験計画 (DoE) の Skill 化（目安 55〜65 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第10章 | 古典的実験計画の Skill 化 — Full factorial / Fractional factorial / Randomization / Blocking | 要因計画・分割区画・**完全ランダム化 vs blocked randomization**、randomization の seed provenance、pyDOE2 の実装、**エージェントが実験計画表を提案する場合の効率 vs 情報量トレードオフ** | 要因計画 Skill、DoE provenance テンプレ |
| 第11章 | 応答曲面法とタグチメソッドの Skill 化 | Central Composite Design、Box-Behnken、応答曲面フィッティング（linear model → GBM → GP へのパス）、**タグチメソッド（品質工学、SN 比、直交表 L8/L18 等）**、ARIM 前処理条件最適化での使いどころ、**「エージェントが応答曲面を勝手に外挿する」失敗**の予防 | 応答曲面 Skill、タグチ Skill、外挿禁止契約 |
| 第12章 | Bayesian 実験計画 — 事前分布と情報利得で Skill 化 | Bayesian D-optimality / A-optimality の骨格、**PyMC で「事前分布 → 実験計画 → 事後分布」のループ**、**vol-02 第11章の階層モデルを DoE 内で使う**、**情報利得を Skill の意思決定変数にする**、逐次計画への橋渡し（詳細は vol-05） | Bayesian DoE Skill |

### 第IV部　総合ハンズオンと運用（目安 40〜50 ページ）

| 章 | タイトル | 責務 | 成果物 |
|---|---|---|---|
| 第13章 | 総合ハンズオン（Advanced Capstone）— 観測データで DAG 仮説化 → DoE で検証 → 反実仮想で最適条件提案 | ARIM 風合成実験データで、**Phase 1: 観測データから DAG を仮説化して CATE 推定** → **Phase 2: 仮説を検証する DoE を設計・実行（合成データ上）** → **Phase 3: 反実仮想シミュレーションで最適条件を提示**。**エージェントは 3 箇所で Human 承認を求める**（DAG 承認・DoE 承認・介入推薦承認） | 複合 Skill (capstone) |
| 第14章 | 因果 × Agentic 特有の失敗パターンと監査 | **セクション 1（因果推論一般の失敗）**：DAG の misspecification、collider bias、positivity violation、外挿の unwarranted、CATE の過剰個別化、refutation スキップ。**セクション 2（DoE 一般の失敗）**：randomization の破綻、blocking の失敗、応答曲面の外挿誤用、タグチ SN 比の誤解釈。**セクション 3（Agentic 特有）**：**エージェントが DAG を勝手に書き換える、confounder を勝手に削除する、感度分析を skip、"介入した" を勝手に記録、reproducibility seed を上書き、CATE 推薦を Human 未承認で外部に送信** | 因果 × Agentic 失敗チェックリスト |
| 第15章 | 組織展開と終章 — 因果的主張の責任分担・実験計画の組織運用・次巻への道しるべ | **因果的主張の責任分担**（Skill 提案 vs 研究者判断）、**実験リソースの割当と DoE 権限**、**vol-04 の到達点**、**vol-05（ベイズ最適化・逐次実験計画）・vol-06（生成モデル・逆設計）への道しるべ**、ARIM 施設としての因果推論運用（データ共有・identification 戦略のレビュー） | 組織運用方針、次巻ロードマップ |

### 付録（目安 40〜50 ページ）

| 付録 | タイトル | 内容 |
|---|---|---|
| 付録A | 因果 × Agentic Skill テンプレート集 | ATE / ATT / CATE 推定 Skill、DiD Skill、IV Skill、DAG 提案 Skill、DoE 生成 Skill、応答曲面 Skill、Bayesian DoE Skill、感度分析 Skill の雛形。**vol-02/03 の provenance を因果 × 実験計画 × Agentic 向けに拡張**（`causal_graph_uri`, `identification_strategy`, `refutation_tests_passed`, `intervention_authorization`, `experimental_design_provenance` 等） |
| 付録B | DoWhy / EconML / CausalPy / pgmpy / pyDOE2 チートシートと **MCP サーバ実装（因果推論版）** | よく使う API、DAG の記法、Refutation の実行、DoE の生成と可視化、**MCP Python SDK で「因果推論と DoE を Skill 化する」ミニマル実装例**、**介入承認ゲートを MCP レベルで組み込む方法** |
| 付録C | 因果推論・実験計画演習データ候補 | **ARIM 風合成因果データの生成スクリプト**（真の DAG と ground truth ATE / CATE を持つ）、公開データセット（Materials Project の DFT データ、Lalonde、IHDP、Twins、synthetic causal benchmark）、**ARIM 実データを持ち込む際の匿名化ガイドと identification 戦略の事前レビュー手順** |

## 責務分離マップ（重複防止）

### vol-04 内での分離

| 論点 | 予防・設計側 | 実行・事例側 |
|---|---|---|
| 識別戦略の選択 | 第5章 (DAG) | 第6-7章 (推定) |
| ML-based 推定器 | 第6章 (DR-Learner, DML) | 第8章 (CATE) |
| 感度分析 | 第9章 | 第14章（refutation 未実施の失敗） |
| DoE の randomization | 第10章 | 第14章（randomization 破綻の事例） |
| Bayesian causal | 第12章（DoE 側） | 第9章の refutation でも触れる |
| Agentic 権限 | 第4章 (intervention_authorization) | 第14章（バイパス事例）・付録B（MCP 実装） |

### vol-01/02/03 との分離

| 論点 | 既刊側 | vol-04 側 |
|---|---|---|
| 予測 Skill | vol-01/02/03 全体 | 「予測 → 介入 → 反実仮想」のラダーとしての位置付けを第1章で提示 |
| 統計モデル | vol-02 第9-12章 (PyMC) | Bayesian causal inference / Bayesian DoE として第7・12章で活用 |
| 深層特徴 | vol-03 第7章 (転移学習) | CATE の共変量として第8章で使用 |
| 階層モデル | vol-02 第11章 (合成階層データ) | DoE で装置差・オペレータ差を blocking factor として組む（第10-11章）、CATE の階層 (第8章) |
| Human-in-the-loop | vol-01 第6章 / vol-03 第4章 (深層版) | **介入実行の Human 承認**（`intervention_authorization`、第4章）、**DAG 承認・DoE 承認**（第13章 capstone） |
| Agentic 権限 | vol-03 第4章 (学習権限 3 段階) | **因果的判断の 3 段階権限**：観測データからの推定は自律 / DoE 提案は準自律 / **実際の介入実行は必ず Human 承認** |
| 失敗パターン | vol-01 第14章 / vol-02 第14章 / vol-03 第14章 | 因果推論一般 + DoE 一般 + Agentic 特有（第14章 3 セクション構成） |

## 各ハンズオン章の共通構成

vol-03 と同形式。**第6章で丁寧に説明し、以降の章は差分中心**。「エージェント役割」節を各章に必ず含める。

- この章で作る Skill の概要
- **エージェント役割**（Skill は何を推定し、何を Human に投げるか、**介入は誰が実行するか**）
- 入力仕様 / 出力仕様 / 制約条件（vol-02 拡張 + vol-03 拡張 + vol-04 因果 × 実験計画拡張に準拠）
- **因果 × 実験計画特有の評価基準**（identification validity / refutation pass / positivity / external validity / DoE efficiency / randomization integrity）
- 実行例 / 失敗例（**因果的失敗と Agentic 失敗を含む**）/ 改善版
- 他データ型・他ライブラリ（DoWhy ↔ EconML ↔ CausalPy）への転用方法

## 特に注意する重点管理項目

| 項目 | 予防・設計 | 事例・検証 |
|---|---|---|
| **エージェントが DAG を勝手に書き換える** | 第4章 `causal_graph_sha256` の pin | 第14章 |
| **エージェントが confounder を勝手に削除する** | 第5章 confounder declaration の必須化 | 第14章 |
| **感度分析 (E-value / refutation) をスキップ** | 第9章の pass 契約 | 第14章 |
| **介入を Human 未承認で外部に "推薦" として送信** | 第4章 `intervention_authorization` | 第14章 |
| **応答曲面を外挿誤用** | 第11章 外挿禁止契約 | 第14章 |
| **randomization seed を上書き** | 第10章 seed provenance の必須化 | 第14章 |
| **CATE 推薦を過剰個別化** | 第8章 external validity の閾値 | 第14章 |
