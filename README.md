# Research proposal: Early warning of cancer drug tolerance

## One-sentence idea

Use a **protein-graph world model** to predict a cancer cell population's response to 24 hours of drug treatment, then ask whether the treated state resembles cells that survive longer treatment. The model also points to proteins that may help explain that response.

## 1. Motivation and concrete research question

Some cancer cells survive a drug without having a permanent resistance mutation. These **drug-tolerant persister (DTP)** cells can become a starting point for later resistance. This is an active problem in cancer research, discussed in [Nature Reviews Cancer](https://www.nature.com/articles/s41568-024-00737-z) and studied experimentally in [BT-474 breast cancer cells treated with lapatinib](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84896).

**Research question:** Given a cancer cell line, its matched vehicle-control cell states, and a drug at a chosen dose, can we predict (1) the distribution of cell states after 24 hours, (2) the fraction of cells showing an **early DTP-like program**, and (3) the protein pathways associated with that response?

The practical use is an **early warning for drug discovery**: identify treatments that produce many escape-like cells and nominate proteins for follow-up experiments. The paper should test whether this early warning relates to true long-term persister states in independent data. It must not call a 24-hour expression pattern “proven resistance.”

This is different from predicting drug synergy. The unit of prediction is **one cell line + one drug + one dose**, and the output is a **change in a population of cells**, with an interpretable protein-pathway explanation. It is also more specific than generic drug-response prediction: [chemCPA at NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html) and [MAP in Nature Machine Intelligence](https://www.nature.com/articles/s42256-026-01286-w) already study prediction of drug-induced expression, including Tahoe data. The proposed contribution must be the **early escape question, careful connection to later DTP data, and protein-level explanations**, not merely another expression predictor.

## 2. Why the data, graph, and world model fit

The main resource is [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M), published in [*Cell*](https://pubmed.ncbi.nlm.nih.gov/42753697/). It contains single-cell RNA measurements from 50 cancer cell lines exposed to hundreds of compounds and roughly 1,100 drug–dose conditions. Cells were measured after about **24 hours** of treatment; the public tables include drug, dose, cell-line, plate, and vehicle-control information.

The world model learns a **one-step intervention effect** from two measured experimental conditions:

```text
Reference state Sref: vehicle-control cell population + graph features
Action A:  drug and dose
Outcome state Sdrug: drug-treated cell population measured after 24 hours

World model: P(Sdrug | Sref, A)
```

Here, a **state** is a distribution of cells, not a single average cell. It records which cell programs are common or rare, gene-expression summaries for each program, and cell-line mutations. Tahoe's vehicle and drug groups contain **different cells measured at the same endpoint**. Thus `Sref` is a matched reference for the untreated condition, **not a direct measurement before drug addition**. The model learns a population-level treatment contrast; it does not track an individual cell or observe a literal before/after trajectory.

Use a fixed human protein-interaction graph, for example the [STRING physical interaction network](https://en.string-db.org/cgi/download), with pathway names from [Reactome](https://reactome.org/download-data). Put three kinds of information on its protein nodes:

| Graph information | Where it comes from | Meaning |
|---|---|---|
| Baseline features | Control-cell RNA and cell-line metadata | Which genes are expressed, variable, or altered in this cell line |
| Drug signal | Drug targets, dose, and chemical structure | Where the treatment enters the graph |
| Predicted next features | World model | Which gene programs should rise or fall after treatment |

Tahoe measures **RNA, not protein abundance or protein activity**. Mapping gene expression to protein nodes gives a useful graph representation, but the resulting node values must be called *RNA-derived features*. The graph's edges come from external knowledge; Tahoe does not directly measure edges changing after treatment. Drug-target entries should be checked against a source such as [ChEMBL](https://www.ebi.ac.uk/chembl/) because some Tahoe target annotations are predicted or curated at different confidence levels.

**Feasibility boundary:** Tahoe supplies true 24-hour treated and matched vehicle states, but it does **not** supply a measured pre-treatment state or a long-term persister or relapse label for each drug–cell-line condition. Public DTP studies supply later states for selected settings. The first benchmark is therefore **early DTP-like change**; evidence that it predicts later persistence is a separate validation question. A true multi-step treatment planner would need additional longitudinal or combined-intervention data.

## 3. Methodology: the graph-to-world-model pipeline

The graph is built **once, before model training**. For each drug question, we then place that cell line and drug on the graph. The **world model runs after the graph encoder** and predicts the treated cell population. The DTP score is calculated from that predicted population.

```text
ONE-TIME SETUP
STRING protein interactions + Reactome pathways
       ↓
Fixed protein graph G

FOR ONE CELL LINE + DRUG + DOSE
Matched DMSO cells ──→ reference state Sref ──┐
                                               ├→ features on G
Drug targets + dose + molecular structure ────┘
                                                    ↓
                                  target diffusion + graph neural network
                                                    ↓
                                     graph-aware cell/drug representation
                                                    ↓
                                      WORLD MODEL Tθ  ← this is the state step
                                                    ↓
                                  predicted treated-cell distribution Ŝdrug
                                                    ↓
                     predicted DTP-like fraction + changed pathways/proteins

TRAINING ONLY: compare Ŝdrug with Tahoe's measured treated cells Sdrug.
```

| Stage | When it happens | What goes in | What happens and comes out |
|---|---|---|---|
| **1. Build graph `G`** | Once, before training | Human [STRING physical interactions](https://en.string-db.org/cgi/download), [Reactome](https://reactome.org/download-data) pathways, gene-to-protein IDs | Proteins are nodes; physical interactions are edges. Keep this graph fixed across experiments. |
| **2. Form reference state `Sref`** | For each cell line and plate | Matched Tahoe DMSO cells | A compact description of the cell population: per-cell embeddings, cell-state proportions, and RNA summaries for graph genes. This is the world model's starting state. |
| **3. Put the action on the graph** | For each drug and dose | Drug targets, SMILES, concentration, and cell-line alterations | Mark target proteins and dose; attach baseline RNA and mutation features to protein nodes. Spread target signal to nearby proteins using bounded diffusion, following the idea in the [GRASP paper](</Users/nguyenhuyhai/Downloads/Graph_Grounded_Autonomous_Multi_channel_Reasoning_Agent_for_Interpretable_Open_World_Drug_Synergy_Prediction (7).pdf>). |
| **4. Encode the graph** | Immediately before the world model | The now-annotated graph | A graph neural network mixes information across connected proteins. Its output tells the world model how this drug meets this cell line's biological state. |
| **5. Apply world model `Tθ`** | Once for each proposed action | `Sref`, graph representation, drug structure, and dose | Predict the **distribution** of treated cells: how common each cell program will be and its gene-expression pattern. This is `P(Sdrug | Sref, drug, dose, G)`, not just a single response score. |
| **6. Read out escape and explanations** | After the predicted state | Predicted treated-cell distribution and graph paths | Apply a fixed DTP gene signature to predicted cells; return the predicted high-score fraction and its change from DMSO. Rank short paths from known drug targets to changed programs. These proteins are follow-up hypotheses. |
| **7. Learn and check** | During training and evaluation | Actual Tahoe treated cells; independent DTP data | Train the world model to match observed gene changes and cell-state proportions. Evaluate on held-out drugs and drug–cell-line pairs; check the DTP interpretation against later persister data. |

The transition in stage 5 can be implemented as a **mixture model**. First group control-cell embeddings into several cell programs. The world model predicts how the **size** and **RNA profile** of each program change after the drug. A decoder turns those predictions into a treated-cell distribution. This matters because a rare escape-like population can grow even when the average expression changes little.

For the DTP readout, fix a signature from [GSE84896](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84896): genes higher in long-term BT-474 lapatinib persisters and genes lower in them. Score each predicted cell as `mean(up genes) − mean(down genes)`. Call a cell “high DTP-like” only if its score exceeds a threshold set from matched DMSO cells. Then:

```text
predicted early-escape shift
  = predicted fraction of high-scoring treated cells
  − fraction of high-scoring matched DMSO cells
```

This is a **24-hour RNA-signature proxy**, not observed long-term survival. The BT-474 signature is most defensible for BT-474/lapatinib; other contexts need their own validated references. Tahoe's control and treated cells were measured in separate wells at the same endpoint, so stage 5 models a **one-step intervention effect**, not a tracked before/after cell trajectory. [GSE156246](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156246) provides an independent 14-day DTP check. Protein rankings are graph-based hypotheses, not proven causal targets.

## 4. Public data and one concrete input/output example

| Public resource | Exact role | Required tables/files |
|---|---|---|
| [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) | Main observed 24-hour transitions | `expression_data`, `sample_metadata`, `gene_metadata`, `cell_line_metadata`, `drug_metadata`; optionally `obs_metadata` for quality control |
| [GSE84896](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84896) | Build a BT-474/lapatinib DTP gene signature | `GSE84896_BT474_Parental_Persister_CuffDiff.txt.gz` |
| [GSE156246](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156246) | Independent later-DTP check | BT-474 untreated, 6-hour lapatinib, and 14-day DTP single-cell data |
| [STRING](https://en.string-db.org/cgi/download) + [Reactome](https://reactome.org/download-data) | Protein graph and pathway names | Human physical links and pathway annotations |

### 4.1 A real Tahoe example: BT-474 + lapatinib

The following are **real public metadata entries**, not invented observations. Tahoe's [sample metadata](https://huggingface.co/datasets/tahoebio/Tahoe-100M/viewer/sample_metadata/train) identifies the wells; its drug and cell-line metadata identify the targets and genotype. The wells contain mixed cell lines, so select only `cell_line_id = CVCL_0179` expression rows inside each sample and retain the example only if those rows pass cell-count and quality checks.

| Role | Tahoe sample ID | Plate | Treatment | Dose | Cell-line filter |
|---|---|---|---|---|---|
| Reference input | `smp_1685` | `plate2` | `DMSO_TF` | 0 µM | BT-474, `CVCL_0179` |
| Reference input | `smp_1686` | `plate2` | `DMSO_TF` | 0 µM | BT-474, `CVCL_0179` |
| Measured outcome during training | `smp_1601` | `plate2` | Lapatinib ditosylate | 0.5 µM | BT-474, `CVCL_0179` |

**Exact model input for this row** (the RNA arrays are the filtered cells from the two DMSO wells):

| Input field | Specific example value | How the model uses it |
|---|---|---|
| Cell line | `BT-474`; Cellosaurus `CVCL_0179`; DepMap `ACH-000927`; breast cancer | Selects the biological context |
| Reference cell population `Sref` | The per-cell `genes` and `expressions` arrays from BT-474 cells in `smp_1685` and `smp_1686` | Produces baseline RNA features and the distribution of cell states |
| Cell-line alterations | `ERBB2` gain; `PIK3CA p.K111N`; `TP53 p.E285K` | Placed as annotations on those protein nodes; these are entries in Tahoe cell-line metadata |
| Drug action | `Lapatinib ditosylate`, `0.5 µM`, PubChem CID `11557040` | Structure and dose encode the action |
| Drug target nodes | `EGFR` and `ERBB2` | Both are marked as drug targets on the graph, according to Tahoe drug metadata; verify target evidence before final analysis |
| Graph | Human STRING physical interactions plus Reactome pathway labels | A fixed network that connects drug targets to other proteins |

For example, the **ERBB2 protein node** receives its BT-474 control-cell RNA summary, an `ERBB2 gain` annotation, and a `drug target = yes` marker. The **PIK3CA node** receives its control-cell RNA summary and `p.K111N` annotation, but no direct lapatinib-target marker. These node features are assembled **before** the graph neural network and world model run.

**Exact measured output format for this row:**

| Output field | Where the real value comes from | What one stored value looks like |
|---|---|---|
| Treated cells `Sdrug` | Tahoe `expression_data`, filtered to `sample = smp_1601` and `cell_line_id = CVCL_0179` | One row per treated cell: a `genes` token-ID array paired with an `expressions` raw-count array |
| Treated population state | Summary of those treated-cell rows | Cell-program proportions and gene-expression summaries; compared with the two DMSO groups |
| Early-escape label | Fixed GSE84896 DTP signature applied to the treated and control cells | One real number: `treated high-score fraction − control high-score fraction` |

The actual RNA arrays and early-escape number for `smp_1601` must be calculated from the full Tahoe expression table. The public **sample IDs, dose, targets, and genotype above are verified**; no RNA count or escape value is invented and presented as a measurement.

### 4.2 Small numerical example showing the final input and output

**Every cell count and prediction in this next table is illustrative, not a measured Tahoe result.** It shows the exact kind of record the proposed benchmark would contain once the expression rows are processed.

| Part of the example record | Concrete value | Meaning |
|---|---|---|
| Input: control state | 800 BT-474 DMSO cells; 40 score above the fixed threshold | Control high-score fraction `40 / 800 = 0.05` |
| Input: action | Lapatinib ditosylate, 0.5 µM; targets `EGFR`, `ERBB2` | Drug action placed on the graph |
| World-model output | Predicted treated-state distribution: high DTP-like `0.23`, other states `0.77` | Model expects 23% of treated cells to look DTP-like |
| World-model output | Predicted early-escape shift `0.23 − 0.05 = +0.18` | Final numerical prediction for this task |
| Measured training output | 600 treated BT-474 cells; 150 score above threshold | Observed high-score fraction `150 / 600 = 0.25` |
| Derived training label | `0.25 − 0.05 = +0.20` | Target value used to judge the prediction `+0.18` |
| Explanation output | Ranked target-to-pathway graph paths, with protein IDs and contribution scores | Protein hypotheses to test experimentally; Tahoe has no ground-truth label for these paths |

A positive shift means **more cells with a DTP-like RNA program after 24 hours**. It does not mean that 20% more cells were shown to survive nine days. For true later DTP status, [GSE84896](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84896) labels parental BT-474 samples (for example `GSM2253664`) separately from nine-day lapatinib persisters (for example `GSM2253666`); these are **different experiments and different samples** from Tahoe.

## Main claim the paper could honestly make

“From a matched vehicle-control cancer cell population, a drug, and a protein graph, we predict the **observed 24-hour population response** and identify conditions that increase an externally defined, early persister-like program. Independent long-term studies test whether that program is relevant to true drug tolerance.”

This follows the [world-model definition](</Users/nguyenhuyhai/Downloads/world_model_pharma_summary.md>) of **state + action → changed state** as a *one-step intervention model*. Tahoe's matched vehicle state stands in for the starting condition; a future dataset with measurements before and after treatment would be needed to establish a literal time trajectory.
