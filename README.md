# Early warning of cancer drug tolerance

## Idea

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

## 3. Detailed methodology

### Step A — Build one training example

For each `(cell line, drug, dose, treated sample, plate)`, collect its treated single cells. Find `DMSO_TF` control cells from the **same cell line and plate**. Apply the same cell-quality filter to both. Keep the sample and plate IDs, because thousands of cells from one well are not thousands of independent experiments. Tahoe's `sample_metadata` gives the dose, `expression_data` gives per-cell raw RNA counts, `gene_metadata` maps gene tokens to genes, and `cell_line_metadata` gives cell-line identifiers and some genotype information.

Normalize the cell profiles and summarize the control cells as (a) a distribution of learned cell-state embeddings and (b) mean, variation, and detection rate for each gene. This is the input reference state `Sref`. Build the same summaries from treated cells to obtain the observed output state `Sdrug`. Do **not** pass treated cells to the model at prediction time.

### Step B — Make an external, fixed DTP reference

Use [GSE84896](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84896), which compares parental BT-474 cells with cells that persisted after **at least nine days of 2 µM lapatinib**. From its processed differential-expression file, select reliably increased and decreased genes using a rule fixed before testing Tahoe, such as adjusted `p < 0.05` plus a minimum effect size. This creates an **up-gene list** and a **down-gene list**.

For each Tahoe cell, compute a simple score: average normalized expression of up genes minus average normalized expression of down genes. Within each cell line and plate, set the “high DTP-like” threshold from its DMSO controls, for example the control 95th percentile. For a drug condition, define:

```text
early-escape shift =
    fraction of treated cells above the threshold
  - fraction of matched control cells above the threshold
```

This number is an **RNA-signature proxy**, not an observed count of living persister cells. The BT-474 signature is most defensible for BT-474/lapatinib. For other drugs or cancer types, either obtain relevant public DTP references and test transfer, or leave this label missing. Do not assign a universal DTP label to all Tahoe conditions from one BT-474 study.

### Step C — Learn the graph-grounded transition

1. **Ground the drug.** Mark its known protein targets on the graph. Encode the molecule from its SMILES and its dose. Spread a limited amount of target information through nearby protein edges, as the original [GRASP paper](</Users/nguyenhuyhai/Downloads/Graph_Grounded_Autonomous_Multi_channel_Reasoning_Agent_for_Interpretable_Open_World_Drug_Synergy_Prediction (7).pdf>) does with drug-target diffusion.
2. **Ground the cell line.** Put control-cell RNA summaries and known mutations on the matching protein nodes. A graph neural network combines these baseline features with the drug signal. The same drug can therefore act differently in different cell lines.
3. **Predict the treated state.** A conditional transition network takes the graph representation, drug, dose, and `Sref`. It predicts both the drug-versus-vehicle gene/program changes and the **proportions of cell states** in `Sdrug`. A probabilistic output allows several possible treated cell states rather than one average profile.
4. **Train from measured changes.** Use the many Tahoe conditions to penalize errors in treated-versus-control gene changes, changes in cell-state proportions, and the shape of the treated-cell distribution. Apply the fixed DTP scoring rule to the **predicted** treated distribution to obtain its predicted early-escape shift. This avoids needing DTP labels for every Tahoe condition. Weight each condition as an experimental unit so a well with more sequenced cells does not automatically dominate training.
5. **Explain a prediction.** Trace a short path from the drug's known target through graph proteins to the predicted changing program. Test whether masking a proposed protein or edge changes the model prediction. Report such proteins as **hypotheses**: graph attention or masking alone does not prove that inhibiting a protein will stop persistence.

The model can be written simply as:

```text
predicted 24-hour state, early-escape score
    = WorldModel(control-cell state, drug, dose, protein graph)
```

### Step D — Test whether the answer is useful

- Hold out **entire drugs** and **drug–cell-line combinations**, not random cells from the same well. Compare with a control-mean predictor, a model without the protein graph, and established perturbation models such as chemCPA or MAP where feasible.
- Measure prediction of expression changes **and** rare/high-score cell fractions. Test graph-pathway explanations against known drug targets and pathway annotations. Remove or randomize graph edges as an ablation.
- Use [GSE156246](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156246), an independent BT-474/HCC1419 lapatinib study with untreated, 6-hour, and **14-day DTP** samples, to check whether the early program points toward a real later persister program. This tests biological transfer; the experiments differ in dose, timing, and laboratory conditions, so it is not a perfect matched outcome study.
- Report performance separately by cell line, drug type, and dose. Check whether the score is just generic stress, cell-cycle arrest, or cell death rather than a specific persister-related program.

## 4. Public data and one concrete input/output example

| Public resource | Exact role | Required tables/files |
|---|---|---|
| [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) | Main observed 24-hour transitions | `expression_data`, `sample_metadata`, `gene_metadata`, `cell_line_metadata`, `drug_metadata`; optionally `obs_metadata` for quality control |
| [GSE84896](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84896) | Build a BT-474/lapatinib DTP gene signature | `GSE84896_BT474_Parental_Persister_CuffDiff.txt.gz` |
| [GSE156246](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156246) | Independent later-DTP check | BT-474 untreated, 6-hour lapatinib, and 14-day DTP single-cell data |
| [STRING](https://en.string-db.org/cgi/download) + [Reactome](https://reactome.org/download-data) | Protein graph and pathway names | Human physical links and pathway annotations |

**Real metadata keys for an example:** Tahoe's BT-474 line has Cellosaurus ID `CVCL_0179` and DepMap ID `ACH-000927`. Its **lapatinib ditosylate, 0.5 µM** treatment is sample `smp_1601` on `plate2`. Two `plate2` DMSO samples are `smp_1685` and `smp_1686`, as shown in [Tahoe sample metadata](https://huggingface.co/datasets/tahoebio/Tahoe-100M/viewer/sample_metadata/train). These IDs identify the candidate input and outcome groups; retain this row only after checking that both groups have enough good BT-474 cells. (Tahoe also lists lapatinib at 0.05 µM as `smp_1505` and 5 µM as `smp_1697`; each dose needs controls from its own plate.)

| Field for this example | Specific value or construction |
|---|---|
| **Input** | BT-474 control-cell RNA profiles from `CVCL_0179` in plate-2 `DMSO_TF` samples `smp_1685` and `smp_1686`; BT-474 mutation metadata; lapatinib ditosylate structure and checked targets; dose `0.5 µM`; fixed protein graph |
| **Measured output** | BT-474 treated-cell RNA profiles from `smp_1601`, summarized as 24-hour gene changes and the distribution of cell programs |
| **Derived training label** | The treated-minus-control fraction with a high score for the **GSE84896-derived** DTP program |
| **Model output** | Predicted treated-cell distribution, predicted early-escape shift, uncertainty, and a ranked list of graph proteins/pathways for experimental follow-up |

For clarity, **the following numbers are an invented calculation example, not Tahoe results**. Suppose 40 of 800 control cells and 150 of 600 treated cells exceed the pre-set signature threshold. The control fraction is `0.05`, the treated fraction is `0.25`, and the early-escape label is `0.25 − 0.05 = +0.20`. The real value for `smp_1601` must be computed from downloaded expression rows. A positive value means more **DTP-like RNA states at 24 hours**; it does **not** mean that 20% more cells were experimentally shown to survive nine days.

## Main claim the paper could honestly make

“From a matched vehicle-control cancer cell population, a drug, and a protein graph, we predict the **observed 24-hour population response** and identify conditions that increase an externally defined, early persister-like program. Independent long-term studies test whether that program is relevant to true drug tolerance.”

This follows the [world-model definition](</Users/nguyenhuyhai/Downloads/world_model_pharma_summary.md>) of **state + action → changed state** as a *one-step intervention model*. Tahoe's matched vehicle state stands in for the starting condition; a future dataset with measurements before and after treatment would be needed to establish a literal time trajectory.
