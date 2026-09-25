# Proposal: Predict early drug escape from cancer-cell changes

## 1. Motivation and research question

A cancer drug can change most cells while leaving a smaller group with a different pattern of gene activity. Some such cells may later survive treatment. An early warning would help researchers decide which drugs and proteins deserve closer testing.

**Research question:** Given a cancer cell line and a drug dose, can we predict the cell population after 24 hours, identify an early pattern linked to later drug survival, and point to proteins that may explain the change?

The main task is to predict the **24-hour treated cell state**. “Early escape” is a further question that we test only where a separate, longer treatment study provides a relevant reference. Tahoe has many cell lines; BT-474 is one reference case, not the entire study.

## 2. Specific input and output, with real values

**What enters the model:** a group of control cells from one cell line, a drug, its dose, and a protein network. **What the model predicts:** the mix of cells and their gene activity after drug treatment. During training, we compare that prediction with cells that Tahoe actually measured after treatment.

Here is one **real Tahoe condition**. The drug and control samples are on the same plate. DMSO means the cells received the drug solvent but no active drug.

| Part | Real value |
|---|---|
| Cell line | **A549** lung cancer cells; Tahoe ID **CVCL_0023** |
| Control-cell samples | **smp_1589** and **smp_1590**; DMSO; **plate1** |
| Drug sample | **smp_1566**; **4EGI-1** at **0.05 µM**; **plate1** |
| Drug target listed by Tahoe | **EIF4E** |
| Model input state | Gene activity of A549 cells in the two control samples, including how common each cell pattern is |
| Drug action | 4EGI-1, dose 0.05 µM, with EIF4E marked on the protein network |

The next table contains **real observed output values** from Tahoe's public precomputed gene-change table for this exact A549, 4EGI-1, 0.05 µM condition. Negative values mean the gene was less active in treated cells than in the matched controls.

| Measured output | Real value |
|---|---:|
| Treated cells used in the comparison | **1,378** |
| Control cells used in the comparison | **4,862** |
| Change in **CD99** gene activity, log2 scale | **−0.3455** |
| Change in **CD38** gene activity, log2 scale | **−0.5645** |

These two gene values are **two entries in the observed treated state**, not the whole state. The full output also includes the other measured genes and the mix of treated-cell patterns. The model would predict these values from the control cells, drug, dose, and protein graph; Tahoe's measured values are the training answer. The source rows are **23** and **50** in the [Tahoe precomputed gene-change table](https://huggingface.co/datasets/tahoebio/Tahoe-100M/viewer/pseudobulk_differential_expression/train).

**What about the escape result?** Tahoe does not publish a ready-made “survived for days” label for this A549 condition. We must not call these gene changes proof of escape. For the escape part of the study, a real matched example is **BT-474 + lapatinib**: Tahoe has 24-hour samples **smp_1685** and **smp_1686** (controls) and **smp_1601** (0.5 µM lapatinib); [GSE84896](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84896) has BT-474 cells that survived at least nine days of lapatinib. A real **early escape-like score** for Tahoe sample smp_1601 still has to be calculated from its single-cell data. There is no honest numeric value to put in the table before that calculation.

## 3. Methodology: state, graph, and world-model pipeline

### What is the “state”?

The state is a description of a **group of cells**, rather than one cell or one number. For a cell line on a given plate, we represent the control group with:

| State component | Meaning |
|---|---|
| Cell-pattern proportions | The fraction of cells showing each learned pattern of gene activity |
| Gene activity within each pattern | Which genes are active in each group of cells |
| Cell-line traits | Known gene changes of that cell line, placed on the corresponding protein nodes |

Keeping the **proportions** matters: if a rare cell pattern grows after treatment, a single average across all cells can hide it.

Tahoe measures control and drug-treated cells **after the same 24-hour period**. They are different cells. Thus the control state is a matched reference, not a measurement of the very same cells before drug addition. The world model learns a **one-step drug effect on a cell population**. It cannot, from Tahoe alone, track individual cells or simulate several days of treatment.

### How the model is applied

~~~text
Build one fixed protein graph
        ↓
For each cell line + drug + dose:
  control cells → reference state
  cell-line gene activity and changes → protein nodes
  drug targets and dose → protein nodes
        ↓
Graph model combines information across connected proteins
        ↓
World model predicts the new cell-pattern proportions
and gene activity after 24 hours
        ↓
Compare with Tahoe's actual treated cells
        ↓
Where a long-term reference exists: calculate an early escape-like score
~~~

| Step | Detailed operation |
|---|---|
| **1. Build the graph once** | Use [STRING](https://en.string-db.org/cgi/download) to connect proteins that interact. Add pathway names from [Reactome](https://reactome.org/download-data). Keep these connections fixed for all experiments. |
| **2. Encode the control state** | A cell encoder turns each control cell's gene-activity list into a short numerical description. Group similar descriptions into cell patterns. Record each pattern's size and average gene activity. This is the reference state. |
| **3. Add cell and drug information to the graph** | For each protein, attach the matching control-cell gene activity and any known cell-line change. Mark the drug's known protein targets and include its dose. For A549 + 4EGI-1, the **EIF4E** node is marked as a target. |
| **4. Run the graph model** | Pass information along protein connections. The output is a summary of the parts of the network most relevant to this drug and cell line. This follows the drug-target grounding idea in the supplied [GRASP paper](</Users/nguyenhuyhai/Downloads/Graph_Grounded_Autonomous_Multi_channel_Reasoning_Agent_for_Interpretable_Open_World_Drug_Synergy_Prediction (7).pdf>). |
| **5. Run the world model** | Give it the reference state, graph summary, drug, and dose. For each cell pattern, it predicts **how its proportion changes** and **how its gene activity changes**. The collection of these predictions is the new, treated state. A pattern rare in controls can become common after treatment. |
| **6. Train using real outcomes** | Compare the predicted treated state with Tahoe's measured treated cells. Check both the overall gene changes and the predicted mix of cell patterns. Train on whole drug–cell-line conditions, since cells from one sample are not independent experiments. |
| **7. Examine early escape where justified** | From a long-term survivor study such as GSE84896, make a fixed list of genes that mark later survivors. Apply that list to the **predicted** 24-hour cells and to Tahoe's **measured** 24-hour cells. Compare the fraction of high-scoring cells with matched controls. This is an **early escape-like pattern**, not a measured survival outcome. |
| **8. Explain and test** | Trace which drug-target-to-protein paths most affect the prediction. Check whether removing a path changes the prediction. Test accuracy on held-out drugs and cell-line–drug pairs. The protein paths are hypotheses for lab experiments, not proof of cause. |

Tahoe measures **RNA**, a readout of gene activity. It does not measure protein amount or activity in these samples. Putting RNA values on protein nodes helps the model use known protein connections, but does not turn those values into protein measurements.

## 4. Public dataset sources

| Source | Public data used | Role |
|---|---|---|
| [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) | **expression_data**, **sample_metadata**, **gene_metadata**, **drug_metadata**, **cell_line_metadata**, and **pseudobulk_differential_expression** | Control and treated cells, drug doses and targets, cell-line traits, and measured 24-hour gene changes. The published atlas covers 50 cancer cell lines. |
| [GSE84896](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84896) | Parental and long-term lapatinib-surviving BT-474 cells | Build a BT-474/lapatinib survivor gene pattern. Its dose and time differ from Tahoe's 24-hour experiment. |
| [GSE156246](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156246) | Untreated, short-treatment, and 14-day lapatinib-surviving BT-474 cells | Check whether the proposed early pattern relates to later survivors in another study. |
| [STRING](https://en.string-db.org/cgi/download) and [Reactome](https://reactome.org/download-data) | Protein connections and pathway names | Build and explain the protein graph. |
