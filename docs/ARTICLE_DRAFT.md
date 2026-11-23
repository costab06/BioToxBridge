# When Adding More Biological Information Doesn't Help—and When It Does

*Two toxicity-prediction experiments, one reusable execution architecture, and a useful reminder about negative results.*

A molecular structure can be represented in many ways. One of the simplest is a fingerprint: a fixed-length vector encoding fragments of the molecule. Another possibility is to ask previously trained models what the molecule might do in a collection of biological assays, then add those predictions as features. Intuitively, those bioactivity predictions ought to contain information that structure alone does not make immediately accessible.

That was the hypothesis behind BioToxBridge. We tested it in two different settings, using the same three representation families: **A**, a 2,048-bit Morgan fingerprint; **B**, 12 continuous predictions from pretrained Tox21 models; and **D**, the two concatenated. The outcomes differed. That contrast is more informative than either experiment alone.

## First experiment: liver toxicity, and a negative result

The first task was regression: predict a liver lowest-observed-adverse-effect level (LOAEL), expressed on a logarithmic scale. The frozen dataset contains 177 chemicals, divided by molecular structure group into 142 training and 35 test chemicals. Structure groups are kept separate to reduce the risk of evaluating on close structural relatives of training examples.

The result was disappointing in an instructive way. On the held-out partition, the fingerprint-only model produced an MAE of **0.5945**, an RMSE of **0.7811**, and an R² of **−0.0201**. The combined representation slightly improved MAE to **0.5855** and RMSE to **0.7752**, but its R² remained negative (**−0.0049**). Its Spearman rank correlation, **0.228**, was also below the fingerprint-only model's **0.466**.

It would be misleading to say that the additional features had no effect at all: two error metrics moved in the favorable direction. But with negative R² and inconsistent improvement across metrics, this experiment did **not** demonstrate a meaningful predictive benefit from the additional Tox21 features. A small test set also limits how confidently we can generalize the outcome.

That is a legitimate research result. A feature can be biologically motivated and still fail to add useful information for a particular endpoint, sample, and modeling procedure.

## Second experiment: assay activity, and a more encouraging result

We then asked a different question. Instead of predicting liver LOAEL, could the same predicted-bioactivity features help classify activity in a ToxCast assay, AEID252?

Here the development cross-validation result was more promising. The combined small neural-network model, **D_small**, reached mean fold average precision of **0.6638**. A Tox21-only logistic baseline reached **0.6428**. The difference—about **0.021** average-precision points—suggests that the combined representation may be useful for this assay.

There are important boundaries around that statement. This is a development cross-validation comparison, not confirmation on a new independent holdout. The model families also differ, so the comparison does not isolate feature addition from model architecture. The right conclusion is a **promising experiment-specific result**, not a general claim that combining fingerprints and Tox21 predictions is superior.

## Separating scientific models from execution infrastructure

The two experiments also tested an engineering principle: scientific methods should be independent of the infrastructure that runs them. BioToxBridge owns the scientific questions and methods: chemical structure handling, Tox21-derived features, split definitions, model families, evaluation, and provenance. A regression experiment and a binary-classification experiment can share these conventions without pretending that they share identical algorithms.

OpenAgentFabric addresses a different problem. It provides reusable orchestration, durable execution, checkpointing, error handling, and observability. Where that infrastructure is valuable, BioToxBridge can integrate through adapters. Where it is unnecessary, the scientific framework can run on its own. There is no reason to move chemical featurization or model evaluation into the orchestration layer.

The common experiment runner is deliberately modest: it reads experiment configurations, invokes the corresponding scientific implementation, and records execution metadata. It is not a claim that both experiments run through one OAF-native graph. The separation matters because a scientific experiment should remain intelligible, testable, and reproducible without depending on the runtime that happens to execute it.

For the liver experiment, we also removed the historical inference service from the reproduction path by using saved Tox21 predictions and the original frozen split. That lets us rerun the scientific comparison without requiring a legacy server or silently selecting a new test set. It also makes the negative result easier to preserve and audit.

## What the two outcomes mean

The larger lesson is not that more features are good or bad. It is that **the value of a representation is conditional on the prediction target, the data, and the evaluation design**. In one task, adding predicted biological activity failed to demonstrate meaningful value. In another, it coincided with a modest improvement in development cross-validation.

The infrastructure should make both outcomes easy to run, inspect, and report. Scientific software earns trust not by manufacturing positive results, but by making negative results just as reproducible.
