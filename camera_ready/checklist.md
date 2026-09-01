# Reviewing Schedule

## Title
A Scaled-Up Empirical Study of Syntactic Language Models

## Authors

## Keywords
syntactic language models, tree linearization, structural attention masks, constituency parsing, language model scaling, empirical evaluation

## TL;DR
We conduct a scaled-up empirical study of 12 controlled configurations (a terminal baseline and 11 SLM variants), comparing different types of linearization and masks; Pushdown is included separately as an architectural reference baseline outside this design space.

## Abstract
Syntactic language models (SLMs) that generate sentences along with their syntactic tree structures have shown promise in both syntactic generalization and downstream tasks. Existing studies, however, are relatively limited in scale regarding model size, training data size, and task diversity. In addition, the design space of tree linearization and attention masks has not been fully explored. In this paper, we conduct a scaled-up empirical study of 12 controlled configurations---a terminal baseline and 11 SLM variants---comparing different types of linearization and masks. Pushdown is included separately as an architectural reference baseline rather than a model variant. The experimental results show that SLMs with standard causal attention match or exceed the performance of those with structural masks across downstream and syntactic benchmarks, while preserving FlashAttention compatibility with zero architectural overhead. We also observe the advantage of SLMs when scaling up at the 1B-parameter scale across 11 downstream tasks.

## Paper Type
Long

## Research Area
(?)

## Research Area Keywords
(?)

## Contribution Types
Model analysis & interpretability (?)
NLP engineering experiment

## Languages Studied
English

## Reassignment Request Area Chair
No

## Reassignment Request Reviewers
No

## Software
## Data
仓库要传吗（？）

## Preprint
Yes

## Preprint Status
We plan to release a non-anonymous preprint in the next two months (i.e., during the reviewing process).

## Preferred Venue
EMNLP

## Visa Needs*
Yes

## Country Of Origin
CN

## A1 Limitations Section
This paper has a limitations section.

## A2 Potential Risks
No
The current draft does not include a dedicated risks or societal impact discussion. The work is an empirical pretraining/evaluation study rather than a deployed system. The training corpus we selected is primarily sourced from news content, with inappropriate internet data filtered out. But possible risks include misuse if trained models are deployed without safeguards.

## B Use Or Create Scientific Artifacts
Yes

## B1 Cite Creators Of Artifacts
Yes
5.2 Evaluation Methods, 5.3 Experiment 1, 5.4 Experiment 2, A.1 Implementation and Hyperparameters, B Evaluation Setup Details

## B2 Discuss The License For Artifacts
No
The code and dataset we use are both subject to open source licenses, and the relevant open source license has been attached to our repository.

## B3 Artifact Use Consistent With Intended Use
Yes
In Appendix A and B, the purposes of all the artifacts we use have been declared and are consistent with their original purposes.

## B4 Data Contains Personally Identifying Info Or Offensive Content
No
All training corpora involved in this paper come from BBC News content included in FineWeb-Edu and FineWeb, and have been partially filtered. We did not construct new manually annotated data nor release new datasets containing personal information, but the training corpora may still carry inherent risks associated with web-based corpora. Given that our work is more oriented toward model comparison studies, the impact of this issue is relatively minor.

## B5 Documentation Of Artifacts
Yes
5.3 Experiment 1, 5.4 Experiment 2, Limitations

## B6 Statistics For Data
Yes
3.2 Tree Linearization, 5.3 Experiment 1, 5.4 Experiment 2, B Evaluation Setup Details

## C Computational Experiments
Yes

## C1 Model Size And Budget
Yes
A.1 Implementation and Hyperparameters, A.2 Computational Costs

## C2 Experimental Setup And Hyperparameters
Yes
A.1 Implementation and Hyperparameters, B Evaluation Setup Details

## C3 Descriptive Statistics
Yes
For the 1B FineWeb-Edu downstream evaluation, Appendix C reports 95% bootstrap confidence intervals from 10,000 resamples, paired bootstrap differences and two-sided p-values for the principal model contrasts, macro-average sensitivity to task-suite composition, and bootstrap ranking probabilities. The paper also states that these intervals quantify evaluation-set uncertainty rather than variation across pretraining seeds, since each 1B configuration is represented by one pretraining run. The per-task tests are identified as exploratory and uncorrected for multiple comparisons.

## C4 Parameters For Packages
Yes
5.1 Setup, A.1 Implementation and Hyperparameters, A.3 Tokenizer and Structural Tokens, B Evaluation Setup Details

## D Human Subjects Including Annotators
No

## D1 Instructions Given To Participants
N/A

## D2 Recruitment And Payment
N/A

## D3 Data Consent
N/A

## D4 Ethics Review Board Approval
N/A

## E Ai Assistants In Research Or Writing
Yes

## E1 Information About Use Of Ai Assistants
No
We used AI assistants only for language-level assistance, such as proofreading, grammar correction, and minor polishing of author-written text. The assistants were not used to generate research ideas, claims, experimental results, analyses, citations, code, or substantive scientific content.

## Author Submission Checklist
Yes

## Association For Computational Linguistics - Blind Submission License Agreement
On behalf of all authors, I agree

## EMNLP 2026 AI Reviewing Experiment
No
