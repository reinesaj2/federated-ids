# UNSW Class Imbalance

Data source: `/scratch/reinesaj/datasets/unsw_full/UNSW_Flow.parquet` (2,059,415 rows).

Full dataset class distribution (10 classes):
- BENIGN: 1,959,772 (95.1616%)
- EXPLOITS: 27,599 (1.3401%)
- GENERIC: 25,378 (1.2323%)
- FUZZERS: 21,795 (1.0583%)
- RECONNAISSANCE: 13,357 (0.6486%)
- DOS: 5,665 (0.2751%)
- ANALYSIS: 2,184 (0.1060%)
- BACKDOOR: 1,983 (0.0963%)
- SHELLCODE: 1,511 (0.0734%)
- WORMS: 171 (0.0083%)

Class distribution for the 8-class subset used in the UNSW SimpleNet runs
(ANALYSIS and SHELLCODE excluded, 2,055,720 rows):
- BENIGN: 1,959,772 (95.3326%)
- EXPLOITS: 27,599 (1.3425%)
- RECONNAISSANCE: 13,357 (0.6497%)
- DOS: 5,665 (0.2756%)
- GENERIC: 25,378 (1.2345%)
- FUZZERS: 21,795 (1.0602%)
- WORMS: 171 (0.0083%)
- BACKDOOR: 1,983 (0.0965%)

Implication: the dataset is highly imbalanced (BENIGN ~95%, each attack class <2%,
several <0.1%), which severely limits training signal for attack classes and
depresses macro F1 even when BENIGN/GENERIC performance is reasonable.
