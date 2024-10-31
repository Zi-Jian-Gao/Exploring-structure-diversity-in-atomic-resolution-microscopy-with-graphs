## Exploring structure diversity in atomic resolution microscopy with graphs

## Abstract
The emergence of deep learning (DL) has provided great opportunities for the high-throughput analysis of atomic-resolution micrographs. However, the DL models trained by fixed image patches generally lack efficiency and flexibility when processing images containing diversified atomic configurations. 
Herein, inspired by the similarity between the atomic structures and graphs, we describe a few-shot learning framework based on an equivariant graph neural network (EGNN) to analyze a library of atomic structures (vacancies, phases, grain boundaries, doping, etc.), showing significantly promoted robustness and three orders of magnitude reduced computing parameters compared to the image-based learning frameworks, which is especially evident for those aggregated vacancy lines with a broad range of lattice distortion. 
Besides, the intuitiveness of graphs facilitates a straightforward structure parametrization from the model outputs, supporting to reveal the evolution dynamics of vacancy lines under continuous electron beam irradiation. 
We further establish a versatile model toolkit via assembling each trained EGNN sub-model to process complex systems in the form of a task chain, contributing to the discovery of novel doping configurations with superior electrocatalytic properties for hydrogen evolution reactions. This work provides a powerful tool to explore structure diversity in a fast, accurate, and intelligent manner.

## Installation and Usage

### Prerequisites
```bash
pip install -e requirements.txt
```

### Running the Code to reproduce the results of our task chain:
```bash
bash auto_analyze.sh
```

### For the sulfur vacancy recognize and visual results:
```bash
cd recognize_sv_linesv/egnn/ && \
python test_pl_vor.py --path='xxx'  && \
python metricse2e_vor.py --json_path=./logs/0/version_0/test.json --save_path='xxx' && \
python plot.py  --result_path='xxx' && \
```
