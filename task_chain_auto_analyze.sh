#!/bin/bash

#atom recognize
cd recognize_atom/msunet/ && \
python test_pl_unequal.py --path=../../img/ && \
python post.py --json_path=./logs/0/version_0/test.json --save_path=../../img/ && \
python test_pl_unequal.py --path=../../img/85/ && \
python post.py --json_path=./logs/0/version_1/test.json --save_path=../../img/85/ && \
cd .. && cd .. && \
mkdir -p atom && \
cp img/*.jpg atom/ && cp img/*.json atom/ && cp img/85/* atom/ && \
echo "\033[31mRecognition atom process completed. Results are stored in the 'atom' directory.\033[0m" && \
echo "\033[31mPress Enter to continue...\033[0m"
read input

#dophant atom recognize and select
mkdir -p dophant_atom/raw && \
cp atom/* dophant_atom/raw && \
cd recognize_dophant/egnn/ && \
python test_pl_vor.py --path=../../dophant_atom/  && \
python metricse2e_vor.py --json_path=./logs/0/version_0/test.json --save_path=../../dophant_atom/raw && \
python cz_exam.py  --save_path=../../dophant_atom/raw && \
cd .. && cd .. && \
echo "\033[31mRecognition dophant atom process completed. Results are stored in the 'dophant_atom' directory.\033[0m" && \
echo "\033[31mPress Enter to continue...\033[0m"
read input

#phase recognize and select
cd recognize_phase/egnn/ && \
python test_pl_vor.py --path=../../phase/  && \
python metricse2e_vor.py --json_path=./logs/0/version_0/test.json --save_path=../../phase/raw && \
python xj_exam.py  --save_path=../../phase/raw && \
cd .. && cd .. && \
echo "\033[31mRecognition phase process completed. Results are stored in the 'phase' directory.\033[0m" && \
echo "\033[31mPress Enter to continue...\033[0m"
read input

#grain boundary recognize
cd recognize_grain/egnn/ && \
python test_pl_vor_aug.py --path=../../gb/raw  && \
python metricse2e.py --json_path=./logs/0/version_0/test.json --save_path=../../gb/raw && \
echo "\033[31mRecognition grain boundary process completed. Results are stored in the 'gb' directory.\033[0m" && \
python jj_exam_visual.py  --save_path=../../gb/raw && \
cd .. && cd .. && \
echo "\033[31mVisualization process completed. Results are stored in the 'gb_visual' directory.\033[0m" && \
echo "\033[31mPress Enter to continue...\033[0m"
read input

#smov recognize
cd recognize_smov/egnn/ && \
python test_pl_vor_aug.py --path=../../smov/raw  && \
python metricse2e_vor.py --json_path=./logs/0/version_0/test.json --save_path=../../smov/raw && \
echo "\033[31mRecognition smov process completed. Results are stored in the 'smov' directory.\033[0m" && \
python kw_exam.py  --save_path=../../smov/raw && \
echo "\033[31mVisualization smov process completed. Results are stored in the 'smov_visual' directory.\033[0m" && \
echo "\033[31mPress Enter to continue...\033[0m"
read input && \
python kw_exam_visual.py  --save_path=../../smov/raw && \
echo "\033[31mVisualization gb&smov process completed. Results are stored in the 'gb+smov_visual' directory.\033[0m" && \
cd .. && cd .. && \
echo "\033[31mPress Enter to continue...\033[0m"
read input
