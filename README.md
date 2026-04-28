# Retrieval-Augmented Multimodal Framework for Enzyme–Substrate Interaction Prediction Under Low-Homology Shift
## Introduction: 
RAMMESI is a Retrieval-Augmented MultiModal framework for Enzyme–Substrate Interaction prediction.

### Usage

#### Install 
1. Create environment
```bash
conda create -n rammesi python==3.10.12
```
2. Install environment dependency
```bash
# We strongly recommend prioritizing the installation of FAISS.
# https://github.com/facebookresearch/faiss 
# We used the Bash method for installation, and ref the follow document:
# https://ai.meta.com/tools/faiss/ 
# https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 pytorch=2.3.0=*cuda* pytorch-cuda=12.1 numpy 
pip install pytorch-lightning==2.5.1
# install requirements
pip install -r ./requirements.txt
```

### Training
The source data are from ESP[1] (remove the molecular that cannot be embedded by the molecular encoder)and Reactzyme[2] (remove the substrates such as water, gases, and metal ions).
[1] Kroll A, Ranjan S, Engqvist M K M, et al. A general model to predict small molecule substrates of enzymes based on machine and deep learning[J]. Nature communications, 2023, 14(1): 2787.

[2] Hua C, Zhong B, Luan S, et al. Reactzyme: A benchmark for enzyme-reaction prediction[J]. Advances in Neural Information Processing Systems, 2024, 37: 26415-26442.

### Training
```bash
# put the needed file in corresponding folder
mv ./scripts/train_ESP.sh ./
bash ./train_ESP.sh
# same with Reactzyme
```

### Retrieval
```bash
# put the needed file in corresponding folder
mv ./scripts/retrieval_ESP_enzyme.sh ./
bash ./retrieval_ESP_enzyme.sh
# same with Reactzyme
```
