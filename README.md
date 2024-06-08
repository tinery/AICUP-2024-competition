# AICUP 2024 Competition

Team 5093: Luu Van Tin , Ngo Duc Thang, 林垣志, Nguyen Quang Sang

![image](https://github.com/nycu-acm/Density-Imbalance-Eased/assets/135048598/00455271-3601-4332-a5ef-36be8ad8ed39)


### Install

This implementation uses Python 3.8, [Pytorch](http://pytorch.org/),  Cuda 11.3. 
```shell
# Copy/Paste the snippet in a terminal
git clone https://github.com/nycu-acm/Density-Imbalance-Eased.git
cd Density-Imbalance-Eased

#Dependencies
conda create -n atlasnet python=3.8 --yes
conda activate Tracker
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --user --requirement  requirements.txt # pip dependencies
```


### Usage

*Demo:*    ```./run_submit.sh```