# Code for **R-TPT**

**R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning**

Our implementation is based on [TPT](https://github.com/azshue/TPT) and [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch).

### Prerequisites:
- python == 3.8.5
- pytorch == 1.12.1
- torchvision == 0.13.1

### Dataset:
Please follow [CoOp](https://github.com/KaiyangZhou/CoOp) and manually download the require datasets.
Replace /path/to/dataset with your dataset folder root in below commands and check the path of json file in fewshot_datasets.py.

### Test-time adaptation:
1. ##### Adaptation on clean dataset (R-TPT)
	```python
	python rtpt.py /path/to/dataset --test_sets DTD -a RN50 -b 64 --gpu 0 --ctx_init a_photo_of_a -p 50 --eps 0.0 --output_dir 'output_results/rtpt' --method rtpt
	```

2. ##### Adaptation on adversarial dataset (R-TPT)
	```python
	python rtpt.py /path/to/dataset --test_sets DTD -a RN50 -b 64 --gpu 0 --ctx_init a_photo_of_a -p 50 --eps 1.0 --step 7 --output_dir 'output_results/rtpt' --method rtpt
	```


3. ### Test-time adaptation based on TeCoA pretrained encoder:

    If you want to load TeCoA pretrained encoder, please download the TeCoA from [this repo](https://github.com/TreeLLi/APT) and copy it into `pretrain/tecoa` folder, and the command is:

    ```python
	python rtpt.py /path/to/dataset --test_sets DTD -a RN50 -b 64 --gpu 0 --ctx_init a_photo_of_a -p 50 --eps 1.0 --step 7 --output_dir 'output_results/rtpt' --method rtpt --load_tecoa 'RN50-eps1'
	```

```aiignore
conda create -n atpt
pip install tqdm
pip install ftfy
pip install regex
pip install scipy
pip install "numpy<2"
pip install matplotlib
pip install huggingface_hub
pip install tensorflow-cpu

pip install tensorflow-text
pip install einops
pip install pycountry==24.6.1
pip install packaging==24.2
pip install gdown==5.2.0
pip install git+https://github.com/fra31/auto-attack

```
