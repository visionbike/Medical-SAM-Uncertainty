### Set up environments

```shell
pip install -r requirements.txt
```

### Download datasets

```shell
chmod +x ./scripts/download_datasets.sh
```

- Download **ISIC** dataset

```shell
./scripts/download_datasets.sh ISIC2016
```

- Download **REFUGE** dataset

```shell 
./scripts/download_datasets.sh REFUGE
```

- Download **DDTI** dataset

```shell
./scripts/download_datasets.sh DDTI
```

- Download **STARE** dataset

```shell
./scripts/download_datasets.sh STARE
```

- Download **DRIVE** dataset

```shell
./scripts/download_datasets.sh DRIVE
````

- Extract IDRiD dataset (can be downloaded from [here](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)).

```shell
./scripts/download_datasets.sh IDRiD
```

- Download and preprocess FLARE22 (can be downloaded from [here](https://www.kaggle.com/datasets/prathamkumar0011/miccai-flare22-challenge-dataset)). Since it is 3D dataset, it will be processed to slice a volume into N slice.

```shell
# download FLARE22
./scripts/download_datasets.sh FLARE22
# preprocess FLARE22
python preprocess_flare22.py
```

- Download and preprocess LiTS17 (can be downloaded from [here](https://www.kaggle.com/code/ojshavsaxena/liver-tumor-segmentation-u-net)). Since it is 3D dataset, it will be processed to slice a volume into N slice.

```shell
# download LiTS17
./scripts/download_datasets.sh LiTS17 
# preprocess LiTS17
python preprocess_lits17.py
```

- Unzip and preprocess Multi-site Dataset for Prostate MRI Segmentation (can be downloaded from [here](https://liuquande.github.io/SAML/)). Since it is 3D dataset, it will be processed to slice a volume into N slice
- The Multi-site Dataset for Prostate MRI Segmentation include 6 datasets (RUNMC, BMC, I2CVB, BIDMC, UCL, HK) from different sites.


```shell
# download LiTS17
./scripts/download_datasets.sh PROSTATE
# preprocess LiTS17
python preprocess_prostate.py

### Directory Structure

The project's directories consist of:

├── project-directory/
│   ├── cfgs/
│   ├── data/
│   ├── dataset/
│   ├── graph/
│   │   ├── logs/
│   │   ├── metric/
│   │   ├── optimizer/
│   │   ├── network/
│   │   │   ├── decoder/
│   │   │   ├── encoder/
│   │   │   ├── layer/
│   │   │   ├── mobile_sam_v2/
│   │   │   └── sam/
│   │   └── mobile_sam_v2/
│   ├── model/
│   ├── pretrain_models/
│   ├── scripts/
│   ├── utils/
│   ├── train.py
│   ├── README.md
│   └── requirements.txt

### Train SAM model

Please take a look in `cfgs/default_train.yaml` to change the training parameters based on 
the configuration as described in [Medical SAM Adapter](https://github.com/SuperMedIntel/Medical-SAM-Adapter/tree/main) repo.

Run the training code as bellows:

```shell
python train.py -cfg cfgs/default_train.yaml
```

In `default_train.yaml`, you can consider and modify these arguments: `gpu_device`, `pretrain`, `epochs`, `val_freq`, `vis_freq`, 
`dataset`, `path`, `batch_size`, `multimask_output`, `lr`, `loss`.

You can check the results in `logs` folder. In `logs`, each experiment folder includes:
- `ckpt` contains checkpoint files. The best checkpoint will be stored under the name `checkpoint_best.pth`
- `log` contains log files.
- `run` contains tensorboard visualization.
- `sample` contains output's visualization.

### Test SAM model

Please take a look in `cfgs/default_test.yaml` to change the training parameters based on 
the configuration as described in [Medical SAM Adapter](https://github.com/SuperMedIntel/Medical-SAM-Adapter/tree/main) repo.

You NEED to set the trained model's checkpoint path in `ckpt` argument.

Run the training code as bellows:

```shell
python test.py -cfg cfgs/default_test.yaml
```