### Set up environments

```shell
pip install -r requirements.txt
```

### Download datasets

- Download ISIC dataset

```shell
chmod +x scripts/download_dataset.sh
bash scripts/download_dataset.sh ISIC2016
```

- Download REFUGE dataset

```shell
chmod +x scripts/download_dataset.sh 
bash scripts/download_dataset.sh REFUGE
```

### Train SAM model

Please take a look in `cfgs/default.yaml` to change the training parameters based on 
the configuration as described in [Medical SAM Adapter](https://github.com/SuperMedIntel/Medical-SAM-Adapter/tree/main) repo.

Run the training code as bellows:

```shell
python train.py -cfg cfgs/default.yaml
```

You can check the results in `logs` folder.