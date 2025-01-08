### Set up environments

```shell
pip install -r requirements.txt
```

### Download datasets

```shell
chmod +x scripts/download_dataset.sh
```

- Download **ISIC** dataset

```shell
bash scripts/download_dataset.sh ISIC2016
```

- Download **REFUGE** dataset

```shell 
bash scripts/download_dataset.sh REFUGE
```

- Download **DDTI** dataset

```shell
bash scripts/download_dataset.sh DDTI
```

- Download **STARE** dataset

```shell
bash scripts/download_dataset.sh STARE
```

- Extract IDRiD dataset (can be downloaded from [here](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)).
```shell
bash script/download.sh IDRiD
``` 

### Train SAM model

Please take a look in `cfgs/default.yaml` to change the training parameters based on 
the configuration as described in [Medical SAM Adapter](https://github.com/SuperMedIntel/Medical-SAM-Adapter/tree/main) repo.

Run the training code as bellows:

```shell
python train.py -cfg cfgs/default.yaml
```

In "default.yaml", you can consider and modify these arguments: `gpu_device`, `pretrain`, `epochs`, `val_freq`, `vis_freq`, 
`dataset`, `path`, `batch_size`, `multimask_output

You can check the results in `logs` folder. In `logs`, each experiment folder includes:
- `ckpt` contains checkpoint files.
- `log` contains log files.
- `run` contains tensorboard visualization.
- `sample` contains output's visualization.