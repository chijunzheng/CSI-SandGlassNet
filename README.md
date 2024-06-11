# DL-CSI
```bash
usage: main.py [-h] [--batchsize BATCHSIZE] [--nepoch NEPOCH] [--lr LR] [--test_only] [--ckpt CKPT] [--device {cpu,cuda}]
               [--datadir DATADIR] [--label LABEL]

Deep Learning for CSI

options:
  -h, --help            show this help message and exit
  --batchsize BATCHSIZE
                        Batch Size. Default is 200.
  --nepoch NEPOCH       number training epoch. Default is 1000
  --lr LR               learning rate. Default is 0.001
  --test_only           evaluate checkpoint
  --ckpt CKPT           checkpoint to be evaluated, only effective with --test_only
  --device {cpu,cuda}   any of ['cpu', 'cuda']
  --datadir DATADIR     path to the root of COST2100 raw data
  --label LABEL         Optional label for the run. Defaults to None. Output directory is always prefixed by date time
```

### Run
Following reproduces CSINet.
```
python main.py
```

### Usage 
1. add new model architecture to `model/` folder
2. revise `create_model` in `main.py` to ensure model creation
3. revise training hyperparameters (LR) as needed 

### Dependency
pip install torch wandb numpy scipy tensorboard

### TODO
[] hardcoded to indoor environment data pipeline, to generalize
