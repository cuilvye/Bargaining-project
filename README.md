# Inferring Private Valuations from Behavioral Data in Bilateral Sequential Bargaining 

This repository is the specific implementation for the published paper on IJCAI-2023.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Sellers Clustering

### Clustering 

To perform K-Loss clustering on synthetic datasets, run the two commands:

```train
python KLoss_clustering_on_syndata.py --fold_lambda 'SynData_Uniform' --rand_idx 1 --iters 10 --save_root <save root path> 
```

```train
python KLoss_clustering_on_syndata.py --fold_lambda 'SynData_Skellam_vs_54_vb_54' --rand_idx 1 --iters 10 --save_root <save root path> 
```

To perform K-Loss clustering on real data, run this command:

```train
python KLoss_clustering_on_realdata.py --k 3 --pattern_iter 50 --save_root <save root path> 
```


## Valuation Inference

### Training 

To train the BLUE-C models with synthetic and real bargaining datasets, run the two commands:
```train
python train_BLUE_C_on_syndata.py --fold_lambda <syndata type> --alpha 0.6 --k 3 --split_idx 1 --save_root <save root path>
```

```train
python train_BLUE_C_on_realdata.py --cluster_path <clustering result path> --k 3 --iter_num 7 --alpha 0.6 --split_idx 1 --save_root <save root path>
```

To train the BLUE models with synthetic and real bargaining datasets, run the two commands:

```train
python train_BLUE_on_syndata.py --fold_lambda <syndata type> --alpha 0.6 --split_idx 1 --save_root <save root path>
```

```train
python train_BLUE_on_realdata.py --alpha 0.6 --split_idx 1 --save_root <save root path>
```


### Evaluation

To evaluate inference models on synthetic datasets, run the commands:

```eval
python eval_on_syn_testdata.py --fold_lambda 'SynData_Uniform' --split_idx 1 --model_root <trained models root>
```

```eval
python eval_on_syn_testdata.py --fold_lambda 'SynData_Skellam_vs_54_vb_54' --split_idx 1 --model_root <trained models root>
```

To evaluate inference models on the real dataset, run this command:

```eval
python eval_on_real_testdata.py --k 3 --iter_num 7 --split_idx 1 --model_root <trained models root>
```

