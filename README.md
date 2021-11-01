# MLGNN

Multi-level Graph Neural Network for Drug-Drug Interaction Prediction.

You can get the drug external multimodal features from [DrugBank](https://go.drugbank.com/drugs).

To run the code, you need the following dependencies:

```
pytorch==1.9.0
networkx==2.4
scipy==1.4.1
chemprop==1.1.0
```

You can run this framwork by

```
python main_miner.py 
--save_dir=checkpoints/ChChMiner 
--data_path=dataProcess/ChChMiner_train.csv 
--separate_val_path=dataProcess/ChChMiner_valid.csv 
--separate_test_path=dataProcess/ChChMiner_test.csv 
--vocab_path=dataProcess/drug_list_miner.csv
```
