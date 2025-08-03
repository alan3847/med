#### Install requirements
Create a new virtual environment and install all dependencies by:
```
pip install -r requirement.txt
```
#### Data preparation
Download the origin dataset from their corresponding official website.

Enter the `dataset_conversion` fold and find the dataset you want to use and the corresponding dimension (2d or 3d)

Edit the `src_path` and `tgt_path` the in `xxxdataset.py`, where the `src_path` is the path to the origin dataset, and `tgt_path` is the target path to store the processed dataset.

Then, `python xxxdataset.py`

After processing is finished, put the processed dataset into `dataset/` folder or use a soft link.

#### Configuration
Enter `config/xxxdataset/` and find the model and dimension (2d or 3d) you want to use. The training details, e.g. model hyper-parameters, training epochs, learning rate, optimizer, data augmentation, etc., can be altered here. You can try your own congiguration or use the default configure, the one we used in the MedFormer paper, which should have a decent performance. The only thing to care is the `data_root`, make sure it points to the processed dataset directory.

#### Training
We can start training after the data and configuration is done. Several arguments can be parsed in the command line, see in the `get_parser()` function in the `train.py` and `train_ddp.py`. You need to specify the model, the dimension, the dataset, whether use pretrain weights, batch size, and the unique experiment name. Our code will find the corresponding configuration and dataset for training.

Here is an example to train 3D MedFormer with one gpu on AMOS MR:

`python train.py --model medformer --dimension 3d --dataset amos_mr --batch_size 3 --unique_name amos_mr_3d_medformer --gpu 0`

This command will start the cross validation on AMOS MR. The training loss and evaluation performance will be logged by tensorboard. You can find them in the `log/dataset/unique_name` folder. All the standard output, the configuration, and model weigths will be saved in the `exp/dataset/unique_name` folder. The results of cross validation will be saved in `exp/dataset/unique_name/cross_validation.txt`.

Besides training with a single GPU, we also provide distributed training (DDP) and automatic mixed precision (AMP) training in the `train_ddp.py`. The `train_ddp.py` is the same as `train.py` except it supports DDP and AMP. We recomend you to start with `train.py` to make sure the whole train and eval pipeline is correct, and then use `train_ddp.py` for faster training or larger batch size.

Example of using DDP:

`python train_ddp.py --model medformer --dimension 3d --dataset amos_mr --batch_size 16 --unique_name amos_mr_3d_medformer_ddp --gpu 0,1,2,3`

Example of using DDP and AMP:

`python train_ddp.py --model medformer --dimension 3d --dataset amos_mr --batch_size 32 --unique_name amos_mr_3d_medformer_ddp_amp --gpu 0,1,2,3 --amp`

We have not fully benchmark if AMP can speed up training, but AMP can reduce the GPU memory consumption a lot.
