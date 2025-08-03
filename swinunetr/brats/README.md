# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

# Pre-trained Models

We provide the self-supervised pre-trained weights for Swin UNETR backbone (CVPR paper [1]) in this <a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"> link</a>.
In the following, we describe steps for pre-training the model from scratch.

# Training

## Distributed Multi-GPU Pre-Training

To pre-train a `Swin UNETR` encoder using multi-gpus:

```bash
python -m torch.distributed.launch --nproc_per_node=<Num-GPUs> --master_port=11223 main.py
--batch_size=<Batch-Size> --num_steps=<Num-Steps> --lrdecay --eval_num=<Eval-Num> --logdir=<Exp-Num> --lr=<Lr>
```

The following was used to pre-train Swin UNETR on 8 X 32G V100 GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=11223 main.py
--batch_size=1 --num_steps=100000 --lrdecay --eval_num=500 --lr=6e-6 --decay=0.1
```

## Single GPU Pre-Training with Gradient Check-pointing

To pre-train a `Swin UNETR` encoder using a single gpu with gradient-checkpointing and a specified patch size:

```bash
python main.py --use_checkpoint --batch_size=<Batch-Size> --num_steps=<Num-Steps> --lrdecay
--eval_num=<Eval-Num> --logdir=<Exp-Num> --lr=<Lr> --roi_x=<Roi_x> --roi_y=<Roi_y> --roi_z=<Roi_z>
```
