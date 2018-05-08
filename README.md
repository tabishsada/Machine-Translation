# Machine-Translation

## Transformer based model

File description:
* transformer_hyperparams.py includes all hyper parameters that are needed.
* transformer_preprocess.py creates vocabulary files for the source and the target.
* transformer_data_load.py contains functions regarding loading and batching data.
* transformer_modules.py has all building blocks for encoder/decoder networks.
* transformer_train.py has the model.
* transformer_live.py is for evaluation.

Training and Evaluation:
1. Adjust hyper parameters in transformer_hyperparams.py.
2. Run transformer_preprocess.py to generate vocab files.
3. Run transformer_train.py which will generate checkpoints.
4. Run transformer_live.py to evaluate and get BLEU scores.



## RNN based NMT
Script to get training data:

```
srun nmt/scripts/wmt16_en_de.sh nmt_data
```

To train 4 layer LSTMs for 1024 units:

```
cd nmt
python nmt.py \
    --src=de --tgt=en \
    --vocab_prefix=/scratch/vbg221/tmp3/nmt_data/vocab.bpe.32000  \
    --train_prefix=/scratch/vbg221/tmp3/nmt_data/train.tok.clean.bpe.32000 \
    --dev_prefix=/scratch/vbg221/tmp3/nmt_data/newstest2013.tok.bpe.32000  \
    --test_prefix=scratch/vbg221/tmp3/nmt_data/newstest2015.tok.bpe.32000 \
    --out_dir=/scratch/vbg221/tmp3/nmt_vbg221 \
    --num_train_steps=35000 \
    --steps_per_stats=100 \
    --num_layers=4 \
    --num_units=1024 \
    --dropout=0.2 \
    --metrics=bleu \
    --num_gpus=1 \
    --learning_rate=0.1 \
    --batch_size=128 \
    --colocate_gradients_with_ops=true \
    --encoder_type=gnmt \
    --forget_bias=1.0 \
    --infer_batch_size=32 \
    --init_weight=0.1 \
    --max_gradient_norm=5.0 \
    --num_buckets=5 \
    --optimizer=sgd \
    --residual=true \
    --subword_option=bpe \
    --beam_width=10 \
    --length_penalty_weight=1.0
```
