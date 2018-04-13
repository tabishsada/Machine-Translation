# Machine-Translation

Data can be found at:


To train 4 layer nmt model:

python -m nmt.nmt \
    --src=vi --tgt=en \
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
