cd /N/slate/tnn3/DucHGA/meteor-foundation
python Src_model/train_job_base.py \
    --seed 42 \
    --batch_size 32 \
    --num_workers 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --max_epochs 2 \
    --mode -1 \
    --checkpoint last best_r2 best_rmse \
    --inp_dir ./Data/merra/dataset/sample_dataset \
    --out_dir ./model_result/test_result \
    --export_result regression_results