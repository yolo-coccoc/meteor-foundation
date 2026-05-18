import os

from lightning.pytorch.callbacks import ModelCheckpoint

def save_checkpoint_callback(out_dir):
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        save_last=True,
    )

    loss_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename='best_loss',
        monitor='val_step_loss',
        mode='min',
        save_top_k=1,
        verbose=True
    )

    f1_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename='best_rmse',
        monitor='val_epoch_rmse',
        mode='min',
        save_top_k=1,
        verbose=True   
    )

    freq_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename='model_epoch_{epoch:02d}',
        every_n_epochs=20,
        save_top_k=-1,    
    )
    
    return [last_checkpoint_callback, loss_checkpoint_callback, f1_checkpoint_callback, freq_checkpoint_callback]