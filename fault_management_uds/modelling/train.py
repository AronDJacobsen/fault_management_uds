import torch
import pytorch_lightning as pl
from fault_management_uds.utilities import get_accelerator



def train_model(model, train_loader, val_loader, callbacks, logger, training_args, save_folder):
    accelerator = get_accelerator()

    # Skip optimizer if requested
    if training_args['skip_optimizer']:
        checkpoint_path = save_folder / 'best_model.ckpt'
        # Save the model checkpoint with additional information
        torch.save({
            'state_dict': model.state_dict(),
            # Add other items if needed, like optimizer state, epoch, etc.
        }, checkpoint_path)
        # add to the callbacks
        callbacks[0].best_model_path = checkpoint_path
        callbacks[0].last_model_path = checkpoint_path
        callbacks[0].best_k_models = [checkpoint_path]
        return model, callbacks, logger


    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=training_args['max_epochs'],
        max_steps=training_args['max_steps'],
        log_every_n_steps=training_args['log_every_n_steps'],
        val_check_interval=training_args['val_check_interval'],  
        check_val_every_n_epoch=1,  # Ensure it evaluates at least once per epoch
        callbacks=callbacks,
        logger=logger,
        accelerator=accelerator,
        devices="auto",
        )
    trainer.fit(model, train_loader, val_loader)
    return model, callbacks, logger 



