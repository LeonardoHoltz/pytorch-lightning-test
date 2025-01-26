# Training Hyperparameters
learning_rate = 0.001
num_epochs = 3

# Dataset
data_dir = "dataset/"
num_workers = 4
batch_size = 64
input_size = 784
num_classes = 10

# Compute related
accelerator = "gpu"
devices = [0]
precision = 32

# Trainer callbacks
callbacks = [
    dict(
        type="RichProgressBar",
        leave=True
    ),
    dict(
        type="RichModelSummary",
    )
]

# default Lightning Trainer
trainer = dict(
    type="Trainer",
    accelerator=accelerator,
    devices=devices,
    min_epochs=1,
    max_epochs=num_epochs,
    precision=precision,
    callbacks=callbacks
)