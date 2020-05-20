import torch

import train
from unet import Classic_UNet


# Declare Global Settings
root_dir = "/home/luigifcruz/Sandbox/DataScience/vimeo_triplet/sequences"
input_size = (448, 256, 3)
learning_rate = 0.01
min_lr = 0.0000001
patience = 2
batch_size = 16
epochs = 250
percentages = [0.85, 0.10, 0.05] # Training, Validation, Testing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Declare Interation Settings
interations = [
    {"model": Classic_UNet, "save_dir": 'runs/A2', "net_size": 6},
]

if __name__ == "__main__":
    for data in interations:
        train.run(data['model'], data['net_size'], root_dir, data['save_dir'],
                  input_size, batch_size, learning_rate, min_lr, epochs, percentages, device, patience)
