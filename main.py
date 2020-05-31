import torch

import train
from unet import Classic_UNet


# Declare Global Settings
root_dir = "/home/luigifcruz/Sandbox/DataScience/vimeo_triplet/sequences"
input_size = (448, 256, 3)
learning_rate = 0.0001
min_lr = 0.000001
patience = 1
batch_size = 16
epochs = 350
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Declare Interation Settings
interations = [
    {"model": Classic_UNet, "save_dir": 'runs/B5', "net_size": 16},
]

if __name__ == "__main__":
    for data in interations:
        train.run(data['model'], data['net_size'], root_dir, data['save_dir'],
                  input_size, batch_size, learning_rate, min_lr, epochs, device, patience)
