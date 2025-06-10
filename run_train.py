from src.modeling import DocformerV2Model
from src.configuration import DocformerV2Config
from src.dataset import create_feature
import os
from datasets import load_dataset
from safetensors.torch import save_file, load_file

from transformers import AutoTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class RVLCDIPData(Dataset):

    def __init__(self, images, labels, tokenizer, target_size=(300, 400), max_len=512, transform=None):
        super().__init__()

        self.images = images
        self.labels = labels
        self.target_size = target_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image.filename = f"text_{idx}.png"
        encoding = create_feature(
            image,
            self.tokenizer,
            add_batch_dim=False,
            target_size=self.target_size,
            max_seq_length=self.max_len,
            path_to_save=None,
            save_to_disk=False,
            apply_mask_for_mlm=False,
            use_ocr = True
        )

        encoding['label'] = torch.as_tensor(label, dtype=torch.int64) # int64
        return encoding
    
def collate_fn(batches):

    '''
    A function for the dataloader to return a batch dict of given keys

    batches: List of dictionary
    '''

    dict_batches = {}

    for i in batches:
        for (key, value) in i.items():
            if key not in dict_batches:
                dict_batches[key] = []
            dict_batches[key].append(value)

    for key in list(dict_batches.keys()):
        if key == "pixel_values":
            dict_batches[key] = torch.cat(dict_batches[key], dim=0)
        else:
            dict_batches[key] = torch.stack(dict_batches[key], dim=0)

    return dict_batches


class DataModule:
    
    def __init__(self, train_dataset, val_dataset, batch_size: int = 4):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, 
                          collate_fn = collate_fn, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, 
                          collate_fn = collate_fn, shuffle = False)
    

class DocFormerV2ForClassification(nn.Module):
  
    def __init__(self, config):
      super().__init__()

      self.config = config
      len_label = 16
      
      self.encoder = DocformerV2Model(config=config)
      self.flatten = nn.Flatten()
      self.linear_layer = nn.Linear(in_features=config.max_seq_length*config.hidden_size, out_features=len_label)  ## Number of Classes
      self.softmax = nn.Softmax(dim=-1)

    def forward(self, batch_dict):

      out = self.encoder(batch_dict)
      out = self.flatten(out)
      out = self.linear_layer(out)
      out = self.softmax(out)
      return out
    

import torch
from torch import nn

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        config,
        datamodule,
        device: str = "cpu",
        epoch_per_device: int = 1, 
        lr: float = 0.5
    ):
        self.model = model
        # self.model.to(dtype=torch.float16)
        self.training_loader = datamodule.train_dataloader()
        self.validation_loader = datamodule.val_dataloader()
        self.config = config
        self.EPOCHS = epoch_per_device
        # Initializing in a separate cell so we can easily add more epochs to the same run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('runs/fashion_trainer_{}'.format(self.timestamp))

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.training_loader):
            # Every data instance is an input + label pair
            inputs = data
            labels = data['label']

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if (i+1) % 5 == 0:
                last_loss = running_loss / 5 # loss per batch , at 5 batch take 5 min.
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    
    def train(self):
        epoch_number = 0
        for epoch in range(self.EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, self.writer)


            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            self.writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            self.writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(self.timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1
    

def main():
    config = DocformerV2Config()
    
    path_model = "google/mt5-large"
    tokenizer = AutoTokenizer.from_pretrained(path_model)

    dataset = load_dataset(
        path="aharley/rvl_cdip", 
        data_dir="/Users/thirawarit/Projects/Dataset/", 
        split=("train", "validation", "test"),
        trust_remote_code=True,
    )
    train_dataset, val_dataset, test_dataset = dataset

    train_dataset = train_dataset.select(range(1000))
    val_dataset = val_dataset.select(range(1000))
    test_dataset = test_dataset.select(range(1000))

    train_ds = RVLCDIPData(train_dataset['image'], train_dataset['label'], 
                        tokenizer,
                        )
    val_ds = RVLCDIPData(val_dataset['image'], val_dataset['label'], 
                        tokenizer,
                        )
    # test_ds = RVLCDIPData(test_dataset['image'], test_dataset['label'], 
    #                     tokenizer, max_len=config.max_seq_len,
    #                     )

    datamodule = DataModule(train_ds, val_ds)

    model = DocFormerV2ForClassification(config)

    trainer = Trainer(model, config, datamodule)
    trainer.train()

if __name__ == "__main__":
    main()