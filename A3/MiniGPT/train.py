"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig


MODEL = "minigpt"  # bigram or minigpt

if MODEL == "bigram":
    config = BigramConfig
    model = BigramLanguageModel(config)
elif MODEL == "minigpt":
    config = MiniGPTConfig
    model = MiniGPT(config)
else:
    raise ValueError("Invalid model name")


# Initialize wandb if you want to use it
if config.to_log:
    wandb.init(project="dl2_proj3")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_dataset = TinyStoriesDataset(
    config.path_to_data,
    mode="train",
    context_length=config.context_length,
)
eval_dataset = TinyStoriesDataset(
    config.path_to_data, mode="test", context_length=config.context_length
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=config.batch_size, pin_memory=True
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))


if not Path.exists(config.save_path):
    Path.mkdir(MiniGPTConfig.save_path, parents=True, exist_ok=True)


### ==================== START OF YOUR CODE ==================== ###
"""
You are required to implement the training loop for the model.

Please keep the following in mind:
- You will need to define an appropriate loss function for the model.
- You will need to define an optimizer for the model.
- You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
- It is recommended that you save the model weights every `config.save_iterations` iterations you can also just save the model with the best training loss.

Please check the config file to see the different configurations you can set for the model.
NOTE : 
The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
not a required part of the assignment. 
Feel free to experiment with the parameters and I would be happy to talk to you about them if interested :)
"""

import random

model = model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

model.train()

for i, (context, target) in enumerate(train_dataloader):
    context, target = context.to(device), target.to(device)

    optimizer.zero_grad()

    logits = model(context)
    
    logits = logits.view(-1, config.vocab_size)
    target = target.view(-1)
    
    loss_val = loss(logits, target)

    loss_val.backward()
    optimizer.step()

    if i % config.log_interval == 0:
        print(f"Iter: {i}, Loss: {loss_val.item()}")
        if config.to_log:
            wandb.log({"iter": i, "loss": loss_val.item()})

    if i % config.save_iterations == 0:
        torch.save(model.state_dict(), config.save_path / f"model_{i}.pth")
        print(f"Model saved at {config.save_path / f'model_{i}.pth'}")

    if i % (config.log_interval * 5) == 0:
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for j, (context, target) in enumerate(eval_dataloader):
                context, target = context.to(device), target.to(device)
                logits = model(context)

                logits = logits.view(-1, config.vocab_size)
                target = target.view(-1)

                loss_val = loss(logits, target)
                total_loss += loss_val.item()
                if j >= 2:
                  break
            avg_loss = total_loss / 3
            
            print(f"Eval Loss: {avg_loss}")

            if config.to_log:
                wandb.log({"eval_loss": avg_loss})

    
    if i >= config.max_iter:
        break
