import torch
import torchvision
import os
import matplotlib.pyplot as plt
from model import Net
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from tqdm import tqdm
import argparse
import time

CURR_PATH = os.getcwd()

def create_data_loaders(rank, world_size, batch_size):
  transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize((0.1307,), (0.3081,))
      ])

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(f'{CURR_PATH}/files/', train=False, download=True,
                              transform=transform),
    batch_size=batch_size_test, shuffle=True)

  # Download and initialize MNIST train dataset
  train_dataset = torchvision.datasets.MNIST(f'{CURR_PATH}/files/',
                                download=True,
                                train=True,
                                transform=transform)

  # Create distributed sampler pinned to rank
  sampler = torch.utils.data.DistributedSampler(train_dataset,
                              num_replicas=world_size,
                              rank=rank,
                              shuffle=True,  # May be True
                              seed=42)

  # Wrap train dataset into DataLoader
  train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=False,  # Must be False!
                            num_workers=4,
                            sampler=sampler,
                            pin_memory=True)

  return train_loader, test_loader

def plot(train_counter, train_losses, test_counter, test_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig.savefig('images/train_test_curve_dist.png')

def main(rank: int,
         epochs: int,
         model: torch.nn.Module,
         train_loader: torch.utils.data.DataLoader,
         test_loader: torch.utils.data.DataLoader) -> torch.nn.Module:
    model = DDP(model)

    # initialize optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss = torch.nn.CrossEntropyLoss()

    # train the model
    for i in range(epochs):
        model.train()
        train_loader.sampler.set_epoch(i)

        epoch_loss = 0
        # train the model for one epoch
        pbar = tqdm(train_loader)
        batch_idx = -1
        for data, target in pbar:
            batch_idx+=1
            optimizer.zero_grad()
            y_hat = model(data)
            batch_loss = loss(y_hat, target)
            batch_loss.backward()
            optimizer.step()
            batch_loss_scalar = batch_loss.item()
            epoch_loss += batch_loss_scalar / data.shape[0]
            pbar.set_description(f'training batch_loss={batch_loss_scalar:.4f}')
            train_losses.append(batch_loss_scalar)
            train_counter.append((batch_idx*64) + ((i)*len(train_loader.dataset)))

        # calculate validation loss
        with torch.no_grad():
            model.eval()
            val_loss = 0
            correct = 0
            pbar = tqdm(test_loader)
            for data, target in pbar:
                y_hat = model(data)
                batch_loss = loss(y_hat, target)
                batch_loss_scalar = batch_loss.item()
                val_loss += batch_loss_scalar

                pred = y_hat.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

                pbar.set_description(f'validation batch_loss={batch_loss_scalar:.4f}')
        
        val_loss /= len(test_loader.dataset)
        test_losses.append(val_loss)

        accuracy = 100. * correct / len(test_loader.dataset)
        time_elapsed = time.time() - t0
        print(f"Epoch={i}, train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, accuracy={accuracy:.2f}, \
          time_elapsed={time_elapsed:.4f}")

    return model.module

if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.02
    momentum = 0.5
    log_interval = 10

    # distributed training settings
    world_size = 2
    rank = 1 # for instance-1
    batch_size = int(batch_size_train/world_size)

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    epochs = 6

    rank = args.local_rank
    world_size = 2

    os.environ['MASTER_ADDR'] = '10.128.0.2'      # ip of your master process (rank 0 instance) should here
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    train_loader, test_loader = create_data_loaders(rank, world_size, batch_size)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(epochs)]

    model = main(rank=rank,
                 epochs=epochs,
                 model=Net(),
                 train_loader=train_loader,
                 test_loader=test_loader)

    plot(train_counter, train_losses, test_counter, test_losses)

    if rank == 0:
        torch.save(model.state_dict(), 'model.pt')
