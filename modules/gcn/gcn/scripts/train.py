import os
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.tensorboard import SummaryWriter

import gcn
from gcn.learning.models import nGCN


# Define the training loop
def train(model, data, index, train_writer, test_writer, scheduler, device='cpu', args=None):
    torch.manual_seed(8616)
    
    optimizer.zero_grad()

    out = model.forward(data['x'], data['edge'], device)
    loss = model.loss(out[data['train_mask']], data['y'][data['train_mask']].to(device),
                      train_writer, index)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        test_out = model.forward(data['x'], data['edge'], device=device)
        test_loss = model.loss(test_out[data['test_mask']], data['y'][data['test_mask']].to(device),
                               test_writer, index)
    # Log the learning rate
    test_writer.add_scalar('learning_rate',
                           scheduler.get_last_lr()[-1],
                           index)
    
    return loss, test_loss


if __name__ == "__main__":
    # Get the necessary hyper-parameters and other required parameters in
    # the form of arguments.
    args = gcn.utils.get_parser()
    
    train_writer_str = 'train'
    train_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, train_writer_str))
    test_writer_str = 'test'
    test_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, test_writer_str))

    cuda = torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    df_ratings = pd.read_csv('/data/ratings.csv')
    df_movies = pd.read_csv('/data/movies.csv')

    merged = pd.merge(df_ratings, df_movies, on='movieId', how='left')

    R = pd.pivot_table(data=merged, index='userId', columns='title', values='rating')

    # Calculate the average rating per user to use a feature for the nodes
    user_avg_rating = R.apply(lambda row: np.mean(row[row.notnull()]), axis=1)

    # Calculate the average rating per movie to use a feature for the nodes
    movie_avg_rating = R.apply(lambda col: np.mean(col[col.notnull()]), axis=0)

    # Initialize the model and optimizer
    device = torch.device("cuda" if cuda else "cpu")
    print(device)

    limit = args.num_steps
    learning_rate = args.lr
    lr_decay = args.weight_decay
    epoch = args.epochs

    model = nGCN(hidden_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch, gamma=lr_decay)

    NUM_ROWs = R.shape[0]
    NUM_COLs = R.shape[1]

    train_loss_vec = []
    test_loss_vec = []

    counter = 0
    while counter < limit:
        row = random.randint(0, NUM_ROWs - 1)
        col = random.randint(0, NUM_COLs - 1)
        if 0 <= R.iloc[row, col] <= 5:
            datum = gcn.utils.get_graph(R, row, col, user_avg_rating, movie_avg_rating)
            if datum is None:
                continue
            loss, test_loss = train(
                model, datum, counter, train_writer, test_writer, 
                scheduler, device, args)
            scheduler.step()
            counter += 1
            print(f"Step {counter}, Train-Loss: {loss:.4f}")
            if counter % 10 == 0:
                print(f"Step {counter}, Test-Loss: {test_loss:.4f}")

    # Saving the model after training
    torch.save(model.state_dict(),
               os.path.join(args.save_dir, 'model.pt'))
