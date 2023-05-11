import os
import torch
import numpy as np
import pandas as pd
import random

import gcn
from gcn.learning.models import nGCN

if __name__ == "__main__":
    # Get the necessary hyper-parameters and other required parameters in
    # the form of arguments.
    args = gcn.utils.get_parser()
    # Write starting seed to the log file
    logfile = os.path.join(args.save_dir, args.logfile)

    cuda = torch.cuda.is_available()
    random.seed(args.seed)
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
    model.load_state_dict(torch.load(args.network_file, map_location=device))
    model.eval()
    model.to(device)

    NUM_ROWs = R.shape[0]
    NUM_COLs = R.shape[1]
    counter = 0
    limit = 5000
    while counter < limit:
        row = random.randint(0, NUM_ROWs - 1)
        col = random.randint(0, NUM_COLs - 1)
        if 0 <= R.iloc[row, col] <= 5:
            datum = gcn.utils.get_graph(R, row, col, user_avg_rating, movie_avg_rating)
            if datum is None:
                continue
            with torch.no_grad():
                out = model.forward(datum['x'], datum['edge'], device)
                loss = model.loss(out[datum['test_mask']], datum['y'][datum['test_mask']].to(device))
                loss = loss.detach().cpu().numpy()
                out = out.detach().cpu().numpy()
                actual = datum['y'].numpy()[1]
                predicted = out[1][0]
                counter += 1
                # print(f"Prediction: #[{counter}], Loss: {loss:.4f}, Actual: {actual:.4f}, Predicted: {predicted:.4f}")
                with open(logfile, "a+") as f:
                    f.write(f"[#]: {counter}"
                            f" | Loss: {loss:.4f}"
                            f" | Actual: {actual:.4f}"
                            f" | Predicted: {predicted:.4f}\n")
