# docker-gcn
Attempted implementation of ''Semi-Supervised Classification with Graph Convolutional Networks'' paper using PyTorch inside docker.. 

You need to have nvidia-docker installed.

Run the following command from inside command line from the root directory of the project.

To run for single iteration with no early stopping tolerance and without batch normalization
```sudo make build && sudo make run-1```

To run for single iteration with early stopping tolerance but without batch normalization
```sudo make build && sudo make run-2```

To run for single iteration with no early stopping tolerance but with batch normalization
```sudo make build && sudo make run-3```

To run for 100 iteration with no early stopping tolerance but with batch normalization
```sudo make build && sudo make run-multi```
