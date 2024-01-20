import torch
from torch import nn


def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    Hint: use nn.Linear
    """
    model = nn.Linear(input_size, output_size)
    return model


def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def fit_regression_model(X, y):
    learning_rate = 0.0001
    num_epochs = 10000
    input_features = X.shape[1]
    output_features = y.shape[1]
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")

    for epoch in range(1, num_epochs):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
        if abs(previous_loss - loss.item()) < 0.0001:
            break
        previous_loss = loss.item()
    return model, loss

