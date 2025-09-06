import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models_HLS import *


def train_model(X, Y, ModelClass, porder, band, kern_type,
                hidden_dim=32, lr=0.02, epochs=10000, patience=700, stop_threshold=1e-4):
    """
    Trains Model1_new using a precomputed covariance matrix.

    Args:
        X: Tensor of shape (d, J)
        Y: Tensor of shape (y_shape_0, y_shape_1)
        ModelClass: typically Model1_new
        porder, band, kern_type: covariance hyperparameters

    Returns:
        model: trained model
        losses: training loss history
        cov_matrix: used covariance matrix
    """
    d, J = X.shape

    # Step 1: Precompute long-run covariance matrix (J x J)
    with torch.no_grad():
        cov_matrix, _, _ = cov_l(X, porder, band, kern_type)

    # Step 2: Initialize model with fixed cov_matrix
    model = ModelClass(
        x_shape_0=d,
        x_shape_1=J,
        cov_dim=J,
        hidden_dim=hidden_dim,
        y_shape_0=Y.shape[0],
        y_shape_1=Y.shape[1],
        cov_matrix=cov_matrix
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_model_state = None
    losses = []
    wait = 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        Y_hat = model(X)
        loss = criterion(Y_hat, Y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if abs(loss.item()) < stop_threshold:
            print(f"Stopping early at Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            break

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at Epoch [{epoch+1}/{epochs}] - No improvement for {patience} epochs.")
            break

        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print("\nRestoring best model weights...")
    model.load_state_dict(best_model_state)

    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    return model, losses, cov_matrix
