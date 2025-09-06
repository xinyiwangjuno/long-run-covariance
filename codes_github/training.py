import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import *

def train_model(X, Y, WeightNetwork, ModelClass, LR_cov_X, m, q, trunc_q, hidden_dim=32, lr=0.02, epochs=10000, patience=700, stop_threshold=1e-4):
    
    # Define the weight network and the main model
    #weight_net = WeightNetwork(m_init=m, q_init=q)  # Neural network to estimate weights
    weight_net = WeightNetwork(m_init=m)
    model = ModelClass(
        X.shape[0], X.shape[1], X.shape[1], hidden_dim, Y.shape[0], Y.shape[1], weight_net, trunc_q
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()) + list(weight_net.parameters()), lr=lr)
    
    # Initialize tracking for best model
    best_loss = float('inf')
    best_model_state = None
    best_weight_net_state = None
    best_cov_matrix = None
    wait = 0
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Recompute covariance matrix with updated weights
        cov_matrix, cov_matrix_w1, cov_matrix3 = LR_cov_X(X, weight_net, trunc_q)
        
        Y_hat = model(X)  # Predicted Y(t)
        loss = criterion(Y_hat, Y)  # Compute L2 loss
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Check if the loss is sufficiently small
        if abs(loss.item()) < stop_threshold:
            print(f"Stopping early at Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            break  # Exit loop when loss is small enough
        
        # Save the best model and covariance matrix
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()  # Save model state
            best_weight_net_state = weight_net.state_dict()  # Save weight_net state
            best_cov_matrix = cov_matrix.clone()  # Save best covariance matrix
            wait = 0  # Reset patience counter
        else:
            wait += 1  # Increment patience counter if loss does not improve
        
        if wait >= patience:
            print(f"Early stopping at Epoch [{epoch+1}/{epochs}] - No improvement for {patience} epochs.")
            break  # Stop training if no improvement for `patience` epochs
        
        # Print loss every 200 epochs
        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Restore the best model and weight network
    print("\nRestoring best model and weight network...")
    model.load_state_dict(best_model_state)
    weight_net.load_state_dict(best_weight_net_state)

    # Extract learned weights
    learned_weights = {name: param.detach().cpu().numpy() for name, param in weight_net.named_parameters()}
    
    
    # Plot the loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.show()
    
    return model, weight_net,learned_weights
