import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Generate dummy data
def generate_dummy_data(num_samples=1000, input_size=10, num_classes=2):
    np.random.seed(42)
    X = np.random.rand(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

# Explainable AI method: Feature importance (simple gradient-based explanation)
def explain_model(model, input_data):
    input_data.requires_grad = True
    output = model(input_data)
    target_class = output.argmax(dim=1)
    target_output = output[range(len(target_class)), target_class]
    target_output.backward(torch.ones_like(target_output))
    explanations = input_data.grad
    return explanations

# Proxy task: Evaluate user decision-making based on explanations
def proxy_task(explanations, true_labels):
    # Simulate a user's ability to make decisions based on explanations
    # Here, we assume the user can correctly classify if the explanation aligns with the true label
    simulated_user_decisions = (explanations.sum(dim=1) > 0).long()
    accuracy = (simulated_user_decisions == true_labels).float().mean().item()
    return accuracy

if __name__ == '__main__':
    # Parameters
    input_size = 10
    hidden_size = 20
    output_size = 2
    num_samples = 1000
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.01

    # Generate dummy data
    X, y = generate_dummy_data(num_samples, input_size, output_size)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Test the model and generate explanations
    test_X, test_y = generate_dummy_data(num_samples=200, input_size=input_size, num_classes=output_size)
    test_X_tensor = torch.tensor(test_X, requires_grad=True)
    test_y_tensor = torch.tensor(test_y)

    # Generate explanations
    explanations = explain_model(model, test_X_tensor)

    # Evaluate proxy task
    accuracy = proxy_task(explanations, test_y_tensor)
    print(f"Proxy Task Accuracy (simulated user decision-making): {accuracy:.4f}")