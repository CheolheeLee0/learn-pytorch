import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def test_tensor_operations(device):
    print("\nTesting Tensor Operations:")
    
    # Create a tensor and move it to the GPU
    tensor = torch.randn(1000, 1000, device=device)
    print(f"Tensor shape: {tensor.shape}")
    
    # Perform a simple operation to ensure that it's using the GPU
    result = tensor @ tensor
    print(f"Result shape: {result.shape}")

def test_data_transfer(device):
    print("\nTesting Data Transfer:")
    
    # Create a tensor on CPU and transfer it to GPU
    tensor_cpu = torch.randn(1000, 1000)
    tensor_gpu = tensor_cpu.to(device)
    print(f"Tensor on GPU: {tensor_gpu.device}")

    # Perform an operation and transfer the result back to CPU
    result_gpu = tensor_gpu @ tensor_gpu
    result_cpu = result_gpu.cpu()
    print(f"Result transferred to CPU: {result_cpu.device}")

def test_model_training(device):
    print("\nTesting Model Training:")
    
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(1000, 1000)
            self.fc2 = nn.Linear(1000, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create dummy data
    data = torch.randn(64, 1000, device=device)
    target = torch.randint(0, 10, (64,), device=device)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Training step completed. Loss: {loss.item()}")

def test_multi_gpu(device_count):
    print("\nTesting Multi-GPU Setup:")
    
    if device_count < 2:
        print("Multiple GPUs not available.")
        return

    # Define a simple model for multi-GPU testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(1000, 1000)
            self.fc2 = nn.Linear(1000, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Use DataParallel to handle multiple GPUs
    model = SimpleModel()
    model = nn.DataParallel(model)
    model.to(f'cuda:{0}')  # Move model to GPU 0
    
    # Create dummy data
    data = torch.randn(64, 1000).to(f'cuda:{0}')
    target = torch.randint(0, 10, (64,), device=f'cuda:{0}')
    
    # Training step
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Multi-GPU training step completed. Loss: {loss.item()}")

def check_gpu_memory(device):
    print("\nChecking GPU Memory Usage:")
    
    # Print GPU memory statistics
    print(f"Memory Allocated: {torch.cuda.memory_allocated(device)} bytes")
    print(f"Memory Cached: {torch.cuda.memory_reserved(device)} bytes")

def test_h100():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices found. Please check your GPU installation.")

    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    device = torch.device("cuda:0")
    
    # Run various tests
    test_tensor_operations(device)
    test_data_transfer(device)
    test_model_training(device)
    test_multi_gpu(device_count)
    check_gpu_memory(device)

if __name__ == "__main__":
    test_h100()
