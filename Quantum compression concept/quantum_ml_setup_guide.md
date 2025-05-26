# Quantum Machine Learning - Quick Setup Guide

## 🚀 Step-by-Step Installation

### 1. **Base Python Environment**
```bash
# Create virtual environment
python -m venv quantum_ml_env
source quantum_ml_env/bin/activate  # Linux/Mac
# quantum_ml_env\Scripts\activate   # Windows

# Update pip
pip install --upgrade pip
```

### 2. **Core Quantum Frameworks**

#### **Qiskit (IBM Quantum)**
```bash
# Basic Qiskit
pip install qiskit qiskit-aer

# Machine Learning extensions
pip install qiskit-machine-learning

# Optimization tools
pip install qiskit-optimization

# IBM Quantum access
pip install qiskit-ibmq-provider
```

#### **PennyLane (Differentiable Programming)**
```bash
# Core PennyLane
pip install pennylane

# Device plugins
pip install pennylane-sf          # Strawberry Fields
pip install pennylane-qiskit      # Qiskit backend
pip install pennylane-cirq        # Cirq backend
pip install pennylane-qulacs      # Fast simulator
```

#### **Cirq (Google Quantum)**
```bash
pip install cirq
pip install cirq-google  # For Google Quantum hardware
```

### 3. **Classical ML Libraries**
```bash
# Deep learning
pip install torch torchvision
pip install tensorflow

# Scientific computing
pip install numpy scipy matplotlib
pip install scikit-learn pandas

# Visualization
pip install seaborn plotly
```

### 4. **Complete Installation (All-in-One)**
```bash
pip install qiskit qiskit-aer qiskit-machine-learning \
            pennylane pennylane-qiskit pennylane-sf \
            cirq torch tensorflow numpy scipy \
            matplotlib seaborn jupyter notebook
```

## 🔧 **Environment Configuration**

### **1. Jupyter Notebook Setup**
```bash
# Install Jupyter
pip install jupyter notebook jupyterlab

# Quantum extensions
pip install qiskit[visualization]
jupyter nbextension enable --py widgetsnbextension

# Start Jupyter
jupyter notebook
```

### **2. IBM Quantum Account Setup**
```python
# In Python/Jupyter
from qiskit import IBMQ

# Save account (get token from quantum-computing.ibm.com)
IBMQ.save_account('YOUR_IBM_QUANTUM_TOKEN')

# Load account
IBMQ.load_account()
provider = IBMQ.get_provider()

# List available backends
print(provider.backends())
```

### **3. Basic Environment Test**
```python
# Test script - save as test_quantum_env.py
import qiskit
import pennylane as qml
import torch
import numpy as np

print("✅ Environment Test")
print(f"Qiskit version: {qiskit.__version__}")
print(f"PennyLane version: {qml.version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

# Test quantum device
dev = qml.device('default.qubit', wires=2)
print("✅ PennyLane device created successfully")

# Test Qiskit backend
backend = qiskit.Aer.get_backend('qasm_simulator')
print("✅ Qiskit backend loaded successfully")

print("🚀 All systems ready for Quantum ML!")
```

## 📚 **Learning Resources**

### **Tutorials & Documentation**
- **Qiskit Textbook**: [qiskit.org/textbook](https://qiskit.org/textbook/)
- **PennyLane Demos**: [pennylane.ai/qml](https://pennylane.ai/qml/)
- **Cirq Documentation**: [quantumai.google/cirq](https://quantumai.google/cirq)

### **Online Courses**
- **IBM Qiskit Global Summer School**
- **Xanadu Quantum Machine Learning Course**
- **Microsoft Quantum Development Kit**

### **Research Papers**
- "Quantum Machine Learning" - Biamonte et al.
- "Variational Quantum Eigensolver" - Peruzzo et al.
- "Quantum Neural Networks" - Killoran et al.

## 🏗️ **Project Structure Template**

```
quantum_ml_project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_classical_baseline.ipynb
│   └── 03_quantum_model.ipynb
├── src/
│   ├── quantum_models/
│   │   ├── __init__.py
│   │   ├── qnn.py
│   │   └── autoencoder.py
│   ├── classical_models/
│   └── utils/
├── experiments/
├── results/
├── requirements.txt
└── README.md
```

## 🎯 **First Quantum ML Project Template**

### **requirements.txt**
```
qiskit>=0.45.0
qiskit-aer>=0.12.0
qiskit-machine-learning>=0.7.0
pennylane>=0.32.0
pennylane-qiskit>=0.30.0
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
jupyter>=1.0.0
scikit-learn>=1.0.0
```

### **Basic Quantum Neural Network (starter.py)**
```python
import pennylane as qml
from pennylane import numpy as np

# Quantum device
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev)
def quantum_neural_network(inputs, weights):
    # Encode inputs
    for i, x in enumerate(inputs):
        qml.RY(x * np.pi, wires=i)
    
    # Variational layer
    for i in range(4):
        qml.RY(weights[i], wires=i)
    
    # Entangling layer
    for i in range(3):
        qml.CNOT(wires=[i, i+1])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Usage example
inputs = np.array([0.1, 0.2, 0.3, 0.4])
weights = np.random.uniform(0, 2*np.pi, 4)
output = quantum_neural_network(inputs, weights)
print(f"QNN output: {output}")
```

## 🐛 **Common Issues & Solutions**

### **Issue 1: Installation Conflicts**
```bash
# Solution: Use fresh environment
conda create -n quantum_ml python=3.9
conda activate quantum_ml
pip install --no-cache-dir qiskit pennylane torch
```

### **Issue 2: Jupyter Widget Issues**
```bash
# Enable widgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### **Issue 3: Memory Issues with Simulators**
```python
# Use smaller quantum systems
dev = qml.device('default.qubit', wires=6)  # Max ~10 qubits on laptop

# Or use lightning simulator
dev = qml.device('lightning.qubit', wires=12)  # Faster for larger systems
```

### **Issue 4: Slow Training**
```python
# Use parameter-shift rule efficiently
optimizer = qml.AdamOptimizer(stepsize=0.01)
# Batch your training data
# Use fewer qubits for prototyping
```

## 🔬 **Development Workflow**

### **1. Prototype Phase**
```python
# Start simple
qubits = 3
layers = 1
data_size = 50

# Test classical baseline first
from sklearn.neural_network import MLPRegressor
classical_model = MLPRegressor(hidden_layer_sizes=(10,))
```

### **2. Quantum Development**
```python
# Gradual complexity increase
dev = qml.device('default.qubit', wires=qubits)

# Simple ansatz first
def simple_ansatz(weights):
    for i in range(qubits):
        qml.RY(weights[i], wires=i)
```

### **3. Scaling Up**
```python
# Move to better simulators
dev = qml.device('lightning.qubit', wires=8)

# Then to real hardware
dev = qml.device('qiskit.ibmq', wires=5, backend='ibmq_manila')
```

## 📊 **Performance Monitoring**

### **Track Key Metrics**
```python
import time
import psutil

def monitor_training(cost_function, params):
    start_time = time.time()
    start_memory = psutil.virtual_memory().used / 1024**3
    
    result = cost_function(params)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used / 1024**3
    
    print(f"Time: {end_time - start_time:.2f}s")
    print(f"Memory: {end_memory - start_memory:.2f}GB")
    
    return result
```

## 🚨 **Hardware Considerations**

### **Local Development**
- **CPU**: 8+ cores recommended
- **RAM**: 16GB minimum, 32GB preferred
- **GPU**: Optional, but helps with classical parts

### **Cloud Options**
- **IBM Quantum Experience**: Free tier available
- **Google Colab**: Good for prototyping
- **AWS Braket**: Pay-per-use quantum simulators/hardware
- **Azure Quantum**: Microsoft's quantum cloud

## 🎉 **You're Ready!**

With this setup, you can:
- ✅ Run quantum algorithms on simulators
- ✅ Access real quantum hardware
- ✅ Build hybrid quantum-classical models
- ✅ Experiment with quantum machine learning
- ✅ Contribute to quantum AI research

**Next steps**: Run the provided tutorial codes and start experimenting!