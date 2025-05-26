#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Machine Learning with Real Frameworks
Practical examples using Qiskit and PennyLane

Installation:
pip install qiskit qiskit-aer pennylane pennylane-qiskit torch numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# QISKIT QUANTUM MACHINE LEARNING
# ============================================================================

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit import execute, Aer, transpile
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.algorithms.optimizers import SPSA, COBYLA, L_BFGS_B
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import Estimator
    
    # Qiskit Machine Learning (if available)
    try:
        from qiskit_machine_learning.neural_networks import CircuitQNN
        from qiskit_machine_learning.algorithms.classifiers import VQC
        QISKIT_ML_AVAILABLE = True
    except ImportError:
        print("Qiskit Machine Learning not available. Install with: pip install qiskit-machine-learning")
        QISKIT_ML_AVAILABLE = False
    
    QISKIT_AVAILABLE = True
    print("✅ Qiskit available")
    
except ImportError:
    print("❌ Qiskit not available. Install with: pip install qiskit qiskit-aer")
    QISKIT_AVAILABLE = False

# ============================================================================
# PENNYLANE QUANTUM MACHINE LEARNING
# ============================================================================

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    import torch
    import torch.nn as nn
    
    PENNYLANE_AVAILABLE = True
    print("✅ PennyLane available")
    
except ImportError:
    print("❌ PennyLane not available. Install with: pip install pennylane torch")
    PENNYLANE_AVAILABLE = False

# ============================================================================
# QISKIT IMPLEMENTATIONS
# ============================================================================

class QiskitQuantumNeuralNetwork:
    """Quantum Neural Network using Qiskit"""
    
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = Aer.get_backend('statevector_simulator')
        
        # Create parameterized circuit
        self.params = ParameterVector('θ', n_qubits * n_layers * 3)
        self.circuit = self._create_circuit()
        
    def _create_circuit(self) -> QuantumCircuit:
        """Create parameterized quantum circuit"""
        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0
        
        for layer in range(self.n_layers):
            # Rotation layer
            for qubit in range(self.n_qubits):
                qc.rx(self.params[param_idx], qubit)
                param_idx += 1
                qc.ry(self.params[param_idx], qubit)
                param_idx += 1
                qc.rz(self.params[param_idx], qubit)
                param_idx += 1
            
            # Entangling layer
            for qubit in range(self.n_qubits - 1):
                qc.cx(qubit, qubit + 1)
            
            # Ring connectivity
            if self.n_qubits > 2:
                qc.cx(self.n_qubits - 1, 0)
        
        return qc
    
    def encode_data(self, data: np.ndarray) -> QuantumCircuit:
        """Encode classical data into quantum circuit"""
        encoding_circuit = QuantumCircuit(self.n_qubits)
        
        # Amplitude encoding (simplified)
        for i, value in enumerate(data[:self.n_qubits]):
            angle = value * np.pi  # Scale to [0, π]
            encoding_circuit.ry(angle, i)
        
        return encoding_circuit
    
    def run_circuit(self, params: np.ndarray, input_data: np.ndarray) -> np.ndarray:
        """Execute quantum circuit with given parameters"""
        # Create full circuit
        full_circuit = self.encode_data(input_data)
        full_circuit = full_circuit.compose(self.circuit)
        
        # Bind parameters
        param_dict = {self.params[i]: params[i] for i in range(len(params))}
        bound_circuit = full_circuit.bind_parameters(param_dict)
        
        # Execute
        job = execute(bound_circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector()
        
        # Extract probabilities
        probabilities = np.abs(statevector) ** 2
        return probabilities
    
    def cost_function(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Cost function for training"""
        total_cost = 0.0
        
        for i, (x_i, y_i) in enumerate(zip(X, y)):
            output = self.run_circuit(params, x_i)
            # Use first few probabilities as features
            prediction = output[:len(y_i)]
            cost = np.sum((prediction - y_i) ** 2)
            total_cost += cost
        
        return total_cost / len(X)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, maxiter: int = 100):
        """Train the quantum neural network"""
        print("Training Qiskit Quantum Neural Network...")
        
        # Initialize parameters
        initial_params = np.random.uniform(0, 2*np.pi, len(self.params))
        
        # Use SPSA optimizer
        optimizer = SPSA(maxiter=maxiter)
        
        # Optimize
        result = optimizer.minimize(
            fun=lambda params: self.cost_function(params, X_train, y_train),
            x0=initial_params
        )
        
        self.trained_params = result.x
        print(f"Training completed. Final cost: {result.fun:.6f}")
        
        return result
    
    def predict(self, X_test: np.ndarray) -> List[np.ndarray]:
        """Make predictions"""
        predictions = []
        for x in X_test:
            output = self.run_circuit(self.trained_params, x)
            predictions.append(output[:4])  # Return first 4 probabilities
        return predictions

class QiskitQuantumAutoencoder:
    """Quantum Autoencoder using Qiskit"""
    
    def __init__(self, n_qubits: int, n_latent: int):
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.backend = Aer.get_backend('statevector_simulator')
        
        # Create encoder and decoder circuits
        self.encoder_params = ParameterVector('enc', n_qubits * 2)
        self.decoder_params = ParameterVector('dec', n_qubits * 2)
        
        self.encoder_circuit = self._create_encoder()
        self.decoder_circuit = self._create_decoder()
    
    def _create_encoder(self) -> QuantumCircuit:
        """Create encoder circuit"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Encoding layer
        for i in range(self.n_qubits):
            qc.ry(self.encoder_params[i], i)
        
        # Compression layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Second encoding layer
        for i in range(self.n_qubits):
            qc.ry(self.encoder_params[i + self.n_qubits], i)
        
        return qc
    
    def _create_decoder(self) -> QuantumCircuit:
        """Create decoder circuit"""
        qc = QuantumCircuit(self.n_qubits)
        
        # Decoding layer (reverse of encoder)
        for i in range(self.n_qubits):
            qc.ry(self.decoder_params[i], i)
        
        # Decompression layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Final decoding layer
        for i in range(self.n_qubits):
            qc.ry(self.decoder_params[i + self.n_qubits], i)
        
        return qc
    
    def encode_classical_data(self, data: np.ndarray) -> QuantumCircuit:
        """Encode classical data"""
        qc = QuantumCircuit(self.n_qubits)
        for i, value in enumerate(data[:self.n_qubits]):
            qc.ry(value * np.pi, i)
        return qc
    
    def run_autoencoder(self, enc_params: np.ndarray, dec_params: np.ndarray, 
                       input_data: np.ndarray) -> np.ndarray:
        """Run full autoencoder"""
        # Data encoding
        full_circuit = self.encode_classical_data(input_data)
        
        # Encoder
        encoder = self.encoder_circuit.bind_parameters(
            {self.encoder_params[i]: enc_params[i] for i in range(len(enc_params))}
        )
        full_circuit = full_circuit.compose(encoder)
        
        # Decoder
        decoder = self.decoder_circuit.bind_parameters(
            {self.decoder_params[i]: dec_params[i] for i in range(len(dec_params))}
        )
        full_circuit = full_circuit.compose(decoder)
        
        # Execute
        job = execute(full_circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector()
        
        return np.abs(statevector) ** 2
    
    def train_autoencoder(self, training_data: List[np.ndarray], maxiter: int = 50):
        """Train the quantum autoencoder"""
        print("Training Qiskit Quantum Autoencoder...")
        
        def cost_function(params):
            enc_params = params[:len(self.encoder_params)]
            dec_params = params[len(self.encoder_params):]
            
            total_cost = 0.0
            for data in training_data:
                output = self.run_autoencoder(enc_params, dec_params, data)
                # Reconstruction loss
                target_probs = np.zeros(len(output))
                target_probs[:len(data)] = data / np.sum(data) if np.sum(data) > 0 else 0
                cost = np.sum((output - target_probs) ** 2)
                total_cost += cost
            
            return total_cost / len(training_data)
        
        # Initialize parameters
        n_params = len(self.encoder_params) + len(self.decoder_params)
        initial_params = np.random.uniform(0, 2*np.pi, n_params)
        
        # Optimize
        optimizer = SPSA(maxiter=maxiter)
        result = optimizer.minimize(cost_function, initial_params)
        
        # Store trained parameters
        self.trained_enc_params = result.x[:len(self.encoder_params)]
        self.trained_dec_params = result.x[len(self.encoder_params):]
        
        print(f"Training completed. Final cost: {result.fun:.6f}")
        return result

# ============================================================================
# PENNYLANE IMPLEMENTATIONS
# ============================================================================

class PennyLaneQuantumNeuralNetwork:
    """Quantum Neural Network using PennyLane"""
    
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Initialize parameters
        self.n_params = n_qubits * n_layers * 3
        self.params = pnp.random.uniform(0, 2*pnp.pi, self.n_params, requires_grad=True)
        
        # Create quantum node
        self.qnode = qml.QNode(self._circuit, self.dev)
    
    def _circuit(self, params, inputs):
        """Quantum circuit definition"""
        # Encode input data
        for i, x in enumerate(inputs[:self.n_qubits]):
            qml.RY(x * pnp.pi, wires=i)
        
        # Variational layers
        param_idx = 0
        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                qml.RX(params[param_idx], wires=wire)
                param_idx += 1
                qml.RY(params[param_idx], wires=wire)
                param_idx += 1
                qml.RZ(params[param_idx], wires=wire)
                param_idx += 1
            
            # Entangling gates
            for wire in range(self.n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
        
        # Return expectations
        return [qml.expval(qml.PauliZ(i)) for i in range(min(4, self.n_qubits))]
    
    def forward(self, inputs):
        """Forward pass"""
        return self.qnode(self.params, inputs)
    
    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        """Train the network"""
        print("Training PennyLane Quantum Neural Network...")
        
        optimizer = qml.AdamOptimizer(stepsize=learning_rate)
        
        def cost_function(params):
            total_cost = 0.0
            for x, y in zip(X_train, y_train):
                pred = self.qnode(params, x)
                cost = pnp.sum((pnp.array(pred) - pnp.array(y[:len(pred)])) ** 2)
                total_cost += cost
            return total_cost / len(X_train)
        
        costs = []
        for epoch in range(epochs):
            self.params, cost = optimizer.step_and_cost(cost_function, self.params)
            costs.append(cost)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.6f}")
        
        return costs
    
    def predict(self, X_test):
        """Make predictions"""
        predictions = []
        for x in X_test:
            pred = self.forward(x)
            predictions.append(pred)
        return predictions

class PennyLaneQuantumAutoencoder:
    """Quantum Autoencoder using PennyLane"""
    
    def __init__(self, n_qubits: int, n_latent: int):
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        
        # Quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Parameters
        self.n_params = n_qubits * 4  # Simplified for demo
        self.params = pnp.random.uniform(0, 2*pnp.pi, self.n_params, requires_grad=True)
        
        # Quantum nodes
        self.encoder_qnode = qml.QNode(self._encoder_circuit, self.dev)
        self.decoder_qnode = qml.QNode(self._decoder_circuit, self.dev)
        self.autoencoder_qnode = qml.QNode(self._autoencoder_circuit, self.dev)
    
    def _encoder_circuit(self, params, inputs):
        """Encoder circuit"""
        # Data encoding
        for i, x in enumerate(inputs[:self.n_qubits]):
            qml.RY(x * pnp.pi, wires=i)
        
        # Encoding layers
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Return latent representation
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_latent)]
    
    def _decoder_circuit(self, params, latent):
        """Decoder circuit"""
        # Initialize with latent representation
        for i, val in enumerate(latent[:self.n_qubits]):
            angle = pnp.arccos(pnp.clip(val, -1, 1))  # Convert expectation to angle
            qml.RY(angle, wires=i)
        
        # Decoding layers
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        for i in range(self.n_qubits):
            qml.RY(params[i + self.n_qubits], wires=i)
        
        # Return reconstruction
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def _autoencoder_circuit(self, params, inputs):
        """Full autoencoder circuit"""
        # Data encoding
        for i, x in enumerate(inputs[:self.n_qubits]):
            qml.RY(x * pnp.pi, wires=i)
        
        # Encoder
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Decoder
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        for i in range(self.n_qubits):
            qml.RY(params[i + self.n_qubits], wires=i)
        
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def encode(self, inputs):
        """Encode data to latent space"""
        return self.encoder_qnode(self.params, inputs)
    
    def decode(self, latent):
        """Decode from latent space"""
        return self.decoder_qnode(self.params, latent)
    
    def forward(self, inputs):
        """Full autoencoder forward pass"""
        return self.autoencoder_qnode(self.params, inputs)
    
    def train(self, training_data, epochs=50, learning_rate=0.01):
        """Train the autoencoder"""
        print("Training PennyLane Quantum Autoencoder...")
        
        optimizer = qml.AdamOptimizer(stepsize=learning_rate)
        
        def cost_function(params):
            total_cost = 0.0
            for data in training_data:
                reconstruction = self.autoencoder_qnode(params, data)
                # Reconstruction loss
                target = pnp.tanh(pnp.array(data))  # Convert to [-1,1] range
                cost = pnp.sum((pnp.array(reconstruction) - target[:len(reconstruction)]) ** 2)
                total_cost += cost
            return total_cost / len(training_data)
        
        costs = []
        for epoch in range(epochs):
            self.params, cost = optimizer.step_and_cost(cost_function, self.params)
            costs.append(cost)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.6f}")
        
        return costs

# ============================================================================
# HYBRID TORCH + QUANTUM MODELS
# ============================================================================

class HybridTorchQuantumModel(nn.Module):
    """Hybrid PyTorch + Quantum model"""
    
    def __init__(self, n_qubits: int, n_classical: int):
        super().__init__()
        
        # Classical layers
        self.classical_net = nn.Sequential(
            nn.Linear(n_classical, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Tanh()
        )
        
        # Quantum layer (PennyLane)
        if PENNYLANE_AVAILABLE:
            self.quantum_layer = PennyLaneQuantumNeuralNetwork(n_qubits, 2)
        
        # Output layer
        self.output = nn.Linear(8 + 4, 1)  # Classical + quantum features
        
    def forward(self, classical_input, quantum_input):
        """Forward pass through hybrid model"""
        # Classical processing
        classical_features = self.classical_net(classical_input)
        
        # Quantum processing
        if PENNYLANE_AVAILABLE:
            quantum_features = torch.tensor(
                self.quantum_layer.forward(quantum_input.numpy()), 
                dtype=torch.float32
            )
        else:
            quantum_features = torch.zeros(4)
        
        # Combine features
        combined = torch.cat([classical_features, quantum_features])
        
        # Final output
        output = self.output(combined)
        return output

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_qiskit_qml():
    """Demonstrate Qiskit quantum machine learning"""
    if not QISKIT_AVAILABLE:
        print("Qiskit not available. Skipping Qiskit demo.")
        return
    
    print("🔵 Qiskit Quantum Machine Learning Demo")
    print("-" * 40)
    
    # Create sample data
    X_train = [
        np.array([0.2, 0.3, 0.4]),
        np.array([0.7, 0.8, 0.1]),
        np.array([0.1, 0.9, 0.5]),
        np.array([0.6, 0.2, 0.8])
    ]
    
    y_train = [
        np.array([0.8, 0.6, 0.4, 0.2]),
        np.array([0.2, 0.4, 0.6, 0.8]),
        np.array([0.9, 0.1, 0.5, 0.5]),
        np.array([0.3, 0.7, 0.3, 0.7])
    ]
    
    # Train Quantum Neural Network
    qnn = QiskitQuantumNeuralNetwork(n_qubits=3, n_layers=2)
    qnn.train(X_train, y_train, maxiter=30)
    
    # Test prediction
    test_input = np.array([0.4, 0.5, 0.6])
    prediction = qnn.predict([test_input])
    print(f"Test prediction: {prediction[0][:4]}")
    
    # Train Quantum Autoencoder
    print("\nTraining Quantum Autoencoder...")
    autoencoder_data = [
        np.array([0.8, 0.2, 0.6, 0.4]),
        np.array([0.1, 0.9, 0.3, 0.7]),
        np.array([0.5, 0.5, 0.8, 0.2])
    ]
    
    qautoencoder = QiskitQuantumAutoencoder(n_qubits=4, n_latent=2)
    qautoencoder.train_autoencoder(autoencoder_data, maxiter=20)
    print("Qiskit autoencoder training completed!")

def demonstrate_pennylane_qml():
    """Demonstrate PennyLane quantum machine learning"""
    if not PENNYLANE_AVAILABLE:
        print("PennyLane not available. Skipping PennyLane demo.")
        return
    
    print("\n🟢 PennyLane Quantum Machine Learning Demo")
    print("-" * 40)
    
    # Create sample data
    X_train = [
        pnp.array([0.2, 0.3, 0.4]),
        pnp.array([0.7, 0.8, 0.1]),
        pnp.array([0.1, 0.9, 0.5]),
        pnp.array([0.6, 0.2, 0.8])
    ]
    
    y_train = [
        pnp.array([0.8, 0.6, 0.4, 0.2]),
        pnp.array([0.2, 0.4, 0.6, 0.8]),
        pnp.array([0.9, 0.1, 0.5, 0.5]),
        pnp.array([0.3, 0.7, 0.3, 0.7])
    ]
    
    # Train Quantum Neural Network
    qnn = PennyLaneQuantumNeuralNetwork(n_qubits=3, n_layers=2)
    costs = qnn.train(X_train, y_train, epochs=50, learning_rate=0.01)
    
    # Test prediction
    test_input = pnp.array([0.4, 0.5, 0.6])
    prediction = qnn.predict([test_input])
    print(f"Test prediction: {prediction[0]}")
    
    # Train Quantum Autoencoder
    print("\nTraining PennyLane Quantum Autoencoder...")
    autoencoder_data = [
        pnp.array([0.8, 0.2, 0.6, 0.4]),
        pnp.array([0.1, 0.9, 0.3, 0.7]),
        pnp.array([0.5, 0.5, 0.8, 0.2])
    ]
    
    qautoencoder = PennyLaneQuantumAutoencoder(n_qubits=4, n_latent=2)
    autoencoder_costs = qautoencoder.train(autoencoder_data, epochs=30)
    
    # Test compression
    test_data = pnp.array([0.7, 0.3, 0.8, 0.2])
    encoded = qautoencoder.encode(test_data)
    reconstructed = qautoencoder.forward(test_data)
    
    print(f"Original: {test_data}")
    print(f"Encoded: {encoded}")
    print(f"Reconstructed: {reconstructed}")

def demonstrate_hybrid_model():
    """Demonstrate hybrid quantum-classical model"""
    if not (PENNYLANE_AVAILABLE and torch):
        print("PyTorch or PennyLane not available. Skipping hybrid demo.")
        return
    
    print("\n🟡 Hybrid Quantum-Classical Model Demo")
    print("-" * 40)
    
    # Create hybrid model
    model = HybridTorchQuantumModel(n_qubits=3, n_classical=4)
    
    # Sample data
    classical_data = torch.randn(10, 4)
    quantum_data = torch.rand(10, 3)
    targets = torch.randn(10, 1)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(20):
        optimizer.zero_grad()
        
        total_loss = 0
        for i in range(len(classical_data)):
            output = model(classical_data[i], quantum_data[i])
            loss = criterion(output, targets[i].unsqueeze(0))
            total_loss += loss
        
        avg_loss = total_loss / len(classical_data)
        avg_loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss.item():.6f}")
    
    print("Hybrid model training completed!")

def main():
    """Main demonstration function"""
    print("🚀 Quantum Machine Learning with Real Frameworks")
    print("=" * 55)
    
    # Check available frameworks
    print("Framework Availability:")
    print(f"  Qiskit: {'✅' if QISKIT_AVAILABLE else '❌'}")
    print(f"  PennyLane: {'✅' if PENNYLANE_AVAILABLE else '❌'}")
    print(f"  PyTorch: {'✅' if 'torch' in globals() else '❌'}")
    
    # Run demonstrations
    try:
        demonstrate_qiskit_qml()
    except Exception as e:
        print(f"Qiskit demo error: {e}")
    
    try:
        demonstrate_pennylane_qml()
    except Exception as e:
        print(f"PennyLane demo error: {e}")
    
    try:
        demonstrate_hybrid_model()
    except Exception as e:
        print(f"Hybrid model demo error: {e}")
    
    print("\n" + "=" * 55)
    print("🎯 Key Takeaways:")
    print("1. Qiskit provides low-level quantum control")
    print("2. PennyLane offers automatic differentiation")
    print("3. Hybrid models combine quantum and classical advantages")
    print("4. Start with simulators, then move to real hardware")
    print("5. Quantum advantage depends on problem structure")

if __name__ == "__main__":
    main()
