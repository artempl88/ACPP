#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Machine Learning Tutorial
Complete guide to creating quantum ML models for data compression

This tutorial covers:
1. Basic quantum circuits
2. Variational Quantum Circuits (VQC)
3. Quantum Neural Networks
4. Hybrid quantum-classical models
5. Application to data compression
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import random
import math

# Try to import quantum libraries (install with: pip install qiskit pennylane cirq)
try:
    import qiskit
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import execute, Aer
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.algorithms.optimizers import SPSA, COBYLA, Adam
    QISKIT_AVAILABLE = True
except ImportError:
    print("Qiskit not available. Install with: pip install qiskit")
    QISKIT_AVAILABLE = False

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    print("PennyLane not available. Install with: pip install pennylane")
    PENNYLANE_AVAILABLE = False

# Fallback quantum simulator for educational purposes
class SimpleQuantumSimulator:
    """
    Simple quantum simulator for educational purposes
    Simulates basic quantum operations without external dependencies
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.state = np.zeros(self.n_states, dtype=complex)
        self.state[0] = 1.0  # Initialize to |00...0⟩
    
    def reset(self):
        """Reset to |00...0⟩ state"""
        self.state = np.zeros(self.n_states, dtype=complex)
        self.state[0] = 1.0
    
    def rx(self, qubit: int, theta: float):
        """Rotation around X-axis"""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        
        new_state = np.zeros_like(self.state)
        for i in range(self.n_states):
            # Check if qubit is 0 or 1
            if (i >> qubit) & 1 == 0:  # qubit is 0
                new_state[i] += cos_half * self.state[i]
                new_state[i | (1 << qubit)] += -1j * sin_half * self.state[i]
            else:  # qubit is 1
                new_state[i] += cos_half * self.state[i]
                new_state[i & ~(1 << qubit)] += -1j * sin_half * self.state[i]
        
        self.state = new_state
    
    def ry(self, qubit: int, theta: float):
        """Rotation around Y-axis"""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        
        new_state = np.zeros_like(self.state)
        for i in range(self.n_states):
            if (i >> qubit) & 1 == 0:  # qubit is 0
                new_state[i] += cos_half * self.state[i]
                new_state[i | (1 << qubit)] += sin_half * self.state[i]
            else:  # qubit is 1
                new_state[i] += cos_half * self.state[i]
                new_state[i & ~(1 << qubit)] += -sin_half * self.state[i]
        
        self.state = new_state
    
    def rz(self, qubit: int, theta: float):
        """Rotation around Z-axis"""
        for i in range(self.n_states):
            if (i >> qubit) & 1 == 1:  # qubit is 1
                self.state[i] *= np.exp(1j * theta / 2)
            else:  # qubit is 0
                self.state[i] *= np.exp(-1j * theta / 2)
    
    def cx(self, control: int, target: int):
        """CNOT gate"""
        new_state = np.zeros_like(self.state)
        for i in range(self.n_states):
            if (i >> control) & 1 == 1:  # control is 1
                # Flip target
                new_i = i ^ (1 << target)
                new_state[new_i] = self.state[i]
            else:  # control is 0
                new_state[i] = self.state[i]
        
        self.state = new_state
    
    def measure_all(self) -> List[int]:
        """Measure all qubits and return classical result"""
        probabilities = np.abs(self.state) ** 2
        result = np.random.choice(self.n_states, p=probabilities)
        
        # Convert to binary representation
        bits = []
        for i in range(self.n_qubits):
            bits.append((result >> i) & 1)
        
        return bits
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities"""
        return np.abs(self.state) ** 2

class QuantumNeuralNetwork:
    """
    Quantum Neural Network using parameterized quantum circuits
    """
    
    def __init__(self, n_qubits: int, n_layers: int, use_entanglement: bool = True):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_entanglement = use_entanglement
        
        # Initialize parameters
        self.n_parameters = n_qubits * n_layers * 3  # 3 rotation angles per qubit per layer
        if use_entanglement:
            self.n_parameters += (n_qubits - 1) * n_layers  # Entangling gates
        
        self.parameters = np.random.uniform(0, 2*np.pi, self.n_parameters)
        self.simulator = SimpleQuantumSimulator(n_qubits)
    
    def create_circuit(self, parameters: np.ndarray, input_data: np.ndarray) -> None:
        """
        Create parameterized quantum circuit
        
        Args:
            parameters: Circuit parameters
            input_data: Classical input data encoded into quantum state
        """
        self.simulator.reset()
        param_idx = 0
        
        # Encode classical data into quantum state
        for i, data_point in enumerate(input_data[:self.n_qubits]):
            # Amplitude encoding
            angle = data_point * np.pi / 2  # Normalize to [0, π/2]
            self.simulator.ry(i, angle)
        
        # Variational layers
        for layer in range(self.n_layers):
            # Rotation gates for each qubit
            for qubit in range(self.n_qubits):
                self.simulator.rx(qubit, parameters[param_idx])
                param_idx += 1
                self.simulator.ry(qubit, parameters[param_idx])
                param_idx += 1
                self.simulator.rz(qubit, parameters[param_idx])
                param_idx += 1
            
            # Entangling gates
            if self.use_entanglement:
                for qubit in range(self.n_qubits - 1):
                    # Parameterized entangling gate
                    self.simulator.cx(qubit, qubit + 1)
                    if param_idx < len(parameters):
                        self.simulator.rz(qubit + 1, parameters[param_idx])
                        param_idx += 1
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass through quantum circuit"""
        self.create_circuit(self.parameters, input_data)
        probabilities = self.simulator.get_probabilities()
        
        # Extract relevant features (e.g., first few probabilities)
        return probabilities[:min(8, len(probabilities))]
    
    def compute_loss(self, inputs: List[np.ndarray], targets: List[np.ndarray]) -> float:
        """Compute loss function"""
        total_loss = 0.0
        for input_data, target in zip(inputs, targets):
            output = self.forward(input_data)
            # Mean squared error
            loss = np.mean((output[:len(target)] - target) ** 2)
            total_loss += loss
        
        return total_loss / len(inputs)
    
    def train(self, train_inputs: List[np.ndarray], train_targets: List[np.ndarray], 
              epochs: int = 100, learning_rate: float = 0.01):
        """
        Train the quantum neural network
        """
        print(f"Training Quantum Neural Network...")
        print(f"Parameters: {len(self.parameters)}, Epochs: {epochs}")
        
        losses = []
        
        for epoch in range(epochs):
            # Compute gradients using parameter shift rule
            gradients = np.zeros_like(self.parameters)
            
            for i in range(len(self.parameters)):
                # Forward shift
                self.parameters[i] += np.pi / 2
                loss_plus = self.compute_loss(train_inputs, train_targets)
                
                # Backward shift
                self.parameters[i] -= np.pi
                loss_minus = self.compute_loss(train_inputs, train_targets)
                
                # Restore parameter
                self.parameters[i] += np.pi / 2
                
                # Parameter shift rule gradient
                gradients[i] = (loss_plus - loss_minus) / 2
            
            # Update parameters
            self.parameters -= learning_rate * gradients
            
            # Compute current loss
            current_loss = self.compute_loss(train_inputs, train_targets)
            losses.append(current_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {current_loss:.6f}")
        
        return losses

class QuantumAutoencoder:
    """
    Quantum Autoencoder for data compression
    """
    
    def __init__(self, n_input_qubits: int, n_latent_qubits: int):
        self.n_input = n_input_qubits
        self.n_latent = n_latent_qubits
        self.n_total = n_input_qubits  # Total qubits needed
        
        # Encoder: maps n_input to n_latent
        self.encoder = QuantumNeuralNetwork(n_input_qubits, 3, True)
        
        # Decoder: maps n_latent back to n_input
        self.decoder = QuantumNeuralNetwork(n_input_qubits, 3, True)
    
    def encode(self, input_data: np.ndarray) -> np.ndarray:
        """Encode input data to compressed representation"""
        encoded = self.encoder.forward(input_data)
        return encoded[:self.n_latent]  # Take only latent dimensions
    
    def decode(self, latent_data: np.ndarray) -> np.ndarray:
        """Decode compressed representation back to original space"""
        # Pad latent data to full input size
        padded_input = np.zeros(self.n_input)
        padded_input[:len(latent_data)] = latent_data
        
        decoded = self.decoder.forward(padded_input)
        return decoded[:self.n_input]  # Return original dimensionality
    
    def train_autoencoder(self, training_data: List[np.ndarray], epochs: int = 50):
        """Train the quantum autoencoder"""
        print("Training Quantum Autoencoder...")
        
        # Create training targets (same as inputs for autoencoder)
        train_targets = training_data.copy()
        
        # Train encoder and decoder jointly
        combined_params = np.concatenate([self.encoder.parameters, self.decoder.parameters])
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for input_data in training_data:
                # Forward pass through encoder
                encoded = self.encode(input_data)
                
                # Forward pass through decoder
                reconstructed = self.decode(encoded)
                
                # Reconstruction loss
                loss = np.mean((reconstructed - input_data) ** 2)
                total_loss += loss
            
            avg_loss = total_loss / len(training_data)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Reconstruction Loss = {avg_loss:.6f}")
        
        return avg_loss

class HybridQuantumClassicalModel:
    """
    Hybrid model combining quantum and classical components
    """
    
    def __init__(self, n_qubits: int, n_classical_features: int):
        self.n_qubits = n_qubits
        self.n_classical = n_classical_features
        
        # Quantum component
        self.quantum_layer = QuantumNeuralNetwork(n_qubits, 2, True)
        
        # Classical neural network component
        self.classical_weights = np.random.normal(0, 0.1, (n_classical_features, 8))
        self.classical_bias = np.zeros(8)
        
        # Output layer
        quantum_output_dim = min(8, 2**n_qubits)
        total_features = quantum_output_dim + 8  # Quantum + classical features
        self.output_weights = np.random.normal(0, 0.1, total_features)
        self.output_bias = 0.0
    
    def forward(self, quantum_input: np.ndarray, classical_input: np.ndarray) -> float:
        """Forward pass through hybrid model"""
        # Quantum processing
        quantum_features = self.quantum_layer.forward(quantum_input)
        
        # Classical processing
        classical_features = np.tanh(np.dot(classical_input, self.classical_weights) + self.classical_bias)
        
        # Combine quantum and classical features
        combined_features = np.concatenate([quantum_features, classical_features])
        
        # Final output
        output = np.dot(combined_features, self.output_weights) + self.output_bias
        return output
    
    def train_hybrid(self, quantum_inputs: List[np.ndarray], 
                    classical_inputs: List[np.ndarray], 
                    targets: List[float], epochs: int = 50):
        """Train hybrid quantum-classical model"""
        print("Training Hybrid Quantum-Classical Model...")
        
        learning_rate = 0.01
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Simple gradient descent (simplified for demo)
            for q_input, c_input, target in zip(quantum_inputs, classical_inputs, targets):
                # Forward pass
                output = self.forward(q_input, c_input)
                
                # Loss
                loss = (output - target) ** 2
                total_loss += loss
                
                # Simplified parameter updates (in practice, use proper backprop)
                error = output - target
                
                # Update output layer
                quantum_features = self.quantum_layer.forward(q_input)
                classical_features = np.tanh(np.dot(c_input, self.classical_weights) + self.classical_bias)
                combined_features = np.concatenate([quantum_features, classical_features])
                
                self.output_weights -= learning_rate * error * combined_features
                self.output_bias -= learning_rate * error
            
            avg_loss = total_loss / len(targets)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

class QuantumCompressionModel:
    """
    Quantum machine learning model specifically for data compression
    """
    
    def __init__(self, input_dim: int, compression_ratio: float = 0.5):
        self.input_dim = input_dim
        self.compressed_dim = max(1, int(input_dim * compression_ratio))
        
        # Determine number of qubits needed
        self.n_qubits = max(3, int(np.ceil(np.log2(input_dim))))
        
        # Quantum autoencoder for compression
        self.autoencoder = QuantumAutoencoder(self.n_qubits, self.compressed_dim)
        
        # Pattern recognition network
        self.pattern_network = QuantumNeuralNetwork(self.n_qubits, 2, True)
        
        # Learned dictionaries
        self.quantum_dictionary = {}
        self.pattern_frequencies = {}
    
    def compress_data(self, data: np.ndarray) -> Dict:
        """
        Compress data using quantum machine learning
        
        Returns:
            Dictionary containing compressed representation
        """
        # Normalize input data
        normalized_data = self._normalize_data(data)
        
        # Quantum compression
        compressed_repr = self.autoencoder.encode(normalized_data)
        
        # Pattern analysis
        patterns = self._analyze_patterns(normalized_data)
        
        # Create compressed representation
        compressed_data = {
            'quantum_compressed': compressed_repr,
            'patterns': patterns,
            'metadata': {
                'original_shape': data.shape,
                'compression_ratio': len(compressed_repr) / len(data),
                'quantum_features': len(compressed_repr)
            }
        }
        
        return compressed_data
    
    def decompress_data(self, compressed_data: Dict) -> np.ndarray:
        """Decompress data using quantum model"""
        # Extract quantum representation
        quantum_repr = compressed_data['quantum_compressed']
        
        # Quantum decompression
        decompressed = self.autoencoder.decode(quantum_repr)
        
        # Apply learned patterns
        patterns = compressed_data.get('patterns', {})
        enhanced_data = self._apply_patterns(decompressed, patterns)
        
        # Denormalize
        final_data = self._denormalize_data(enhanced_data, compressed_data['metadata'])
        
        return final_data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data for quantum processing"""
        # Simple min-max normalization
        if len(data) == 0:
            return data
        
        min_val, max_val = np.min(data), np.max(data)
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(data)
    
    def _denormalize_data(self, data: np.ndarray, metadata: Dict) -> np.ndarray:
        """Denormalize data after quantum processing"""
        # In practice, would store normalization parameters in metadata
        return data  # Simplified for demo
    
    def _analyze_patterns(self, data: np.ndarray) -> Dict:
        """Analyze patterns using quantum network"""
        patterns = {}
        
        # Use quantum pattern network to identify recurring patterns
        for i in range(0, len(data) - 3, 4):
            chunk = data[i:i+4]
            if len(chunk) == 4:
                pattern_id = self.pattern_network.forward(chunk)
                pattern_key = str(np.argmax(pattern_id))
                
                if pattern_key not in patterns:
                    patterns[pattern_key] = []
                patterns[pattern_key].append(i)
        
        return patterns
    
    def _apply_patterns(self, data: np.ndarray, patterns: Dict) -> np.ndarray:
        """Apply learned patterns to enhance decompression"""
        # Simplified pattern application
        enhanced_data = data.copy()
        
        # In practice, would use patterns to refine decompressed data
        return enhanced_data
    
    def train_compression_model(self, training_data: List[np.ndarray], epochs: int = 30):
        """Train the quantum compression model"""
        print("Training Quantum Compression Model...")
        
        # Prepare training data
        normalized_data = [self._normalize_data(data) for data in training_data]
        
        # Train autoencoder
        self.autoencoder.train_autoencoder(normalized_data, epochs)
        
        # Update pattern dictionary
        for data in normalized_data:
            patterns = self._analyze_patterns(data)
            for pattern_id, positions in patterns.items():
                if pattern_id not in self.pattern_frequencies:
                    self.pattern_frequencies[pattern_id] = 0
                self.pattern_frequencies[pattern_id] += len(positions)
        
        print(f"Training completed. Found {len(self.pattern_frequencies)} unique patterns.")

def demonstrate_quantum_ml():
    """Demonstrate quantum machine learning concepts"""
    print("🚀 Quantum Machine Learning Demonstration")
    print("=" * 50)
    
    # 1. Basic Quantum Neural Network
    print("\n1. Basic Quantum Neural Network")
    print("-" * 30)
    
    qnn = QuantumNeuralNetwork(n_qubits=3, n_layers=2)
    
    # Create synthetic training data
    train_inputs = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6]),
        np.array([0.7, 0.8, 0.9]),
        np.array([0.2, 0.4, 0.6])
    ]
    
    train_targets = [
        np.array([0.9, 0.8, 0.7, 0.6]),
        np.array([0.6, 0.5, 0.4, 0.3]),
        np.array([0.3, 0.2, 0.1, 0.0]),
        np.array([0.8, 0.6, 0.4, 0.2])
    ]
    
    # Train the network
    losses = qnn.train(train_inputs, train_targets, epochs=20, learning_rate=0.05)
    
    # Test the trained network
    test_input = np.array([0.3, 0.4, 0.5])
    output = qnn.forward(test_input)
    print(f"Test input: {test_input}")
    print(f"Network output: {output[:4]}")
    
    # 2. Quantum Autoencoder
    print("\n2. Quantum Autoencoder for Compression")
    print("-" * 40)
    
    autoencoder = QuantumAutoencoder(n_input_qubits=4, n_latent_qubits=2)
    
    # Create training data (small vectors to compress)
    compression_data = [
        np.array([0.8, 0.6, 0.4, 0.2]),
        np.array([0.1, 0.3, 0.7, 0.9]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.9, 0.1, 0.8, 0.2])
    ]
    
    # Train autoencoder
    autoencoder.train_autoencoder(compression_data, epochs=20)
    
    # Test compression and decompression
    test_data = np.array([0.7, 0.3, 0.6, 0.4])
    compressed = autoencoder.encode(test_data)
    reconstructed = autoencoder.decode(compressed)
    
    print(f"Original data: {test_data}")
    print(f"Compressed (latent): {compressed}")
    print(f"Reconstructed: {reconstructed}")
    print(f"Compression ratio: {len(compressed) / len(test_data):.2f}")
    
    # 3. Hybrid Quantum-Classical Model
    print("\n3. Hybrid Quantum-Classical Model")
    print("-" * 35)
    
    hybrid_model = HybridQuantumClassicalModel(n_qubits=3, n_classical_features=4)
    
    # Create hybrid training data
    quantum_inputs = [
        np.array([0.2, 0.4, 0.6]),
        np.array([0.8, 0.1, 0.9]),
        np.array([0.3, 0.7, 0.2])
    ]
    
    classical_inputs = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.5, 1.5, 2.5, 3.5]),
        np.array([2.0, 1.0, 4.0, 0.5])
    ]
    
    targets = [0.8, 0.3, 0.6]
    
    # Train hybrid model
    hybrid_model.train_hybrid(quantum_inputs, classical_inputs, targets, epochs=20)
    
    # Test hybrid model
    test_q = np.array([0.4, 0.5, 0.3])
    test_c = np.array([1.5, 2.5, 1.0, 3.0])
    hybrid_output = hybrid_model.forward(test_q, test_c)
    
    print(f"Hybrid model output: {hybrid_output:.4f}")
    
    # 4. Quantum Compression Model
    print("\n4. Quantum Compression Model")
    print("-" * 30)
    
    compression_model = QuantumCompressionModel(input_dim=8, compression_ratio=0.4)
    
    # Create data to compress
    data_samples = [
        np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.4, 0.6]),
        np.array([0.9, 0.8, 0.2, 0.3, 0.7, 0.1, 0.6, 0.4]),
        np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.2, 0.8, 0.2]),
        np.array([0.3, 0.7, 0.4, 0.6, 0.1, 0.9, 0.2, 0.8])
    ]
    
    # Train compression model
    compression_model.train_compression_model(data_samples, epochs=15)
    
    # Test compression
    test_data = np.array([0.6, 0.4, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2])
    
    # Compress
    compressed_result = compression_model.compress_data(test_data)
    print(f"Original data: {test_data}")
    print(f"Compressed size: {len(compressed_result['quantum_compressed'])}")
    print(f"Compression ratio: {compressed_result['metadata']['compression_ratio']:.2f}")
    
    # Decompress
    decompressed_result = compression_model.decompress_data(compressed_result)
    print(f"Decompressed: {decompressed_result}")
    
    # Calculate reconstruction error
    mse = np.mean((test_data - decompressed_result) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    print("\n🎯 Quantum ML Demo Complete!")
    print("This demonstrates the basics of quantum machine learning.")
    print("For production use, consider using Qiskit, PennyLane, or Cirq.")

def setup_instructions():
    """Print setup instructions for quantum ML libraries"""
    print("📋 Setup Instructions for Quantum Machine Learning")
    print("=" * 55)
    print()
    
    print("1. Install Qiskit (IBM's quantum framework):")
    print("   pip install qiskit qiskit-aer qiskit-ibmq-provider")
    print()
    
    print("2. Install PennyLane (Differentiable quantum programming):")
    print("   pip install pennylane pennylane-sf pennylane-qiskit")
    print()
    
    print("3. Install Cirq (Google's quantum framework):")
    print("   pip install cirq")
    print()
    
    print("4. Additional ML libraries:")
    print("   pip install torch tensorflow scikit-learn")
    print()
    
    print("5. For quantum hardware access:")
    print("   - IBM Quantum: Sign up at quantum-computing.ibm.com")
    print("   - Google Quantum AI: Apply for access")
    print("   - Amazon Braket: AWS quantum computing service")
    print()
    
    print("6. Quantum simulators for testing:")
    print("   - Qiskit Aer (local simulation)")
    print("   - PennyLane simulators")
    print("   - Cirq simulators")

if __name__ == "__main__":
    # Run demonstrations
    setup_instructions()
    print("\n" + "="*60 + "\n")
    
    demonstrate_quantum_ml()
    
    print("\n" + "="*60)
    print("🚀 Next Steps:")
    print("1. Install quantum ML frameworks (Qiskit, PennyLane)")
    print("2. Experiment with real quantum hardware")
    print("3. Explore quantum advantage in your specific domain")
    print("4. Join quantum computing communities and forums")
    print("5. Consider quantum machine learning courses")
