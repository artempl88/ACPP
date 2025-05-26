#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running and testing ACPP algorithm
"""

import os
import time
from acpp import ACPPCompressor  # Import main class

def test_text_compression():
    """Test text data compression"""
    print("=== TEXT COMPRESSION TEST ===\n")
    
    # Test data
    test_texts = [
        "Hello world! This is a test of the data compression algorithm.",
        """
        The ACPP algorithm represents an innovative solution for data compression.
        Main advantages of the algorithm include adaptivity and high efficiency.
        The algorithm analyzes context and predicts next symbols.
        Symbol prediction allows achieving high compression ratios.
        High compression ratios are especially important for large data volumes.
        """,
        "a" * 1000,  # Repeating characters
        "The quick brown fox jumps over the lazy dog. " * 20,  # Repeating text
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"--- Test {i} ---")
        test_single_compression(text.encode('utf-8'))
        print()

def test_file_compression(filename):
    """Test file compression"""
    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return
        
    print(f"=== COMPRESSING FILE {filename} ===\n")
    
    with open(filename, 'rb') as f:
        data = f.read()
    
    print(f"File size: {len(data)} bytes")
    test_single_compression(data)

def test_single_compression(data):
    """Test compression of single data block"""
    compressor = ACPPCompressor(max_context_length=6, prediction_depth=3)
    
    # Time measurement
    start_time = time.time()
    
    try:
        compressed = compressor.compress(data)
        compression_time = time.time() - start_time
        
        # Get statistics
        stats = compressor.get_compression_stats(len(data), len(compressed))
        
        # Display results
        print(f"Original size: {stats['original_size']:,} bytes")
        print(f"Compressed size: {stats['compressed_size']:,} bytes")
        print(f"Space savings: {stats['space_savings']}")
        print(f"Compression ratio: {stats['compression_ratio']:.3f}")
        print(f"Compression time: {compression_time:.3f} sec")
        print(f"Contexts found: {stats['contexts_created']:,}")
        print(f"Patterns found: {stats['patterns_found']:,}")
        
        # Speed
        speed = len(data) / compression_time / 1024 / 1024  # MB/sec
        print(f"Speed: {speed:.2f} MB/s")
        
    except Exception as e:
        print(f"Compression error: {e}")

def interactive_mode():
    """Interactive mode"""
    print("=== ACPP INTERACTIVE MODE ===\n")
    
    while True:
        print("\nChoose action:")
        print("1. Compress text")
        print("2. Compress file")
        print("3. Performance tests")
        print("4. Algorithm settings")
        print("0. Exit")
        
        choice = input("\nYour choice: ").strip()
        
        if choice == '1':
            text = input("\nEnter text to compress:\n")
            if text:
                test_single_compression(text.encode('utf-8'))
                
        elif choice == '2':
            filename = input("\nEnter file path: ").strip()
            test_file_compression(filename)
            
        elif choice == '3':
            test_text_compression()
            
        elif choice == '4':
            configure_algorithm()
            
        elif choice == '0':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice!")

def configure_algorithm():
    """Configure algorithm parameters"""
    print("\n=== ALGORITHM SETTINGS ===")
    
    try:
        context_length = int(input("Context length (1-12, default 8): ") or "8")
        prediction_depth = int(input("Prediction depth (1-8, default 3): ") or "3")
        
        print(f"\nNew settings:")
        print(f"Context length: {context_length}")
        print(f"Prediction depth: {prediction_depth}")
        
        # Test with new settings
        test_text = "Test with new ACPP algorithm parameters."
        compressor = ACPPCompressor(max_context_length=context_length, 
                                  prediction_depth=prediction_depth)
        
        compressed = compressor.compress(test_text.encode('utf-8'))
        stats = compressor.get_compression_stats(len(test_text.encode('utf-8')), len(compressed))
        
        print(f"Test result: {stats['space_savings']} savings")
        
    except ValueError:
        print("Error: enter valid numbers!")

def benchmark_mode():
    """Benchmark mode"""
    print("=== ACPP BENCHMARKS ===\n")
    
    # Different data types for testing
    benchmarks = {
        "Repeating text": "Hello World! " * 100,
        "Random characters": os.urandom(1000).decode('latin1', errors='ignore'),
        "JSON-like data": '{"name": "test", "value": 123, "active": true}' * 50,
        "Python code": '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
''' * 20,
    }
    
    print("Data type".ljust(25) + "Size".ljust(10) + "Compressed".ljust(12) + "Savings".ljust(10) + "Speed")
    print("-" * 70)
    
    for name, data in benchmarks.items():
        data_bytes = data.encode('utf-8')
        
        start_time = time.time()
        compressor = ACPPCompressor()
        compressed = compressor.compress(data_bytes)
        comp_time = time.time() - start_time
        
        stats = compressor.get_compression_stats(len(data_bytes), len(compressed))
        speed = len(data_bytes) / comp_time / 1024 if comp_time > 0 else 0
        
        print(f"{name[:24]:<25}{len(data_bytes):<10}{len(compressed):<12}{stats['space_savings'][:6]:<10}{speed:.1f} KB/s")

def main():
    """Main function"""
    print("🗜️  ACPP Compressor - Adaptive Contextual Pattern Prediction")
    print("=" * 60)
    
    if len(os.sys.argv) > 1:
        # Command line mode
        command = os.sys.argv[1].lower()
        
        if command == 'test':
            test_text_compression()
        elif command == 'benchmark':
            benchmark_mode()
        elif command == 'file' and len(os.sys.argv) > 2:
            test_file_compression(os.sys.argv[2])
        else:
            print("Usage:")
            print("  python acpp_runner.py test          # Run tests")
            print("  python acpp_runner.py benchmark     # Run benchmarks")
            print("  python acpp_runner.py file <path>   # Compress file")
            print("  python acpp_runner.py               # Interactive mode")
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
