#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal ACPP (Adaptive Contextual Pattern Prediction) Compressor
Universal compression algorithm for files of any format and size
"""

import os
import struct
import hashlib
import mmap
from typing import Dict, List, Tuple, Optional, BinaryIO
from collections import defaultdict, deque
import heapq
import threading
import time

class UniversalACPPCompressor:
    """
    Universal ACPP compressor for files of any type and size
    
    Features:
    - Streaming processing (doesn't load entire file into memory)
    - Works with binary data of any type
    - Adaptive compression window
    - Scalable to terabyte-sized files
    - Full compression/decompression support
    """
    
    # Algorithm constants
    MAGIC_SIGNATURE = b'UACPP'
    VERSION = 2
    DEFAULT_BLOCK_SIZE = 64 * 1024  # 64KB blocks
    DEFAULT_WINDOW_SIZE = 1024 * 1024  # 1MB search window
    MAX_MATCH_LENGTH = 258
    MIN_MATCH_LENGTH = 4
    HASH_SIZE = 65536  # 64K hash table
    
    def __init__(self, 
                 block_size: int = DEFAULT_BLOCK_SIZE,
                 window_size: int = DEFAULT_WINDOW_SIZE,
                 compression_level: int = 6):
        """
        Initialize compressor
        
        Args:
            block_size: Block size for processing
            window_size: Search window size for LZ algorithm
            compression_level: Compression level (1-9)
        """
        self.block_size = block_size
        self.window_size = window_size
        self.compression_level = max(1, min(9, compression_level))
        
        # Adaptive parameters based on compression level
        self.max_chain_length = 32 << (compression_level - 1)
        self.good_match_length = 4 + compression_level
        self.max_lazy_match = 16 + compression_level * 2
        
        # Working data structures
        self.hash_table = [0] * self.HASH_SIZE
        self.prev_table = [0] * self.window_size
        self.match_buffer = bytearray()
        
        # Statistics
        self.stats = {
            'bytes_processed': 0,
            'bytes_compressed': 0,
            'blocks_processed': 0,
            'matches_found': 0,
            'start_time': 0,
            'compression_ratio': 0.0
        }

    def _hash_function(self, data: bytes, pos: int) -> int:
        """Fast hash function for 3-byte sequences"""
        if pos + 2 >= len(data):
            return 0
        return ((data[pos] << 10) ^ (data[pos+1] << 5) ^ data[pos+2]) & (self.HASH_SIZE - 1)

    def _find_longest_match(self, data: bytes, pos: int, prev_pos: int) -> Tuple[int, int]:
        """
        Find the longest match
        
        Returns:
            (distance, length) - distance and length of match
        """
        if prev_pos <= 0 or pos - prev_pos > self.window_size:
            return (0, 0)
            
        max_len = min(self.MAX_MATCH_LENGTH, len(data) - pos)
        best_len = 0
        best_dist = 0
        
        # Quick check for minimum match
        if pos + self.MIN_MATCH_LENGTH > len(data):
            return (0, 0)
            
        current_len = 0
        i = 0
        
        # Find maximum match
        while (i < max_len and 
               pos + i < len(data) and 
               prev_pos + i < len(data) and 
               data[pos + i] == data[prev_pos + i]):
            i += 1
            
        current_len = i
        
        if current_len >= self.MIN_MATCH_LENGTH:
            best_len = current_len
            best_dist = pos - prev_pos
            
        return (best_dist, best_len)

    def _update_hash_tables(self, data: bytes, pos: int):
        """Update hash tables"""
        if pos + 2 < len(data):
            hash_val = self._hash_function(data, pos)
            self.prev_table[pos & (self.window_size - 1)] = self.hash_table[hash_val]
            self.hash_table[hash_val] = pos

    def _encode_length_distance(self, length: int, distance: int) -> bytes:
        """Encode length and distance of match"""
        result = bytearray()
        
        # Encode with special marker
        result.append(255)  # LZ match marker
        
        # Encode length (1-2 bytes)
        if length < 128:
            result.append(length)
        else:
            result.append(128 | (length & 127))
            result.append(length >> 7)
            
        # Encode distance (2-4 bytes depending on size)
        if distance < 256:
            result.append(1)  # 1 byte
            result.append(distance)
        elif distance < 65536:
            result.append(2)  # 2 bytes
            result.extend(struct.pack('<H', distance))
        else:
            result.append(4)  # 4 bytes
            result.extend(struct.pack('<I', distance))
            
        return bytes(result)

    def _compress_block(self, data: bytes) -> bytes:
        """Compress a single block of data"""
        if not data:
            return b''
            
        compressed = bytearray()
        pos = 0
        
        # Initialize hash tables for new block
        for i in range(min(3, len(data))):
            self._update_hash_tables(data, i)
            
        while pos < len(data):
            best_length = 0
            best_distance = 0
            
            # Search for matches only if sufficient data available
            if pos + self.MIN_MATCH_LENGTH <= len(data):
                hash_val = self._hash_function(data, pos)
                prev_pos = self.hash_table[hash_val]
                
                # Check match chain
                chain_length = 0
                while (prev_pos > 0 and 
                       chain_length < self.max_chain_length and
                       pos - prev_pos <= self.window_size):
                    
                    distance, length = self._find_longest_match(data, pos, prev_pos)
                    
                    if length > best_length:
                        best_length = length
                        best_distance = distance
                        
                    if length >= self.good_match_length:
                        break
                        
                    prev_pos = self.prev_table[prev_pos & (self.window_size - 1)]
                    chain_length += 1
            
            # Decision: use match or literal
            if best_length >= self.MIN_MATCH_LENGTH:
                # Use LZ match
                lz_encoded = self._encode_length_distance(best_length, best_distance)
                compressed.extend(lz_encoded)
                
                # Update hash tables for all positions in match
                for i in range(best_length):
                    if pos + i < len(data):
                        self._update_hash_tables(data, pos + i)
                        
                pos += best_length
                self.stats['matches_found'] += 1
                
            else:
                # Add literal
                byte_val = data[pos]
                if byte_val == 255:
                    # Escape marker
                    compressed.extend([254, 255])
                else:
                    compressed.append(byte_val)
                    
                self._update_hash_tables(data, pos)
                pos += 1
                
        return bytes(compressed)

    def _create_header(self, original_size: int, checksum: bytes) -> bytes:
        """Create compressed file header"""
        header = bytearray()
        
        # Signature and version
        header.extend(self.MAGIC_SIGNATURE)
        header.append(self.VERSION)
        
        # Compression parameters
        header.append(self.compression_level)
        header.extend(struct.pack('<I', self.block_size))
        header.extend(struct.pack('<I', self.window_size))
        
        # Original file size
        header.extend(struct.pack('<Q', original_size))
        
        # Checksum
        header.extend(checksum)
        
        # Reserved for future extensions
        header.extend(b'\x00' * 16)
        
        return bytes(header)

    def compress_file(self, input_path: str, output_path: str) -> Dict:
        """
        Compress file
        
        Args:
            input_path: Path to source file
            output_path: Path to compressed file
            
        Returns:
            Dictionary with compression statistics
        """
        self.stats['start_time'] = time.time()
        
        try:
            # Get file size
            file_size = os.path.getsize(input_path)
            
            with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
                # Calculate hash of original file for integrity check
                hasher = hashlib.sha256()
                
                # Use memory mapping for large files
                if file_size > self.block_size * 10:
                    with mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                        return self._compress_memory_mapped(mmapped_file, outfile, file_size)
                else:
                    return self._compress_regular(infile, outfile, file_size)
                    
        except Exception as e:
            raise Exception(f"Error compressing file: {e}")

    def _compress_memory_mapped(self, mmapped_file, outfile: BinaryIO, file_size: int) -> Dict:
        """Compress large files using memory mapping"""
        hasher = hashlib.sha256()
        
        # Reserve space for header
        header_placeholder = b'\x00' * 64
        outfile.write(header_placeholder)
        
        bytes_processed = 0
        bytes_compressed = len(header_placeholder)
        
        print(f"Compressing file of size {file_size:,} bytes...")
        
        while bytes_processed < file_size:
            # Read block
            chunk_size = min(self.block_size, file_size - bytes_processed)
            chunk = mmapped_file[bytes_processed:bytes_processed + chunk_size]
            
            if not chunk:
                break
                
            # Update hash
            hasher.update(chunk)
            
            # Compress block
            compressed_chunk = self._compress_block(chunk)
            
            # Write block size and block data
            block_header = struct.pack('<I', len(compressed_chunk))
            outfile.write(block_header)
            outfile.write(compressed_chunk)
            
            bytes_processed += chunk_size
            bytes_compressed += 4 + len(compressed_chunk)
            self.stats['blocks_processed'] += 1
            
            # Progress
            if bytes_processed % (self.block_size * 100) == 0:
                progress = (bytes_processed / file_size) * 100
                print(f"Progress: {progress:.1f}% ({bytes_processed:,}/{file_size:,} bytes)")
        
        # Write final header
        checksum = hasher.digest()
        header = self._create_header(file_size, checksum)
        
        outfile.seek(0)
        outfile.write(header)
        
        return self._finalize_stats(file_size, bytes_compressed)

    def _compress_regular(self, infile: BinaryIO, outfile: BinaryIO, file_size: int) -> Dict:
        """Compress regular files"""
        hasher = hashlib.sha256()
        
        # Reserve space for header
        header_placeholder = b'\x00' * 64
        outfile.write(header_placeholder)
        
        bytes_processed = 0
        bytes_compressed = len(header_placeholder)
        
        while True:
            chunk = infile.read(self.block_size)
            if not chunk:
                break
                
            hasher.update(chunk)
            compressed_chunk = self._compress_block(chunk)
            
            # Write block size and data
            block_header = struct.pack('<I', len(compressed_chunk))
            outfile.write(block_header)
            outfile.write(compressed_chunk)
            
            bytes_processed += len(chunk)
            bytes_compressed += 4 + len(compressed_chunk)
            self.stats['blocks_processed'] += 1
        
        # Write header
        checksum = hasher.digest()
        header = self._create_header(file_size, checksum)
        
        outfile.seek(0)
        outfile.write(header)
        
        return self._finalize_stats(file_size, bytes_compressed)

    def _finalize_stats(self, original_size: int, compressed_size: int) -> Dict:
        """Finalize statistics"""
        compression_time = time.time() - self.stats['start_time']
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'space_savings_percent': (1 - compression_ratio) * 100,
            'compression_time': compression_time,
            'speed_mbps': (original_size / (1024 * 1024)) / compression_time if compression_time > 0 else 0,
            'blocks_processed': self.stats['blocks_processed'],
            'matches_found': self.stats['matches_found']
        }

    def decompress_file(self, input_path: str, output_path: str) -> Dict:
        """
        Decompress file
        
        Args:
            input_path: Path to compressed file
            output_path: Path to restored file
            
        Returns:
            Dictionary with decompression statistics
        """
        start_time = time.time()
        
        with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
            # Read and validate header
            header = infile.read(64)
            if not self._validate_header(header):
                raise Exception("Invalid file format or version")
                
            original_size, checksum = self._parse_header(header)
            hasher = hashlib.sha256()
            
            bytes_decompressed = 0
            
            print(f"Decompressing file, expected size: {original_size:,} bytes...")
            
            while bytes_decompressed < original_size:
                # Read block size
                size_data = infile.read(4)
                if len(size_data) < 4:
                    break
                    
                block_size = struct.unpack('<I', size_data)[0]
                if block_size == 0:
                    break
                    
                # Read compressed block
                compressed_block = infile.read(block_size)
                if len(compressed_block) != block_size:
                    raise Exception("Unexpected end of file")
                    
                # Decompress block
                decompressed_block = self._decompress_block(compressed_block)
                
                # Write result
                remaining = original_size - bytes_decompressed
                to_write = min(len(decompressed_block), remaining)
                
                outfile.write(decompressed_block[:to_write])
                hasher.update(decompressed_block[:to_write])
                
                bytes_decompressed += to_write
                
                # Progress
                if bytes_decompressed % (self.block_size * 100) == 0:
                    progress = (bytes_decompressed / original_size) * 100
                    print(f"Progress: {progress:.1f}%")
            
            # Check checksum
            if hasher.digest() != checksum:
                raise Exception("Checksum error - file is corrupted")
            
            decompression_time = time.time() - start_time
            
            return {
                'decompressed_size': bytes_decompressed,
                'decompression_time': decompression_time,
                'speed_mbps': (bytes_decompressed / (1024 * 1024)) / decompression_time if decompression_time > 0 else 0,
                'integrity_check': 'PASSED'
            }

    def _decompress_block(self, compressed_data: bytes) -> bytes:
        """Decompress a single block"""
        decompressed = bytearray()
        pos = 0
        
        while pos < len(compressed_data):
            byte_val = compressed_data[pos]
            
            if byte_val == 255:  # LZ match
                pos += 1
                if pos >= len(compressed_data):
                    break
                    
                # Read length
                length = compressed_data[pos]
                pos += 1
                
                if length & 128:  # Two-byte length
                    length = (length & 127) | (compressed_data[pos] << 7)
                    pos += 1
                
                # Read distance size
                if pos >= len(compressed_data):
                    break
                    
                dist_size = compressed_data[pos]
                pos += 1
                
                # Read distance
                if dist_size == 1:
                    distance = compressed_data[pos]
                    pos += 1
                elif dist_size == 2:
                    distance = struct.unpack('<H', compressed_data[pos:pos+2])[0]
                    pos += 2
                elif dist_size == 4:
                    distance = struct.unpack('<I', compressed_data[pos:pos+4])[0]
                    pos += 4
                else:
                    raise Exception("Invalid distance format")
                
                # Copy data
                start_pos = len(decompressed) - distance
                if start_pos < 0:
                    raise Exception("Invalid offset in LZ match")
                    
                for i in range(length):
                    if start_pos + i >= len(decompressed):
                        break
                    decompressed.append(decompressed[start_pos + i])
                    
            elif byte_val == 254:  # Escaped 255
                pos += 1
                if pos < len(compressed_data):
                    decompressed.append(compressed_data[pos])
                    pos += 1
                    
            else:  # Regular byte
                decompressed.append(byte_val)
                pos += 1
        
        return bytes(decompressed)

    def _validate_header(self, header: bytes) -> bool:
        """Validate file header"""
        if len(header) < 64:
            return False
        return header[:5] == self.MAGIC_SIGNATURE and header[5] == self.VERSION

    def _parse_header(self, header: bytes) -> Tuple[int, bytes]:
        """Parse file header"""
        # Skip signature, version and compression parameters
        offset = 14  # 5 + 1 + 1 + 4 + 4 - 1
        
        # Original file size
        original_size = struct.unpack('<Q', header[offset:offset+8])[0]
        offset += 8
        
        # Checksum
        checksum = header[offset:offset+32]
        
        return original_size, checksum

# Utilities for working with compressor
def compress_file(input_file: str, output_file: str = None, compression_level: int = 6) -> Dict:
    """
    Simple function to compress file
    
    Args:
        input_file: Path to source file
        output_file: Path to compressed file (default adds .acpp)
        compression_level: Compression level (1-9)
    """
    if output_file is None:
        output_file = input_file + '.acpp'
    
    compressor = UniversalACPPCompressor(compression_level=compression_level)
    return compressor.compress_file(input_file, output_file)

def decompress_file(input_file: str, output_file: str = None) -> Dict:
    """
    Simple function to decompress file
    
    Args:
        input_file: Path to compressed file (.acpp)
        output_file: Path to restored file
    """
    if output_file is None:
        if input_file.endswith('.acpp'):
            output_file = input_file[:-5]  # Remove .acpp
        else:
            output_file = input_file + '.decompressed'
    
    compressor = UniversalACPPCompressor()
    return compressor.decompress_file(input_file, output_file)

# Example usage
if __name__ == "__main__":
    import sys
    
    def print_stats(stats: Dict, operation: str):
        print(f"\n=== {operation.upper()} COMPLETED ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
    
    # Demonstration
    if len(sys.argv) < 2:
        print("Creating test file...")
        
        # Create test file with various data types
        test_data = bytearray()
        
        # Text data
        text = "This is a test file for demonstrating the universal ACPP algorithm. " * 1000
        test_data.extend(text.encode('utf-8'))
        
        # Repeating bytes
        test_data.extend(b'A' * 5000)
        
        # Random data
        import random
        random.seed(42)
        test_data.extend(bytes([random.randint(0, 255) for _ in range(10000)]))
        
        # Patterns
        pattern = b'\x00\x01\x02\x03\x04\x05\xFF\xFE'
        test_data.extend(pattern * 1000)
        
        with open('test_file.bin', 'wb') as f:
            f.write(test_data)
        
        print(f"Created test file of size {len(test_data):,} bytes")
        
        # Compress
        stats = compress_file('test_file.bin', 'test_file.bin.acpp', compression_level=7)
        print_stats(stats, "COMPRESSION")
        
        # Decompress
        stats = decompress_file('test_file.bin.acpp', 'test_file_restored.bin')
        print_stats(stats, "DECOMPRESSION")
        
        # Check identity
        with open('test_file.bin', 'rb') as f1, open('test_file_restored.bin', 'rb') as f2:
            original = f1.read()
            restored = f2.read()
            
        if original == restored:
            print("\n✅ FILES ARE IDENTICAL - decompression successful!")
        else:
            print("\n❌ ERROR - files differ!")
            
    else:
        # Compress file from command line
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"File {input_file} not found!")
            sys.exit(1)
            
        stats = compress_file(input_file)
        print_stats(stats, "COMPRESSION")
