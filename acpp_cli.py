#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal ACPP CLI - Command Line Interface for Universal Compressor
"""

import os
import sys
import argparse
import time
import threading
from pathlib import Path
from typing import List, Dict

# Import main compressor
from acpp import UniversalACPPCompressor, compress_file, decompress_file

class ProgressBar:
    """Simple progress bar for command line"""
    
    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, amount: int = 1):
        self.current += amount
        self._print_progress()
        
    def _print_progress(self):
        if self.total == 0:
            return
            
        progress = self.current / self.total
        bar_length = 50
        filled = int(bar_length * progress)
        
        bar = '█' * filled + '░' * (bar_length - filled)
        
        elapsed = time.time() - self.start_time
        if progress > 0:
            eta = elapsed / progress - elapsed
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --:--"
            
        percent = progress * 100
        speed = self.current / elapsed if elapsed > 0 else 0
        
        print(f"\r{self.description} |{bar}| {percent:.1f}% ({self.current:,}/{self.total:,}) {speed:.1f} B/s {eta_str}", end='', flush=True)
        
        if progress >= 1.0:
            print()  # New line at end

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"

def analyze_file(filepath: str) -> Dict:
    """Analyze file before compression"""
    file_size = os.path.getsize(filepath)
    
    # Simple heuristic for file type detection
    with open(filepath, 'rb') as f:
        header = f.read(512)
    
    # Determine file type by content
    if header.startswith(b'\x89PNG'):
        file_type = "PNG Image"
        expected_compression = "Low (already compressed)"
    elif header.startswith(b'\xFF\xD8\xFF'):
        file_type = "JPEG Image"
        expected_compression = "Very low (already compressed)"
    elif header.startswith(b'PK'):
        file_type = "ZIP/Office document"
        expected_compression = "Low (already compressed)"
    elif header.startswith(b'%PDF'):
        file_type = "PDF document"
        expected_compression = "Medium"
    elif b'<html' in header.lower() or b'<!doctype' in header.lower():
        file_type = "HTML document"
        expected_compression = "High"
    elif header.startswith(b'{') or header.startswith(b'['):
        file_type = "JSON file"
        expected_compression = "Very high"
    elif all(32 <= b <= 126 or b in [9, 10, 13] for b in header[:100]):
        file_type = "Text file"
        expected_compression = "Very high"
    else:
        file_type = "Binary file"
        expected_compression = "Medium"
    
    # Analyze entropy of first 4KB
    sample_size = min(4096, len(header))
    if sample_size > 0:
        entropy = calculate_entropy(header[:sample_size])
    else:
        entropy = 0
        
    return {
        'size': file_size,
        'type': file_type,
        'expected_compression': expected_compression,
        'entropy': entropy
    }

def calculate_entropy(data: bytes) -> float:
    """Calculate data entropy"""
    if not data:
        return 0.0
        
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    
    length = len(data)
    entropy = 0.0
    
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * (p.bit_length() - 1)  # Fast log2 approximation
    
    return entropy / 8.0  # Normalization

def compress_command(args):
    """Compression command"""
    input_file = args.input
    output_file = args.output or (input_file + '.acpp')
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return 1
    
    print(f"🔍 Analyzing file: {input_file}")
    analysis = analyze_file(input_file)
    
    print(f"   Size: {format_size(analysis['size'])}")
    print(f"   Type: {analysis['type']}")
    print(f"   Expected compression: {analysis['expected_compression']}")
    print(f"   Entropy: {analysis['entropy']:.3f}")
    
    if args.level < 1 or args.level > 9:
        print("❌ Compression level must be between 1 and 9")
        return 1
    
    print(f"\n🗜️  Starting compression (level {args.level})...")
    
    try:
        start_time = time.time()
        stats = compress_file(input_file, output_file, args.level)
        end_time = time.time()
        
        print(f"\n✅ Compression completed successfully!")
        print(f"   Original size: {format_size(stats['original_size'])}")
        print(f"   Compressed size: {format_size(stats['compressed_size'])}")
        print(f"   Space savings: {stats['space_savings_percent']:.1f}%")
        print(f"   Compression ratio: {stats['compression_ratio']:.3f}")
        print(f"   Processing time: {stats['compression_time']:.2f} sec")
        print(f"   Speed: {stats['speed_mbps']:.1f} MB/s")
        print(f"   Blocks processed: {stats['blocks_processed']:,}")
        print(f"   Matches found: {stats['matches_found']:,}")
        print(f"   Output file: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Compression error: {e}")
        return 1

def decompress_command(args):
    """Decompression command"""
    input_file = args.input
    output_file = args.output
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return 1
    
    if not output_file:
        if input_file.endswith('.acpp'):
            output_file = input_file[:-5]
        else:
            output_file = input_file + '.decompressed'
    
    print(f"📤 Decompressing: {input_file} → {output_file}")
    
    try:
        start_time = time.time()
        stats = decompress_file(input_file, output_file)
        end_time = time.time()
        
        print(f"\n✅ Decompression completed successfully!")
        print(f"   Size: {format_size(stats['decompressed_size'])}")
        print(f"   Processing time: {stats['decompression_time']:.2f} sec")
        print(f"   Speed: {stats['speed_mbps']:.1f} MB/s")
        print(f"   Integrity check: {stats['integrity_check']}")
        print(f"   Output file: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Decompression error: {e}")
        return 1

def benchmark_command(args):
    """Benchmark command"""
    print("🏃 Running ACPP algorithm benchmark...")
    
    # Create test files of different types
    test_files = create_test_files()
    
    results = []
    
    for test_name, test_file in test_files.items():
        print(f"\n📊 Testing: {test_name}")
        
        try:
            # Compression
            compress_stats = compress_file(test_file, test_file + '.acpp', args.level)
            
            # Decompression  
            decompress_stats = decompress_file(test_file + '.acpp', test_file + '.restored')
            
            # Integrity check
            with open(test_file, 'rb') as f1, open(test_file + '.restored', 'rb') as f2:
                integrity_ok = f1.read() == f2.read()
            
            results.append({
                'name': test_name,
                'original_size': compress_stats['original_size'],
                'compressed_size': compress_stats['compressed_size'],
                'compression_ratio': compress_stats['compression_ratio'],
                'space_savings': compress_stats['space_savings_percent'],
                'compress_speed': compress_stats['speed_mbps'],
                'decompress_speed': decompress_stats['speed_mbps'],
                'integrity': integrity_ok
            })
            
            # Clean up temporary files
            os.remove(test_file + '.acpp')
            os.remove(test_file + '.restored')
            
        except Exception as e:
            print(f"❌ Error in test {test_name}: {e}")
    
    # Clean up test files
    for test_file in test_files.values():
        try:
            os.remove(test_file)
        except:
            pass
    
    # Display results
    print(f"\n📈 BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Data Type':<20} {'Size':<10} {'Compressed':<10} {'Savings':<8} {'Comp.Speed':<10} {'Decomp.Speed':<12} {'Integrity':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<20} "
              f"{format_size(result['original_size']):<10} "
              f"{format_size(result['compressed_size']):<10} "
              f"{result['space_savings']:.1f}%    "
              f"{result['compress_speed']:.1f} MB/s   "
              f"{result['decompress_speed']:.1f} MB/s     "
              f"{'✅' if result['integrity'] else '❌':<10}")
    
    return 0

def create_test_files() -> Dict[str, str]:
    """Create test files for benchmark"""
    import random
    import string
    
    test_files = {}
    
    # 1. Text file
    text_content = "This is test text for demonstrating the ACPP compression algorithm. " * 1000
    with open('test_text.txt', 'w', encoding='utf-8') as f:
        f.write(text_content)
    test_files['Text'] = 'test_text.txt'
    
    # 2. Repeating data
    repeated_content = b'ABCDEF' * 5000
    with open('test_repeated.bin', 'wb') as f:
        f.write(repeated_content)
    test_files['Repeated'] = 'test_repeated.bin'
    
    # 3. Random data
    random.seed(42)
    random_content = bytes([random.randint(0, 255) for _ in range(30000)])
    with open('test_random.bin', 'wb') as f:
        f.write(random_content)
    test_files['Random'] = 'test_random.bin'
    
    # 4. JSON-like data
    json_content = '{"name": "test", "value": 123, "items": [1, 2, 3, 4, 5], "active": true}' * 500
    with open('test_json.json', 'w') as f:
        f.write(json_content)
    test_files['JSON'] = 'test_json.json'
    
    # 5. Mixed data
    mixed_content = bytearray()
    mixed_content.extend(b'Header section\n' * 100)
    mixed_content.extend(b'\x00\x01\x02\x03' * 2000)
    mixed_content.extend(b"Text section with repeated patterns. " * 500)
    mixed_content.extend(bytes([random.randint(0, 255) for _ in range(5000)]))
    
    with open('test_mixed.bin', 'wb') as f:
        f.write(mixed_content)
    test_files['Mixed'] = 'test_mixed.bin'
    
    return test_files

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Universal ACPP - Universal File Compressor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python acpp_cli.py compress document.pdf                    # Compress PDF
  python acpp_cli.py compress -l 9 bigfile.txt               # Maximum compression
  python acpp_cli.py decompress document.pdf.acpp            # Decompress
  python acpp_cli.py benchmark -l 7                          # Run benchmark
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Compression command
    compress_parser = subparsers.add_parser('compress', help='Compress file')
    compress_parser.add_argument('input', help='Input file')
    compress_parser.add_argument('-o', '--output', help='Output file (default: input + .acpp)')
    compress_parser.add_argument('-l', '--level', type=int, default=6, 
                               help='Compression level (1-9, default: 6)')
    
    # Decompression command
    decompress_parser = subparsers.add_parser('decompress', help='Decompress file')
    decompress_parser.add_argument('input', help='Compressed file (.acpp)')
    decompress_parser.add_argument('-o', '--output', help='Output file')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark')
    benchmark_parser.add_argument('-l', '--level', type=int, default=6,
                                help='Compression level for testing (1-9)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute commands
    if args.command == 'compress':
        return compress_command(args)
    elif args.command == 'decompress':
        return decompress_command(args)
    elif args.command == 'benchmark':
        return benchmark_command(args)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)
