import json
import os
from pathlib import Path
from typing import List, Dict, Any

class STUNIRChunker:
    """Chunks extraction.json into batches for STUNIR processing"""
    
    def __init__(self, max_functions: int = 100):
        self.max_functions = max_functions
    
    def load_extraction(self, filepath: str) -> Dict[str, Any]:
        """Load extraction.json file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def chunk_by_count(self, extraction: Dict[str, Any], output_dir: str) -> List[str]:
        """Chunk extraction into batches based on function count"""
        functions = extraction.get('functions', [])
        total = len(functions)
        
        if total <= self.max_functions:
            # No chunking needed
            output_path = Path(output_dir) / "batch_01.json"
            with open(output_path, 'w') as f:
                json.dump(extraction, f, indent=2)
            return [str(output_path)]
        
        # Chunk into multiple files
        batches = []
        num_batches = (total + self.max_functions - 1) // self.max_functions
        
        for i in range(num_batches):
            start_idx = i * self.max_functions
            end_idx = min((i + 1) * self.max_functions, total)
            
            batch = {
                'source_file': extraction.get('source_file', 'unknown'),
                'batch_info': {
                    'batch_number': i + 1,
                    'total_batches': num_batches,
                    'functions_in_batch': end_idx - start_idx,
                    'total_functions': total
                },
                'functions': functions[start_idx:end_idx]
            }
            
            output_path = Path(output_dir) / f"batch_{i+1:02d}.json"
            with open(output_path, 'w') as f:
                json.dump(batch, f, indent=2)
            batches.append(str(output_path))
            print(f"Created {output_path} with {end_idx - start_idx} functions")
        
        return batches
    
    def chunk_by_file(self, extraction: Dict[str, Any], output_dir: str, 
                      file_groups: List[List[str]]) -> List[str]:
        """Chunk extraction by file groups (for multi-file projects like bc)"""
        functions = extraction.get('functions', [])
        
        batches = []
        for group_idx, file_group in enumerate(file_groups, 1):
            # Filter functions belonging to this file group
            batch_funcs = [
                f for f in functions 
                if any(src in f.get('source_file', '') for src in file_group)
            ]
            
            if not batch_funcs:
                continue
            
            batch = {
                'source_files': file_group,
                'batch_info': {
                    'batch_number': group_idx,
                    'total_batches': len(file_groups),
                    'functions_in_batch': len(batch_funcs)
                },
                'functions': batch_funcs
            }
            
            output_path = Path(output_dir) / f"batch_{group_idx:02d}.json"
            with open(output_path, 'w') as f:
                json.dump(batch, f, indent=2)
            batches.append(str(output_path))
            print(f"Created {output_path} with {len(batch_funcs)} functions from {file_group}")
        
        return batches

def create_bc_batches():
    """Create batches for GNU bc (2 batches)"""
    chunker = STUNIRChunker(max_functions=100)
    
    # Define file groups for bc
    batch1_files = ['execute.c', 'util.c', 'storage.c', 'load.c', 'global.c']
    batch2_files = ['bc.c', 'scan.c', 'main.c', 'warranty.c']
    
    # Note: This assumes we have an extraction.json to work with
    # For now, just return the file group definitions
    return [batch1_files, batch2_files]

def create_lua_batches():
    """Create batches for Lua (10 batches)"""
    # Group files by function count to stay under 100 per batch
    batches = [
        ['lvm.c'],  # VM - largest
        ['ldo.c', 'ldebug.c'],  # Control flow
        ['lapi.c'],  # API
        ['lparser.c'],  # Parser
        ['llex.c', 'lcode.c'],  # Lexer + code gen
        ['ltable.c', 'lstring.c'],  # Data structures
        ['lgc.c', 'lmem.c'],  # GC + memory
        ['lfunc.c', 'lundump.c', 'ldump.c'],  # Functions + dump
        ['lstate.c', 'linit.c'],  # State + init
        ['loslib.c', 'liolib.c', 'lmathlib.c', 'lstrlib.c', 'ltablib.c', 
         'lutf8lib.c', 'lbaselib.c', 'lcorolib.c', 'ldblib.c', 'lauxlib.c']  # Libraries
    ]
    return batches

def create_sqlite_batches():
    """Create batches for SQLite (29+ batches)"""
    # For SQLite amalgamation, we need to split by function count
    # This will be handled by chunk_by_count since it's one big file
    return None  # Use count-based chunking

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python stunir_chunker.py <extraction.json> <output_dir>")
        sys.exit(1)
    
    extraction_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    os.makedirs(output_dir, exist_ok=True)
    
    chunker = STUNIRChunker(max_functions=100)
    extraction = chunker.load_extraction(extraction_file)
    batches = chunker.chunk_by_count(extraction, output_dir)
    
    print(f"\nCreated {len(batches)} batch files in {output_dir}")
