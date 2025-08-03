#!/usr/bin/env python3
"""
Script to fix all import statements after package restructuring
"""

import os
import re
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    # Common package imports
    'from newsies.document_structures': 'from newsies_common.document_structures',
    'from newsies.targets': 'from newsies_common.targets',
    'from newsies.utils': 'from newsies_common.utils',
    'from newsies.visitor': 'from newsies_common.visitor',
    'from newsies.pipelines.task_status': 'from newsies_common.task_status',
    'from newsies.pipelines import TASK_STATUS': 'from newsies_common.task_status import TASK_STATUS',
    
    # Client package imports
    'from newsies.chromadb_client': 'from newsies_clients.chromadb_client',
    'from newsies.redis_client': 'from newsies_clients.redis_client',
    'from newsies.session': 'from newsies_clients.session',
    'from newsies.chroma_client': 'from newsies_clients.chroma_client',
    'from newsies.collections': 'from newsies_clients.collections',
    
    # Service-specific imports (relative within packages)
    'from newsies.ap_news': 'from ..ap_news',
    'from newsies.llm': 'from ..llm',
    'from newsies.classify': 'from ..classify',
    'from newsies.api': 'from ..api',
    'from newsies.cli': 'from ..cli',
    
    # Pipeline imports
    'from newsies.pipelines': 'from newsies_common.task_status',
    'import newsies.pipelines': 'import newsies_common.task_status',
}

def fix_imports_in_file(file_path: Path):
    """Fix imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply import mappings
        for old_import, new_import in IMPORT_MAPPINGS.items():
            content = content.replace(old_import, new_import)
        
        # Fix specific patterns
        # Remove commented imports that are causing issues
        content = re.sub(r'from newsies\.ap_news\.archive import Archive', 
                        '# from newsies.ap_news.archive import Archive', content)
        content = re.sub(r'from newsies\.llm\.batch_set import BatchSet', 
                        '# from newsies.llm.batch_set import BatchSet', content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed imports in: {file_path}")
            return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False
    
    return False

def main():
    """Main function to fix all imports"""
    base_path = Path('.')
    
    # Directories to process
    package_dirs = [
        'newsies-common/newsies_common',
        'newsies-clients/newsies_clients', 
        'newsies-scraper/newsies_scraper',
        'newsies-analyzer/newsies_analyzer',
        'newsies-trainer/newsies_trainer',
        'newsies-api/newsies_api',
        'newsies-cli/newsies_cli',
    ]
    
    total_files = 0
    fixed_files = 0
    
    for package_dir in package_dirs:
        package_path = base_path / package_dir
        if package_path.exists():
            print(f"\nProcessing package: {package_dir}")
            
            # Find all Python files
            for py_file in package_path.rglob('*.py'):
                if py_file.name != '__pycache__':
                    total_files += 1
                    if fix_imports_in_file(py_file):
                        fixed_files += 1
    
    print(f"\nCompleted: Fixed {fixed_files} out of {total_files} files")

if __name__ == '__main__':
    main()
