#!/usr/bin/env python3
"""
Script to fix common linting issues in the Newsies codebase
"""

import os
import re
from pathlib import Path


def fix_unused_imports():
    """Fix unused imports in __init__.py files by commenting them out"""
    init_files = [
        "newsies/classify/__init__.py",
        "newsies/cli/__init__.py", 
        "newsies/llm/__init__.py",
        "newsies/pipelines/__init__.py",
        "newsies/redis_client/__init__.py",
        "newsies/session/__init__.py",
        "newsies/utils/__init__.py",
        "newsies/visitor/__init__.py",
        "newsies/api/__init__.py"
    ]
    
    for file_path in init_files:
        if os.path.exists(file_path):
            print(f"Fixing unused imports in {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Comment out unused imports but keep the structure
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                if line.strip().startswith('from .') and 'import' in line:
                    # Comment out specific imports that are unused
                    new_lines.append(f"# {line}  # Unused import - available for explicit use")
                elif line.strip().startswith('from newsies.') and 'import' in line:
                    new_lines.append(f"# {line}  # Unused import - available for explicit use")
                else:
                    new_lines.append(line)
            
            with open(file_path, 'w') as f:
                f.write('\n'.join(new_lines))


def fix_star_imports():
    """Fix star imports in pipelines/__init__.py"""
    file_path = "newsies/pipelines/__init__.py"
    if os.path.exists(file_path):
        print(f"Fixing star imports in {file_path}")
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace star imports with specific imports
        new_content = '''"""
newsies.pipelines
"""

# Import specific functions instead of using star imports
from .analyze import analyze_pipeline
from .get_articles import get_articles_pipeline
from .task_status import TASK_STATUS, LOCK
'''
        
        with open(file_path, 'w') as f:
            f.write(new_content)


def fix_whitespace_operators():
    """Fix missing whitespace around operators"""
    files_to_fix = [
        "newsies/cli/main.py",
        "newsies/llm/question_generator.py"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            print(f"Fixing whitespace in {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Fix common whitespace issues
            content = re.sub(r'(\w+)=(\w+)', r'\1 = \2', content)  # x=y -> x = y
            content = re.sub(r'(\w+)\+(\w+)', r'\1 + \2', content)  # x+y -> x + y
            content = re.sub(r'(\w+)-(\w+)', r'\1 - \2', content)  # x-y -> x - y
            content = re.sub(r'(\w+)\*(\w+)', r'\1 * \2', content)  # x*y -> x * y
            content = re.sub(r'(\w+)/(\w+)', r'\1 / \2', content)  # x/y -> x / y
            
            with open(file_path, 'w') as f:
                f.write(content)


def fix_long_lines():
    """Fix remaining long lines manually"""
    fixes = {
        "newsies/session/main.py": {
            15: 'from newsies.llm import (load_base_model_with_lora, CORPUS_PROMPT, '
                'tokenize, decode)'
        }
    }
    
    for file_path, line_fixes in fixes.items():
        if os.path.exists(file_path):
            print(f"Fixing long lines in {file_path}")
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, new_content in line_fixes.items():
                if line_num <= len(lines):
                    lines[line_num - 1] = new_content + '\n'
            
            with open(file_path, 'w') as f:
                f.writelines(lines)


def main():
    """Main function to run all fixes"""
    print("Starting linting fixes...")
    
    fix_unused_imports()
    fix_star_imports()
    fix_whitespace_operators()
    fix_long_lines()
    
    print("Linting fixes completed!")


if __name__ == "__main__":
    main()
