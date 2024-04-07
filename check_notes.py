import os
import re
import shutil
import nbformat

backup_folder = './backup'

if not os.path.exists(backup_folder):
    os.makedirs(backup_folder)

def contains_chinese(s):
    return any('\u4e00' <= char <= '\u9fff' for char in s)

def remove_chinese_comments_py(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            if '#' in line:
                code, comment = line.split('#', 1)
                if contains_chinese(comment):
                    file.write(code.rstrip() + '\n')
                else:
                    file.write(line)
            else:
                file.write(line)

def remove_chinese_comments_ipynb(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            notebook = nbformat.read(file, as_version=4)
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                new_source_lines = []
                for line in cell['source'].splitlines():
                    if '#' in line:
                        code, comment = line.split('#', 1)
                        if not contains_chinese(comment):
                            new_source_lines.append(line)
                        else:
                            new_source_lines.append(code.rstrip())
                    else:
                        new_source_lines.append(line)
                cell['source'] = '\n'.join(new_source_lines)
        with open(file_path, 'w', encoding='utf-8') as file:
            nbformat.write(notebook, file)
    except nbformat.reader.NotJSONError:
        shutil.move(file_path, os.path.join(backup_folder, os.path.basename(file_path)))
        print(f'Moved invalid notebook to backup: {file_path}')

for root, dirs, files in os.walk('.'):
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith('.py'):
            remove_chinese_comments_py(file_path)
        elif file.endswith('.ipynb'):
            remove_chinese_comments_ipynb(file_path)

