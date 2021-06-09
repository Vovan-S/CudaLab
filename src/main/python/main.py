import os
import re

import util
from vars import Var, Function


cuda_filepath = '../cpp/Cuda'
prdirname = 'processed'
process_filepath = '../cpp/Runner/' + prdirname
vsfile = '../cpp/Runner/Runner.vcxproj'

# Очистим директорию для созданных классов

for root, dirs, files in os.walk(process_filepath):
    for f in files:
        os.remove(root + '/' + f)

# находим все __global__ функции

global_func = []

for root, dirs, files in os.walk(cuda_filepath):
    for file in files:
        with open(os.path.join(root, file)) as f:
            code = f.read()
            while True:
                m = re.search(r"__global__\s+(\w+)\s+(\w+)\s*\(([\w\s,\*&]*)\)\s*\{", code)
                if m == None:
                    break
            #for m in re.finditer(r"__global__\s+(\w+)\s+(\w+)\s*\(([\w\s,\*&]*)\)\s*\{", code):
                rval = m.group(1)
                name = m.group(2)
                args = m.group(3)
                body = util.get_body(code[m.end():])
                includes = re.findall(r'#include\s*["<][\w\.]+[">]', code)
                f = Function(name, rval, args, body, includes)
                f.create_file()
                global_func.append(f)
                #code = code.replace(m.group() + body + '}', '')
                code = code[:m.start()] + code[m.end() + len(body) + 1:]
        with open(os.path.join(process_filepath, file), mode='w') as f:
            clear_code = util.clear_comments(code).replace('#include "cuda_runtime.h"', '').replace('__global__', '')
            if file.endswith('.h') or file.endswith('.cuh'):
                clear_code += f"\n#include \"{Function.classesfile}\""
            else:
                lines = clear_code.split('\n')
                for i, line in zip(range(len(lines)), lines):
                    if len(line.strip()) and not line.startswith('#'):
                        lines = lines[:i] + [f"#include \"{Function.classesfile}\""] +\
                                lines[i:]
                        clear_code = '\n'.join(lines)
                        break
            f.write(clear_code)

vsheaders = ''
vssources = ''

# Заменяем все __global__ функции классами
for root, dirs, files in os.walk(process_filepath):
    for file in files:
        with open(os.path.join(root, file)) as f:
            code = f.read()
            for m in re.finditer(r"(\w+)\s*<<<([^;]+)>>>\s*\(([^;]+)\);", code):
                fun = next(filter(lambda x: x.name == m.group(1), global_func))
                code = code.replace(m.group(),
                             f"Runner::run({m.group(2)}, &{fun.classname()}({m.group(3)}));", 1)
            code = code.replace('__device__', '')
        with open(os.path.join(root, file), mode='w') as f:
            f.write(code)
        if file.endswith('.h') or file.endswith('.inc'):
            vsheaders += f'    <ClInclude Include="{prdirname}\{file}" />\n'
        else:
            vssources += f'    <ClCompile Include="{prdirname}\{file}" />\n'

with open(vsfile) as f:
    prop = f.read()
#print(prop)
ms = re.search(r'<ItemGroup>([^;]*)</ItemGroup>\s*<ItemGroup>([^;]*)</ItemGroup>', prop)
if ms:
    i = ms.group(1).find('    <ClInclude Include="runner\AbstractMemory.h" />')
    prop = prop[:ms.start(1)] + '\n   ' + vsheaders + prop[ms.start(1) + i : ]
else:
    print('Headers not matched')
ms = re.search(r'<ItemGroup>([^;]*)</ItemGroup>\s*<ItemGroup>([^;]*)</ItemGroup>', prop)
if ms:
    i = ms.group(2).find('    <ClCompile Include="runner\Block.cpp" />')
    prop = prop[:ms.start(2)] + '\n   ' + vssources + prop[ms.start(2) + i:]
else:
    print('Sources not matched')
#print(prop)
with open(vsfile, mode='w') as f:
    f.write(prop)
