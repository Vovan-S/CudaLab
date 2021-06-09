import re
import os

class Var:
    vtype: str
    vname: str

    def __init__(self, entry, explicit_type=None):
        self.array_sizes = re.findall(r'\[([\w\+\s\*]+)\]+', entry)
        no_arrays = re.split(r'\[', entry)[0]
        self.vname = [x.strip() for x in re.split(r'\W+', no_arrays)
                      if x.strip() != ''][-1]
        if explicit_type:
            self.vtype = explicit_type
        else:
            self.vtype = no_arrays[::-1].\
                        replace(self.vname[::-1], "", 1)[::-1].\
                        strip()
    def declare(self):
        return self.get_actual_type() + ' ' + self.vname

    def get_type(self):
        return self.vtype

    def get_actual_type(self):
        return self.vtype + ('*' * len(self.array_sizes))

    def get_name(self):
        return self.vname

    def is_pointer(self):
        return '*' in self.get_actual_type()

    def sizes_array(self):
        return '{ ' + ', '.join(f"(size_t) {x}" for x in self.array_sizes) + ' }'

    def __str__(self):
        return f"name=[{self.vname}], type=[{self.vtype}], sizes={self.array_sizes}"

class Function:
    name:str
    rtype: str
    args: list
    vlocals: list
    vshared: list
    includes: list
    body: str
    rootdir = '../cpp/Runner/processed'
    classesfile = '_all_classes.h'

    @staticmethod
    def add_varline(where, line):
        if isinstance(line, list):
            for l in line:
                Function.add_varline(where, l)
            return
        if ';' in line:
            for l in line.split(';'):
                Function.add_varline(where, l)
            return
        tokens = line.split(',')
        first = Var(tokens[0])
        where.append(first)
        for t in tokens[1:]:
            where.append(Var(t, first.vtype))

    def __init__(self, name, rtype, args, body, includes):
        self.name = name
        self.rtype = rtype
        self.args = [Var(x) for x in args.split(',')]
        lm = re.search(r'//\s*local[ \t\w]*\n([\w\*,;\[\]\s\n]*)', body)
        self.body = body
        self.vlocals = []
        self.vshared = []
        if lm:
            local_vars = [x.strip() for x in lm.group(1).split(';') if '__shared__' not in x and len(x.strip()) > 0]
            shared_vars = [x.replace('__shared__', '').strip()
                            for x in lm.group(1).split(';') if '__shared__' in x and len(x.strip()) > 0]
            self.add_varline(self.vlocals, local_vars)
            self.add_varline(self.vshared, shared_vars)
            self.body = body.replace(lm.group(), "")
        c = 0
        while "__syncthreads();" in self.body:
            c += 1
            self.body = self.body.replace("__syncthreads();", f"current_part = {c}; return 1;\nsync{c}: __asm nop\n", 1)
        if rtype == 'void':
            self.body += '\n\treturn 0;'
        enter_switch = "switch(current_part) {\ncase 0: goto enter; \n"
        for i in range(c):
            enter_switch += f"case {i+1}: goto sync{i+1};\n"
        enter_switch += "default: return -1;\n}\nenter:\n"
        self.body = enter_switch + self.body
        self.includes = includes

    def classname(self):
        return f"global__{self.name}"

    def memoryname(self):
        return f"memory__{self.name}"

    def class_definition(self):
        classname = self.classname()
        string = ''
        string += f'class {classname} : public Thread' + ' {\n\t'
        string += ';\n\t'.join(v.declare()
                               for v in self.args + self.vlocals)
        string += ";\n\n\t AbstractMemory* _shared;\n\n"
        string += "public:\n\t" + classname + \
                  f"({', '.join(v.declare() for v in self.args)}):\n\t\t"
        inits = ["_shared(nullptr)"] + \
                [f"{v.get_name()}({v.get_name()})" for v in self.args] + \
                [f"{v.get_name()}()" for v in self.vlocals]
        string += ',\n\t\t'.join(inits) + " {}\n"
        string += '\n\tbool usingShared() { return ' + \
                  ('true' if len(self.vshared) else 'false') + '; }\n'
        string += '\n\tAbstractMemory* buildSharedMemory() { return new ' + \
                  self.memoryname() + '(); }\n'
        string += '\n\tThread* build(dim3 threadId, AbstractMemory* shared) {\n\t\t'
        temp = f'{classname}* new_thread = new {classname}('
        string += temp + (',\n\t\t' + (' ' * len(temp))).\
                  join(v.get_name() for v in self.args) + ');\n\t\t'
        string += 'new_thread->_shared = shared;\n\t\tnew_thread->m_threadId = threadId;\n'+\
                  '\t\treturn new_thread;\n\t}\n'
        shared_init = ''
        if len(self.vshared) > 0:
            for i, v in zip(range(len(self.vshared)), self.vshared):
                if v.is_pointer():
                    shared_init += f'\n\t\t {v.declare()} = ({v.get_actual_type()}) ' + \
                                 f'_shared->getPtr({i});'
                else:
                    shared_init += f'\n\t\t {v.get_type()}& {v.get_name()} = ' + \
                                  f'*({v.get_type()}*) _shared->getPtr({i});'
            shared_init += '\n'
        string += '\n\tint run() {\n\n' + shared_init + self.body + '\n\t}\n};\n'
        return string

    def memory_definition(self):
        string = ''
        classname = self.memoryname()
        if len(self.vshared) == 0:
            return f'class {classname} : public AbstractMemory' + ' {\npublic:\n\t' +\
                   f'{classname}()' + ' {}\n\tvoid* getPrt(size_t) { return nullptr; }\n};\n'
        string += f"class {classname}" + ' : public AbstractMemory {\n\t'
        string += ';\n\t'.join(v.declare() for v in self.vshared) + ";\n"
        string += f"public:\n\t{classname}():" + ',\n\t\t'.join(v.get_name() + '()'
                                                               for v in self.vshared)
        string += ' {'
        destructor = ''
        for v in self.vshared:
            if not v.is_pointer():
                continue
            string += f'\n\t\tsize_t {v.get_name()}_dims[] = {v.sizes_array()};'
            string += f'\n\t\t{v.get_name()} = ({v.get_actual_type()}) initMultiarray({v.get_name()}_dims, '+\
                      f'{len(v.array_sizes) - 1}, sizeof({v.get_type()}));'
            destructor += f'\n\t\tsize_t {v.get_name()}_dims[] = {v.sizes_array()};'
            destructor += f'\n\t\tdeleteMultiarray((void**){v.get_name()}, {v.get_name()}_dims, '+\
                          f'{len(v.array_sizes) - 1});'
        string += '\n\t}'
        if len(destructor):
            string += f"\n\n\t~{classname}() " + "{" + destructor + "\n\t};\n"
        string += "\n\tvoid* getPtr(size_t index) {"
        for i, v in zip(range(len(self.vshared)), self.vshared):
            string += f'\n\t\tif ({i} == index) return '
            if v.is_pointer():
                string += f'{v.get_name()};'
            else:
                string += f'&{v.get_name()};'
        string += '\n\t\treturn nullptr;\n\t}\n};\n'
        return string

    def create_file(self):
        classname = f"global__{self.name}"
        with open(self.rootdir + f"/{classname}.h", mode='w') as out:
            out.write('#pragma once\n#include "../runner/Thread.h"\n')
            for inc in self.includes:
                out.write(str(inc) + '\n')
            out.write('\n' + self.memory_definition())
            out.write('\n' + self.class_definition())
        classes = self.rootdir + '/' + self.classesfile
        if os.path.exists(classes):
            with open(classes, mode='a') as f:
                f.write(f"#include \"{classname}.h\"\n")
        else:
            with open(classes, mode='w') as f:
                f.write("#pragma once\n" +\
                        "#include \"../runner/PseudoCuda.h\"\n" +\
                        "#include \"../runner/Runner.h\"\n" +\
                        f"#include \"{classname}.h\"\n")


if __name__ == "__main__":
    v = Var("size_t block_triag[THREADS_PER_BLOCK]")
    print(v)
    print(v.get_type())
    print(v.get_actual_type())
