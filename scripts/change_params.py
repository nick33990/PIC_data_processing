import os
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path')
    parser.add_argument('-P', '--params', nargs = '+')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.path, 'r') as f:
        content = f.read()

    eq = ' = ' if not '.cfg' in args.path else '='
    end = ';' if not '.cfg' in args.path else '\n'

    for p in args.params:
        name, value = p.split('=')
        if '.cfg' in args.path:
            values = value.replace('_', ' ')
        match_str = re.compile(rf'{name}{eq}.+{end}')
        res = re.search(match_str, content)
        if not res is None:
            content = content[:res.start() + len(rf'{name}{eq}')] + str(value) + content[res.end() - len(end) :] 
        else:
            print(f'\n\n\n{name} not found\n\n\n')
    
    with open(args.path, 'w') as f:
        f.write(content) 