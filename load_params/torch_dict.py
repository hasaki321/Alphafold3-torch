elements = []
with open('load_params/state_dict.log','r') as f:
    for line in f.readlines():
        if line.strip():
            elements.extend(line.strip().split(' ')[0].split('.'))
print('\n'.join(sorted(list(set(elements)))))