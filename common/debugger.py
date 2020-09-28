def breakpoint():
    import pdb; pdb.set_trace()


def dist_tester():
    print('please use ')

def print_group(values, names=None):
    if names is None:
        for j in range(len(values)):
            print(values[j], '\n')
    else:
        for j in range(len(names)):
            print(names[j], '\n', values[j], '\n')
