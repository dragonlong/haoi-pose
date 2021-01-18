def breakpoint():
    import pdb; pdb.set_trace()

def bp():
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

# def print_group(names, values):
#     for subname, subvalue in zip(names, values):
#         print(subname, ':\n', subvalue)
#         print('')
# # np.savetxt(f'{save_path}/contact_err_{domain}.txt', contacts_err, fmt='%1.5f', delimiter='\n')

def check_h5_keys(hf, verbose=False):
    print('')
    if verbose:
        for key in list(hf.keys()):
            print(key, hf[key][()].shape)
