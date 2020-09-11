def breakpoint():
    import pdb; pdb.set_trace()


def dist_tester():
    print('please use ')

# def save_viz():
#     viz_dict = {}
#     print('!!! Saving visualization data')
#     viz_dict['input'] = np.argmax(vfe_feat[0][0].cpu().numpy().reshape(16, 20, 384, 384), axis=1)
#     viz_dict['label'] = proj_labels[0].cpu().numpy()
#     for key, value in viz_dict.items():
#         print(key, value.shape)
#     np.save(f'{cfg.DIR}/{i}_viz_data.npy', arr=viz_dict)
# 