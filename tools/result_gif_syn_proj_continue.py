from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import _init_paths
from lib.vis_utils import plot3d_pts, hist_show, plot2d_img


# Create the frames
frames = []
category_objs = ['eyeglasses', 'oven', 'washing_machine', 'laptop']
viz_intances  = {'eyeglasses': ['0002', '0005', '0008', '0016', '0036'],
                'oven':['0002', '0003', '0005', '0018', '0026', '0041', '0042'],
                'washing_machine':['0005', '0011', '0013', '0014', '0021', '0025', '0033', '0037', '0042'],
                'laptop': ['0001', '0002', '0009', '0017', '0035']}
viz_all = {}

for category_obj in category_objs:
    viz_all[category_obj] = {}
    # base_path = '/work/cascades/lxiaol9/6DPOSE/shape2motion/results/{}_demo/arti/proj'.format(category_obj)
    base_path = '/work/cascades/lxiaol9/6DPOSE/shape2motion/results/{}_demo/continuous/proj'.format(category_obj)
    for instance in viz_intances[category_obj][0:5]:
        frames_ins = []
        imgs_pred = glob.glob("{}/{}_*_depth.png".format(base_path, instance))
        imgs_pred.sort()
        # for img_pred in imgs_pred:
        for idx in range(2, len(imgs_pred) + 1, 2):
            try:
                img_pred   = "{}/{}_{}_0_depth.png".format(base_path, instance, idx)
                frame_pred = Image.open(img_pred)
                w, h = frame_pred.size
                off_w = 450
                off_h = 200
                frame_pred = frame_pred.crop((off_w, off_h+150, w-off_w, h-off_h))
            except:
                continue
            print('adding ', img_pred)
            frames_ins.append(frame_pred)
        viz_all[category_obj][instance] = frames_ins

# plot & save figure
nrows   = len(category_objs)
ncols   = 5
heights = [50 for a in range(nrows)]
widths  = [50 for a in range(ncols)]
cmaps = [['viridis', 'binary'], ['plasma', 'coolwarm'], ['Greens', 'copper']]
fig_width  = 8  # inches
fig_height = fig_width * sum(heights) / sum(widths)

for k in range(15):
    fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios':heights}, dpi=400) #define to be 2 rows, and 4cols.
    for i in range(nrows):
        category_obj = category_objs[i]
        for j in range(ncols):
            instance = viz_intances[category_obj][j]
            # image= cv2.imread(path_base+object_names[i]+'/output/{:06d}.png'.format(200+j))
            image = viz_all[category_obj][instance][k]
            axes[i, j].imshow(image)
            axes[i, j].axis('off')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0.01, wspace = 0.01)
    # plt.show()
    print('saving ', './results/final_{}.png'.format(k))
    fig.savefig('./results/final_{}.png'.format(k), pad_inches=0)
    plt.close()

# load figure & append
imgs_all = glob.glob("./results/final*.png")
imgs_all.sort()
# for img_pred in imgs_all:
for idx in range(13):
    try:
        img_pred = "./results/final_{}.png".format(idx)
        frame_pred = Image.open(img_pred)
    except:
        continue
    frames.append(frame_pred)

# Save into a GIF file that loops forever
print('saving ', './results/pred_proj_final400.gif')
print('we have {} images ', len(frames))
frames[0].save('./results/pred_proj_final400.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=500, loop=0)
