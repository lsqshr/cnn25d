import argparse
import matplotlib
from matplotlib import pyplot as plt
import os
import h5py
import numpy as np


def plot_history(legends=[], losses=[], imgpath='plot.eps', legend_loc='lower_left'):
    # summarize history for loss
    plt.figure()

    linestyles = ['-', '--', ':', '-.']
    for i, l in enumerate(losses):
        plt.plot(l, linestyle=linestyles[i])

    matplotlib.rcParams.update({'font.size': 18}) 
    plt.yscale('logit')
    plt.xlim(xmin=2)
    plt.ylim(ymax=0.17)
    plt.ylim(ymin=0.0)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legends, loc=legend_loc)
    if imgpath is not None: 
        plt.savefig(imgpath, dpi=1000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments to plot the training history.')

    parser.add_argument(
        '-i',
        '--in_model',
        type=str,
        default=[],
        required=True,
        nargs='+',
        help='the input models to plot. can be multiple h5 file. '
    )

    parser.add_argument(
        '--legends',
        type=str,
        default=[],
        required=False,
        nargs='+',
    )

    parser.add_argument(
        '--legend_loc',
        type=str,
        default='best',
        required=False
    )

    parser.add_argument(
        '-o',
        '--out_img',
        type=str,
        default=None,
        required=False,
        help='The output image path.'
    )

    parser.add_argument('--val', dest='val', action='store_true')
    parser.set_defaults(val=False)

    args = parser.parse_args()
    legends, losses = [] if len(args.legends) == 0 else args.legends, []

    for h5 in args.in_model:
        model = h5py.File(h5, 'r')
        losses.append(model['history/loss'])
        if len(args.legends) == 0:
            legends.append('%s-loss' % os.path.basename(h5))
        if args.val:
            losses.append(model['history/val_loss'])
            if len(args.legends) == 0:
                legends.append('%s-val_loss' % os.path.basename(h5))

    plot_history(legends, losses, args.out_img, legend_loc=args.legend_loc)

    if args.out_img is None:
        plt.show()
