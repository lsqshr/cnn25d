import argparse
from matplotlib import pyplot as plt
import os
import h5py


def plot_history(legends=[], losses=[], imgpath='plot.eps'):
    # summarize history for loss
    plt.figure()

    linestyles = ['-', '--', ':', '-.']
    for i, l in enumerate(losses):
        plt.plot(l, linestyle=linestyles[i])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legends, loc='upper right')
    plt.savefig(imgpath)

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
        help='The input models to plot. Can be multiple H5 file. '
    )

    parser.add_argument(
        '-o',
        '--out_img',
        type=str,
        default='plot.eps',
        required=False,
        help='The output image path.'
    )

    parser.add_argument('--val', dest='val', action='store_true')
    parser.set_defaults(val=False)

    args = parser.parse_args()
    legends, losses = [], []

    for h5 in args.in_model:
        model = h5py.File(h5, 'r')
        losses.append(model['history/loss'])
        legends.append('%s-loss' % os.path.basename(h5))
        if args.val:
            losses.append(model['history/val_loss'])
            legends.append('%s-val_loss' % os.path.basename(h5))

    plot_history(legends, losses, args.out_img)
    plt.show()
