import os
import sys
import glob
import json
import numpy as np
import gflags as flags

FLAGS = flags.FLAGS
flags.DEFINE_string(name='hq_replays_path', default='high_quality_replays',
                    help='Paths for filtered replays')
flags.DEFINE_string(name='save_path', default='train_val_test',
                    help='Path for saving results')
flags.DEFINE_string(name='ratio', default='7:1:2',
                    help='train:val:test')
flags.DEFINE_integer(name='seed', default=1,
                    help='random seed')

def save(replays, prefix, folder):
    print('{}/{}: {}'.format(folder, prefix, len(replays)))
    with open(os.path.join(folder, prefix+'.json'), 'w') as f:
        json.dump(replays, f)

def main():
    np.random.seed(FLAGS.seed)
    ratio = np.asarray([float(i) for i in FLAGS.ratio.split(':')])
    ratio /= np.sum(ratio)

    if not os.path.isdir(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    replay_sets = glob.glob(os.path.join(FLAGS.hq_replays_path, '*.json'))
    for replay_set in replay_sets:
        folder = os.path.join(FLAGS.save_path, os.path.basename(replay_set).split('.')[0])
        if not os.path.isdir(folder):
            os.makedirs(folder)

        with open(replay_set) as f:
            replays = json.load(f)
        np.random.shuffle(replays)

        train_end = int(len(replays)*ratio[0])
        val_end = int(len(replays)*(ratio[0]+ratio[1]))

        save(replays[:train_end], 'train', folder)
        save(replays[train_end:val_end], 'val', folder)
        save(replays[val_end:], 'test', folder)

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()