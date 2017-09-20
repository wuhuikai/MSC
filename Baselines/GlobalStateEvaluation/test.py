import os
import visdom
import pickle
import argparse

import numpy as np

LAMBDA = 0.99

def calc_value_acc(value_pre, value_gt):
    return np.mean(value_pre == value_gt)

def calc_weighted_value_acc(value_pre, value_gt, weight):
    return np.sum((value_pre == value_gt) * np.abs(weight)) / np.sum(np.abs(weight))

def show_test_result(name, phrase, result, steps=10, title=''):
    value_pres, value_gts = result

    ################################## Calc Acc #########################################
    weights = [(value_gt[0]*2-1)*LAMBDA**np.arange(len(value_gt)-1, -1, -1) for value_gt in value_gts]

    value_pres_np = np.hstack(value_pres)
    value_gts_np = np.hstack(value_gts)
    weights_np = np.hstack(weights)

    value_acc = calc_value_acc(value_pres_np, value_gts_np)
    weighted_value_acc = calc_weighted_value_acc(value_pres_np, value_gts_np, weights_np)
    print('\tValue Accuracy: {}%\tWeighted Value Accuracy: {}%'.format(value_acc*100, weighted_value_acc * 100))
    ################################### Plot ###################################################
    vis = visdom.Visdom(env=name + '[{}]'.format(phrase))

    value_pre_result, value_gt_result = [[] for _ in range(steps)], [[] for _ in range(steps)]
    weight_result = [[] for _ in range(steps)]
    for value_pre, value_gt, weight in zip(value_pres, value_gts, weights):
        if len(value_pre) < steps:
            continue

        step = len(value_pre) // steps
        for s in range(steps):
            value_pre_result[s].append(value_pre[s * step:(s + 1) * step])
            value_gt_result[s].append(value_gt[s * step:(s + 1) * step])
            weight_result[s].append(weight[s * step:(s + 1) * step])

    legend = ['Value', 'Weighted Value']
    X = np.repeat(np.arange(steps), len(legend), axis=0).reshape(steps, -1)
    Y = np.zeros((steps, len(legend)))
    for idx, (value_pres, value_gts, weights) in enumerate(
            zip(value_pre_result, value_gt_result, weight_result)):

        value_pres_np = np.hstack(value_pres)
        value_gts_np = np.hstack(value_gts)
        weights_np = np.hstack(weights)

        Y[idx, 0] = calc_value_acc(value_pres_np, value_gts_np)
        Y[idx, 1] = calc_weighted_value_acc(value_pres_np, value_gts_np, weights_np)

    vis.line(X=X, Y=Y,
             opts=dict(title='Acc[{}]'.format(title), legend=legend), win=title)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2C : Starcraft II')
    parser.add_argument('--name', type=str, default='StarCraft II:TvT',
                        help='Experiment name. All outputs will be stored in checkpoints/[name]/')
    parser.add_argument('--phrase', type=str, default='test',
                        help='val|test (default: test)')
    args = parser.parse_args()

    test_path = os.path.join('checkpoints', args.name, args.phrase)
    test_results = sorted([int(os.path.basename(result).split('.')[0].split('_')[-1])
                        for result in os.listdir(test_path)])

    for idx, name_id in enumerate(test_results):
        result_path = os.path.join(test_path, 'model_iter_{}.pth'.format(name_id))
        print('Processing {}/{} ...'.format(idx+1, len(test_results)))
        print('\t'+result_path)

        with open(result_path, 'rb') as f:
            result = pickle.load(f)

        show_test_result(args.name, args.phrase, result, title=idx)