from __future__ import print_function

import os
import json
import time
import pickle
import argparse

import visdom
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

from Baselines.GlobalStateEvaluation.test import show_test_result

from data_loader.BatchEnv import BatchSpatialEnv

class BuildOrderGRU(torch.nn.Module):
    def __init__(self, n_channels, n_features, n_actions):
        super(BuildOrderGRU, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)

        self.linear_g = nn.Linear(n_features, 128)

        self.linear = nn.Linear(1280, 512)
        self.rnn = nn.GRUCell(input_size=512, hidden_size=128)

        self.actor_linear = nn.Linear(128, n_actions)

        self.h = None

    def forward(self, states_S, states_G, require_init):
        batch = states_S.size(1)
        if self.h is None:
            self.h = Variable(states_S.data.new().resize_((batch, 128)).zero_())
        elif True in require_init:
            h= self.h.data
            for idx, init in enumerate(require_init):
                if init:
                    h[idx].zero_()
            self.h = Variable(h)
        else:
            pass

        values = []
        for idx, (state_S, state_G) in enumerate(zip(states_S, states_G)):
            x_s = F.relu(self.conv1(state_S))
            x_s = F.relu(self.conv2(x_s))
            x_s = x_s.view(-1, 1152)

            x_g = F.relu(self.linear_g(state_G))

            x = torch.cat((x_s, x_g), 1)
            x = F.relu(self.linear(x))

            self.h = self.rnn(x, self.h)

            values.append(self.actor_linear(self.h))

        return values

    def detach(self):
        if self.h is not None:
            self.h.detach_()

def train(model, env, args):
    #################################### PLOT ###################################################
    STEPS = 10
    LAMBDA = 0.99
    vis = visdom.Visdom(env=args.name+'[{}]'.format(args.phrase))
    pre_per_replay = [[] for _ in range(args.n_replays)]
    gt_per_replay = [[] for _ in range(args.n_replays)]
    acc = None
    win = vis.line(X=np.zeros(1), Y=np.zeros(1))
    loss_win = vis.line(X=np.zeros(1), Y=np.zeros(1))

    #################################### TRAIN ######################################################
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    gpu_id = args.gpu_id
    with torch.cuda.device(gpu_id):
        model = model.cuda() if gpu_id >= 0 else model
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epoch = 0
    save = args.save_intervel
    env_return = env.step(reward=False, action=True)
    if env_return is not None:
        (states_S, states_G, actions_gt), require_init = env_return
    with torch.cuda.device(gpu_id):
        states_S = torch.from_numpy(states_S).float()
        states_G = torch.from_numpy(states_G).float()
        actions_gt = torch.from_numpy(actions_gt).long().squeeze()
        weight = torch.ones((env.n_actions,))
        weight[-1] = 0.05
        if gpu_id >= 0:
            states_S = states_S.cuda()
            states_G = states_G.cuda()
            actions_gt = actions_gt.cuda()
            weight = weight.cuda()

    while True:
        actions = model(Variable(states_S), Variable(states_G), require_init)
        action_loss = 0
        for action, action_gt in zip(actions, actions_gt):
            action_loss = action_loss + F.cross_entropy(action, Variable(action_gt), weight=weight)
        action_loss = action_loss / len(actions_gt)

        model.zero_grad()
        action_loss.backward()
        optimizer.step()
        model.detach()

        if env.epoch > epoch:
            epoch = env.epoch
            for p in optimizer.param_groups:
                p['lr'] *= 0.5

        ############################ PLOT ##########################################
        vis.updateTrace(X=np.asarray([env.step_count()]),
                        Y=np.asarray(action_loss.data.cpu().numpy()),
                        win=loss_win,
                        name='action')

        actions_np = np.swapaxes(np.asarray([np.argmax(action.data.cpu().numpy(), axis=1) for action in actions]), 0, 1)
        actions_gt_np = np.swapaxes(actions_gt.cpu().numpy(), 0, 1)

        for idx, (action, action_gt, init) in enumerate(zip(actions_np, actions_gt_np, require_init)):
            if init and len(pre_per_replay[idx]) > 0:
                pre_per_replay[idx] = np.asarray(pre_per_replay[idx], dtype=np.uint8)
                gt_per_replay[idx] = np.asarray(gt_per_replay[idx], dtype=np.uint8)

                step = len(pre_per_replay[idx]) // STEPS
                if step > 0:
                    acc_tmp = []
                    for s in range(STEPS):
                        action_pre = pre_per_replay[idx][s*step:(s+1)*step]
                        action_gt = gt_per_replay[idx][s*step:(s+1)*step]
                        acc_tmp.append(np.mean(action_pre == action_gt))

                    acc_tmp = np.asarray(acc_tmp)
                    if acc is None:
                        acc = acc_tmp
                    else:
                        acc = LAMBDA * acc + (1-LAMBDA) * acc_tmp

                    if acc is None:
                        continue
                    for s in range(STEPS):
                        vis.updateTrace(X=np.asarray([env.step_count()]),
                                        Y=np.asarray([acc[s]]),
                                        win=win,
                                        name='{}[{}%~{}%]'.format('action', s*10, (s+1)*10))
                    vis.updateTrace(X=np.asarray([env.step_count()]),
                                    Y=np.asarray([np.mean(acc)]),
                                    win=win,
                                    name='action[TOTAL]')

                pre_per_replay[idx] = []
                gt_per_replay[idx] = []

            pre_per_replay[idx].append(action[-1])
            gt_per_replay[idx].append(action_gt[-1])

        ####################### NEXT BATCH ###################################
        env_return = env.step(reward=False, action=True)
        if env_return is not None:
            (raw_states_S, raw_states_G, raw_rewards), require_init = env_return
            states_S = states_S.copy_(torch.from_numpy(raw_states_S).float())
            states_G = states_G.copy_(torch.from_numpy(raw_states_G).float())
            actions_gt = actions_gt.copy_(torch.from_numpy(raw_rewards).long().squeeze())

        if env.step_count() > save or env_return is None:
            save = env.step_count()+args.save_intervel
            torch.save(model.state_dict(),
                       os.path.join(args.model_path, 'model_iter_{}.pth'.format(env.step_count())))
            torch.save(model.state_dict(), os.path.join(args.model_path, 'model_latest.pth'))
        if env_return is None:
            env.close()
            break

def test(model, env, args):
    ######################### SAVE RESULT ############################
    action_pre_per_replay = [[]]
    action_gt_per_replay = [[]]

    ######################### TEST ###################################
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    gpu_id = args.gpu_id
    with torch.cuda.device(gpu_id):
        model = model.cuda() if gpu_id >= 0 else model
    model.eval()

    env_return = env.step(reward=False, action=True)
    if env_return is not  None:
        (states_S, states_G, actions_gt), require_init = env_return
    with torch.cuda.device(gpu_id):
        states_S = torch.from_numpy(states_S).float()
        states_G = torch.from_numpy(states_G).float()
        actions_gt = torch.from_numpy(actions_gt).float()
        if gpu_id >= 0:
            states_S = states_S.cuda()
            states_G = states_G.cuda()
            actions_gt = actions_gt.cuda()

    while True:
        actions = model(Variable(states_S), Variable(states_G), require_init)
        ############################ PLOT ##########################################
        actions_np = np.squeeze(np.vstack([np.argmax(action.data.cpu().numpy(), axis=1) for action in actions]))
        actions_gt_np = np.squeeze(actions_gt.cpu().numpy())

        if require_init[-1] and len(action_gt_per_replay[-1]) > 0:
            action_pre_per_replay[-1] = np.ravel(np.hstack(action_pre_per_replay[-1]))
            action_gt_per_replay[-1] = np.ravel(np.hstack(action_gt_per_replay[-1]))

            action_pre_per_replay.append([])
            action_gt_per_replay.append([])

        action_pre_per_replay[-1].append(actions_np)
        action_gt_per_replay[-1].append(actions_gt_np)
        ########################### NEXT BATCH #############################################
        env_return = env.step(reward=False, action=True)
        if env_return is not None:
            (raw_states_S, raw_states_G, raw_actions), require_init = env_return
            states_S = states_S.copy_(torch.from_numpy(raw_states_S).float())
            states_G = states_G.copy_(torch.from_numpy(raw_states_G).float())
            actions_gt = actions_gt.copy_(torch.from_numpy(raw_actions).float())
        else:
            action_pre_per_replay[-1] = np.ravel(np.hstack(action_pre_per_replay[-1]))
            action_gt_per_replay[-1] = np.ravel(np.hstack(action_gt_per_replay[-1]))

            env.close()
            break

    return action_pre_per_replay, action_gt_per_replay

def next_path(model_folder, paths):
    models = {int(os.path.basename(model).split('.')[0].split('_')[-1])
                for model in os.listdir(model_folder) if 'latest' not in model}
    models_not_process = models - paths
    if len(models_not_process) == 0:
        return None
    models_not_process = sorted(models_not_process)
    paths.add(models_not_process[0])

    return os.path.join(model_folder, 'model_iter_{}.pth'.format(models_not_process[0]))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Global State Evaluation : StarCraft II')
    parser.add_argument('--name', type=str, default='StarCraft II:TvT[BuildOrder:Spatial]',
                        help='Experiment name. All outputs will be stored in checkpoints/[name]/')
    parser.add_argument('--replays_path', default='train_val_test/Terran_vs_Terran',
                        help='Path for training, validation and test set (default: train_val_test/Terran_vs_Terran)')
    parser.add_argument('--race', default='Terran', help='Which race? (default: Terran)')
    parser.add_argument('--enemy_race', default='Terran', help='Which the enemy race? (default: Terran)')
    parser.add_argument('--phrase', type=str, default='train',
                        help='train|val|test (default: train)')
    parser.add_argument('--gpu_id', default=0, type=int, help='Which GPU to use [-1 indicate CPU] (default: 0)')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')

    parser.add_argument('--n_steps', type=int, default=20, help='# of forward steps (default: 20)')
    parser.add_argument('--n_replays', type=int, default=32, help='# of replays (default: 32)')
    parser.add_argument('--n_epoch', type=int, default=10, help='# of epoches (default: 10)')

    parser.add_argument('--save_intervel', type=int, default=1000000,
                        help='Frequency of model saving (default: 1000000)')
    args = parser.parse_args()

    args.save_path = os.path.join('checkpoints', args.name)
    args.model_path = os.path.join(args.save_path, 'snapshots')

    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))
    print('-------------- End ----------------')

    if args.phrase == 'train':
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        if not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)
        with open(os.path.join(args.save_path, 'config'), 'w') as f:
            f.write(json.dumps(vars(args)))

        env = BatchSpatialEnv()
        env.init(os.path.join(args.replays_path, '{}.json'.format(args.phrase)),
                    './', args.race, args.enemy_race, n_steps=args.n_steps, seed=args.seed,
                        n_replays=args.n_replays, epochs=args.n_epoch)
        model = BuildOrderGRU(env.n_channels, env.n_features, env.n_actions)
        train(model, env, args)
    elif 'val' in args.phrase or 'test' in args.phrase:
        test_result_path = os.path.join(args.save_path, args.phrase)
        if not os.path.isdir(test_result_path):
            os.makedirs(test_result_path)

        dataset_path = 'test.json' if 'test' in args.phrase else 'val.json'
        paths = set()
        while True:
            path = next_path(args.model_path, paths)
            if path is not None:
                print('[{}]Testing {} ...'.format(len(paths), path))

                env = BatchSpatialEnv()
                env.init(os.path.join(args.replays_path, dataset_path),
                            './', args.race, args.enemy_race, n_steps=args.n_steps,
                                            seed=args.seed, n_replays=1, epochs=1)
                model = BuildOrderGRU(env.n_channels, env.n_features, env.n_actions)
                model.load_state_dict(torch.load(path))
                result = test(model, env, args)
                with open(os.path.join(test_result_path, os.path.basename(path)), 'wb') as f:
                    pickle.dump(result, f)
                show_test_result(args.name, args.phrase, result, title=len(paths)-1)
            else:
                time.sleep(60)

if __name__ == '__main__':
    main()
