# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adapatation
# https://[arxiv link]
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import os
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
import torch


def _obs_allegro2hora(obses):
    obs_index = obses[0:4]
    obs_middle = obses[4:8]
    obs_ring = obses[8:12]
    obs_thumb = obses[12:16]
    obses = np.concatenate([obs_index, obs_thumb, obs_middle, obs_ring]).astype(np.float32)
    return obses


def _action_hora2allegro(actions):
    cmd_act = actions.copy()
    cmd_act[[4, 5, 6, 7]] = actions[[8, 9, 10, 11]]
    cmd_act[[12, 13, 14, 15]] = actions[[4, 5, 6, 7]]
    cmd_act[[8, 9, 10, 11]] = actions[[12, 13, 14, 15]]
    return cmd_act


class HardwarePlayer(object):
    def __init__(self, config):
        self.action_scale = 1 / 24
        self.actions_num = 16
        self.device = 'cuda'

        obs_shape = (144,)
        self.network_config = config.train.network
        net_config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'actor_units': self.network_config.mlp.units,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'priv_info': True,
            'proprio_adapt': True,
            'priv_info_dim': 9,
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()
        self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((30, 48)).to(self.device)
        self.sa_mean_std.eval()

    def deploy(self):
        import rospy
        from hora.algo.deploy.robots.allegro import Allegro
        # try to set up rospy
        rospy.init_node('example')
        allegro = Allegro(hand_topic_prefix='allegroHand_0')
        # Wait for connections.
        rospy.sleep(0.5)

        hz = 20
        ros_rate = rospy.Rate(hz)

        pose_init = [
            0.0627, 1.2923, 0.3383, 0.1088,
            0.0724, 1.1983, 0.1551, 0.1499,
            0.1343, 1.1736, 0.5355, 0.2164,
            1.1202, 1.1374, 0.8535, -0.0852,
        ]
        dof_lower = np.array([
            -0.4700, -0.1960, -0.1740, -0.2270, 0.2630, -0.1050, -0.1890, -0.1620,
            -0.4700, -0.1960, -0.1740, -0.2270, -0.4700, -0.1960, -0.1740, -0.2270
        ])
        dof_upper = np.array([
            0.4700, 1.6100, 1.7090, 1.6180, 1.3960, 1.1630, 1.6440, 1.7190,
            0.4700, 1.6100, 1.7090, 1.6180, 0.4700, 1.6100, 1.7090, 1.6180
        ])
        # command to the initial position
        allegro.command_joint_position(pose_init)
        input('press to run policy')

        for t in range(hz * 4):
            print(f'{t} / {hz * 4}')
            allegro.command_joint_position(pose_init)
            ros_rate.sleep()

        obses, _ = allegro.poll_joint_position(wait=True)
        obses = _obs_allegro2hora(obses)
        # hardware deployment buffer
        obs_buf = torch.from_numpy(np.zeros((1, 16 * 3 * 3)).astype(np.float32)).cuda()
        proprio_hist_buf = torch.from_numpy(np.zeros((1, 30, 16 * 3)).astype(np.float32)).cuda()

        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)

        prev_target = torch.from_numpy(obses[None].astype(np.float32)).cuda()
        cur_obs_buf = torch.from_numpy(unscale(obses, dof_lower, dof_upper)[None]).cuda()

        for i in range(3):
            obs_buf[:, i*16+0:i*16+16] = cur_obs_buf.clone()  # joint position
            obs_buf[:, i*16+16:i*16+32] = 0  # previous action
            obs_buf[:, i*16+32:i*16+48] = cur_obs_buf.clone()  # current target (obs_t-1 + s * act_t-1)

        proprio_hist_buf[:, :, :16] = cur_obs_buf.clone()
        proprio_hist_buf[:, :, 32:48] = cur_obs_buf.clone()

        while True:
            obs = self.running_mean_std(obs_buf.clone())
            input_dict = {
                'obs': obs,
                'proprio_hist': self.sa_mean_std(proprio_hist_buf.clone()),
            }
            action = self.model.act_inference(input_dict)
            action = torch.clamp(action, -1.0, 1.0)

            target = prev_target + self.action_scale * action
            target = torch.clip(target, torch.from_numpy(dof_lower).cuda(), torch.from_numpy(dof_upper).cuda())
            prev_target = target.clone()
            commands = target.cpu().numpy()[0]
            commands = _action_hora2allegro(commands)
            allegro.command_joint_position(commands)
            ros_rate.sleep()

            obses, torques = allegro.poll_joint_position(wait=True)
            obses = _obs_allegro2hora(obses)

            cur_obs_buf = torch.from_numpy(unscale(obses, dof_lower, dof_upper)[None]).cuda()
            prev_obs_buf = obs_buf[:, 48:].clone()
            obs_buf[:, :96] = prev_obs_buf
            obs_buf[:, 96:112] = cur_obs_buf.clone()
            obs_buf[:, 112:128] = action.clone()
            obs_buf[:, 128:144] = target.clone()

            priv_proprio_buf = proprio_hist_buf[:, 1:30, :].clone()
            cur_sa_buf = torch.cat([
                cur_obs_buf, action.clone(), target.clone()
            ], dim=-1)[:, None]
            proprio_hist_buf[:] = torch.cat([priv_proprio_buf, cur_sa_buf], dim=1)

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])
        self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])
