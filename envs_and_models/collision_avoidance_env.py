# import julia
# j = julia.Julia()
# x = j.include("test.jl")
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gym
from numpy import random


class UnicycleEnv(gym.Env):
    def __init__(self):
        self.max_vr = 2
        self.max_ar = 4
        self.max_vt = np.pi
        self.dt = 0.1
        self.max_u = np.array([self.max_ar, self.max_vt])
        self.render_initialized = False
        self.step_cnt = 0
        self.obstacle_radius = 0.5
        self.obstacle_center_y = None
        self.time_total = 10
        self.fps = 10
        self.goal = [0, 5, 0, 0.5 * np.pi]
        self.rew_coef = np.array([1., 1., 1., 0.1])
        self.phi = None
        self.sis_info = dict()
        self.observation_space = gym.spaces.Box(low=np.array([-10.0, -10.0, -2.0, -np.pi]),
                                                high=np.array([10.0, 10.0, 2.0, np.pi]))
        self.action_space = gym.spaces.Box(low=np.array([-4.0, -np.pi])
                                           , high=np.array([4.0, np.pi]))
        # self.fig = plt.figure()
        plt.ion()

    def rk4(self, s, u, dt):
        dot_s1 = self.dynamics(s, u, dt)
        dot_s2 = self.dynamics(s + 0.5 * dt * dot_s1, u, dt)
        dot_s3 = self.dynamics(s + 0.5 * dt * dot_s2, u, dt)
        dot_s4 = self.dynamics(s + dt * dot_s3, u, dt)
        dot_s = (dot_s1 + 2 * dot_s2 + 2 * dot_s3 + dot_s4) / 6.0
        return dot_s

    def dynamics(self, s, u, dt):
        x = s[0]
        y = s[1]
        v = s[2]
        theta = s[3]

        # v = s[2] + 0.5*u[0]*dt
        # theta = s[3] + 0.5*u[1]*dt

        dot_x = v * np.cos(theta)
        dot_y = v * np.sin(theta)
        dot_v = u[0]
        dot_theta = u[1]

        dot_s = np.array([dot_x, dot_y, dot_v, dot_theta])
        return dot_s

    def step(self, u):
        u = self.filt_action(u)
        self.action = u
        dot_state = self.rk4(self.state, u, self.dt)
        # dot_state = self.dynamics(self.state, u, self.dt)
        self.state = self.state + dot_state * self.dt
        self.state = self.filt_state(self.state)
        rew = self.compute_reward()
        self.step_cnt += 1
        done = False
        if self.step_cnt >= 200:
            done = True
        # state, reward, done, info
        info = {}
        info.update(dict(cost=self.compute_cost()))
        old_phi = self.phi
        self.phi = self.adaptive_safety_index()
        if old_phi <= 0:
            delta_phi = max(self.phi, 0)
        else:
            delta_phi = self.phi - old_phi

        # update info dict
        info.update({'delta_phi': delta_phi})
        info.update(self.sis_info)
        return np.squeeze(np.array(self.state)), rew, done, info

    def reset(self):
        self.state = np.array([0., -1.5, 1. + random.random(),
                               random.random() * np.pi / 2 + np.pi / 4]) # random.random() * np.pi / 2 + np.pi / 4
        self.obstacle_center_y = random.random() - 0.5
        self.phi = self.adaptive_safety_index()
        self.ref = np.zeros_like(self.state)
        return self.state

    def compute_reward(self, mode='linear'):
        # x0 = [0, -1.5, 1+rand(), π/2+rand()*π/2-π/4]
        # xg = [0,5,0,-π]
        self.ref = self.get_ref()
        error = np.abs(self.state - np.array(self.ref))
        reward = - np.dot(error, self.rew_coef)
        return reward

    def compute_cost(self):
        obstacle_pos = np.array([0, self.obstacle_center_y])
        rela_pos = self.state[:2] - obstacle_pos
        d = np.linalg.norm(rela_pos)
        if d <= self.obstacle_radius:
            return 1.0
        else:
            return 0.0

    def get_ref(self):
        dt = 1 / self.fps
        dp = [self.goal[0] - self.state[0], self.goal[1] - self.state[1]]
        da = self.goal[3] - self.state[3]
        a = np.arctan(dp[1] / (dp[0] + 1e-8))
        if dp[1] > 0 and a < 0:
            a += np.pi
        elif dp[1] < 0 and a > 0:
            a += np.pi
        elif dp[1] < 0 and a < 0:
            a += 2 * np.pi
        v = np.linalg.norm(dp, ord=2) / self.time_total #todo: time total always?
        v = max(min(v, 1), -1)
        vx = v * np.cos(a)
        vy = v * np.sin(a)
        xref = [self.state[0] + vx * dt, self.state[1] + vy * dt, v, a]
        # xref[-1][2] = 0
        return xref


    def sample_action(self):
        action = np.random.uniform(-self.max_u, self.max_u)
        return action

    def filt_action(self, u):
        u = np.clip(u, self.action_space.low, self.action_space.high)
        return u

    def filt_state(self, x):
        while x[3] > np.pi:
            x[3] = x[3] - 2 * np.pi
        while x[3] < - np.pi:
            x[3] = x[3] + 2 * np.pi
        return x

    def adaptive_safety_index(self, k=2, sigma=0.04, n=2):
        '''
        synthesis the safety index that ensures the valid solution
        '''
        # initialize safety index

        '''
        function phi(index::CollisionIndex, x, obs)
            o = [obs.center; [0,0]]
            d = sqrt((x[1]-o[1])^2 + (x[2]-o[2])^2)
            dM = [x[1]-o[1], x[2]-o[2], x[3]*cos(x[4])-o[3], x[3]*sin(x[4])-o[4]]
            dim = 2
            dp = dM[[1,dim]]
            dv = dM[[dim+1,dim*2]]
            dot_d = dp'dv / d
            return (index.margin + obs.radius)^index.phi_power - d^index.phi_power - index.dot_phi_coe*dot_d
        end
        '''
        phi = -1e8
        sis_info_t = self.sis_info.get('sis_data', [])
        sis_info_tp1 = []

        obstacle_pos = np.array([0, self.obstacle_center_y])
        rela_pos = self.state[:2] - obstacle_pos
        d = np.linalg.norm(rela_pos)
        robot_to_hazard_angle = np.arctan((-rela_pos[1])/(-rela_pos[0]))
        vel_rela_angle = self.state[-1] - robot_to_hazard_angle
        dotd = self.state[2] * np.cos(vel_rela_angle)

        # if dotd <0, then we are getting closer to hazard
        sis_info_tp1.append((d, dotd))

        # compute the safety index
        phi_tmp = sigma + self.obstacle_radius ** n - d ** n - k * dotd
        # select the largest safety index
        if phi_tmp > phi:
            phi = phi_tmp

        # sis_info is a list consisting of tuples, len is num of obstacles
        self.sis_info.update(dict(sis_data=sis_info_tp1, sis_trans=(sis_info_t, sis_info_tp1)))
        return phi

    def get_unicycle_plot(self):
        theta = self.state[3]
        ang = (-self.state[3] + np.pi / 2) / np.pi * 180
        s = self.state[[0, 1]]
        t = self.state[[0, 1]] + np.hstack([np.cos(theta), np.sin(theta)])
        c = s
        s = s - (t - s)
        return np.hstack([s[0], t[0]]), np.hstack([s[1], t[1]])

    def render(self, mode='human'):
        plt.clf()
        ax = plt.axes()
        ax.add_patch(plt.Rectangle((-3.5,-2), 7.0, 7.0, edgecolor='black',
                                   facecolor='none', linewidth=2))
        plt.axis("equal")
        plt.axis('off')
        plt.arrow(self.state[0], self.state[1],
                  0.2 * np.cos(self.state[3]), 0.5 * np.sin(self.state[3]),
                  color='b', head_width=0.2)
        plt.plot([0,0], [-2, 5],color='b')
        plt.plot([-3,3], [0, 0], color='b')
        ax.add_patch(plt.Circle((0, self.obstacle_center_y), 0.5,
                                 edgecolor='black',facecolor='none',))
        ax.add_patch(plt.Circle(self.goal[:2], 0.2,
                                edgecolor='none', facecolor='red', ))
        # self.fig.canvas.flush_events()
        text_x, text_y_start = -6, 4
        ge = iter(range(0, 1000, 4))
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'ego_x: {:.2f}m'.format(self.state[0]))
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'ego_y: {:.2f}m'.format(self.state[1]))
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'ego_v: {:.2f}m/s'.format(self.state[2]))
        plt.text(text_x, text_y_start - 0.1 * next(ge),
                 r'ego_angle: ${:.2f}\degree$'.format(self.state[3] / np.pi * 180.))

        next(ge)
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'ref_x: {:.2f}m'.format(self.ref[0]))
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'ref_y: {:.2f}m'.format(self.ref[1]))
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'ref_v: {:.2f}m/s'.format(self.ref[2]))
        plt.text(text_x, text_y_start - 0.1 * next(ge),
                 r'ref_angle: ${:.2f}\degree$'.format(self.ref[3] / np.pi * 180.))
        next(ge)
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'action_v: {:.2f}m'.format(self.action[0]))
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'action_a: {:.2f}m'.format(self.action[1]))
        next(ge)
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'reward: {:.2f}m'.format(self.compute_reward()))
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'cost: {:.2f}m'.format(self.compute_cost()))
        next(ge)
        d = self.sis_info.get('sis_data')[0][0]
        dotd = self.sis_info.get('sis_data')[0][1]
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'd: {:.2f}m'.format(d))
        plt.text(text_x, text_y_start - 0.1 * next(ge), 'dotd: {:.2f}m/s'.format(dotd))
        time.sleep(1)
        plt.show()
        plt.pause(0.01)


def try_env():
    env = UnicycleEnv()
    env.reset()
    u = np.array([-1.0,0.0])
    while True:
        env.step(u)
        env.render()

if __name__ == '__main__':
    # plt.ion()
    try_env()