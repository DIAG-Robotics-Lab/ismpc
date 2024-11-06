import numpy as np
from utils import *

class FootstepPlanner:
    def __init__(self, vref, initial_lfoot, initial_rfoot, params):
        default_ss_duration = params['ss_duration']
        default_ds_duration = params['ds_duration']

        unicycle_pos   = (initial_lfoot[3:5] + initial_rfoot[3:5]) / 2.
        unicycle_theta = (initial_lfoot[2]   + initial_rfoot[2]  ) / 2.
        support_foot   = params['first_swing']
        self.plan = []

        for j in range(len(vref)):
            # set step duration
            ss_duration = default_ss_duration
            ds_duration = default_ds_duration

            # exception for first step
            if j == 0:
                ss_duration = 0
                ds_duration = (default_ss_duration + default_ds_duration) * 2

            # exception for last step
            # to be added

            # move virtual unicycle
            for i in range(ss_duration + ds_duration):
                if j > 1:
                    unicycle_theta += vref[j][2] * params['world_time_step']
                    R = np.array([[np.cos(unicycle_theta), - np.sin(unicycle_theta)],
                                  [np.sin(unicycle_theta),   np.cos(unicycle_theta)]])
                    unicycle_pos += R @ vref[j][:2] * params['world_time_step']

            # compute step position
            displacement = 0.1 if support_foot == 'lfoot' else - 0.1
            displ_x = - np.sin(unicycle_theta) * displacement
            displ_y =   np.cos(unicycle_theta) * displacement
            pos = np.array((
                unicycle_pos[0] + displ_x, 
                unicycle_pos[1] + displ_y,
                0.))
            ang = np.array((0., 0., unicycle_theta))

            # add step to plan
            self.plan.append({
                'pos'        : pos,
                'ang'        : ang,
                'ss_duration': ss_duration,
                'ds_duration': ds_duration,
                'foot_id'    : support_foot
                })
            
            # switch support foot
            support_foot = 'rfoot' if support_foot == 'lfoot' else 'lfoot'

    def get_step_index_at_time(self, time):
        t = 0
        for i in range(len(self.plan)):
            t += self.plan[i]['ss_duration'] + self.plan[i]['ds_duration']
            if t > time: return i
        return None

    def get_start_time(self, step_index):
        t = 0
        for i in range(step_index):
            t += self.plan[i]['ss_duration'] + self.plan[i]['ds_duration']
        return t

    def get_phase_at_time(self, time):
        step_index = self.get_step_index_at_time(time)
        start_time = self.get_start_time(step_index)
        time_in_step = time - start_time
        if time_in_step < self.plan[step_index]['ss_duration']:
            return 'ss'
        else:
            return 'ds'