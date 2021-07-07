# Gradient Mappi
# ng Sampler
## Gaussian Sampler
# 2D configuration space infeasible set

import numpy as np
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import osqp
import copy


from gym.envs.robotics.utils import reset_mocap2body_xpos



import safe_rl.pg.run_agent

def quadprog(H, f, A=None, b=None,
             initvals=None, verbose=False):
    qp_P = sparse.csc_matrix(H)
    qp_f = np.array(f)
    qp_l = -np.inf * np.ones(len(b))
    qp_A = sparse.csc_matrix(A)
    qp_u = np.array(b)
    model = osqp.OSQP()
    model.setup(P=qp_P, q=qp_f,
                A=qp_A, l=qp_l, u=qp_u, verbose=verbose)
    if initvals is not None:
        model.warm_start(x=initvals)
    results = model.solve()
    return results.x, results.info.status


def chk_unsafe(s, point, dt_ratio, dt_adamba, env, threshold, margin, adaptive_k, adaptive_n, adaptive_sigma, trigger_by_pre_execute, pre_execute_coef):
    #safe_rl.pg.run_agent.CHECK_CNT += 1
    # flag=1 is unsafe, flag=0 is safe


    action = [point[0], point[1]]

    # save state of env
    stored_state = copy.deepcopy(env.sim.get_state())


    #stored_robot_position = env.robot_pos
    #mujoco_id = env.sim.model.body_name2id('robot')
    #stored_robot_body_jacp = copy.deepcopy(env.sim.data.body_jacp[mujoco_id])
    # print()
    # print("env state:",env.sim.get_state())
    # vel = env.sim.data.get_body_xvelp('robot')
    # print("robot vel:", env.sim.data.get_body_xvelp('robot'))
    # print("robot pos:",env.robot_pos)
    # print("hazards pos:", env.hazards_pos)
    # print("robot jacp:", stored_robot_body_jacp)

    

    #cost_now = env.cost()['cost']
    #projection_cost_now = env.projection_cost_max(margin=margin)
    safe_index_now = env.adaptive_safety_index(k=adaptive_k, sigma=adaptive_sigma, n=adaptive_n)

    # simulate the action
    s_new = env.step(action, simulate_in_adamba=True)
    #vel_after_tmp_action = env.sim.data.get_body_xvelp('robot')

    #cost_future = env.cost()['cost']
    #projection_cost_future = env.projection_cost_max(margin=margin)
    safe_index_future = env.adaptive_safety_index(k=adaptive_k, sigma=adaptive_sigma, n=adaptive_n)
    #dphi = cost_future - cost_now
    # projection_dphi
    # dphi = projection_cost_future - projection_cost_now
    dphi = safe_index_future - safe_index_now
    # if dphi < 0:
    #     print(dphi)

    # print(dphi)
    # print(threshold * dt_adamba)
    # print()
    assert dt_adamba == 0.02

    if trigger_by_pre_execute:
        if safe_index_future < pre_execute_coef:
            flag = 0  # safe
        else:
            flag = 1  # unsafe
    else:
        if dphi <= threshold * dt_adamba: #here dt_adamba = dt_env
            flag = 0  # safe
        else:
            flag = 1  # unsafe



    # reset env

    # set qpos and qvel
    env.sim.set_state(stored_state)
    
    # Note that the position-dependent stages of the computation must have been executed for the current state in order for these functions to return correct results. So to be safe, do mj_forward and then mj_jac. If you do mj_step and then call mj_jac, the Jacobians will correspond to the state before the integration of positions and velocities took place.
    env.sim.forward()
        
    
    # reset xpos
    # no need to to this again, sim.forward() is good and simple enough
    # mujoco_id = env.sim.model.body_name2id('robot')
    # env.sim.data.body_xpos[mujoco_id] = stored_robot_position
    # env.sim.data.body_jacp[mujoco_id] = stored_robot_body_jacp


    # print("\n --------after recovery------- \n")
    # print("env state2:",env.sim.get_state())
    # vel2 = env.sim.data.get_body_xvelp('robot')
    # print("robot vel_after_tmp_action:", vel_after_tmp_action)
    # print("robot vel2:", env.sim.data.get_body_xvelp('robot'))
    # print("vel_after_tmp_action angle:", np.arctan2(vel_after_tmp_action[1], vel_after_tmp_action[0]))
    # print("vel2_robot angle:", np.arctan2(env.sim.data.get_body_xvelp('robot')[1], env.sim.data.get_body_xvelp('robot')[0]))
    #angle_dif = (np.arctan2(env.sim.data.get_body_xvelp('robot')[1], env.sim.data.get_body_xvelp('robot')[0]) - np.arctan2(vel_after_tmp_action[1], vel_after_tmp_action[0]) )/np.pi *180
    # angle_dif = (np.arccos(np.dot(env.sim.data.get_body_xvelp('robot')[0:2], vel_after_tmp_action[0:2])/ (np.linalg.norm(env.sim.data.get_body_xvelp('robot')[0:2]) * np.linalg.norm(vel_after_tmp_action[0:2]))))/np.pi *180
    # if not np.isnan(angle_dif):

    #     safe_rl.pg.run_agent.ANGLE_DIF += abs(angle_dif)
    #     safe_rl.pg.run_agent.ANGLE_CNT += 1
    #     print("angle dif:", angle_dif)
    #     print("DIF:", safe_rl.pg.run_agent.ANGLE_DIF)
    #     print("CNT:",safe_rl.pg.run_agent.ANGLE_CNT )

    # print("angle dif:",  angle_dif)
    # print("robot pos2:",env.robot_pos)
    # print("hazards pos2:", env.hazards_pos)
    # print("robot jacp:", env.sim.data.body_jacp[mujoco_id])
    # print("robot jacp:", env.sim.data.get_body_jacp('robot', jacp=stored_robot_body_jacp))
    # if (vel2 != vel).any() or (env.robot_pos != stored_robot_position).any():
    #     safe_rl.pg.run_agent.VEL_CNT += 1
    #     print("VEL_CNT/CHECK_CNT: ", safe_rl.pg.run_agent.VEL_CNT, "/",safe_rl.pg.run_agent.CHECK_CNT)
    #     print("!"*10 + "  vel changed after recovery  " + "!"*10)
    #     exit(0)
    # print("\n\n\n\n")




    # exit(0)

    
    
    return flag, env


def outofbound(limit, p):
    # limit, dim*2
    # p, dim
    # flag=1 is out of bound
    flag = 0
    assert len(limit[0]) == 2
    for i in range(len(limit)):
        assert limit[i][1] > limit[i][0]
        if p[i] < limit[i][0] or p[i] > limit[i][1]:
            flag = 1
            break
    return flag


# def step(s, u, dt):
#     fx = np.array(s)
#     theta = s[2]
#     gx = np.array([[np.cos(theta)*dt, 0],
#           [np.sin(theta)*dt, 0],
#           [0, dt]])
#     u = np.array(u).T
#
#     # print(s)
#     # print(u)
#     # print(dt)
#     # print(gx)
#     # exit(0)
#
#     s_next = fx + np.dot(gx, u)
#     return s_next


def AdamBA_SC(s, u, env, threshold=0, dt_ratio=1.0, ctrlrange=10.0, margin=0.4, adaptive_k=3, adaptive_n=1, adaptive_sigma=0.04, trigger_by_pre_execute=False, pre_execute_coef=0.0):
    
    # start_time = time.time()
    infSet = []
    u = np.clip(u, -ctrlrange, ctrlrange)
    np.random.seed(0)

    # 2d case, 2 dimensional control signal
    # uniform sampling
    # offset = [0.5 0.5];
    # scale = [0.5 0.5];
    # action = scale. * rand(1, 2) + offset;

    # 这里就不转置了
    action_space_num = 2
    action = np.array(u).reshape(-1, action_space_num)

    dt_adamba = 0.002 * env.frameskip_binom_n * dt_ratio

    assert dt_ratio == 1
    # print("dt of env step= ", 0.002 * env.frameskip_binom_n, "s")
    # print("dt of adamBA= ", dt_adamba, "s")
    # exit(0)
    # limits = [-100, 100, -0.1 * pi / dt, 0.1 * pi / dt] # each row define th limits for one dimensional action
    limits = [[-ctrlrange, ctrlrange], [-ctrlrange, ctrlrange]]  # each row define the limits for one dimensional action
    NP = []

    # no need crop since se only need on sample

    # NP = np.clip(action, np.array([[-1,-1]]), np.array([[1, 1]]))
    NP = action
    # print(NP)
    # exit(0)

    # start_time = time.time()

    # generate direction
    NP_vec_dir = []
    NP_vec = []
    sigma_vec = [[1, 0], [0, 1]]
    vec_num = 10

    # num of actions input, default as 1
    for t in range(0, NP.shape[0]):
        vec_set = []
        vec_dir_set = []
        for m in range(0, vec_num):
            # vec_dir = np.random.multivariate_normal(mean=[0, 0], cov=sigma_vec)
            theta_m = m * (2 * np.pi / vec_num)
            vec_dir = np.array([np.sin(theta_m), np.cos(theta_m)]) / 2
            #vec_dir = vec_dir / np.linalg.norm(vec_dir)
            vec_dir_set.append(vec_dir)
            vec = NP[t]
            vec_set.append(vec)
            # print(vec)
            # print(vec_set)
            # print(vec_dir)
            # print(vec_dir_set)
            # exit(0)

        NP_vec_dir.append(vec_dir_set)
        NP_vec.append(vec_set)

    # print(NP_vec)
    # print(NP_vec_dir)
    # exit(0)
    bound = 0.0001

    # record how many boundary points have been found
    # collected_num = 0
    valid = 0
    cnt = 0
    out = 0
    yes = 0
    for n in range(0, NP.shape[0]):
        NP_vec_tmp = NP_vec[n]
        NP_vec_dir_tmp = NP_vec_dir[n]
        # print("NP_vec:\n",NP_vec)
        # print("NP_vec_dir_tmp:\n",NP_vec_dir_tmp)
        for v in range(0, vec_num):
            # if collected_num >= 2:
            #     break
            # collected_num = collected_num + 1  # one more instance
            # update NP_vec
            NP_vec_tmp_i = NP_vec_tmp[v]
            NP_vec_dir_tmp_i = NP_vec_dir_tmp[v]
            # print(NP_vec_tmp_i)
            # print(NP_vec_dir_tmp_i)
            # exit(0)
            eta = bound
            decrease_flag = 0
            # print(eta)
            
            while True: 
                # chk_start_time = time.time()
                flag, env = chk_unsafe(s, NP_vec_tmp_i, dt_ratio=dt_ratio, dt_adamba=dt_adamba, env=env,
                                       threshold=threshold, margin=margin, adaptive_k=adaptive_k, adaptive_n=adaptive_n, adaptive_sigma=adaptive_sigma,
                                       trigger_by_pre_execute=trigger_by_pre_execute, pre_execute_coef=pre_execute_coef)

                # chk_end_time = time.time()
                #safe_rl.pg.run_agent.TIME_SIMULATION += (chk_end_time - chk_start_time)
                # if flag == 0:
                #     print("v=",v)
                #     print(NP_vec_tmp_i)

                # print(flag)
                # print(eta)
                    # exit(0)
                # check if the action is out of limit, if yes, the abandon

                # safety gym env itself has clip operation inside
                if outofbound(limits, NP_vec_tmp_i):
                    # print("\nout\n")
                    # collected_num = collected_num - 1  # not found, discard the recorded number
                    break

                if eta < bound and flag==0:
                    break

                # AdamBA procudure
                if flag == 1 and decrease_flag == 0:
                    # outreach
                    NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                    eta = eta * 2
                    continue
                # monitor for 1st reaching out boundary
                if flag == 0 and decrease_flag == 0:
                    decrease_flag = 1
                    eta = eta * 0.25  # make sure decrease step start at 0.5 of last increasing step
                    continue
                # decrease eta
                if flag == 1 and decrease_flag == 1:
                    NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                    eta = eta * 0.5
                    continue
                if flag == 0 and decrease_flag == 1:
                    NP_vec_tmp_i = NP_vec_tmp_i - eta * NP_vec_dir_tmp_i
                    eta = eta * 0.5
                    continue

            NP_vec_tmp[v] = NP_vec_tmp_i

        # exit(0)
        # discard those points that are out of boundary and not expanded

        NP_vec_tmp_new = []
        # print("NP_vec_tmp: ",NP_vec_tmp)
        # exit(0)

        # print(u)
        # print(NP_vec_tmp)
        for vnum in range(0, len(NP_vec_tmp)):
            # print(vnum)
            # print(NP_vec_tmp[vnum])
            # print(NP_vec_tmp)
            # print(len(NP_vec_tmp))
            # exit(0)
            cnt += 1
            if outofbound(limits, NP_vec_tmp[vnum]):
                # print("out")
                out += 1
                continue
            if NP_vec_tmp[vnum][0] == u[0][0] and NP_vec_tmp[vnum][1] == u[0][1]:
                # print("yes")
                yes += 1
                continue

            valid += 1
            NP_vec_tmp_new.append(NP_vec_tmp[vnum])
        # print("out = ", out)
        # print("yes = ", yes)
        # print("valid = ", valid)
        # update NP_vec

        NP_vec[n] = NP_vec_tmp_new
        # print(NP_vec_tmp_new)
        # exit(0)

    # print("collected_num: ",collected_num)
    # end_time = time.time()
    # exit(0)

    # start to get the A and B for the plane
    # print(NP_vec)

    NP_vec_tmp = NP_vec[0]
    # print("NP_vec: ", NP_vec)
    # print("NP_vec_tmp: ", NP_vec_tmp)
    if valid > 0:
        valid_adamba_sc = "adamba_sc success"
    elif valid == 0 and yes==vec_num:
        valid_adamba_sc = "itself satisfy"
    elif valid == 0 and out==vec_num:
        valid_adamba_sc = "all out"
    else:
        valid_adamba_sc = "exception"
        print("out = ", out)
        print("yes = ", yes)
        print("valid = ", valid)
        
    # print("\n"*3)
    if len(NP_vec_tmp) > 0:  # at least we have one sampled action satisfying the safety index 
        min_distance = 1e8
        index = 0
        for i in range(0, len(NP_vec_tmp)):
            assert u.shape[0] == 1 # assume only one action is passed to adamba_sc
            action_distance = np.linalg.norm(NP_vec_tmp[i] - u[0]) 
            if action_distance < min_distance:
                min_distance = action_distance

                index = i

        
        # safe_rl.pg.run_agent.DIS += min_distance
        # safe_rl.pg.run_agent.DIS_CNT += 1
        # end_time = time.time()
        # safe_rl.pg.run_agent.TIME_ALL += (end_time - start_time)
        return NP_vec_tmp[index], valid_adamba_sc, env, NP_vec_tmp
    else:
        # end_time = time.time()
        # safe_rl.pg.run_agent.TIME_ALL += (end_time - start_time)
        return None, valid_adamba_sc, env, None
