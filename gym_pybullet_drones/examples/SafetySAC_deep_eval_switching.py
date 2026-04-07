'''
Deep batch evaluation for the switching safety filter.

Runs num_runs headless episodes (in parallel across num_workers processes) and
reports aggregate safety rate, success rate, and irrecoverable exit rate.

Key design choices vs. the single-run eval script:
  - No GUI / no Logger / no plot — headless batch only.
  - Each worker owns one CtrlAviary and one SafetySAC model; the env is reused
    (reset) between episodes rather than recreated.
  - Irrecoverable exits (hard-bound breach) terminate the episode immediately
    and are counted as a separate failure mode.
  - Each episode uses seed = run_index for reproducible but varied resets.
  - Targets for all runs are generated upfront in the main process, then
    distributed to workers as plain data.
'''

import os
import argparse
from collections import deque
import numpy as np
import multiprocessing as mp

import torch as th

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.isaacs.safety_sac import SafetySAC
from gym_pybullet_drones.envs.SafeHoverAviary import DisturbedEnvWrapper

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_DURATION_SEC = 6
DEFAULT_NUM_SEGMENTS = 1
DEFAULT_SEGMENT_PATH = True
DEFAULT_MODEL_PATH = "results/best_model.zip"
DEFAULT_RANDOM_INIT = False
DEFAULT_GUARD_DURATION = 10
DEFAULT_SAFETY_THRESHOLD = 0.25
DEFAULT_SAFE_TARGET = False
DEFAULT_BIAS_TARGET = False
DEFAULT_NUM_RUNS = 100
DEFAULT_NUM_WORKERS = max(1, mp.cpu_count() - 1)
DEFAULT_DISTURBED = False
DEFAULT_DISTURBANCE_BOUND = 0.2

SAFE_CTRL_FREQ = 30
ACTION_BUFFER_SIZE = SAFE_CTRL_FREQ // 2


##### Helper Functions #######

def segment(init_xyzs, goal_pos, segments):
    target_waypoints = np.zeros((segments, 3))
    tot_offset = goal_pos - init_xyzs
    seg_offset = tot_offset / segments
    target_waypoints[0] = init_xyzs
    for i in range(segments):
        target_waypoints[i] = target_waypoints[i - 1] + seg_offset
    return target_waypoints

def create_action_buffer():
    actions = [np.zeros((1, 4), dtype=np.float32) for _ in range(ACTION_BUFFER_SIZE)]
    return deque(actions, maxlen=ACTION_BUFFER_SIZE)

def build_safe_obs(state, act_buffer):
    base_state = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16]])
    obs = base_state.reshape(1, 12).astype(np.float32)
    for action in act_buffer:
        obs = np.hstack([obs, action])
    return obs

def normalized_to_rpm(action, hover):
    return hover * (1 + .05 * action)

def rpm_to_normalized(rpm, hover):
    return np.clip((rpm / hover - 1.0) / .05, -1, 1).astype(np.float32)

def random_target():
    return np.array([[np.random.uniform(-.75, .75),
                      np.random.uniform(-.75, .75),
                      np.random.uniform(0.2, 2.0)]])

def biased_target():
    intervals = [[-.9, -.5], [.5, .9]]
    z_intervals = [[.1, .5], [1.5, 1.9]]
    x_int = np.random.choice([0, 1])
    y_int = np.random.choice([0, 1])
    z_int = np.random.choice([0, 1])
    pos = np.array([[np.random.uniform(intervals[x_int][0], intervals[x_int][1]),
                     np.random.uniform(intervals[y_int][0], intervals[y_int][1]),
                     np.random.uniform(z_intervals[z_int][0], z_intervals[z_int][1])]])
    return pos, np.zeros((1, 3))

def safe_random_init(model, threshold):
   
    verified = False
    while not verified:
        candidate_pos, candidate_rpy = biased_target()
        # Drone at rest at candidate_pos — indices match build_safe_obs expectations
        state = np.zeros(16)
        state[0:3] = candidate_pos[0]
        init_obs = build_safe_obs(state, create_action_buffer())
        action, _ = model.predict(init_obs, deterministic=True)
        obs_t, _ = model.policy.obs_to_tensor(init_obs)
        act_t = th.as_tensor(action, device=model.device).float()
        with th.no_grad():
            q1, q2 = model.policy.critic(obs_t, act_t)
        if np.mean([q1.item(), q2.item()]) > threshold:
            verified = True
        else:
            print("unsafe config:", candidate_pos, "resampling!")
    return candidate_pos, candidate_rpy

### EPISODE LOOP ########################
def run_episode(env, model, ctrl, HOVER_RPM, target, start_pos, start_rpy,
                duration_sec, num_segments, segment_path, guard_duration,
                safety_threshold, seed):
   
    env.reset(seed=seed, options={})
    for c in ctrl:
        c.reset()

    waypoints = np.array([segment(start_pos[0], target[0], num_segments)])
    waypoint_ct = 0
    waypoint_dur = duration_sec / num_segments

    guard_active = False
    guard_timesteps = guard_duration
    guard_activated = 0
    safety_violations = 0
    reached = False
    action_buffer = create_action_buffer()
    action = np.zeros((1, 4))

    for i in range(int(duration_sec * env.CTRL_FREQ)):
        env.step(action)
        state = env.unwrapped._getDroneStateVector(0)
        drone_x, drone_y, drone_z = state[0], state[1], state[2]

        # Hard bounds exceeded: count as irrecoverable and end episode
        if abs(drone_x) > 1.5 or abs(drone_y) > 1.5 or drone_z > 2.5 or drone_z < 0:
            return guard_activated, safety_violations, reached, True

        if i != 0 and segment_path and (i % int(waypoint_dur * env.CTRL_FREQ) == 0):
            waypoint_ct = min(waypoint_ct + 1, num_segments - 1)

        # PID candidate action
        candidate = np.zeros((1, 4))
        target_wp = waypoints[0, waypoint_ct] if segment_path else target[0]
        candidate[0, :], _, _ = ctrl[0].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP, state=state,
            target_pos=target_wp, target_rpy=start_rpy[0],
        )

        # Query safety filter
        safe_obs = build_safe_obs(state, action_buffer)
        norm_candidate = rpm_to_normalized(candidate, HOVER_RPM)
        obs_t, _ = model.policy.obs_to_tensor(safe_obs)
        act_t = th.as_tensor(norm_candidate, device=model.device).float()
        with th.no_grad():
            q1, q2 = model.policy.critic(obs_t, act_t)
        q_mean = np.mean([q1.item(), q2.item()])

        if q_mean < safety_threshold and not guard_active:
            guard_active = True
            guard_timesteps = guard_duration
            guard_activated += 1

        if guard_active:
            norm_fallback, _ = model.predict(safe_obs, deterministic=True)
            action = normalized_to_rpm(norm_fallback, HOVER_RPM)
            norm_output = norm_fallback
            guard_timesteps -= 1
            if guard_timesteps <= 0:
                guard_active = False
                guard_timesteps = guard_duration
        else:
            action = candidate
            norm_output = norm_candidate

        action_buffer.append(norm_output.reshape(1, 4).astype(np.float32))

        margin = min(drone_z, 2 - drone_z, 1 - drone_y, 1 + drone_y, 1 - drone_x, 1 + drone_x)
        if margin <= 0:
            safety_violations += 1

        if not reached:
            if (abs(drone_x - target[0][0]) < .1 and
                    abs(drone_y - target[0][1]) < .1 and
                    abs(drone_z - target[0][2]) < .1):
                reached = True

    return guard_activated, safety_violations, reached, False


# ---- Worker process state (one env + model per process) ----

_worker_env = None
_worker_model = None
_worker_ctrl = None
_worker_hover_rpm = None
_worker_start_pos = None
_worker_start_rpy = None

def _worker_init(model_path, start_pos, start_rpy, drone, physics, sim_freq, disturbed, disturbance_bound):
    global _worker_env, _worker_model, _worker_ctrl, _worker_hover_rpm
    global _worker_start_pos, _worker_start_rpy
    th.set_num_threads(1)  # avoid thread contention across workers
    _worker_model = SafetySAC.load(model_path)
    _worker_env = CtrlAviary(
        drone_model=drone,
        num_drones=1,
        initial_xyzs=start_pos,
        initial_rpys=start_rpy,
        physics=physics,
        pyb_freq=sim_freq,
        ctrl_freq=SAFE_CTRL_FREQ,
        gui=False,
        user_debug_gui=False,
    )
    if disturbed:
        _worker_env = DisturbedEnvWrapper(_worker_env, disturbance_bound=disturbance_bound)
    _worker_hover_rpm = _worker_env.HOVER_RPM
    _worker_ctrl = [DSLPIDControl(drone_model=drone)]
    _worker_start_pos = start_pos
    _worker_start_rpy = start_rpy

def _worker_run(task):
    seed, target, duration_sec, num_segments, segment_path, guard_duration, safety_threshold = task
    return run_episode(
        _worker_env, _worker_model, _worker_ctrl, _worker_hover_rpm,
        target, _worker_start_pos, _worker_start_rpy,
        duration_sec, num_segments, segment_path, guard_duration, safety_threshold, seed,
    )


# ---- Entry point ----

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='Deep batch evaluation of switching safety filter')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,             type=DroneModel, metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,            type=Physics,    metavar='', choices=Physics)
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,        metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,       type=int,        metavar='')
    parser.add_argument('--segment_path',       default=DEFAULT_SEGMENT_PATH,       type=str2bool,   metavar='')
    parser.add_argument('--num_segments',       default=DEFAULT_NUM_SEGMENTS,       type=int,        metavar='')
    parser.add_argument('--model_path',         default=DEFAULT_MODEL_PATH,         type=str)
    parser.add_argument('--random_init',        default=DEFAULT_RANDOM_INIT,        type=str2bool,   metavar='')
    parser.add_argument('--guard_duration',     default=DEFAULT_GUARD_DURATION,     type=int,        metavar='')
    parser.add_argument('--safety_threshold',   default=DEFAULT_SAFETY_THRESHOLD,   type=float,      metavar='')
    parser.add_argument('--safe_target',        default=DEFAULT_SAFE_TARGET,        type=str2bool,   metavar='')
    parser.add_argument('--bias_target',        default=DEFAULT_BIAS_TARGET,        type=str2bool,   metavar='')
    parser.add_argument('--num_runs',           default=DEFAULT_NUM_RUNS,           type=int,        metavar='')
    parser.add_argument('--num_workers',        default=DEFAULT_NUM_WORKERS,        type=int,        metavar='', help='parallel worker processes (default: cpu_count-1)')
    parser.add_argument('--disturbed',          default=DEFAULT_DISTURBED,          type=str2bool,   metavar='', help='Whether to wrap env with DisturbedEnvWrapper during eval')
    parser.add_argument('--disturbance_bound',  default=DEFAULT_DISTURBANCE_BOUND,  type=float,      metavar='', help='Uniform disturbance bound applied to actions (symmetric, per-dim)')
    ARGS = parser.parse_args()

    if not os.path.isfile(ARGS.model_path):
        print(f"[ERROR] Model file not found: {ARGS.model_path}")
        exit(1)

    # Fixed start position — env is reused across episodes, so initial_xyzs must be constant
    if ARGS.random_init:
        start_pos = np.array([[np.random.uniform(-1, 1),
                               np.random.uniform(-1, 1),
                               np.random.uniform(0.1, 2.0)]])
    else:
        start_pos = np.array([[0., 0., 0.1]])
    start_rpy = np.zeros((1, 3))

    # Load model in main process only for target pre-generation, then free it
    model = SafetySAC.load(ARGS.model_path)
    print(f"[INFO] Generating {ARGS.num_runs} targets...")
    targets = []
    for i in range(ARGS.num_runs):
        if ARGS.safe_target:
            pos, _ = safe_random_init(model, ARGS.safety_threshold)
            targets.append(pos)
        elif ARGS.bias_target:
            pos, _ = biased_target()
            targets.append(pos)
        else:
            targets.append(random_target())
    del model  # free before forking workers

    tasks = [
        (i, targets[i], ARGS.duration_sec, ARGS.num_segments, ARGS.segment_path,
         ARGS.guard_duration, ARGS.safety_threshold)
        for i in range(ARGS.num_runs)
    ]

    print(f"[INFO] Running {ARGS.num_runs} episodes on {ARGS.num_workers} worker(s)...")
    initargs = (ARGS.model_path, start_pos, start_rpy,
                ARGS.drone, ARGS.physics, ARGS.simulation_freq_hz,
                ARGS.disturbed, ARGS.disturbance_bound)

    with mp.Pool(processes=ARGS.num_workers,
                 initializer=_worker_init,
                 initargs=initargs) as pool:
        results = list(pool.imap(_worker_run, tasks, chunksize=max(1, ARGS.num_runs // (ARGS.num_workers * 4))))
        # imap with chunksize gives progress as results come in; list() waits for all

    total = len(results)
    violations    = sum(1 for _, sv, _, _  in results if sv > 0)
    reached_num   = sum(1 for _, _,  r, _  in results if r)
    irrecov_num   = sum(1 for _, _,  _, ir in results if ir)

    print(f"\nNum Runs         = {total}")
    print(f"[RESULTS] Safety Rate:      {(total - violations) / total:.3f}  ({total - violations}/{total} violation-free)")
    print(f"[RESULTS] Success Rate:     {reached_num / total:.3f}  ({reached_num}/{total} reached target)")
    print(f"[RESULTS] Irrecoverable:    {irrecov_num / total:.3f}  ({irrecov_num}/{total} exited hard bounds)")
