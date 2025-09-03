import os
import random
import argparse
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch

from DAG import DAGTasks
from util import OffloadEnv, make_agents, AverageMeter
from model import PrioritizedReplayBuffer
from federated_learning import build_federated_hierarchy, PrivacyBudgetAllocator

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def make_writer(log_root: str):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root, ts)
    os.makedirs(log_dir, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    print("TensorBoard log dir:", os.path.abspath(log_dir))
    return SummaryWriter(log_dir=log_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda"])
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--num_tasks", type=int, default=20)
    p.add_argument("--num_vehicles", type=int, default=8)
    p.add_argument("--num_rsus", type=int, default=2)
    p.add_argument("--agg_every", type=int, default=10)
    p.add_argument("--log_root", type=str, default="./runs/fed_iov_dp_dqn_v3")
    p.add_argument("--epsilon_total", type=float, default=20.0)  # 提高预算
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    print(f"Using device: {device}")

    num_tasks = args.num_tasks
    max_layers = 6
    num_vehicles = args.num_vehicles
    num_rsus = args.num_rsus
    episodes = args.episodes
    steps_per_episode_cap = num_tasks

    buffer_capacity = 12000
    batch_size = args.batch_size
    start_learning_after = max(400, batch_size * 3)
    train_freq = 1
    target_update_freq = 100

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_episodes = 150

    clip_C = 0.1
    alpha_rdp = 64.0
    delta = 1e-5
    total_epsilon_budget = args.epsilon_total
    lambda_priv = 0.01
    mu_deadline = 2.0

    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = make_writer(args.log_root)

    dag = DAGTasks(num_tasks=num_tasks, max_layers=max_layers, seed=SEED)
    env = OffloadEnv(
        dag=dag,
        num_vehicles=num_vehicles,
        num_rsus=num_rsus,
        lambda_priv=lambda_priv,
        mu_deadline=mu_deadline,
        seed=SEED
    )

    agents = make_agents(num_vehicles, state_dim=env.state_dim, action_dim=env.action_dim, device=device)

    init_eps_vehicle = total_epsilon_budget / max(1, num_vehicles)
    vehicles, rsus, bs = build_federated_hierarchy(
        num_vehicles=num_vehicles,
        num_rsus=num_rsus,
        agents=agents,
        clip_C=clip_C,
        alpha_rdp=alpha_rdp,
        delta=delta,
        init_eps_vehicle=init_eps_vehicle,
        device=device
    )

    task_weights = [dag.nodes[i].alpha for i in dag.get_topo_order()]
    allocator = PrivacyBudgetAllocator(
        total_eps=total_epsilon_budget,
        num_tasks=num_tasks,
        task_weights=task_weights,
        num_vehicles=num_vehicles
    )

    buffers = [PrioritizedReplayBuffer(capacity=buffer_capacity) for _ in range(num_vehicles)]

    from tqdm import tqdm
    pbar = tqdm(total=episodes, dynamic_ncols=True)

    global_step = 0
    episode_rewards = AverageMeter()
    episode_delays = AverageMeter()
    episode_eps_used_proxy = AverageMeter()

    global_sd = None

    for ep in range(episodes):
        state = env.reset()
        eps_map = {i: allocator.get_task_remaining(i) for i in dag.get_topo_order()}
        env.set_per_task_eps_remaining(eps_map)

        done = False
        ep_reward = 0.0
        ep_delay = 0.0
        ep_eps_proxy = 0.0
        step_count = 0

        eps = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (ep / max(1, epsilon_decay_episodes)))
        vehicle_idx = 0

        while not done and step_count < steps_per_episode_cap:
            agent = vehicles[vehicle_idx].agent
            if state is None:
                break
            action = agent.act(state, eps=eps)
            next_state, reward, done, info = env.step(action)

            ep_reward += reward
            ep_delay += info.get("T", 0.0)
            ep_eps_proxy += info.get("eps_used_proxy", 0.0)

            terminal = 1.0 if done else 0.0
            buffers[vehicle_idx].push((state, action, reward, next_state if next_state is not None else state, terminal))

            if len(buffers[vehicle_idx]) >= max(batch_size, start_learning_after) and (global_step % train_freq == 0):
                batch, indices, weights = buffers[vehicle_idx].sample(batch_size)
                loss, td_abs = agent.compute_td_loss(batch, weights)
                stats = vehicles[vehicle_idx].local_update_with_dp(
                    loss, batch_size=batch_size, buffer_len=len(buffers[vehicle_idx])
                )
                buffers[vehicle_idx].update_priorities(indices, td_abs + 1e-6)

                writer.add_scalar("privacy/sigma", stats["sigma"], global_step)
                writer.add_scalar("privacy/rdp_eps_total", stats["eps_rdp_total"], global_step)
                writer.add_scalar("privacy/epsilon_remain", stats["eps_remain"], global_step)
                writer.add_scalar("privacy/frozen_update", stats["frozen"], global_step)
                writer.add_scalar("privacy/low_eps_mode", stats["low_eps"], global_step)

            state = next_state
            step_count += 1
            global_step += 1
            vehicle_idx = (vehicle_idx + 1) % num_vehicles

            if global_step % target_update_freq == 0:
                for ag in agents:
                    ag.soft_update()

        if (ep + 1) % args.agg_every == 0:
            with torch.no_grad():
                per_rsu_states: List[List[Dict[str, torch.Tensor]]] = [[] for _ in rsus]
                for i, v in enumerate(vehicles):
                    sd = {k: val.detach().cpu() for k, val in v.agent.get_state_dict().items()}
                    per_rsu_states[i % len(rsus)].append(sd)
                rsu_agg_states = []
                for i in range(len(rsus)):
                    keys = per_rsu_states[i][0].keys()
                    agg_sd = {}
                    for k in keys:
                        agg_sd[k] = sum([sd[k] for sd in per_rsu_states[i]]) / float(len(per_rsu_states[i]))
                    rsu_agg_states.append(agg_sd)
                keys = rsu_agg_states[0].keys()
                global_sd = {}
                for k in keys:
                    global_sd[k] = sum([sd[k] for sd in rsu_agg_states]) / float(len(rsu_agg_states))
                for v in vehicles:
                    v.agent.load_state_dict(global_sd)

        allocator.dynamic_redistribute(completed=list(env.completed))

        episode_rewards.update(ep_reward)
        episode_delays.update(ep_delay)
        episode_eps_used_proxy.update(ep_eps_proxy)

        writer.add_scalar("episode/reward", ep_reward, ep)
        writer.add_scalar("episode/avg_reward", episode_rewards.avg, ep)
        writer.add_scalar("episode/total_delay", ep_delay, ep)
        writer.add_scalar("episode/eps_proxy_used", ep_eps_proxy, ep)
        writer.add_scalar("train/epsilon", eps, ep)

        pbar.update(1)
        pbar.set_postfix({
            "ep": ep,
            "R": f"{ep_reward:.1f}",
            "Delay": f"{ep_delay:.3f}",
            "eps": f"{eps:.2f}",
        })

        if (ep + 1) % 50 == 0 and global_sd is not None:
            ckpt_path = os.path.join("./checkpoints", f"global_ep{ep+1}.pt")
            torch.save(global_sd, ckpt_path)

    if global_sd is None:
        global_sd = vehicles[0].agent.get_state_dict()
    torch.save(global_sd, os.path.join("./checkpoints", "global_final.pt"))

    pbar.close()
    writer.close()
    print("Training finished. TensorBoard root:", os.path.abspath(args.log_root))


if __name__ == "__main__":
    main()