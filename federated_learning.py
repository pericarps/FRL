# federated_learning.py
import torch
import copy
from typing import List, Dict, Optional
from agent import DoubleDQNAgent

class HierFL:
    def __init__(self, 
                 num_vehicles: int, 
                 device: str = 'cpu', 
                 num_rsus: int = 2):
        self.num_vehicles = num_vehicles
        self.device = device
        self.global_model = None
        self.num_rsus = max(1, int(num_rsus))
        self.rsu_members: Dict[int, List[int]] = {i: [] for i in range(self.num_rsus)}
        for vid in range(num_vehicles):
            rsu_id = vid % self.num_rsus
            self.rsu_members[rsu_id].append(vid)
        self.rsu_models: Dict[int, torch.nn.Module] = {}

    def set_global_model(self, agent: DoubleDQNAgent):
        # 处理 Opacus GradSampleModule 包装
        # GradSampleModule 包装后，原始模型在 ._module 属性中
        if hasattr(agent.q_network, '_module'):
            # Opacus wrapped: q_network._module 就是 FusedQNetwork
            self.global_model = copy.deepcopy(agent.q_network._module)
        else:
            # Not wrapped: q_network 就是 FusedQNetwork
            self.global_model = copy.deepcopy(agent.q_network)
        self.global_model.eval()

        self.rsu_models = {
            rsu_id: copy.deepcopy(self.global_model).eval() for rsu_id in self.rsu_members.keys()
        }

    def _fedavg_state_dict(self, models: List[torch.nn.Module]) -> Dict:
        """
        联邦平均：对多个模型的参数求平均
        处理 Opacus GradSampleModule 包装的情况
        """
        assert len(models) > 0
        first_sd = models[0].state_dict()
        keys = first_sd.keys()
        # 检查是否是 Opacus 包装的模型（key 以 _module. 开头）
        is_wrapped = any(k.startswith('_module.') for k in keys)
        
        # 对所有模型的参数求平均
        out = {}
        for k in keys:
            out[k] = torch.stack([m.state_dict()[k].float() for m in models], dim=0).mean(dim=0)

        # 如果是包装的，移除 _module. 前缀
        if is_wrapped:
            clean_out = {}
            for k, v in out.items():
                if k.startswith('_module.'):
                    clean_key = k.replace('_module.', '')
                    clean_out[clean_key] = v
                else:
                    clean_out[k] = v
            return clean_out
        return out

    def aggregate_models(self, agents: List[DoubleDQNAgent]):  
        """聚合所有agent的critic模型（仅DDQN，不包含transformer）"""
        if self.global_model is None:
            self.set_global_model(agents[0])
            return

        # RSU级别聚合
        for rsu_id, members in self.rsu_members.items():
            if not members:
                continue
            local_models = [agents[i].q_network for i in members]
            agg_sd = self._fedavg_state_dict(local_models)
            self.rsu_models[rsu_id].load_state_dict(agg_sd)
            
        # 全局聚合
        rsu_model_list = [self.rsu_models[rid] for rid in sorted(self.rsu_models.keys()) if self.rsu_members[rid]]
        if rsu_model_list:
            global_sd = self._fedavg_state_dict(rsu_model_list)
            self.global_model.load_state_dict(global_sd)

    def distribute_model(self, agents: List[DoubleDQNAgent]):
        """
        将全局critic模型分发到各个agent（仅DDQN，不包含transformer）
        处理 Opacus GradSampleModule 包装的情况
        """
        if self.global_model is None:
            return
        for rsu_id, members in self.rsu_members.items():
            src_model = self.rsu_models.get(rsu_id, self.global_model)
            sd = src_model.state_dict()
            for vid in members:
                agent = agents[vid]
                # 如果 q_network 是 Opacus 包装的，需要添加 _module. 前缀
                if hasattr(agent.q_network, '_module'):
                    sd_with_prefix = {}
                    for k, v in sd.items():
                        sd_with_prefix[f'_module.{k}'] = v
                    agent.q_network.load_state_dict(sd_with_prefix)
                    # target_network 没有包装，直接加载
                    agent.target_network.load_state_dict(sd)
                else:
                    agent.q_network.load_state_dict(sd)
                    agent.target_network.load_state_dict(sd)

    def flat_distribute(self, agents: List[DoubleDQNAgent]):
        """
        扁平化分发：将全局模型分发给所有 agent（不通过 RSU 层级）
        处理 Opacus GradSampleModule 包装的情况
        """
        if self.global_model is None:
            return
        sd = self.global_model.state_dict()
        for agent in agents:
            # 如果 q_network 是 Opacus 包装的，需要添加 _module. 前缀
            if hasattr(agent.q_network, '_module'):
                sd_with_prefix = {}
                for k, v in sd.items():
                    sd_with_prefix[f'_module.{k}'] = v
                agent.q_network.load_state_dict(sd_with_prefix)
                # target_network 没有包装，直接加载
                agent.target_network.load_state_dict(sd)
            else:
                agent.q_network.load_state_dict(sd)
                agent.target_network.load_state_dict(sd)