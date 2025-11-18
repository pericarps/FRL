# dag.py
import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import networkx as nx

@dataclass
class Node:
    id: int
    C: float
    d: float
    l: float
    p: float
    layer: int
    pre: Set[int] = field(default_factory=set)
    suc: Set[int] = field(default_factory=set)
    alpha_path: float = 0.0
    alpha_prio: float = 0.0
    alpha_lay: float = 0.0
    alpha: float = 0.0
    task_type: str = "normal" 
    accuracy_requirement: float = 1.0  
    privacy_sensitivity: float = 0.5
    deadline_pressure: float = 0.5  # 截止时间压力 [0, 1]，0=宽松，1=紧迫  
    @property
    def priority(self) -> float:
        return self.p

    @priority.setter
    def priority(self, value: float):
        self.p = value

class DAGTasks:
    def __init__(self, num_tasks: int, max_layers: int, seed: Optional[int], task_config: dict):
        self.num_tasks = num_tasks
        self.max_layers = max_layers
        self.seed = seed
        self.cfg = task_config
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.G = nx.DiGraph()
        self.nodes: Dict[int, Node] = {}
        self.layers: Dict[int, List[int]] = {}
        self._build_random_dag_safe()
        self._recompute_layers()
        self._assign_task_types() 

    def _build_random_dag_safe(self, retry: int = 8):
        for _try in range(max(1, retry)):
            self.nodes.clear()
            self.layers.clear()
            ok = self._try_build_once()
            if ok:
                return

        self.G.clear()
        self.nodes.clear()
        self.layers.clear()
        per_layer = max(1, self.num_tasks // max(1, self.max_layers))
        remaining = self.num_tasks
        nid = 0

        for l in range(self.max_layers):
            cnt = min(per_layer, remaining) if l < self.max_layers - 1 else remaining
            self.layers[l] = []
            for _ in range(cnt):
                C = float(np.random.uniform(*self.cfg["cycles_range"]))
                d = float(np.random.uniform(*self.cfg["data_range"]))
                base = float(np.random.uniform(*self.cfg["deadline_base_range"]))
                ldl = base + l * float(self.cfg["deadline_layer_offset"])
                node = Node(nid, C, d, ldl, p=0.0, layer=l)
                self.nodes[nid] = node
                self.layers[l].append(nid)
                self.G.add_node(nid)
                nid += 1
            remaining -= cnt
            if remaining <= 0:
                break

        for l in range(self.max_layers - 1):
            srcs = self.layers.get(l, [])
            dsts = self.layers.get(l + 1, [])
            if not srcs or not dsts:
                continue
            for s in srcs:
                t = random.choice(dsts)
                if not self.G.has_edge(s, t):
                    self.G.add_edge(s, t)
                    self.nodes[s].suc.add(t)
                    self.nodes[t].pre.add(s)
        assert nx.is_directed_acyclic_graph(self.G), "fallback DAG should be acyclic"

    def _try_build_once(self) -> bool:
        for i in range(self.num_tasks):
            layer = random.randint(0, max(1, self.max_layers) - 1)
            C = float(np.random.uniform(*self.cfg["cycles_range"]))
            d = float(np.random.uniform(*self.cfg["data_range"]))
            base = float(np.random.uniform(*self.cfg["deadline_base_range"]))
            ldl = base + layer * float(self.cfg["deadline_layer_offset"])
            node = Node(i, C, d, ldl, p=0.0, layer=layer)
            self.nodes[i] = node
            self.layers.setdefault(layer, []).append(i)
            self.G.add_node(i)

        if 0 not in self.layers or len(self.layers[0]) == 0:
            if not self.nodes: 
                 return True 
            any_id = random.randrange(self.num_tasks)
            old_l = self.nodes[any_id].layer
            if any_id in self.layers.get(old_l, []):
                self.layers[old_l].remove(any_id)
            self.nodes[any_id].layer = 0
            self.layers.setdefault(0, []).append(any_id)

        for l in range(self.max_layers - 1):
            srcs = self.layers.get(l, [])
            if not srcs:
                continue
            higher = []
            for j in range(l + 1, self.max_layers):
                higher.extend(self.layers.get(j, []))
            if not higher:
                continue
            for s in srcs:
                k = random.randint(1, min(3, len(higher)))
                for t in random.sample(higher, k):
                    if s == t or self.nodes[s].layer >= self.nodes[t].layer:
                        continue

                    if not self.G.has_edge(s, t):
                        self.G.add_edge(s, t)
                        self.nodes[s].suc.add(t)
                        self.nodes[t].pre.add(s)

        for nid, nd in self.nodes.items():
            if nd.layer == 0:
                continue
            if len(nd.pre) == 0:
                lower = []
                for l in range(0, nd.layer):
                    lower.extend(self.layers.get(l, []))
                if not lower:
                    continue

                s = random.choice(lower)

                if s != nid and not self.G.has_edge(s, nid):
                    self.G.add_edge(s, nid)
                    self.nodes[s].suc.add(nid)
                    self.nodes[nid].pre.add(s)

        if not nx.is_directed_acyclic_graph(self.G):
            return False
        return True

    def _recompute_layers(self):
        for n in self.nodes.values():
            n.pre.clear(); n.suc.clear()
        for u, v in self.G.edges():
            self.nodes[u].suc.add(v)
            self.nodes[v].pre.add(u)

        try:
            top = list(nx.topological_sort(self.G))
        except nx.NetworkXUnfeasible:
            self.max_layers = 0
            return

        dist = {u: 0 for u in top}

        for u in top:
            for v in self.G.successors(u):
                dist[v] = max(dist[v], dist[u] + 1)
        max_layer = max(dist.values()) if dist else 0
        for i in self.nodes:
            self.nodes[i].layer = dist[i]
        self.max_layers = max_layer + 1

    def _compute_priorities(self, current_time: float):
        for n in self.nodes.values():
            n.p = current_time / max(n.l, 1e-9)

    def _compute_alpha(self):
        w1, w2, w3 = self.cfg.get("alpha_weights", (0.3, 0.3, 0.4))
        try:
            top = list(nx.topological_sort(self.G))
        except nx.NetworkXUnfeasible:
            return 

        longest_to_sink = {u: 0 for u in top[::-1]}

        for u in reversed(top):
            m = 0
            for v in self.G.successors(u):
                m = max(m, 1 + longest_to_sink[v])
            longest_to_sink[u] = m
        max_overall = max(longest_to_sink.values()) if longest_to_sink else 1

        for i, node in self.nodes.items():
            node.alpha_path = (longest_to_sink[i] / max(1, max_overall)) if longest_to_sink[i] > 0 else 0.0
            node.alpha_prio = 1.0 / (1.0 + np.exp(-1.0 * (node.p - 1.0)))
            lambda_lay = np.log(10.0) / max(1, self.max_layers - 1) if self.max_layers > 1 else 0.0
            node.alpha_lay = np.exp(-lambda_lay * node.layer)
            node.alpha = w1 * node.alpha_path + w2 * node.alpha_prio + w3 * node.alpha_lay

    def update_dynamic_scores(self, current_time: float):
        self._compute_priorities(current_time)
        self._compute_alpha()

    def _topo(self) -> List[int]:
        try:
            return list(nx.topological_sort(self.G))
        except nx.NetworkXUnfeasible:
            return []

    def get_ready(self, completed: Set[int]) -> List[int]:
        ready = []
        for i in self._topo():
            if i in completed:
                continue
            if self.nodes[i].pre.issubset(completed):
                ready.append(i)
        return ready

    def node_features(self, i: int):
        n = self.nodes[i]
        return n.C, n.d, n.l, n.p, n.layer, n.alpha

    def get_predecessors(self, task_id: int) -> List[int]:
        return list(self.nodes[task_id].pre)

    def get_successors(self, task_id: int) -> List[int]:
        return list(self.nodes[task_id].suc)   

    def add_edge(self, from_id: int, to_id: int):
        if not self.G.has_edge(from_id, to_id):
            self.G.add_edge(from_id, to_id)
            self.nodes[from_id].suc.add(to_id)
            self.nodes[to_id].pre.add(from_id)

    def get_ready_sorted(self, completed: Set[int]) -> List[int]:
        ready = []
        for i in self._topo():
            if i in completed:
                continue
            if self.nodes[i].pre.issubset(completed):
                ready.append(i)
        layer_map = {}

        for tid in ready:
            layer = self.nodes[tid].layer
            layer_map.setdefault(layer, []).append(tid)
        sorted_ready = []

        for layer in sorted(layer_map.keys()):
            group = layer_map[layer]
            group_sorted = sorted(group, key=lambda tid: self.nodes[tid].priority, reverse=True)
            sorted_ready.extend(group_sorted)
        return sorted_ready

    def _assign_task_types(self):
        num_accuracy = max(1, int(self.num_tasks * 0.3))
        num_privacy = max(1, int(self.num_tasks * 0.3))
        
        # 随机选择任务ID
        all_task_ids = list(range(self.num_tasks))
        random.shuffle(all_task_ids)
        
        accuracy_tasks = all_task_ids[:num_accuracy]
        privacy_tasks = all_task_ids[num_accuracy:num_accuracy + num_privacy]
        normal_tasks = all_task_ids[num_accuracy + num_privacy:]
        
        # 分配任务类型和特征
        for task_id in accuracy_tasks:
            node = self.nodes[task_id]
            node.task_type = "accuracy_critical"
            node.accuracy_requirement = random.uniform(0.85, 1.0)  
            node.privacy_sensitivity = random.uniform(0.1, 0.3)
            node.deadline_pressure = random.uniform(0.3, 0.7)  # 中等压力
        
        for task_id in privacy_tasks:
            node = self.nodes[task_id]
            node.task_type = "privacy_sensitive"
            node.accuracy_requirement = random.uniform(0.4, 0.7)   
            node.privacy_sensitivity = random.uniform(0.7, 1.0)
            node.deadline_pressure = random.uniform(0.5, 0.9)  # 较高压力（隐私敏感任务通常更紧急）
        
        for task_id in normal_tasks:
            node = self.nodes[task_id]
            node.task_type = "normal"
            node.accuracy_requirement = random.uniform(0.5, 0.8)   
            node.privacy_sensitivity = random.uniform(0.3, 0.7)
            node.deadline_pressure = random.uniform(0.2, 0.8)  # 广泛范围    
