import random
from typing import List, Dict, Tuple
import networkx as nx
import numpy as np


class TaskNode:
    def __init__(self, idx: int, c: int, d: int, l: float, p: float, layer: int):
        self.idx = idx
        self.c = c  # cycles
        self.d = d  # bits
        self.l = l  # deadline
        self.p = p  # priority [0,1]
        self.pre = set()
        self.layer = layer
        self.alpha_path = 0.0
        self.alpha_prio = p
        self.alpha_lay = 0.0
        self.alpha = 0.0


class DAGTasks:
    def __init__(self, num_tasks: int, max_layers: int, seed: int = 42):
        self.num_tasks = num_tasks
        self.max_layers = max_layers
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.G = nx.DiGraph()
        self.nodes: Dict[int, TaskNode] = {}
        self.layers: Dict[int, List[int]] = {}
        self._build_random_dag()

    def _build_random_dag(self):
        # Assign layers
        for i in range(self.num_tasks):
            layer = random.randint(0, self.max_layers - 1)
            c = random.randint(2_0_0_0_0, 8_0_0_0_0)  # cycles (20k-80k)
            d = random.randint(50_000, 500_000)  # bits (50kb-500kb)
            l = random.uniform(0.1, 1.5) + layer * 0.1  # deadline grows with layer
            p = random.random()
            node = TaskNode(i, c, d, l, p, layer)
            self.nodes[i] = node
            if layer not in self.layers:
                self.layers[layer] = []
            self.layers[layer].append(i)
            self.G.add_node(i)

        # Ensure at least one root in layer 0
        if 0 not in self.layers:
            self.layers[0] = [0]
            self.nodes[0].layer = 0
            if not self.G.has_node(0):
                self.G.add_node(0)

        # Add edges forward in layers to ensure DAG
        for l in range(self.max_layers - 1):
            srcs = self.layers.get(l, [])
            dsts = []
            for j in range(l + 1, self.max_layers):
                dsts.extend(self.layers.get(j, []))
            for s in srcs:
                # Each node connects to 1-3 successors (if available)
                k = random.randint(0, min(3, len(dsts)))
                succ = random.sample(dsts, k) if k > 0 else []
                for t in succ:
                    if not self.G.has_edge(s, t):
                        self.G.add_edge(s, t)
                        self.nodes[t].pre.add(s)

        # Remove cycles if any (shouldn't exist), enforce DAG
        assert nx.is_directed_acyclic_graph(self.G), "Generated graph is not a DAG."

        # Topological sort to recompute layers (lay) if needed
        top_order = list(nx.topological_sort(self.G))
        # Recompute layer by longest distance from sources
        dist = {u: 0 for u in top_order}
        for u in top_order:
            for v in self.G.successors(u):
                dist[v] = max(dist[v], dist[u] + 1)
        max_layer = max(dist.values()) if dist else 0
        for i in self.nodes:
            self.nodes[i].layer = dist[i]
        self.max_layers = max_layer + 1

        # Compute alpha components
        self._compute_alpha_scores()

    def _compute_alpha_scores(self, w1: float = 0.5, w2: float = 0.3, w3: float = 0.2):
        # alpha_path: normalize by longest path length among all nodes to any successor
        top_order = list(nx.topological_sort(self.G))
        # compute longest path to any descendant using DP from sinks backward
        longest_to_sink = {u: 0 for u in top_order[::-1]}
        for u in reversed(top_order):
            max_succ = 0
            for v in self.G.successors(u):
                max_succ = max(max_succ, 1 + longest_to_sink[v])
            longest_to_sink[u] = max_succ
        max_overall = max(longest_to_sink.values()) if longest_to_sink else 1
        for i, node in self.nodes.items():
            node.alpha_path = (longest_to_sink[i] / max(1, max_overall))
            node.alpha_lay = 1.0 - (node.layer / max(1, self.max_layers - 1))
            node.alpha = w1 * node.alpha_path + w2 * node.alpha_prio + w3 * node.alpha_lay

    def get_topo_order(self) -> List[int]:
        return list(nx.topological_sort(self.G))

    def get_ready(self, completed: set) -> List[int]:
        ready = []
        for i in self.get_topo_order():
            if i in completed:
                continue
            if self.nodes[i].pre.issubset(completed):
                ready.append(i)
        return ready

    def node_features(self, i: int):
        node = self.nodes[i]
        return node.c, node.d, node.l, node.p, node.layer, node.alpha

    def total_alpha_sum(self) -> float:
        return sum([n.alpha for n in self.nodes.values()])

    def info(self) -> str:
        return f"DAG | tasks={self.num_tasks}, layers={self.max_layers}, edges={self.G.number_of_edges()}"