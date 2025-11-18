"""
GACE (Graph-Aware Context Encoder) 独立评估脚本
用于加载训练好的模型并评估编码质量

使用方法:
    python evaluate_gace.py --model_path <模型路径> --num_episodes <测试episode数>
    
示例:
    python evaluate_gace.py --model_path model/experiments/1000iters_10.0eps_10vehicles_7tasks_fullmodel/agent_0_final.pth --num_episodes 100
"""

import torch
import numpy as np
import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
import seaborn as sns

from agent import DoubleDQNAgent
from env import OffloadEnv
from dag import DAGTasks
from network import GraphAwareContextEncoder


# ============================================================================
# GACE质量评估指标类（内嵌版本）
# ============================================================================

class GACEQualityMetrics:
    """GACE编码质量评估器"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.encoding_history = []
        self.task_labels = []
        
    def evaluate_encoding_quality(self, 
                                   context_encoded: torch.Tensor,
                                   task_features: torch.Tensor,
                                   privacy_features: torch.Tensor,
                                   dag_features: torch.Tensor) -> Dict[str, float]:
        """
        评估GACE编码质量
        
        Args:
            context_encoded: GACE输出的编码 [B, L, d_model]
            task_features: 任务特征 [B, L, 15]
            privacy_features: 隐私特征 [B, L, 6]
            dag_features: DAG特征 [B, L, 4]
            
        Returns:
            质量指标字典
        """
        metrics = {}
        
        # 1. 编码范数分析（编码强度）
        encoding_norm = torch.norm(context_encoded, p=2, dim=-1).mean().item()
        metrics['encoding_norm'] = encoding_norm
        
        # 2. 编码方差（表示丰富度）
        encoding_var = context_encoded.var(dim=-1).mean().item()
        metrics['encoding_variance'] = encoding_var
        
        # 3. 特征利用率（非零激活比例）
        activation_ratio = (torch.abs(context_encoded) > 1e-3).float().mean().item()
        metrics['activation_ratio'] = activation_ratio
        
        # 4. 编码稳定性（批次内方差）
        if context_encoded.shape[0] > 1:
            batch_std = context_encoded.std(dim=0).mean().item()
            metrics['batch_stability'] = 1.0 / (1.0 + batch_std)
        
        # 5. 层次结构保持度（相邻任务相似性）
        if context_encoded.shape[1] > 1:
            similarity_scores = []
            for b in range(context_encoded.shape[0]):
                for i in range(context_encoded.shape[1] - 1):
                    v1 = context_encoded[b, i].detach().cpu().numpy()
                    v2 = context_encoded[b, i+1].detach().cpu().numpy()
                    sim = 1.0 - cosine(v1, v2) if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0 else 0.0
                    similarity_scores.append(sim)
            metrics['temporal_consistency'] = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # 6. 信息增益（与原始特征的距离）
        original_concat = torch.cat([task_features, privacy_features, dag_features], dim=-1)
        d_model = context_encoded.shape[-1]
        
        import torch.nn as nn
        proj = nn.Linear(25, d_model).to(self.device)
        with torch.no_grad():
            original_projected = proj(original_concat.to(self.device))
            information_gain = torch.norm(context_encoded - original_projected, p=2, dim=-1).mean().item()
        metrics['information_gain'] = information_gain
        
        return metrics
    
    def evaluate_feature_separability(self, 
                                       encodings: torch.Tensor, 
                                       labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        评估特征可分性（聚类质量）
        
        Args:
            encodings: 编码向量 [N, d_model]
            labels: 任务类别标签 [N]
            
        Returns:
            可分性指标
        """
        metrics = {}
        
        if encodings.shape[0] < 2:
            return metrics
        
        encodings_np = encodings.detach().cpu().numpy()
        
        # 1. 轮廓系数（需要标签）
        if labels is not None and len(np.unique(labels)) > 1:
            try:
                silhouette = silhouette_score(encodings_np, labels)
                metrics['silhouette_score'] = silhouette
            except:
                pass
        
        # 2. 编码空间分散度
        variance_per_dim = np.var(encodings_np, axis=0)
        metrics['encoding_spread'] = np.sum(variance_per_dim)
        
        # 3. 内聚度（样本间平均距离）
        pairwise_dists = []
        for i in range(min(100, encodings_np.shape[0])):
            for j in range(i+1, min(100, encodings_np.shape[0])):
                dist = np.linalg.norm(encodings_np[i] - encodings_np[j])
                pairwise_dists.append(dist)
        metrics['avg_pairwise_distance'] = np.mean(pairwise_dists) if pairwise_dists else 0.0
        
        return metrics


# ============================================================================
# GACE评估器主类
# ============================================================================

class GACEEvaluator:
    """GACE编码质量独立评估器"""
    
    def __init__(self, model_path: str, config_path: str = None, device: str = 'cpu'):
        """
        初始化评估器
        
        Args:
            model_path: 模型权重文件路径
            config_path: 配置文件路径（可选）
            device: 计算设备
        """
        self.device = device
        self.model_path = model_path
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                import yaml
                self.params = yaml.safe_load(f)
        else:
            # 使用默认配置
            self.params = self._get_default_params()
        
        # 加载模型
        self.agent = self._load_agent()
        self.gace_model = self._extract_gace_model()
        
        # 创建评估指标收集器
        self.metrics_collector = GACEQualityMetrics(device=device)
        
        # 存储评估结果
        self.all_encodings = []
        self.all_task_labels = []
        self.encoding_history = []
        
    def _get_default_params(self) -> Dict:
        """获取默认参数"""
        return {
            'state_size': 15,
            'action_size': 3,
            'hidden_size': 256,
            'use_gace': True,
            'd_model': 128,
            'nhead': 4,
            'num_layers': 2,
            'num_tasks': 7,
            'max_layers': 3,
            'learning_rate': 1e-5,
            'gamma': 0.99,
            'epsilon_start': 0.0,
            'epsilon_end': 0.0,
            'epsilon_decay': 1.0,
            'batch_size': 64,
            'memory_capacity': 10000,
            'critic_noise_multiplier': 1.0,
        }
    
    def _load_agent(self) -> DoubleDQNAgent:
        """加载训练好的Agent"""
        print(f"正在加载模型: {self.model_path}")
        
        # 创建agent
        agent = DoubleDQNAgent(
            state_size=self.params['state_size'],
            action_size=self.params['action_size'],
            hidden_size=self.params['hidden_size'],
            use_gace=self.params['use_gace'],
            d_model=self.params['d_model'],
            nhead=self.params['nhead'],
            num_layers=self.params['num_layers'],
            params=self.params,
            device=self.device
        )
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'critic_state_dict' in checkpoint:
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            print("✓ 加载critic权重")
        elif 'model_state_dict' in checkpoint:
            agent.critic.load_state_dict(checkpoint['model_state_dict'])
            print("✓ 加载模型权重")
        else:
            agent.critic.load_state_dict(checkpoint)
            print("✓ 加载权重")
        
        agent.critic.eval()  # 设置为评估模式
        print("✓ 模型加载完成\n")
        
        return agent
    
    def _extract_gace_model(self) -> GraphAwareContextEncoder:
        """提取GACE模型"""
        if hasattr(self.agent.critic, 'context_encoder'):
            gace = self.agent.critic.context_encoder
            print("✓ 成功提取GACE模型")
            return gace
        else:
            print("⚠ 模型中未找到GACE，use_gace可能为False")
            return None
    
    def run_evaluation(self, num_episodes: int = 100) -> Dict:
        """
        运行评估
        
        Args:
            num_episodes: 评估的episode数量
            
        Returns:
            评估结果字典
        """
        print(f"{'='*60}")
        print(f"开始GACE编码质量评估 (共{num_episodes}个episodes)")
        print(f"{'='*60}\n")
        
        if self.gace_model is None:
            print("错误: 无法评估，模型中不包含GACE")
            return {}
        
        all_metrics = {
            'encoding_quality': [],
            'separability': [],
            'attention_patterns': []
        }
        
        for ep in range(num_episodes):
            # 创建新的DAG和环境
            dag = self._create_dag()
            env = OffloadEnv(dag, self.params, device=self.device)
            env.reset()
            
            # 收集一个episode的编码
            episode_encodings = []
            episode_task_features = []
            episode_privacy_features = []
            episode_dag_features = []
            episode_attention_weights = []
            
            ready_tasks = env.get_ready_tasks()
            step = 0
            max_steps = 50
            
            while ready_tasks and step < max_steps:
                task_id = ready_tasks[0]
                
                # 准备输入
                state = env.build_state(task_id)
                inputs = env.prepare_transformer_inputs(ready_tasks)
                
                if inputs is not None:
                    tf, pf, df, adj, attn_mask, total_budget = inputs
                    
                    aux = {
                        'task_features': tf,
                        'privacy_features': pf,
                        'dag_features': df,
                        'dag_adjacency': adj,
                        'attention_mask': attn_mask,
                        'total_budget': total_budget
                    }
                    
                    # 获取GACE编码（带中间结果）
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        local_global = self.agent.critic.state_encoder(state_tensor)
                        
                        # 调用GACE forward并获取中间激活
                        aligned, intermediates = self.gace_model.forward(
                            local_global, 
                            aux, 
                            return_intermediates=True
                        )
                        
                        if intermediates:
                            episode_encodings.append(intermediates['context_encoded'])
                            episode_task_features.append(intermediates['task_features'])
                            episode_privacy_features.append(intermediates['privacy_features'])
                            episode_dag_features.append(intermediates['dag_features'])
                            
                            # 记录任务标签（用于可分性分析）
                            # 使用任务优先级或计算密集度作为标签
                            node = dag.nodes[task_id]
                            label = self._get_task_label(node)
                            self.all_task_labels.append(label)
                
                # 执行动作
                action = self.agent.act(state, aux, sigma=1.0)
                next_state, reward, done, info = env.step(task_id, action, {"sigma": 1.0, "q": 0.01})
                
                ready_tasks = env.get_ready_tasks()
                step += 1
                
                if done:
                    break
            
            # 评估本episode
            if episode_encodings:
                ep_metrics = self._evaluate_episode(
                    episode_encodings,
                    episode_task_features,
                    episode_privacy_features,
                    episode_dag_features
                )
                
                for key, value in ep_metrics.items():
                    if key in all_metrics:
                        all_metrics[key].append(value)
            
            if (ep + 1) % 10 == 0:
                print(f"已完成 {ep + 1}/{num_episodes} episodes")
        
        # 汇总结果
        summary = self._summarize_metrics(all_metrics)
        
        print(f"\n{'='*60}")
        print("评估完成！")
        print(f"{'='*60}\n")
        
        return summary
    
    def _evaluate_episode(self, encodings, task_features, privacy_features, dag_features) -> Dict:
        """评估单个episode的编码"""
        metrics = {}
        
        # 合并所有step的编码
        context_encoded = torch.cat(encodings, dim=0)  # [total_steps, L, d_model]
        tf = torch.cat(task_features, dim=0)
        pf = torch.cat(privacy_features, dim=0)
        df = torch.cat(dag_features, dim=0)
        
        # 存储编码用于后续分析
        for i in range(context_encoded.shape[0]):
            for j in range(context_encoded.shape[1]):
                self.all_encodings.append(context_encoded[i, j].cpu())
        
        # 评估编码质量
        quality = self.metrics_collector.evaluate_encoding_quality(
            context_encoded, tf, pf, df
        )
        metrics['encoding_quality'] = quality
        
        # 评估可分性（如果有足够样本）
        if len(self.all_encodings) > 10:
            encodings_tensor = torch.stack(self.all_encodings)
            labels = np.array(self.all_task_labels) if self.all_task_labels else None
            
            separability = self.metrics_collector.evaluate_feature_separability(
                encodings_tensor, labels
            )
            metrics['separability'] = separability
        
        return metrics
    
    def _summarize_metrics(self, all_metrics: Dict) -> Dict:
        """汇总所有episode的指标"""
        summary = {}
        
        # 编码质量指标
        if all_metrics['encoding_quality']:
            quality_keys = all_metrics['encoding_quality'][0].keys()
            for key in quality_keys:
                values = [m[key] for m in all_metrics['encoding_quality'] if key in m]
                if values:
                    summary[f'quality_{key}_mean'] = float(np.mean(values))
                    summary[f'quality_{key}_std'] = float(np.std(values))
        
        # 可分性指标
        if all_metrics['separability']:
            sep_keys = all_metrics['separability'][0].keys()
            for key in sep_keys:
                values = [m[key] for m in all_metrics['separability'] if key in m]
                if values:
                    summary[f'sep_{key}_mean'] = float(np.mean(values))
                    summary[f'sep_{key}_std'] = float(np.std(values))
        
        return summary
    
    def _create_dag(self) -> DAGTasks:
        """创建测试DAG"""
        task_config = {
            "cycles_range": (1e7, 1e8),
            "data_range": (1e5, 1e6),
            "deadline_range": (0.5, 2.0),
            "accuracy_requirement_range": (0.6, 0.95),
            "privacy_sensitivity_range": (0.3, 0.9),
        }
        dag = DAGTasks(
            num_tasks=self.params.get('num_tasks', 7),
            max_layers=self.params.get('max_layers', 3),
            config=task_config
        )
        return dag
    
    def _get_task_label(self, node) -> int:
        """获取任务标签用于分类"""
        # 基于计算密集度分为3类
        C = float(node.C)
        if C < 3e7:
            return 0  # 低计算
        elif C < 7e7:
            return 1  # 中等计算
        else:
            return 2  # 高计算
    
    def print_summary(self, summary: Dict):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("GACE编码质量评估报告")
        print("="*60 + "\n")
        
        print("【编码质量指标】")
        quality_metrics = {k: v for k, v in summary.items() if k.startswith('quality_')}
        for key, value in quality_metrics.items():
            metric_name = key.replace('quality_', '').replace('_mean', '').replace('_std', '')
            if '_mean' in key:
                std_key = key.replace('_mean', '_std')
                std_value = summary.get(std_key, 0)
                print(f"  {metric_name:30s}: {value:8.4f} ± {std_value:.4f}")
        
        print("\n【特征可分性指标】")
        sep_metrics = {k: v for k, v in summary.items() if k.startswith('sep_')}
        for key, value in sep_metrics.items():
            metric_name = key.replace('sep_', '').replace('_mean', '').replace('_std', '')
            if '_mean' in key:
                std_key = key.replace('_mean', '_std')
                std_value = summary.get(std_key, 0)
                print(f"  {metric_name:30s}: {value:8.4f} ± {std_value:.4f}")
        
        print("\n" + "="*60)
        
        # 诊断建议
        self._print_diagnosis(summary)
    
    def _print_diagnosis(self, summary: Dict):
        """打印诊断建议"""
        print("\n【诊断建议】")
        
        issues = []
        suggestions = []
        
        # 检查编码强度
        norm = summary.get('quality_encoding_norm_mean', 0)
        if norm < 1.0:
            issues.append("✗ 编码强度过低 (encoding_norm < 1.0)")
            suggestions.append("  → 增加训练轮数或学习率")
        elif norm > 10.0:
            issues.append("✗ 编码强度过高 (encoding_norm > 10.0)")
            suggestions.append("  → 检查梯度裁剪或降低学习率")
        else:
            print("✓ 编码强度正常")
        
        # 检查编码多样性
        variance = summary.get('quality_encoding_variance_mean', 0)
        if variance < 0.05:
            issues.append("✗ 编码缺乏多样性 (variance < 0.05)")
            suggestions.append("  → 检查GACE是否正确参与训练")
        else:
            print("✓ 编码多样性良好")
        
        # 检查激活率
        activation = summary.get('quality_activation_ratio_mean', 0)
        if activation < 0.2:
            issues.append("✗ 激活率过低，神经元死亡 (activation < 0.2)")
            suggestions.append("  → 检查激活函数或初始化方法")
        elif activation > 0.8:
            issues.append("⚠ 激活率过高，可能过拟合")
        else:
            print("✓ 特征利用率正常")
        
        # 检查可分性
        silhouette = summary.get('sep_silhouette_score_mean', -1)
        if silhouette > 0:
            if silhouette < 0.3:
                issues.append("✗ 特征可分性差 (silhouette < 0.3)")
                suggestions.append("  → 增加GACE层数或维度")
            else:
                print("✓ 特征可分性良好")
        
        if issues:
            print("\n发现的问题:")
            for issue in issues:
                print(issue)
            print("\n建议:")
            for suggestion in suggestions:
                print(suggestion)
        else:
            print("\n✓ 所有指标正常，GACE工作良好！")
        
        print("="*60 + "\n")
    
    def visualize_encoding_space(self, output_path: str = None):
        """可视化编码空间"""
        if len(self.all_encodings) < 10:
            print("编码样本不足，无法可视化")
            return
        
        print("正在生成编码空间可视化...")
        
        # 转换为numpy
        encodings_np = torch.stack(self.all_encodings).detach().cpu().numpy()
        labels_np = np.array(self.all_task_labels) if self.all_task_labels else None
        
        # t-SNE降维
        print("  执行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(encodings_np)-1))
        encodings_2d = tsne.fit_transform(encodings_np[:1000])  # 限制样本数避免过慢
        
        # 绘图
        plt.figure(figsize=(12, 8))
        
        if labels_np is not None:
            labels_subset = labels_np[:1000]
            scatter = plt.scatter(
                encodings_2d[:, 0], 
                encodings_2d[:, 1], 
                c=labels_subset,
                cmap='viridis',
                alpha=0.6,
                s=20
            )
            plt.colorbar(scatter, label='任务类别')
            plt.title('GACE编码空间可视化 (t-SNE)\n颜色表示任务计算密集度', fontsize=14)
        else:
            plt.scatter(encodings_2d[:, 0], encodings_2d[:, 1], alpha=0.6, s=20)
            plt.title('GACE编码空间可视化 (t-SNE)', fontsize=14)
        
        plt.xlabel('t-SNE维度 1', fontsize=12)
        plt.ylabel('t-SNE维度 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ 可视化已保存: {output_path}")
        else:
            plt.savefig('gace_encoding_space.png', dpi=300, bbox_inches='tight')
            print("✓ 可视化已保存: gace_encoding_space.png")
        
        plt.close()
    
    def save_results(self, summary: Dict, output_dir: str):
        """保存评估结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON
        json_path = os.path.join(output_dir, 'gace_evaluation_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✓ 评估结果已保存: {json_path}")
        
        # 保存详细报告
        report_path = os.path.join(output_dir, 'gace_evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("GACE编码质量评估报告\n")
            f.write("="*60 + "\n\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"评估样本数: {len(self.all_encodings)}\n\n")
            
            f.write("【编码质量指标】\n")
            quality_metrics = {k: v for k, v in summary.items() if k.startswith('quality_')}
            for key, value in quality_metrics.items():
                if '_mean' in key:
                    metric_name = key.replace('quality_', '').replace('_mean', '')
                    std_key = key.replace('_mean', '_std')
                    std_value = summary.get(std_key, 0)
                    f.write(f"  {metric_name:30s}: {value:8.4f} ± {std_value:.4f}\n")
            
            f.write("\n【特征可分性指标】\n")
            sep_metrics = {k: v for k, v in summary.items() if k.startswith('sep_')}
            for key, value in sep_metrics.items():
                if '_mean' in key:
                    metric_name = key.replace('sep_', '').replace('_mean', '')
                    std_key = key.replace('_mean', '_std')
                    std_value = summary.get(std_key, 0)
                    f.write(f"  {metric_name:30s}: {value:8.4f} ± {std_value:.4f}\n")
        
        print(f"✓ 评估报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='GACE编码质量独立评估')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--config_path', type=str, default=None,
                        help='配置文件路径（可选）')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='评估的episode数量')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认为模型所在目录/evaluation）')
    parser.add_argument('--device', type=str, default='cpu',
                        help='计算设备 (cpu/cuda)')
    parser.add_argument('--visualize', action='store_true',
                        help='生成编码空间可视化')
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir is None:
        model_dir = Path(args.model_path).parent
        args.output_dir = str(model_dir / 'evaluation')
    
    # 创建评估器
    evaluator = GACEEvaluator(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )
    
    # 运行评估
    summary = evaluator.run_evaluation(num_episodes=args.num_episodes)
    
    # 打印结果
    evaluator.print_summary(summary)
    
    # 保存结果
    evaluator.save_results(summary, args.output_dir)
    
    # 可视化
    if args.visualize:
        vis_path = os.path.join(args.output_dir, 'gace_encoding_space.png')
        evaluator.visualize_encoding_space(vis_path)
    
    print(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
