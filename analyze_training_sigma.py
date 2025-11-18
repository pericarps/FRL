"""
训练过程中Sigma使用情况分析脚本
从TensorBoard日志或训练记录中提取和分析sigma预测的实际表现

功能:
1. 从TensorBoard事件文件中提取sigma相关数据
2. 分析sigma与奖励、时延、任务特征的相关性
3. 评估sigma预测对训练效果的影响
4. 可视化sigma在训练过程中的演化

使用方法:
    python analyze_training_sigma.py --log_dir runs/experiment_xxx --output_dir results/sigma_analysis

示例:
    python analyze_training_sigma.py \
        --log_dir model/experiments/run_20250117 \
        --output_dir results/training_sigma_analysis \
        --visualize
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠ TensorBoard未安装，将尝试从CSV文件读取")


class TrainingSigmaAnalyzer:
    """训练过程中Sigma使用情况分析器"""
    
    def __init__(self, log_dir: str):
        """
        初始化分析器
        
        Args:
            log_dir: TensorBoard日志目录或实验目录
        """
        self.log_dir = Path(log_dir)
        self.data = defaultdict(list)
        self.episode_data = []  # 每个episode的汇总数据
        
    def load_data(self):
        """加载训练数据"""
        print(f"{'='*60}")
        print(f"正在加载训练数据: {self.log_dir}")
        print(f"{'='*60}\n")
        
        # 尝试多种数据源
        loaded = False
        
        # 1. 尝试从TensorBoard事件文件加载
        if TENSORBOARD_AVAILABLE:
            loaded = self._load_from_tensorboard()
        
        # 2. 尝试从导出的CSV加载
        if not loaded:
            loaded = self._load_from_csv()
        
        # 3. 尝试从训练日志文件加载
        if not loaded:
            loaded = self._load_from_logs()
        
        if not loaded:
            raise FileNotFoundError(
                f"无法在 {self.log_dir} 中找到可用的训练数据\n"
                "支持的数据源:\n"
                "  - TensorBoard事件文件 (events.out.tfevents.*)\n"
                "  - 导出的CSV文件 (tensorboard_data.csv)\n"
                "  - 训练日志 (training_log.json)"
            )
        
        print(f"✓ 数据加载完成\n")
        self._summarize_data()
        
    def _load_from_tensorboard(self) -> bool:
        """从TensorBoard事件文件加载"""
        event_files = list(self.log_dir.rglob("events.out.tfevents.*"))
        
        if not event_files:
            return False
        
        print(f"找到 {len(event_files)} 个TensorBoard事件文件")
        
        for event_file in event_files:
            try:
                ea = event_accumulator.EventAccumulator(str(event_file))
                ea.Reload()
                
                # 提取所有标量数据
                for tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    for event in events:
                        self.data[tag].append({
                            'step': event.step,
                            'value': event.value,
                            'wall_time': event.wall_time
                        })
                
                print(f"  ✓ 已加载: {event_file.name}")
                
            except Exception as e:
                print(f"  ✗ 加载失败 {event_file.name}: {e}")
        
        return len(self.data) > 0
    
    def _load_from_csv(self) -> bool:
        """从导出的CSV文件加载"""
        csv_files = list(self.log_dir.glob("*.csv"))
        
        if not csv_files:
            return False
        
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # 假设CSV格式: step, tag, value
                if 'tag' in df.columns and 'value' in df.columns:
                    for tag in df['tag'].unique():
                        tag_df = df[df['tag'] == tag]
                        self.data[tag] = tag_df.to_dict('records')
                    
                    print(f"  ✓ 已加载: {csv_file.name}")
                    return True
                
                # 备选格式: 列名直接是tag
                elif 'step' in df.columns:
                    for col in df.columns:
                        if col != 'step':
                            self.data[col] = [
                                {'step': row['step'], 'value': row[col]}
                                for _, row in df.iterrows()
                                if pd.notna(row[col])
                            ]
                    
                    print(f"  ✓ 已加载: {csv_file.name}")
                    return True
                
            except Exception as e:
                print(f"  ✗ 加载失败 {csv_file.name}: {e}")
        
        return False
    
    def _load_from_logs(self) -> bool:
        """从JSON训练日志加载"""
        log_files = list(self.log_dir.glob("*training_log*.json"))
        
        if not log_files:
            return False
        
        print(f"找到 {len(log_files)} 个日志文件")
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                
                # 解析日志结构
                if isinstance(logs, list):
                    for entry in logs:
                        step = entry.get('episode', entry.get('step', 0))
                        for key, value in entry.items():
                            if isinstance(value, (int, float)):
                                self.data[key].append({
                                    'step': step,
                                    'value': value
                                })
                
                print(f"  ✓ 已加载: {log_file.name}")
                return True
                
            except Exception as e:
                print(f"  ✗ 加载失败 {log_file.name}: {e}")
        
        return False
    
    def _summarize_data(self):
        """汇总数据概览"""
        print(f"\n数据概览:")
        print(f"  总标签数: {len(self.data)}")
        
        # 查找sigma相关标签
        sigma_tags = [tag for tag in self.data.keys() if 'sigma' in tag.lower()]
        if sigma_tags:
            print(f"  Sigma相关标签: {len(sigma_tags)}")
            for tag in sigma_tags[:5]:  # 只显示前5个
                print(f"    - {tag}: {len(self.data[tag])} 条记录")
            if len(sigma_tags) > 5:
                print(f"    ... 还有 {len(sigma_tags)-5} 个标签")
        else:
            print(f"  ⚠ 未找到sigma相关标签")
        
        # 查找奖励、时延等关键指标
        key_metrics = ['reward', 'delay', 'completion', 'violation']
        found_metrics = []
        for metric in key_metrics:
            tags = [tag for tag in self.data.keys() if metric in tag.lower()]
            if tags:
                found_metrics.append(f"{metric}({len(tags)})")
        
        if found_metrics:
            print(f"  关键指标: {', '.join(found_metrics)}")
    
    def prepare_episode_data(self):
        """准备episode级别的数据用于分析"""
        print(f"\n正在准备episode级别数据...")
        
        # 提取所有episode的数据
        episodes = set()
        for tag_data in self.data.values():
            for record in tag_data:
                episodes.add(record['step'])
        
        episodes = sorted(list(episodes))
        
        for ep in episodes:
            ep_record = {'episode': ep}
            
            # 提取该episode的所有指标
            for tag, tag_data in self.data.items():
                matching_records = [r for r in tag_data if r['step'] == ep]
                if matching_records:
                    # 取平均值（如果有多条记录）
                    ep_record[tag] = np.mean([r['value'] for r in matching_records])
            
            self.episode_data.append(ep_record)
        
        print(f"✓ 准备了 {len(self.episode_data)} 个episode的数据\n")
    
    def analyze_sigma_statistics(self) -> Dict:
        """分析Sigma统计特性"""
        print(f"{'='*60}")
        print("Sigma统计分析")
        print(f"{'='*60}\n")
        
        results = {}
        
        # 查找所有sigma相关标签
        sigma_tags = [tag for tag in self.data.keys() if 'sigma' in tag.lower()]
        
        if not sigma_tags:
            print("⚠ 未找到sigma相关数据\n")
            return results
        
        # 对每个sigma标签进行分析
        for tag in sigma_tags:
            values = [r['value'] for r in self.data[tag]]
            
            stats = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
            }
            
            results[tag] = stats
            
            print(f"【{tag}】")
            print(f"  记录数: {stats['count']}")
            print(f"  均值±标准差: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  中位数: {stats['median']:.4f}")
            print(f"  四分位数: [{stats['q25']:.4f}, {stats['q75']:.4f}]")
            print()
        
        return results
    
    def analyze_sigma_performance_correlation(self) -> Dict:
        """分析Sigma与性能指标的相关性"""
        print(f"{'='*60}")
        print("Sigma与性能指标相关性分析")
        print(f"{'='*60}\n")
        
        if not self.episode_data:
            print("⚠ 无episode数据可用\n")
            return {}
        
        df = pd.DataFrame(self.episode_data)
        results = {}
        
        # 查找sigma列
        sigma_cols = [col for col in df.columns if 'sigma' in col.lower()]
        
        # 查找性能指标列
        perf_cols = []
        for keyword in ['reward', 'delay', 'time', 'violation', 'completion', 'success']:
            perf_cols.extend([col for col in df.columns if keyword in col.lower()])
        perf_cols = list(set(perf_cols))  # 去重
        
        if not sigma_cols or not perf_cols:
            print("⚠ 未找到足够的sigma或性能指标数据\n")
            return results
        
        # 计算相关性
        for sigma_col in sigma_cols:
            results[sigma_col] = {}
            
            print(f"【{sigma_col}】")
            
            for perf_col in perf_cols:
                # 过滤有效数据
                valid_mask = df[sigma_col].notna() & df[perf_col].notna()
                
                if valid_mask.sum() < 10:
                    continue
                
                sigma_vals = df.loc[valid_mask, sigma_col].values
                perf_vals = df.loc[valid_mask, perf_col].values
                
                try:
                    pearson_r, pearson_p = pearsonr(sigma_vals, perf_vals)
                    spearman_r, spearman_p = spearmanr(sigma_vals, perf_vals)
                    
                    results[sigma_col][perf_col] = {
                        'pearson_r': float(pearson_r),
                        'pearson_p': float(pearson_p),
                        'spearman_r': float(spearman_r),
                        'spearman_p': float(spearman_p),
                        'n_samples': int(valid_mask.sum())
                    }
                    
                    sig = "***" if pearson_p < 0.001 else ("**" if pearson_p < 0.01 else ("*" if pearson_p < 0.05 else ""))
                    print(f"  vs {perf_col:30s}: r={pearson_r:7.4f} {sig:3s} (p={pearson_p:.4f}, n={valid_mask.sum()})")
                    
                except Exception as e:
                    print(f"  vs {perf_col:30s}: 计算失败 ({e})")
            
            print()
        
        return results
    
    def analyze_sigma_evolution(self) -> Dict:
        """分析Sigma随训练过程的演化"""
        print(f"{'='*60}")
        print("Sigma训练过程演化分析")
        print(f"{'='*60}\n")
        
        results = {}
        
        sigma_tags = [tag for tag in self.data.keys() if 'sigma' in tag.lower()]
        
        for tag in sigma_tags:
            records = sorted(self.data[tag], key=lambda x: x['step'])
            
            if len(records) < 10:
                continue
            
            steps = [r['step'] for r in records]
            values = [r['value'] for r in records]
            
            # 分段分析（训练初期、中期、后期）
            n = len(values)
            early = values[:n//3]
            mid = values[n//3:2*n//3]
            late = values[2*n//3:]
            
            evolution = {
                'total_records': len(records),
                'early_mean': float(np.mean(early)),
                'mid_mean': float(np.mean(mid)),
                'late_mean': float(np.mean(late)),
                'early_std': float(np.std(early)),
                'mid_std': float(np.std(mid)),
                'late_std': float(np.std(late)),
                'trend': 'increasing' if np.mean(late) > np.mean(early) else 'decreasing',
                'variance_change': float(np.std(late) - np.std(early)),
            }
            
            results[tag] = evolution
            
            print(f"【{tag}】")
            print(f"  训练初期: {evolution['early_mean']:.4f} ± {evolution['early_std']:.4f}")
            print(f"  训练中期: {evolution['mid_mean']:.4f} ± {evolution['mid_std']:.4f}")
            print(f"  训练后期: {evolution['late_mean']:.4f} ± {evolution['late_std']:.4f}")
            print(f"  趋势: {evolution['trend']}")
            print(f"  方差变化: {evolution['variance_change']:.6f}")
            print()
        
        return results
    
    def analyze_sigma_task_adaptation(self) -> Dict:
        """分析Sigma对不同任务类型的适配（如果有任务类型数据）"""
        print(f"{'='*60}")
        print("Sigma任务适配性分析")
        print(f"{'='*60}\n")
        
        if not self.episode_data:
            print("⚠ 无episode数据可用\n")
            return {}
        
        df = pd.DataFrame(self.episode_data)
        results = {}
        
        # 查找任务特征相关列
        task_cols = []
        for keyword in ['privacy', 'accuracy', 'computation', 'task_type', 'layer']:
            task_cols.extend([col for col in df.columns if keyword in col.lower()])
        
        if not task_cols:
            print("⚠ 未找到任务特征数据，跳过适配性分析\n")
            return results
        
        sigma_cols = [col for col in df.columns if 'sigma' in col.lower()]
        
        for sigma_col in sigma_cols:
            results[sigma_col] = {}
            
            print(f"【{sigma_col}】")
            
            for task_col in task_cols:
                valid_mask = df[sigma_col].notna() & df[task_col].notna()
                
                if valid_mask.sum() < 10:
                    continue
                
                sigma_vals = df.loc[valid_mask, sigma_col].values
                task_vals = df.loc[valid_mask, task_col].values
                
                try:
                    r, p = pearsonr(sigma_vals, task_vals)
                    
                    results[sigma_col][task_col] = {
                        'correlation': float(r),
                        'p_value': float(p),
                        'n_samples': int(valid_mask.sum())
                    }
                    
                    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                    print(f"  vs {task_col:30s}: r={r:7.4f} {sig:3s} (p={p:.4f})")
                    
                except Exception as e:
                    continue
            
            print()
        
        return results
    
    def run_analysis(self) -> Dict:
        """运行完整分析"""
        self.load_data()
        self.prepare_episode_data()
        
        results = {
            'sigma_statistics': self.analyze_sigma_statistics(),
            'performance_correlation': self.analyze_sigma_performance_correlation(),
            'evolution': self.analyze_sigma_evolution(),
            'task_adaptation': self.analyze_sigma_task_adaptation(),
        }
        
        return results
    
    def save_results(self, results: Dict, output_dir: str):
        """保存分析结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON
        json_path = os.path.join(output_dir, 'training_sigma_analysis.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ 分析结果已保存: {json_path}")
        
        # 保存episode数据
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            csv_path = os.path.join(output_dir, 'episode_data.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✓ Episode数据已保存: {csv_path}")
    
    def visualize_results(self, output_dir: str):
        """生成可视化图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 图1: Sigma随训练过程演化
        self._plot_sigma_evolution(output_dir)
        
        # 图2: Sigma分布（训练初期 vs 后期）
        self._plot_sigma_distribution_comparison(output_dir)
        
        # 图3: Sigma与奖励的关系
        self._plot_sigma_vs_reward(output_dir)
        
        # 图4: Sigma与时延的关系
        self._plot_sigma_vs_delay(output_dir)
        
        # 图5: 相关性矩阵
        self._plot_correlation_matrix(output_dir)
        
        # 图6: Sigma统计箱线图
        self._plot_sigma_boxplot(output_dir)
        
        print(f"\n✓ 所有可视化图表已保存到: {output_dir}")
    
    def _plot_sigma_evolution(self, output_dir):
        """Sigma演化图"""
        sigma_tags = [tag for tag in self.data.keys() if 'sigma' in tag.lower()]
        
        if not sigma_tags:
            return
        
        fig, axes = plt.subplots(len(sigma_tags), 1, figsize=(14, 4*len(sigma_tags)))
        
        if len(sigma_tags) == 1:
            axes = [axes]
        
        for idx, tag in enumerate(sigma_tags):
            records = sorted(self.data[tag], key=lambda x: x['step'])
            steps = [r['step'] for r in records]
            values = [r['value'] for r in records]
            
            # 原始曲线
            axes[idx].plot(steps, values, alpha=0.3, color='steelblue', linewidth=0.5)
            
            # 平滑曲线
            if len(values) > 10:
                from scipy.ndimage import uniform_filter1d
                smoothed = uniform_filter1d(values, size=min(50, len(values)//10))
                axes[idx].plot(steps, smoothed, color='darkblue', linewidth=2, label='平滑曲线')
            
            # 均值线
            mean_val = np.mean(values)
            axes[idx].axhline(mean_val, color='red', linestyle='--', 
                            label=f'均值={mean_val:.4f}', linewidth=1.5)
            
            axes[idx].set_xlabel('Episode/Step', fontsize=11)
            axes[idx].set_ylabel('Sigma Value', fontsize=11)
            axes[idx].set_title(f'{tag} - 训练过程演化', fontsize=13, fontweight='bold')
            axes[idx].legend(loc='upper right')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_sigma_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sigma_distribution_comparison(self, output_dir):
        """初期vs后期分布对比"""
        sigma_tags = [tag for tag in self.data.keys() if 'sigma' in tag.lower()]
        
        if not sigma_tags:
            return
        
        fig, axes = plt.subplots(1, len(sigma_tags), figsize=(6*len(sigma_tags), 5))
        
        if len(sigma_tags) == 1:
            axes = [axes]
        
        for idx, tag in enumerate(sigma_tags):
            records = sorted(self.data[tag], key=lambda x: x['step'])
            values = [r['value'] for r in records]
            
            n = len(values)
            early = values[:n//3]
            late = values[2*n//3:]
            
            axes[idx].hist(early, bins=30, alpha=0.5, color='blue', label='训练初期', density=True)
            axes[idx].hist(late, bins=30, alpha=0.5, color='red', label='训练后期', density=True)
            
            axes[idx].set_xlabel('Sigma Value', fontsize=11)
            axes[idx].set_ylabel('密度', fontsize=11)
            axes[idx].set_title(f'{tag}\n分布对比', fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_sigma_distribution_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sigma_vs_reward(self, output_dir):
        """Sigma vs 奖励"""
        if not self.episode_data:
            return
        
        df = pd.DataFrame(self.episode_data)
        sigma_cols = [col for col in df.columns if 'sigma' in col.lower()]
        reward_cols = [col for col in df.columns if 'reward' in col.lower()]
        
        if not sigma_cols or not reward_cols:
            return
        
        fig, axes = plt.subplots(1, min(len(sigma_cols), 3), figsize=(18, 5))
        
        if len(sigma_cols) == 1:
            axes = [axes]
        
        for idx, sigma_col in enumerate(sigma_cols[:3]):
            reward_col = reward_cols[0]  # 使用第一个奖励列
            
            valid_mask = df[sigma_col].notna() & df[reward_col].notna()
            
            if valid_mask.sum() < 10:
                continue
            
            x = df.loc[valid_mask, sigma_col].values
            y = df.loc[valid_mask, reward_col].values
            
            axes[idx].scatter(x, y, alpha=0.5, s=20, color='steelblue')
            
            # 拟合线
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            axes[idx].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            # 相关系数
            r, p_val = pearsonr(x, y)
            axes[idx].text(0.05, 0.95, f'r = {r:.4f}\np = {p_val:.4f}',
                          transform=axes[idx].transAxes, fontsize=10,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            axes[idx].set_xlabel(sigma_col, fontsize=11)
            axes[idx].set_ylabel(reward_col, fontsize=11)
            axes[idx].set_title(f'Sigma vs 奖励', fontsize=12, fontweight='bold')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_sigma_vs_reward.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sigma_vs_delay(self, output_dir):
        """Sigma vs 时延"""
        if not self.episode_data:
            return
        
        df = pd.DataFrame(self.episode_data)
        sigma_cols = [col for col in df.columns if 'sigma' in col.lower()]
        delay_cols = [col for col in df.columns if 'delay' in col.lower() or 'time' in col.lower()]
        
        if not sigma_cols or not delay_cols:
            return
        
        fig, axes = plt.subplots(1, min(len(delay_cols), 3), figsize=(18, 5))
        
        if len(delay_cols) == 1:
            axes = [axes]
        
        sigma_col = sigma_cols[0]  # 使用第一个sigma列
        
        for idx, delay_col in enumerate(delay_cols[:3]):
            valid_mask = df[sigma_col].notna() & df[delay_col].notna()
            
            if valid_mask.sum() < 10:
                continue
            
            x = df.loc[valid_mask, sigma_col].values
            y = df.loc[valid_mask, delay_col].values
            
            axes[idx].scatter(x, y, alpha=0.5, s=20, color='coral')
            
            # 拟合线
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            axes[idx].plot(x_line, p(x_line), "b--", alpha=0.8, linewidth=2)
            
            # 相关系数
            r, p_val = pearsonr(x, y)
            axes[idx].text(0.05, 0.95, f'r = {r:.4f}\np = {p_val:.4f}',
                          transform=axes[idx].transAxes, fontsize=10,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            axes[idx].set_xlabel(sigma_col, fontsize=11)
            axes[idx].set_ylabel(delay_col, fontsize=11)
            axes[idx].set_title(f'Sigma vs {delay_col}', fontsize=12, fontweight='bold')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_sigma_vs_delay.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrix(self, output_dir):
        """相关性矩阵"""
        if not self.episode_data:
            return
        
        df = pd.DataFrame(self.episode_data)
        
        # 选择数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 选择包含sigma, reward, delay的列
        selected_cols = []
        for keyword in ['sigma', 'reward', 'delay', 'time', 'completion', 'violation']:
            selected_cols.extend([col for col in numeric_cols if keyword in col.lower()])
        
        selected_cols = list(set(selected_cols))[:15]  # 最多15个指标
        
        if len(selected_cols) < 2:
            return
        
        corr_matrix = df[selected_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('训练指标相关性矩阵', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '5_correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sigma_boxplot(self, output_dir):
        """Sigma统计箱线图"""
        sigma_tags = [tag for tag in self.data.keys() if 'sigma' in tag.lower()]
        
        if not sigma_tags:
            return
        
        fig, ax = plt.subplots(figsize=(max(8, len(sigma_tags)*2), 6))
        
        data_to_plot = []
        labels = []
        
        for tag in sigma_tags:
            values = [r['value'] for r in self.data[tag]]
            data_to_plot.append(values)
            labels.append(tag.split('/')[-1])  # 简化标签名
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_ylabel('Sigma Value', fontsize=12)
        ax.set_title('训练过程中Sigma统计分布', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '6_sigma_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='训练过程Sigma性能分析')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='TensorBoard日志目录或实验目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化图表')
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir is None:
        args.output_dir = str(Path(args.log_dir) / 'sigma_analysis')
    
    # 创建分析器
    analyzer = TrainingSigmaAnalyzer(args.log_dir)
    
    # 运行分析
    results = analyzer.run_analysis()
    
    # 保存结果
    analyzer.save_results(results, args.output_dir)
    
    # 可视化
    if args.visualize:
        analyzer.visualize_results(args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"分析完成！结果已保存到: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
