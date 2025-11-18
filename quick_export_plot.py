"""
快速导出并绘制TensorBoard数据
使用方法: python quick_export_plot.py <tensorboard_log_dir>
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


def export_and_plot(log_dir: str):
    """导出TensorBoard数据并自动绘图"""
    
    if not os.path.exists(log_dir):
        print(f"错误: 目录不存在: {log_dir}")
        return
    
    print(f"{'='*60}")
    print(f"从TensorBoard日志导出数据并绘图")
    print(f"日志目录: {log_dir}")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    output_dir = os.path.join(log_dir, 'exported_data')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 加载TensorBoard数据
    print("正在加载TensorBoard事件文件...")
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 加载所有标量
        }
    )
    ea.Reload()
    
    # 获取所有标量标签
    tags = ea.Tags()['scalars']
    print(f"找到 {len(tags)} 个指标\n")
    
    # 导出所有数据到字典
    data_dict = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data_dict[tag] = {
            'step': [e.step for e in events],
            'value': [e.value for e in events]
        }
    
    # 创建DataFrame（合并所有指标）
    df_combined = pd.DataFrame()
    for tag, data in data_dict.items():
        if df_combined.empty:
            df_combined['step'] = data['step']
        safe_tag = tag.replace('/', '_').replace('\\', '_')
        df_combined[safe_tag] = data['value']
    
    # 保存CSV
    csv_path = os.path.join(output_dir, 'all_metrics.csv')
    df_combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ CSV数据已保存: {csv_path}\n")
    
    # 设置绘图样式
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    
    print("开始绘制图表...\n")
    
    # ========== 图1: 训练总览 ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('训练指标总览', fontsize=16, fontweight='bold')
    
    # 1.1 Episode奖励
    ax = axes[0, 0]
    if 'reward_avg_episode_reward' in df_combined.columns:
        ax.plot(df_combined['step'], df_combined['reward_avg_episode_reward'], 
                linewidth=2, color='#2E86AB')
    ax.set_xlabel('Episode')
    ax.set_ylabel('奖励')
    ax.set_title('Episode平均奖励')
    ax.grid(True, alpha=0.3)
    
    # 1.2 任务奖励
    ax = axes[0, 1]
    if 'reward_global_avg_task_reward' in df_combined.columns:
        ax.plot(df_combined['step'], df_combined['reward_global_avg_task_reward'], 
                linewidth=2, color='#A23B72', label='全局')
    if 'reward_avg_per_vehicle_task_reward' in df_combined.columns:
        ax.plot(df_combined['step'], df_combined['reward_avg_per_vehicle_task_reward'], 
                linewidth=2, color='#F18F01', label='每车平均')
    ax.set_xlabel('Episode')
    ax.set_ylabel('奖励')
    ax.set_title('任务级奖励')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.3 任务完成率
    ax = axes[0, 2]
    if 'performance_global_task_completion_rate' in df_combined.columns:
        ax.plot(df_combined['step'], df_combined['performance_global_task_completion_rate'], 
                linewidth=2, color='#06A77D', label='全局')
    if 'performance_avg_per_vehicle_completion_rate' in df_combined.columns:
        ax.plot(df_combined['step'], df_combined['performance_avg_per_vehicle_completion_rate'], 
                linewidth=2, color='#D62828', label='每车平均')
    ax.set_xlabel('Episode')
    ax.set_ylabel('完成率 (%)')
    ax.set_title('任务完成率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.4 端到端时延
    ax = axes[1, 0]
    if 'delay_global_avg_end_to_end_delay' in df_combined.columns:
        ax.plot(df_combined['step'], df_combined['delay_global_avg_end_to_end_delay'], 
                linewidth=2, color='#E63946', label='全局')
    if 'delay_avg_per_vehicle_end_to_end_delay' in df_combined.columns:
        ax.plot(df_combined['step'], df_combined['delay_avg_per_vehicle_end_to_end_delay'], 
                linewidth=2, color='#F77F00', label='每车平均')
    ax.set_xlabel('Episode')
    ax.set_ylabel('时延')
    ax.set_title('端到端时延')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.5 时延分解
    ax = axes[1, 1]
    if 'delay_global_avg_exec_time' in df_combined.columns:
        ax.plot(df_combined['step'], df_combined['delay_global_avg_exec_time'], 
                linewidth=2, color='#457B9D', label='执行时延')
    if 'delay_global_avg_waiting_delay' in df_combined.columns:
        ax.plot(df_combined['step'], df_combined['delay_global_avg_waiting_delay'], 
                linewidth=2, color='#2A9D8F', label='等待时延')
    ax.set_xlabel('Episode')
    ax.set_ylabel('时延')
    ax.set_title('时延分解')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.6 Critic损失
    ax = axes[1, 2]
    if 'loss_critic_avg' in df_combined.columns:
        ax.plot(df_combined['step'], df_combined['loss_critic_avg'], 
                linewidth=2, color='#6A4C93')
    ax.set_xlabel('Episode')
    ax.set_ylabel('损失')
    ax.set_title('Critic损失')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    overview_path = os.path.join(plot_dir, 'training_overview.png')
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练总览图: {overview_path}")
    plt.close()
    
    # ========== 图2: 奖励详细对比 ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    reward_cols = [col for col in df_combined.columns if 'reward' in col and col != 'step']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62828']
    
    for i, col in enumerate(reward_cols):
        label = col.replace('reward_', '').replace('_', ' ').title()
        ax.plot(df_combined['step'], df_combined[col], 
                linewidth=2, color=colors[i % len(colors)], label=label, alpha=0.8)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('奖励值', fontsize=12)
    ax.set_title('奖励指标对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    reward_path = os.path.join(plot_dir, 'reward_comparison.png')
    plt.savefig(reward_path, dpi=300, bbox_inches='tight')
    print(f"✓ 奖励对比图: {reward_path}")
    plt.close()
    
    # ========== 图3: 时延全面分析 ==========
    delay_cols = [col for col in df_combined.columns if 'delay' in col and col != 'step']
    if delay_cols:
        fig, ax = plt.subplots(figsize=(14, 6))
        colors_delay = ['#E63946', '#F77F00', '#457B9D', '#2A9D8F', '#06A77D', '#118AB2']
        
        for i, col in enumerate(delay_cols):
            label = col.replace('delay_', '').replace('_', ' ').title()
            ax.plot(df_combined['step'], df_combined[col], 
                    linewidth=2.5, color=colors_delay[i % len(colors_delay)], 
                    label=label, alpha=0.8)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('时延值', fontsize=12)
        ax.set_title('时延指标全面分析', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        delay_path = os.path.join(plot_dir, 'delay_analysis.png')
        plt.savefig(delay_path, dpi=300, bbox_inches='tight')
        print(f"✓ 时延分析图: {delay_path}")
        plt.close()
    
    # ========== 图4: 性能指标 ==========
    perf_cols = [col for col in df_combined.columns if 'performance' in col and col != 'step']
    if perf_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors_perf = ['#06A77D', '#D62828', '#F18F01']
        
        for i, col in enumerate(perf_cols):
            label = col.replace('performance_', '').replace('_', ' ').title()
            # 违反次数用右轴
            if 'violation' in col:
                ax2 = ax.twinx()
                ax2.plot(df_combined['step'], df_combined[col], 
                        linewidth=2, color=colors_perf[i % len(colors_perf)], 
                        label=label, linestyle='--', alpha=0.7)
                ax2.set_ylabel('违反次数', fontsize=12)
                ax2.legend(loc='upper right', fontsize=10)
            else:
                ax.plot(df_combined['step'], df_combined[col], 
                       linewidth=2.5, color=colors_perf[i % len(colors_perf)], 
                       label=label, alpha=0.8)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('完成率 (%)', fontsize=12)
        ax.set_title('任务完成性能', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        perf_path = os.path.join(plot_dir, 'performance_metrics.png')
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        print(f"✓ 性能指标图: {perf_path}")
        plt.close()
    
    # ========== 生成统计摘要 ==========
    summary = {}
    for col in df_combined.columns:
        if col != 'step':
            summary[col] = {
                '最小值': float(df_combined[col].min()),
                '最大值': float(df_combined[col].max()),
                '平均值': float(df_combined[col].mean()),
                '最终值': float(df_combined[col].iloc[-1]),
                '标准差': float(df_combined[col].std())
            }
    
    summary_df = pd.DataFrame(summary).T
    summary_path = os.path.join(output_dir, 'metrics_summary.csv')
    summary_df.to_csv(summary_path, encoding='utf-8-sig')
    print(f"✓ 统计摘要: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"导出完成！")
    print(f"{'='*60}")
    print(f"数据文件: {output_dir}")
    print(f"图表文件: {plot_dir}")
    print(f"\n生成的文件:")
    print(f"  - all_metrics.csv (所有指标数据)")
    print(f"  - metrics_summary.csv (统计摘要)")
    print(f"  - plots/training_overview.png (训练总览)")
    print(f"  - plots/reward_comparison.png (奖励对比)")
    print(f"  - plots/delay_analysis.png (时延分析)")
    print(f"  - plots/performance_metrics.png (性能指标)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法: python quick_export_plot.py <tensorboard_log_dir>")
        print("示例: python quick_export_plot.py runs/experiments/1000iters_10.0eps_10vehicles_7tasks_fullmodel")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    export_and_plot(log_dir)
