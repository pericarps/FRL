"""
TensorBoard数据导出工具
用于将TensorBoard事件文件导出为CSV/JSON格式，方便绘图和分析
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tensorboard.backend.event_processing import event_accumulator


def export_tensorboard_scalars(log_dir: str, output_dir: str = None, 
                                format: str = 'csv', tags: Optional[List[str]] = None):
    """
    从TensorBoard日志目录导出所有标量数据
    
    Args:
        log_dir: TensorBoard日志目录路径
        output_dir: 输出目录，默认为log_dir/exported_data
        format: 导出格式，'csv' 或 'json'
        tags: 要导出的标签列表，None表示导出所有标签
    """
    if not os.path.exists(log_dir):
        print(f"错误: 日志目录不存在: {log_dir}")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(log_dir, 'exported_data')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在从 {log_dir} 导出数据...")
    print(f"输出目录: {output_dir}")
    
    # 加载TensorBoard事件文件
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.SCALARS: 0,  # 0表示加载所有
            event_accumulator.HISTOGRAMS: 0,
        }
    )
    ea.Reload()
    
    # 获取所有标量标签
    available_tags = ea.Tags()['scalars']
    print(f"\n找到 {len(available_tags)} 个标量指标")
    
    # 过滤标签
    if tags is not None:
        available_tags = [tag for tag in available_tags if tag in tags]
        print(f"筛选后保留 {len(available_tags)} 个指标")
    
    # 导出数据
    all_data = {}
    
    for tag in available_tags:
        print(f"  导出: {tag}")
        events = ea.Scalars(tag)
        
        # 提取数据
        steps = [event.step for event in events]
        values = [event.value for event in events]
        wall_times = [event.wall_time for event in events]
        
        all_data[tag] = {
            'step': steps,
            'value': values,
            'wall_time': wall_times
        }
    
    # 保存数据
    if format == 'csv':
        # 每个标签保存为单独的CSV文件
        for tag, data in all_data.items():
            safe_tag = tag.replace('/', '_').replace('\\', '_')
            csv_path = os.path.join(output_dir, f"{safe_tag}.csv")
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 同时保存一个合并的CSV（宽格式）
        combined_df = pd.DataFrame()
        for tag, data in all_data.items():
            if combined_df.empty:
                combined_df['step'] = data['step']
            safe_tag = tag.replace('/', '_').replace('\\', '_')
            combined_df[safe_tag] = data['value']
        
        combined_path = os.path.join(output_dir, 'all_metrics_combined.csv')
        combined_df.to_csv(combined_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ CSV文件已保存到: {output_dir}")
        print(f"  - 单独文件: {len(all_data)} 个")
        print(f"  - 合并文件: all_metrics_combined.csv")
        
    elif format == 'json':
        json_path = os.path.join(output_dir, 'all_metrics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ JSON文件已保存: {json_path}")
    
    # 生成元数据摘要
    metadata = {
        'log_dir': log_dir,
        'export_time': pd.Timestamp.now().isoformat(),
        'total_tags': len(all_data),
        'tags': list(all_data.keys()),
        'tag_categories': {}
    }
    
    # 按类别分组标签
    for tag in all_data.keys():
        category = tag.split('/')[0] if '/' in tag else 'other'
        if category not in metadata['tag_categories']:
            metadata['tag_categories'][category] = []
        metadata['tag_categories'][category].append(tag)
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n指标分类:")
    for category, tags_list in metadata['tag_categories'].items():
        print(f"  {category}: {len(tags_list)} 个指标")
    
    return all_data


def create_plotting_script(output_dir: str):
    """
    创建一个示例绘图脚本
    """
    script_content = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
TensorBoard数据绘图脚本
使用导出的CSV数据绘制训练曲线
\"\"\"

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据目录
data_dir = Path(__file__).parent

# 加载合并数据
df = pd.read_csv(data_dir / 'all_metrics_combined.csv')

# 创建输出目录
plot_dir = data_dir / 'plots'
plot_dir.mkdir(exist_ok=True)

# 示例1: 绘制奖励曲线
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('训练指标总览', fontsize=16)

# 奖励曲线
if 'reward_avg_episode_reward' in df.columns:
    axes[0, 0].plot(df['step'], df['reward_avg_episode_reward'], label='Episode奖励', linewidth=2)
if 'reward_global_avg_task_reward' in df.columns:
    axes[0, 0].plot(df['step'], df['reward_global_avg_task_reward'], label='任务奖励', linewidth=2)
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('奖励')
axes[0, 0].set_title('奖励曲线')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 任务完成率
if 'performance_global_task_completion_rate' in df.columns:
    axes[0, 1].plot(df['step'], df['performance_global_task_completion_rate'], 
                    color='green', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('完成率 (%)')
    axes[0, 1].set_title('任务完成率')
    axes[0, 1].grid(True, alpha=0.3)

# 端到端时延
if 'delay_global_avg_end_to_end_delay' in df.columns:
    axes[1, 0].plot(df['step'], df['delay_global_avg_end_to_end_delay'], 
                    color='orange', linewidth=2, label='端到端时延')
if 'delay_global_avg_exec_time' in df.columns:
    axes[1, 0].plot(df['step'], df['delay_global_avg_exec_time'], 
                    color='blue', linewidth=2, label='执行时延')
if 'delay_global_avg_waiting_delay' in df.columns:
    axes[1, 0].plot(df['step'], df['delay_global_avg_waiting_delay'], 
                    color='red', linewidth=2, label='等待时延')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('时延')
axes[1, 0].set_title('时延分解')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 损失曲线
if 'loss_critic_avg' in df.columns:
    axes[1, 1].plot(df['step'], df['loss_critic_avg'], 
                    color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('损失')
    axes[1, 1].set_title('Critic损失')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir / 'training_overview.png', dpi=300, bbox_inches='tight')
print(f"✓ 训练总览图已保存: {plot_dir / 'training_overview.png'}")

# 示例2: 单独绘制高分辨率奖励曲线
fig, ax = plt.subplots(figsize=(12, 6))
if 'reward_avg_episode_reward' in df.columns:
    ax.plot(df['step'], df['reward_avg_episode_reward'], 
            linewidth=2, color='#2E86AB', label='Episode平均奖励')
if 'reward_global_avg_task_reward' in df.columns:
    ax.plot(df['step'], df['reward_global_avg_task_reward'], 
            linewidth=2, color='#A23B72', label='任务平均奖励')
if 'reward_avg_per_vehicle_task_reward' in df.columns:
    ax.plot(df['step'], df['reward_avg_per_vehicle_task_reward'], 
            linewidth=2, color='#F18F01', label='每车任务平均奖励')

ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('奖励值', fontsize=12)
ax.set_title('训练奖励曲线', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(plot_dir / 'reward_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ 奖励曲线图已保存: {plot_dir / 'reward_curves.png'}")

# 示例3: 时延对比图
fig, ax = plt.subplots(figsize=(12, 6))
if 'delay_global_avg_end_to_end_delay' in df.columns:
    ax.plot(df['step'], df['delay_global_avg_end_to_end_delay'], 
            linewidth=2.5, color='#E63946', label='端到端时延', alpha=0.8)
if 'delay_global_avg_exec_time' in df.columns:
    ax.plot(df['step'], df['delay_global_avg_exec_time'], 
            linewidth=2, color='#457B9D', label='执行时延', alpha=0.8)
if 'delay_global_avg_waiting_delay' in df.columns:
    ax.plot(df['step'], df['delay_global_avg_waiting_delay'], 
            linewidth=2, color='#2A9D8F', label='等待时延', alpha=0.8)

ax.set_xlabel('Episode', fontsize=12)
ax.set_ylabel('时延', fontsize=12)
ax.set_title('任务时延分解曲线', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(plot_dir / 'delay_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ 时延曲线图已保存: {plot_dir / 'delay_curves.png'}")

print(f"\\n所有图表已保存到: {plot_dir}")
"""
    
    script_path = os.path.join(output_dir, 'plot_metrics.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"\n✓ 绘图脚本已生成: {script_path}")
    print(f"  运行方式: cd {output_dir} && python plot_metrics.py")


def main():
    parser = argparse.ArgumentParser(description='导出TensorBoard数据')
    parser.add_argument('log_dir', type=str, help='TensorBoard日志目录路径')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='输出目录，默认为log_dir/exported_data')
    parser.add_argument('--format', type=str, default='csv', 
                        choices=['csv', 'json'], help='导出格式')
    parser.add_argument('--tags', type=str, nargs='+', default=None, 
                        help='要导出的标签列表（空格分隔）')
    parser.add_argument('--create-plot-script', action='store_true', 
                        help='生成绘图脚本')
    
    args = parser.parse_args()
    
    # 导出数据
    all_data = export_tensorboard_scalars(
        args.log_dir, 
        args.output_dir, 
        args.format, 
        args.tags
    )
    
    # 生成绘图脚本
    if args.create_plot_script and all_data:
        output_dir = args.output_dir or os.path.join(args.log_dir, 'exported_data')
        create_plotting_script(output_dir)
    
    print("\n✓ 数据导出完成！")


if __name__ == '__main__':
    main()
