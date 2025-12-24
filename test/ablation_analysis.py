#!/usr/bin/env python3
"""
消融实验数据可视化和统计分析脚本
用于对比 SJTU_cost + optimize_yaw 的效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_csv(filename):
    """加载CSV数据"""
    if not Path(filename).exists():
        print(f"Error: {filename} not found!")
        return None
    
    df = pd.read_csv(filename)
    return df

def generate_line_plot(df, output_file="ablation_line_plot.png"):
    """生成对比折线图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['Frame'], df['Baseline_Error'], 'o-', color='#FF6B6B', 
            label='Baseline (No Optimization)', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(df['Frame'], df['Optimized_Error'], 's-', color='#4ECDC4', 
            label='Optimized (With SJTU_cost + optimize_yaw)', linewidth=2, markersize=4, alpha=0.7)
    
    ax.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reprojection Error (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Reprojection Error Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[Success] Line plot saved to: {output_file}")
    plt.close()

def generate_box_plot(baseline_errors, optimized_errors, output_file="ablation_box_plot.png"):
    """生成箱线图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot([baseline_errors, optimized_errors], 
                     labels=['Baseline', 'Optimized'],
                     patch_artist=True,
                     widths=0.6)
    
    colors = ['#FF6B6B', '#4ECDC4']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Reprojection Error (pixels)', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加均值标记
    means = [np.mean(baseline_errors), np.mean(optimized_errors)]
    ax.scatter([1, 2], means, marker='D', color='red', s=100, zorder=3, 
               label='Mean', edgecolors='black')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[Success] Box plot saved to: {output_file}")
    plt.close()

def generate_improvement_plot(df, output_file="ablation_improvement_plot.png"):
    """生成改进率柱状图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 计算移动平均（窗口大小=30）
    window = 30
    df['Improvement_MA'] = df['Improvement_Rate(%)'].rolling(window=window, center=True).mean()
    
    # 绘制改进率
    ax.bar(df['Frame'], df['Improvement_Rate(%)'], alpha=0.3, color='#95E1D3', 
           label='Instantaneous Improvement Rate')
    ax.plot(df['Frame'], df['Improvement_MA'], 'o-', color='#38ADA9', linewidth=2.5,
            label=f'Moving Average (window={window})', markersize=3)
    
    ax.axhline(y=df['Improvement_Rate(%)'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f"Mean Improvement: {df['Improvement_Rate(%)'].mean():.2f}%")
    
    ax.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Optimization Improvement Rate Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[Success] Improvement plot saved to: {output_file}")
    plt.close()

def generate_histogram(baseline_errors, optimized_errors, output_file="ablation_histogram.png"):
    """生成直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 基线直方图
    axes[0].hist(baseline_errors, bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(baseline_errors), color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {np.mean(baseline_errors):.4f}")
    axes[0].set_xlabel('Error (pixels)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Baseline Error Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 优化直方图
    axes[1].hist(optimized_errors, bins=30, color='#4ECDC4', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(optimized_errors), color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {np.mean(optimized_errors):.4f}")
    axes[1].set_xlabel('Error (pixels)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Optimized Error Distribution', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[Success] Histogram saved to: {output_file}")
    plt.close()

def generate_statistics_table(baseline_errors, optimized_errors, output_file="ablation_statistics.txt"):
    """生成详细统计表"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDY: SJTU_cost & optimize_yaw Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("BASELINE (No Optimization)\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Samples:        {len(baseline_errors)}\n")
        f.write(f"  Mean Error:           {np.mean(baseline_errors):.6f} pixels\n")
        f.write(f"  Median Error:         {np.median(baseline_errors):.6f} pixels\n")
        f.write(f"  Std Dev:              {np.std(baseline_errors):.6f} pixels\n")
        f.write(f"  Min Error:            {np.min(baseline_errors):.6f} pixels\n")
        f.write(f"  Max Error:            {np.max(baseline_errors):.6f} pixels\n")
        f.write(f"  Q1 (25%):             {np.percentile(baseline_errors, 25):.6f} pixels\n")
        f.write(f"  Q3 (75%):             {np.percentile(baseline_errors, 75):.6f} pixels\n")
        f.write(f"  IQR:                  {np.percentile(baseline_errors, 75) - np.percentile(baseline_errors, 25):.6f} pixels\n\n")
        
        f.write("OPTIMIZED (With SJTU_cost + optimize_yaw)\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Samples:        {len(optimized_errors)}\n")
        f.write(f"  Mean Error:           {np.mean(optimized_errors):.6f} pixels\n")
        f.write(f"  Median Error:         {np.median(optimized_errors):.6f} pixels\n")
        f.write(f"  Std Dev:              {np.std(optimized_errors):.6f} pixels\n")
        f.write(f"  Min Error:            {np.min(optimized_errors):.6f} pixels\n")
        f.write(f"  Max Error:            {np.max(optimized_errors):.6f} pixels\n")
        f.write(f"  Q1 (25%):             {np.percentile(optimized_errors, 25):.6f} pixels\n")
        f.write(f"  Q3 (75%):             {np.percentile(optimized_errors, 75):.6f} pixels\n")
        f.write(f"  IQR:                  {np.percentile(optimized_errors, 75) - np.percentile(optimized_errors, 25):.6f} pixels\n\n")
        
        baseline_mean = np.mean(baseline_errors)
        optimized_mean = np.mean(optimized_errors)
        improvement = (baseline_mean - optimized_mean) / baseline_mean * 100
        
        f.write("IMPROVEMENT ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Mean Error Reduction: {baseline_mean - optimized_mean:.6f} pixels\n")
        f.write(f"  Improvement Rate:     {improvement:.2f}%\n")
        f.write(f"  Relative Reduction:   {improvement:.2f}%\n")
        f.write(f"  Error Reduction Std:  {np.std([baseline_errors[i] - optimized_errors[i] for i in range(len(baseline_errors))]):.6f} pixels\n\n")
        
        f.write("PERCENTILE COMPARISON\n")
        f.write("-" * 80 + "\n")
        for p in [10, 25, 50, 75, 90]:
            b_val = np.percentile(baseline_errors, p)
            o_val = np.percentile(optimized_errors, p)
            reduction = (b_val - o_val) / b_val * 100 if b_val != 0 else 0
            f.write(f"  P{p:02d}: Baseline={b_val:.6f}, Optimized={o_val:.6f}, "
                   f"Improvement={reduction:.2f}%\n")
        f.write("\n")
    
    print(f"[Success] Statistics table saved to: {output_file}")

def main():
    """主函数"""
    csv_file = "ablation_study_detailed.csv"
    
    # 加载数据
    df = load_csv(csv_file)
    if df is None:
        return -1
    
    baseline_errors = df['Baseline_Error'].values
    optimized_errors = df['Optimized_Error'].values
    
    print("\n" + "=" * 80)
    print("Ablation Study: SJTU_cost & optimize_yaw")
    print("=" * 80 + "\n")
    
    # 生成各种可视化
    print("[Processing] Generating line plot...")
    generate_line_plot(df, "ablation_line_plot.png")
    
    print("[Processing] Generating box plot...")
    generate_box_plot(baseline_errors, optimized_errors, "ablation_box_plot.png")
    
    print("[Processing] Generating improvement plot...")
    generate_improvement_plot(df, "ablation_improvement_plot.png")
    
    print("[Processing] Generating histogram...")
    generate_histogram(baseline_errors, optimized_errors, "ablation_histogram.png")
    
    print("[Processing] Generating statistics table...")
    generate_statistics_table(baseline_errors, optimized_errors, "ablation_statistics.txt")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Samples: {len(baseline_errors)}")
    print(f"\nBaseline (No Optimization):")
    print(f"  Mean:  {np.mean(baseline_errors):.6f} pixels")
    print(f"  Std:   {np.std(baseline_errors):.6f} pixels")
    print(f"\nOptimized (With SJTU_cost + optimize_yaw):")
    print(f"  Mean:  {np.mean(optimized_errors):.6f} pixels")
    print(f"  Std:   {np.std(optimized_errors):.6f} pixels")
    print(f"\nImprovement:")
    print(f"  Error Reduction: {np.mean(baseline_errors) - np.mean(optimized_errors):.6f} pixels")
    print(f"  Improvement Rate: {(np.mean(baseline_errors) - np.mean(optimized_errors)) / np.mean(baseline_errors) * 100:.2f}%")
    print("\n" + "=" * 80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
