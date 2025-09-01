import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_performance(df, last_n_episodes=100):
    """
    分析最后 N 个回合的性能指标。
    Analyzes performance metrics for the last N episodes.

    Args:
        df (pd.DataFrame): 包含奖励数据的DataFrame。
        last_n_episodes (int): 用于计算最终性能和稳定性的回合数。

    Returns:
        pd.DataFrame: 包含分析结果的DataFrame。
    """
    # 截取最后 N 个回合的数据
    last_episodes_df = df.iloc[-last_n_episodes:]

    # 1. 最终性能 (平均奖励)
    final_performance = last_episodes_df.mean()

    # 2. 稳定性 (奖励的标准差)
    stability = last_episodes_df.std()

    # 将结果合并到一个新的DataFrame中
    analysis_df = pd.DataFrame({
        '最终平均奖励 (后100回合)': final_performance,
        '奖励标准差 (稳定性)': stability
    })

    return analysis_df.sort_values(by='最终平均奖励 (后100回合)', ascending=False)


def analyze_convergence_speed(df, threshold=200, window=30):
    """
    分析收敛速度。
    Analyzes convergence speed.

    定义为：奖励的移动平均值首次持续稳定在阈值以上的回合数。
    Defined as: The episode where the moving average of rewards first
                consistently stays above a threshold.

    Args:
        df (pd.DataFrame): 包含奖励数据的DataFrame。
        threshold (int): 判断收敛的奖励阈值。
        window (int): 用于计算移动平均的窗口大小，以确保稳定性。

    Returns:
        pd.Series: 包含每个优化器收敛速度的Series。
    """
    convergence_speeds = {}
    for optimizer in df.columns:
        # 计算移动平均
        moving_avg = df[optimizer].rolling(window=window, min_periods=1).mean()
        # 找到第一个移动平均值超过阈值的位置
        converged_series = moving_avg[moving_avg > threshold]

        if not converged_series.empty:
            # .index[0] 给出了第一次达标的索引（回合数）
            convergence_point = converged_series.index[0]
            convergence_speeds[optimizer] = convergence_point
        else:
            # 如果从未达到阈值
            convergence_speeds[optimizer] = '未收敛'

    return pd.Series(convergence_speeds, name=f'收敛到{threshold}奖励的回合数').sort_values()


def plot_results(df):
    """
    将结果绘制成图表。
    Plots the results.
    """
    # 解决中文字体显示问题
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    except:
        print("警告：未找到中文字体'SimHei'，图表中的中文可能显示为方块。")
        print("Warning: Chinese font 'SimHei' not found. Chinese characters in the plot may display as squares.")

    plt.figure(figsize=(14, 8))

    # 绘制每个优化器的移动平均奖励曲线
    for optimizer in df.columns:
        moving_avg = df[optimizer].rolling(window=30, min_periods=1).mean()
        plt.plot(moving_avg, label=optimizer, alpha=0.8)

    plt.title('不同优化器在 CartPole-v1 上的性能对比 (30回合移动平均)', fontsize=16)
    plt.xlabel('回合 (Episode)', fontsize=12)
    plt.ylabel('移动平均奖励 (Moving Average Reward)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.ylim(bottom=0)  # 确保Y轴从0开始
    plt.show()


if __name__ == '__main__':
    # 定义CSV文件名
    filename = 'rl_performance_results.csv'

    if not os.path.exists(filename):
        print(f"错误：找不到文件 '{filename}'。请确保该文件与脚本在同一目录下。")
    else:
        # 读取数据，将第一列作为索引
        results_df = pd.read_csv(filename, index_col='Episode')

        print("--- 强化学习性能分析报告 ---\n")

        # 1. 分析最终性能和稳定性
        performance_analysis = analyze_performance(results_df)
        print("1. 最终性能与稳定性分析 (基于最后100回合):\n")
        print(performance_analysis)
        print("\n* '最终平均奖励' 越高越好。")
        print("* '奖励标准差' 越低越好，代表越稳定。")

        print("\n" + "=" * 50 + "\n")

        # 2. 分析收敛速度
        speed_analysis = analyze_convergence_speed(results_df)
        print("2. 收敛速度分析:\n")
        print(speed_analysis)
        print("\n* '收敛回合数' 越低越好，代表学得越快。")

        # 3. 绘制性能曲线图
        print("\n正在生成性能曲线图...")
        plot_results(results_df)
