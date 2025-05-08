import numpy as np
import matplotlib.pyplot as plt


"""
说明：
1. 该代码用于处理睡眠时相数据，计算睡眠指标，并绘制睡眠时相图。
2. 主要功能包括：
   - 将字符串数组转换为数字数组
   - 从txt文件中读取睡眠时相并转换为numpy数组
   - 计算睡眠指标（潜伏期、持续时间、分期等）
   - 渲染睡眠时相图
3. 代码中使用了numpy和matplotlib库。
4. 该代码假设每个帧的时间为0.5分钟（30秒），并且睡眠时相的映射关系为：
   {W: 0, N1: 1, N2: 2, N3: 3, R: 4, ?: 5}
"""


def str2ndarray(str_array):
    """ 将字符串数组转换为数字数组 """
    mapping = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4, '?': 5}
    vectorized_map = np.vectorize(lambda s: mapping[s])
    return vectorized_map(str_array)


def load_hypnogram(path):
    """ 从txt文件中读取睡眠时相并转换为numpy数组 """
    str_array = np.loadtxt(path, dtype=str) 
    id_array = str2ndarray(str_array)
    return id_array


def calculate_hypnogram_metrics(hypnogram: np.ndarray, verbose=False):
    """计算睡眠指标"""

    # 定义各阶段的时间和持续时间
    time_per_frame = 0.5  # 每个帧的时间（单位：min），假设每帧为30秒

    # 获取第一个N1的索引
    try:
        first_n1_index = np.where(hypnogram == 1)[0][0]
    except IndexError:
        raise ValueError("hypnogram 中没有 N1 阶段，无法计算潜伏期。")

    # 1. 计算潜伏期
    def calculate_latency_after_n1(stage, first_n1_index):
        """计算第一个 N1 之后指定阶段的潜伏期，返回时间（min）"""
        try:
            target_index = np.where(hypnogram[first_n1_index:] == stage)[0][0] + first_n1_index
            return (target_index - first_n1_index) * time_per_frame
        except IndexError:
            return np.nan  # 如果阶段不存在，返回 NaN

    # 根据第一个 N1 后的各阶段潜伏期定义
    n1_latency = 0  # N1潜伏期为0
    n2_latency = calculate_latency_after_n1(2, first_n1_index)  # N2潜伏期
    n3_latency = calculate_latency_after_n1(3, first_n1_index)  # N3潜伏期
    rem_latency = calculate_latency_after_n1(4, first_n1_index)  # REM潜伏期

    # 2. 睡眠持续时间
    tst = np.sum(hypnogram > 0) * time_per_frame  # 总睡眠时间 (TST)
    rem_duration = np.sum(hypnogram == 4) * time_per_frame  # REM持续时间
    nrem_duration = np.sum((hypnogram == 1) | (hypnogram == 2) | (hypnogram == 3)) * time_per_frame  # NREM持续时间
    sws_duration = np.sum(hypnogram == 3) * time_per_frame  # SWS持续时间 (N3阶段)

    # 3. 睡眠分期
    # 从第一个N1之后的片段中计算wake持续时间和次数
    wake_after_sleep = hypnogram[first_n1_index:]
    wake_duration = np.sum(wake_after_sleep == 0) * time_per_frame  # 入睡后的清醒时间
    wake_episodes = np.sum((wake_after_sleep[:-1] != 0) & (wake_after_sleep[1:] == 0))  # 入睡后wake次数

    n1_duration = np.sum(hypnogram == 1) * time_per_frame  # N1持续时间
    n2_duration = np.sum(hypnogram == 2) * time_per_frame  # N2持续时间
    n3_duration = sws_duration  # N3持续时间 (SWS持续时间即为N3持续时间)

    # 4. 各阶段的百分比
    total_sleep_time = tst  # 等于 R + N1 + N2 + N3 的总时间
    rem_percentage = (rem_duration / total_sleep_time) * 100
    n1_percentage = (n1_duration / total_sleep_time) * 100
    n2_percentage = (n2_duration / total_sleep_time) * 100
    n3_percentage = (n3_duration / total_sleep_time) * 100

    if verbose:
        # 打印结果
        print("睡眠潜伏期:")
        print(f"N1 潜伏期: {n1_latency} min")
        print(f"N2 潜伏期: {n2_latency} min")
        print(f"N3 潜伏期: {n3_latency} min")
        print(f"REM 潜伏期: {rem_latency} min")

        print("\n睡眠持续时间:")
        print(f"总睡眠时间 (TST): {tst} min")
        print(f"REM 持续时间: {rem_duration} min")
        print(f"NREM 持续时间: {nrem_duration} min")
        print(f"SWS (N3) 持续时间: {sws_duration} min")

        print("\n睡眠分期:")
        print(f"")
        print(f"W (SPT): 次数: {wake_episodes} 持续时间 (W): {wake_duration} min")
        print(f"R 持续时间: {rem_duration} min")
        print(f"N1 持续时间: {n1_duration} min")
        print(f"N2 持续时间: {n2_duration} min")
        print(f"N3 持续时间: {n3_duration} min")

        print("\n睡眠分期百分比:")
        print(f"REM 百分比: {rem_percentage:.2f}%")
        print(f"N1 百分比: {n1_percentage:.2f}%")
        print(f"N2 百分比: {n2_percentage:.2f}%")
        print(f"N3 百分比: {n3_percentage:.2f}%")

    return {
        "n1_latency": n1_latency,   # N1潜伏期
        "n2_latency": n2_latency,   # N2潜伏期
        "n3_latency": n3_latency,   # N3潜伏期
        "rem_latency": rem_latency, # REM潜伏期
        "tst": tst,                 # 总睡眠时间(TST)
        "rem_duration": rem_duration,   # REM持续时间
        "nrem_duration": nrem_duration, # NREM持续时间
        "sws_duration": sws_duration,   # SWS(N3)持续时间
        "wake_duration": wake_duration, # W(SPT)持续时间
        "wake_episodes": wake_episodes, # W(SPT)次数
        "n1_duration": n1_duration,     # N1持续时间
        "n2_duration": n2_duration,     # N2持续时间
        "n3_duration": n3_duration,     # N3持续时间
        "rem_percentage": rem_percentage,   # REM百分比
        "n1_percentage": n1_percentage,     # N1百分比
        "n2_percentage": n2_percentage,     # N2百分比
        "n3_percentage": n3_percentage,     # N3百分比
    }


def render_hypnogram(predicts: np.ndarray, filename):
    # 图形参数
    N = len(predicts)

    # 调整数据顺序，与 MATLAB 中的步骤一致
    predicts = 4 - predicts

    # 创建图形窗口
    fig, ax = plt.subplots(figsize=(20, 5))

    # 配置AI标注的图
    ax.set_title("", fontsize=25)
    for level in range(1, 6):  # 绘制参考线
        ax.axhline(level, linestyle='--', color=[0.8, 0.8, 1], linewidth=0.5)
    ax.stairs(predicts + 1, color='black')
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['R', 'N3', 'N2', 'N1', 'W'], fontsize=15)

    # 设置x轴
    ax.set_xlim([1, N])
    xticks = np.arange(0, N + 1, 120)
    xticklabels = [f'{i//120}h' for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=15)

    # 显示图形
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # 示例用法
    txt_path = r"Demo/data/output/2025021001.txt"  # 模型预测的txt文件路径
    fig_path = r"Demo/data/output/hypnogram.svg"   # 睡眠时相图保存路径

    hypnogram = load_hypnogram(txt_path)
    metrics = calculate_hypnogram_metrics(hypnogram, verbose=True)
    render_hypnogram(hypnogram, fig_path)
