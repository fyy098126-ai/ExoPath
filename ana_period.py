import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# 1. 配置所有数据集的路径 (根据你的目录结构对齐)
dataset_root = "../Time-Series-Library-main/dataset/"
data_map = {
    'ETTh1': dataset_root + 'ETT-small/ETTh1.csv',
    'ETTh2': dataset_root + 'ETT-small/ETTh2.csv',
    'ETTm1': dataset_root + 'ETT-small/ETTm1.csv',
    'ETTm2': dataset_root + 'ETT-small/ETTm2.csv',
    'Weather': dataset_root + 'weather/weather.csv',
    'ECL': dataset_root + 'electricity/electricity.csv',
    'Traffic': dataset_root + 'traffic/traffic.csv'
}

def find_best_period(name, path, max_lag=400):
    if not os.path.exists(path):
        print(f" [跳过] 找不到路径: {path}")
        return

    # 读取数据
    df = pd.read_csv(path)
    # 取最后一列作为目标变量 (通常是 OT 或流量总值)
    target = df.iloc[:, -1].values 
    
    print(f"正在分析 {name}...")

    # 计算自相关系数 (ACF)
    acf_values = sm.tsa.acf(target, nlags=max_lag)

    # 自动寻找周期点逻辑：
    # 忽略前 10 个点（防止瞬时相关性干扰），寻找第一个局部峰值
    # 这里的 10 是经验值，可以根据需要调整
    search_range = acf_values[10:]
    peak_lag = np.argmax(search_range) + 10
    peak_value = acf_values[peak_lag]

    # 绘图展示
    plt.figure(figsize=(10, 4))
    plt.plot(acf_values, color='#2c3e50', lw=1.5, label='ACF Score')
    plt.axvline(x=peak_lag, color='#e74c3c', linestyle='--', alpha=0.8, 
                label=f'Suggested Period: {peak_lag} (Score: {peak_value:.2f})')
    
    plt.title(f"Autocorrelation Analysis - {name}", fontsize=12)
    plt.xlabel("Lags (Time Steps)")
    plt.ylabel("Correlation Strength")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # 保存结果图
    save_path = f"acf_{name}.png"
    plt.savefig(save_path)
    print(f" ✅ {name} 分析完成。建议 --period: {peak_lag}。结果图已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("开始执行全数据集 ACF 周期探测...\n" + "="*40)
    for name, path in data_map.items():
        find_best_period(name, path)
    print("="*40 + "\n所有分析已结束。")