# 项目结构

```text
Demo/
├── data/                               # 示例数据
│   ├── input/                    
|   |   ├── 2025021001.edf              # 输入 PSG 数据
|   |   └── 2025021001.scoredata.xml    # 输入 scoredata 数据，该数据从康迪软件导出，目的是和技师现有分图流程对齐
│   └── ouput/                          # 运行示例（inference.bat 或 inference.sh）后默认输出位置 
├── weights/   
│   ├── LPSGM_cpu.pt                    # CPU版本模型权重
│   └── LPSGM_cuda.pt                   # GPU版本模型权重
├── config.py                           # 通道配置定义（适配杭州市第七人民医院和衢州三院）
├── main.py                             # 主程序入口
├── utils.py                            # 工具函数
├── inference.sh                        # Linux推理脚本
├── inference.bat                       # Windows推理脚本
└── README.md                           # 项目说明
```

# 运行推理

推理主程序入口为 `main.py`，可通过以下脚本调用：
- Linux: `./inference.sh`
- Windows: `双击执行 inference.bat`

## 必需参数
- `--input`  
  输入路径（文件/文件夹）  
  - 文件路径：处理单个.edf文件
  - 文件夹路径：批量处理目录内所有.edf文件

- `--output`  
  输出路径（文件/文件夹）  
  - 文件路径：结果写入指定.txt文件
  - 文件夹路径：自动创建同名.txt文件

## 可选参数
- `--log_file`  
  推理输出的日志文件，省略则不输出日志  

- `--scoredata_xml`  
  兼容康迪软件的.scoredata.xml文件
  - 该文件从康迪软件中导出，目的是和技师现有分图流程对齐，
  - 若指定该参数，则把睡眠分期结果保存为另一个同名的.scoredata.xml文件中，目录与.txt相同。
  - 若指定该参数，则 `--input` 参数必须为单个.edf文件，不能批量处理目录。

## 硬件加速
- 自动检测GPU可用性，优先使用GPU加速
- 无GPU时自动切换CPU模式

## 输出文件
```text
输入文件：data/2025021001.edf
生成文件：
├── 2025021001.txt              # 睡眠分期结果
├── 2025021001.scoredata.xml    # 睡眠分期结果 (康迪软件格式)
└── 2025021001.log              # 运行日志
```

# 项目测试

已在5个计算平台完成验证，性能数据如下：

| 平台   | 操作系统      | CPU 型号                     | GPU 型号               | CUDA  | torch  | 推理速度  |
|--------|---------------|------------------------------|------------------------|-------|--------|-----------|
| 平台1  | Windows 11    | i7-13700F                    | RTX 4060 Ti (8G)       | 11.8  | 2.5.1  | 14秒/例   |
| 平台2  | Windows 11    | i7-13700F                    | 无                     | -     | 2.5.1  | 69秒/例   |
| 平台3  | Windows 11    | i7-13700F                    | 无                     | -     | 2.2.1  | 69秒/例   |
| 平台4  | Ubuntu 20.04  | Xeon Gold 6330 @ 2.00GHz     | NVIDIA A800 (80G)      | 11.7  | 2.0.0  | 22秒/例   |
| 平台5  | Ubuntu 20.04  | Xeon Gold 6330 @ 2.00GHz     | 无                     | -     | 2.0.0  | 47秒/例   |

# 环境配置

## 平台1参考安装流程（Anaconda）
```bash
conda create -n LPSGM python=3.10
conda activate LPSGM
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy tqdm scipy mne pandas
```

## 依赖说明
- GPU版本需额外配置CUDA环境
- CPU版本只需安装基础PyTorch