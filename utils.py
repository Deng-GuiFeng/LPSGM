import numpy as np
import torch
import os
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score


def info(*tensors):
    for t in tensors:
        print(t.shape, t.dtype, t.min(),  t.max())


def str_to_set(s: str) -> set:
    if not s:
        return set()
    return {item.strip() for item in s.split(',') if item.strip()}


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed, force=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if force:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_metric_legacy(y_true, y_pred):

    if type(y_true) == list:
        y_true = np.array(y_true, dtype=np.int64)
        y_pred = np.array(y_pred, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    wake_f1 = f1_score(y_true==0, y_pred==0)
    n1_f1 = f1_score(y_true==1, y_pred==1)
    n2_f1 = f1_score(y_true==2, y_pred==2)
    n3_f1 = f1_score(y_true==3, y_pred==3)
    rem_f1 = f1_score(y_true==4, y_pred==4)
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa


def get_metric(y_true, y_pred, num_classes=5):
    """
    计算分类指标：支持 5/4/3 类别评价

    参数：
        y_true (array-like)：真实标签数组，取值 0–4
        y_pred (array-like)：预测标签数组，取值 0–4
        num_classes (int)：评估的类别数，可选 5、4 或 3

    返回：
        acc (float)：总体准确率
        f1_macro (float)：宏平均 F1 分数
        cm (ndarray)：混淆矩阵，shape=(num_classes, num_classes)
        per-class F1 (tuple)：各类别的 F1 分数，
            - 若 num_classes=5，则顺序为 (wake, N1, N2, N3, REM)
            - 若 num_classes=4，则顺序为 (wake, N1+N2, N3, REM)
            - 若 num_classes=3，则顺序为 (wake, NREM, REM)
        kappa (float)：Cohen’s Kappa 系数
    """
    # 转为 numpy 数组
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    # 根据不同类别数构造重编码映射
    if num_classes == 5:
        # 保持原始五类：0=W,1=N1,2=N2,3=N3,4=REM
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        labels = [0, 1, 2, 3, 4]
    elif num_classes == 4:
        # 合并 N1(1) 与 N2(2) → 新类别 1；W=0, N1+N2=1, N3=2, REM=3
        mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
        labels = [0, 1, 2, 3]
    elif num_classes == 3:
        # 合并 N1/N2/N3 → 新类别 1；W=0, NREM=1, REM=2
        mapping = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
        labels = [0, 1, 2]
    else:
        raise ValueError("num_classes 必须是 5、4 或 3")

    # 应用映射到标签
    y_true_m = np.vectorize(mapping.get)(y_true)
    y_pred_m = np.vectorize(mapping.get)(y_pred)

    # 基本指标
    acc = accuracy_score(y_true_m, y_pred_m)
    f1_macro = f1_score(y_true_m, y_pred_m, average='macro')
    cm = confusion_matrix(y_true_m, y_pred_m, labels=labels)

    # 每个类别的 F1
    per_class_f1 = tuple(
        f1_score(y_true_m == lbl, y_pred_m == lbl)
        for lbl in labels
    )

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true_m, y_pred_m)

    return acc, f1_macro, cm, *per_class_f1, kappa


def model_summary(model):
    print("Total Param:", end='')
    print(sum(p.numel() for p in model.parameters()))
    print("Trainable Param:", end='')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad == True))
    print("Untrainable Param:", end='')
    print(sum(p.numel() for p in model.parameters() if p.requires_grad == False))

    print('\n'*5, '*'*30, "MODEL ARCHITECTURE", '*'*30, '\n')
    print(model)

    print('\n'*5, '*'*30, "PARAM GRADIENT", '*'*30, '\n')
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")


def find_files_with_suffix(root_dir, suffix):
    matched_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(suffix):
                matched_files.append(os.path.join(dirpath, filename))
    return matched_files

