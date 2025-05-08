from sklearn.model_selection import StratifiedKFold, train_test_split
import os


source_datasets_root = r"/nvme1/denggf/SleepStagingProcessedData/"
HANG7_datasets_root = r"/nvme1/denggf/SleepStagingProcessedData/HANG7/"
SYSU_datasets_root = r"/nvme1/denggf/SleepStagingProcessedData/SYSU/"
QU3_datasets_root = r"/nvme1/denggf/SleepStagingProcessedData/QU3_V2/"


Healthy_Control = [
    "20220220-T4-陈仙方", 
    "20210722-T4-110169", 
    "20220128-T2-韦又丹", 
    "20220504-T2-席淑雯", 
    "20220726-T4-虞浩", 
    "20210708-T4-109725", 
    "20220210-T2-潘中炉", 
    "20210711-T4-109786", 
    "20220110-T1-韦又丹", 
    "20220212-T4-黄万凤", 
    "20210713-T3-109874", 
    "20210703-T4-109536", 
    "20210710-T4-109774", 
    "20210802-T4-110475", 
    "20210702-T4-109517", 
    "20220124-T2-莫菲萍", 
    "20220407-T2-李君辉", 
    "20210712-T4-109811", 
    "20211214-T1-曾婧芝", 
    "20220213-T4-施俊珍", 
    "20210704-T4-109563", 
    "20210701-T4-109490", 
    "20210707-T4-109690", 
    "20210706-T4-109641", 
    "20220123-T4-李群潇", 
    "20220218-T4-孙英", 
    "20220719-T1-毛海明", 
    "20220720-T4-李征", 
    "20210802-T3-109490", 
    "20210714-T4-109914", 
    "20210705-T4-109605", 
    "20210721-T4-110145", 
    "20220729-T2-雷毅刚", 
]
Narcolepsy = [
    "20200526-T3-98939", 
    "20191112-T3-94718", 
    "20200421-T3-94747", 
    "20190708-T3-91095", 
    "20190924-T2-93329", 
    "20201006-T4-102664", 
    "20200802-T4-92178", 
    "20191029-T3-94303", 
    "20181016-T3-85216", 
    "20210315-T2-93725", 
    "20200517-T2-98616", 
    "20200114-T3-91901", 
    "20210608-T3-108755", 
    "20210714-T3-109892", 
    "20210201-T3-99816", 
    "20201110-T1-103755", 
    "20210301-T4-78178", 
    "20200506-T3-98277", 
    "20210221-T1-106313", 
    "20210908-T3-111580", 
    "20191217-T1-95673", 
    "20210527-T3-108353", 
    "20210426-T2-100116", 
    "20200722-T4-100622", 
    "20190723-T2-91621", 
    "20210120-T2-88492", 
    "20210627-T4-107490", 
    "20200812-T2-101209", 
    "20210818-T2-110856", 
    "20201029-T1-103408", 
    "20210128-T1-105955", 
    "20190916-T3-93088", 
    "20210719-T1-110040", 
    "20201123-T3-104163", 
    "20210517-T3-104291", 
    "20201126-T2-104251", 
    "20210201-T1-91302", 
    "20210715-T2-109946", 
    "20200729-T3-100643", 
    "20210218-T1-106230", 
    "20200922-T2-102346", 
    "20200716-T2-100459", 
    "20200720-T2-100542", 
    "20200707-T3-100162", 
    "20200907-T1-101895", 
    "20200817-T2-101354", 
    "20201125-T3-104171", 
    "20190917-T3-93135", 
    "20210805-T1-110578", 
    "20210602-T2-108520", 
    "20210614-T3-108923",
]
Depression = [
    "20210722-T1-110167", 
    "20200614-T3-99500", 
    "20210415-T1-107463", 
    "20200521-T2-98779", 
    "20201013-T1-102914", 
    "20210617-T1-109026", 
    "20201103-T3-103548", 
    "20200729-T2-100799", 
    "20210127-T3-105917", 
    "20190604-T3-90240", 
    "20201230-T3-105293", 
    "20210819-T3-110967", 
    "20210105-T2-105431", 
    "20210224-T4-106196", 
    "20200611-T3-99418", 
    "20200706-T2-98871", 
    "20200914-T3-102109", 
    "20200811-T2-101001", 
    "20200813-T4-101240", 
    "20200715-T3-100321", 
    "20201119-T1-104048", 
    "20210520-T1-108133", 
    "20200430-T3-97398", 
    "20210217-T4-106178", 
    "20200817-T1-101342", 
    "20210328-T3-107101", 
    "20210920-T3-111945", 
    "20210826-T3-111178", 
    "20210427-T1-105721", 
    "20201005-T4-102647", 
    "20210304-T2-92147", 
    "20210223-T2-106036", 
    "20201006-T3-102653", 
    "20190710-T3-91283", 
    "20200709-T4-100234", 
    "20200512-T1-98471", 
    "20210729-T3-110373", 
    "20191126-T2-95105", 
    "20201220-T2-103823", 
    "20190702-T2-91026", 
    "20200930-T2-102554", 
    "20210110-T3-105543", 
    "20191205-T3-94737", 
]


def get_HANG7_subjects():
    Subjects = os.listdir(HANG7_datasets_root)
    Subjects = [os.path.join(HANG7_datasets_root, sub) for sub in Subjects]
    SubType = [0 if os.path.basename(sub) in Healthy_Control else 1 if os.path.basename(sub) in Depression else 2 for sub in Subjects]
    return Subjects, SubType


def get_SYSU_datasets():
    Subjects = os.listdir(SYSU_datasets_root)
    Subjects = [os.path.join(SYSU_datasets_root, sub) for sub in Subjects]
    SubType = [0 if '健康' in sub else 1 for sub in Subjects]
    return Subjects, SubType


def get_QU3_subjects():
    Subjects = os.listdir(QU3_datasets_root)
    Subjects = [os.path.join(QU3_datasets_root, sub) for sub in Subjects]
    SubType = ['UNKNOWN' for sub in Subjects]
    return Subjects, SubType


def get_datasets_subjects(domains: dict):
    Subjects, Domains = [], []
    for domain in domains:
        domain_root = os.path.join(source_datasets_root, domain)
        subjects = os.listdir(domain_root)
        Subjects.extend( os.path.join(domain_root, sub_id) for sub_id in subjects )
        Domains.extend( [domain]*len(subjects) )
    return Subjects, Domains


def HANG7_StratifiedKFold(k_folds, n_fold, seed, val_size=0.2):

    subjects, subtypes = get_HANG7_subjects()

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for i, (train_val_idx, test_idx) in enumerate(skf.split(subjects, subtypes)):
        if i == n_fold:
            break

    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=seed,
                                          stratify=[subtypes[idx] for idx in train_val_idx])

    train_subjects = [(subjects[idx], subtypes[idx]) for idx in train_idx]
    val_subjects =   [(subjects[idx], subtypes[idx]) for idx in val_idx]
    test_subjects =  [(subjects[idx], subtypes[idx]) for idx in test_idx]

    return train_subjects, val_subjects, test_subjects


def QU3_StratifiedKFold(k_folds, n_fold, seed, val_size=0.2):

    subjects, subtypes = get_QU3_subjects()

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for i, (train_val_idx, test_idx) in enumerate(skf.split(subjects, subtypes)):
        if i == n_fold:
            break

    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=seed,
                                          stratify=[subtypes[idx] for idx in train_val_idx])

    train_subjects = [(subjects[idx], subtypes[idx]) for idx in train_idx]
    val_subjects =   [(subjects[idx], subtypes[idx]) for idx in val_idx]
    test_subjects =  [(subjects[idx], subtypes[idx]) for idx in test_idx]

    return train_subjects, val_subjects, test_subjects


def SYSU_StratifiedKFold(k_folds, n_fold, seed, val_size=0.2):

    subjects = [f"健康被试-{i}" for i in range(1, 20+1)] + [f"抑郁患者-{i}" for i in range(1, 24+1)]
    subtypes = ['健康'] * 20 + ['抑郁'] * 24

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for i, (train_val_idx, test_idx) in enumerate(skf.split(subjects, subtypes)):
        if i == n_fold:
            break
    
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=seed,
                                          stratify=[subtypes[idx] for idx in train_val_idx])

    train_subjects = [(subjects[idx], subtypes[idx]) for idx in train_idx]
    val_subjects = [(subjects[idx], subtypes[idx]) for idx in val_idx]
    test_subjects = [(subjects[idx], subtypes[idx]) for idx in test_idx]

    train_subjects_path = []
    for sub, sub_type in train_subjects:
        if "健康被试" in sub:
            sub_id = int(sub.split("-")[-1])
            train_subjects_path.extend(
                [(os.path.join(SYSU_datasets_root, f"健康被试-{i}"), sub_type) for i in range((sub_id-1)*4+1, sub_id*4+1)]
            )
        elif "抑郁患者" in sub:
            train_subjects_path.append(
                (os.path.join(SYSU_datasets_root, sub), sub_type)
            )

    val_subjects_path = []
    for sub, sub_type in val_subjects:
        if "健康被试" in sub:
            sub_id = int(sub.split("-")[-1])
            val_subjects_path.extend(
                [(os.path.join(SYSU_datasets_root, f"健康被试-{i}"), sub_type) for i in range((sub_id-1)*4+1, sub_id*4+1)]
            )
        elif "抑郁患者" in sub:
            val_subjects_path.append(
                (os.path.join(SYSU_datasets_root, sub), sub_type)
            )

    test_subjects_path = []
    for sub, sub_type in test_subjects:
        if "健康被试" in sub:
            sub_id = int(sub.split("-")[-1])
            test_subjects_path.extend(
                [(os.path.join(SYSU_datasets_root, f"健康被试-{i}"), sub_type) for i in range((sub_id-1)*4+1, sub_id*4+1)]
            )
        elif "抑郁患者" in sub:
            test_subjects_path.append(
                (os.path.join(SYSU_datasets_root, sub), sub_type)
            )

    return train_subjects_path, val_subjects_path, test_subjects_path


if __name__ == "__main__":
    train, val, test = QU3_StratifiedKFold(5, 0, 2022, 0.2)
    print(train)
    print(val)
    print(test)
    print(len(train), len(val), len(test))