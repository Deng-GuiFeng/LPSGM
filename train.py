import os
from datetime import datetime
import sys
import shutil
import argparse
import warnings

from utils import *
from train.trainer import Trainer
from data.dataset import *
from tester_iterative import Tester

warnings.filterwarnings("ignore", category=UserWarning)

current_time = datetime.now().strftime("%b%d_%H-%M-%S")
root = os.path.split(sys.argv[0])[0]


def prepare_data(args):
    source_subjects, source_domains = get_datasets_subjects(args.source_domains)
    source_subjects = [(sub, dom) for sub, dom in zip(source_subjects, source_domains)]
    args.train_subjects, args.eval_subjects = train_test_split(
        source_subjects, 
        test_size=args.eval_size, 
        random_state=args.seed,
        shuffle=True,
        stratify=source_domains,
    )
    test_subjects, test_domains = get_datasets_subjects(args.target_domain)
    args.test_subjects = [(sub, dom) for sub, dom in zip(test_subjects, test_domains)]


def log_config(args):
    # back up config file
    os.makedirs(args.run_name, exist_ok=True)
    shutil.copy2(
        os.path.realpath(__file__), 
        os.path.join(args.run_name, "config_backup")
        )

    # log data
    sys.stdout = open(os.path.join(args.run_name, 'data.txt'), 'a')
    print(f"Train Subjects: {len(args.train_subjects)}, Eval Subjects: {len(args.eval_subjects)}, Test Subjects: {len(args.test_subjects)}")
    print("\n"*10, "Train Subjects:")
    for sub in args.train_subjects:
        print(sub)
    print("\n"*10, "Eval Subjects:")
    for sub in args.eval_subjects:
        print(sub)
    print("\n"*10, "Test Subjects:")
    for sub in args.test_subjects:
        print(sub)

    # log param
    sys.stdout = open(os.path.join(args.run_name, 'train.txt'), 'a')
    print(
        '--architecture', args.architecture, '\n',
        '--epoch_encoder_dropout', args.epoch_encoder_dropout, '\n',
        '--transformer_num_heads', args.transformer_num_heads, '\n',
        '--transformer_dropout', args.transformer_dropout, '\n',
        '--transformer_attn_dropout', args.transformer_attn_dropout, '\n',
        '--ch_num', args.ch_num, '\n',
        '--seq_len', args.seq_len, '\n',
        '--ch_emb_dim', args.ch_emb_dim, '\n',
        '--seq_emb_dim', args.seq_emb_dim, '\n',
        '--num_transformer_blocks', args.num_transformer_blocks, '\n',

        '--seed', args.seed, '\n',
        '--batch_size', args.batch_size, '\n',
        '--num_workers', args.num_workers, '\n',
        '--num_processes', args.num_processes, '\n',
        '--epochs', args.epochs, '\n',
        '--warmup_epochs', args.warmup_epochs, '\n',
        '--lr0', args.lr0, '\n',
        '--weight_decay', args.weight_decay, '\n',
        '--eta_min', args.eta_min, '\n',
        '--clip_value', args.clip_value, '\n',
        '--save_epoch', args.save_epoch, '\n',

        '--source_domains', args.source_domains, '\n',
        '--target_domain', args.target_domain, '\n',
        '--state_dict_file', args.state_dict_file, '\n',
        '--run_name', args.run_name, '\n',
        '--model_dir', args.model_dir, '\n',
        '--log_dir', args.log_dir, '\n',
        '--cache_root', args.cache_root, '\n',
        '--random_shift_len', args.random_shift_len, '\n',
    )


def run(args):
    # init
    set_seed(args.seed)
    prepare_data(args)
    log_config(args)

    # train
    print("\n\nHere Start the Training ...\n\n")
    model = Trainer(args).train()

    # test
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "eval_best.pth"))['model_state_dict'])
    print("\n\nHere Start the Testing ...\n\n")
    sys.stdout = open(os.path.join(args.run_name, 'test.txt'), 'a')
    Tester(args, model)


def main():
    
    parser = argparse.ArgumentParser(description='Large PSG Model')

    """ general """
    parser.add_argument('--seed', type=int, default=0)

    """ model param """
    parser.add_argument('--architecture', type=str, default='cat_cls')  # 'cat_cls' 'add_cls' 'cat_avg' 'none_cls'
    parser.add_argument('--epoch_encoder_dropout', type=float, default=0)
    parser.add_argument('--transformer_num_heads', type=int, default=8)
    parser.add_argument('--transformer_dropout', type=float, default=0)
    parser.add_argument('--transformer_attn_dropout', type=float, default=0)
    parser.add_argument('--ch_num', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--ch_emb_dim', type=int, default=32)
    parser.add_argument('--seq_emb_dim', type=int, default=64)
    parser.add_argument('--num_transformer_blocks', type=int, default=6)

    """  training  """
    parser.add_argument('--eval_size', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=256)
    parser.add_argument('--num_processes', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=15)
    parser.add_argument('--lr0', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--eta_min', type=float, default=1e-8)
    parser.add_argument('--clip_value', type=float, default=1)
    parser.add_argument('--clamp_value', type=float, default=10)
    parser.add_argument('--state_dict_file', type=str, default=None)
    parser.add_argument('--save_epoch', action='store_true')

    """ data """
    parser.add_argument(
        '--source_domains', type=str_to_set, 
        default={
            'APPLES', 'DCSM', 'DOD-H', 'DOD-O', 'HMC', 'ISRUC', 'P2018', 'SHHS-1', 'SHHS-2', 
            'STAGES-BOGN', 'STAGES-STNF', 'STAGES-MSTR', 'STAGES-GSDV', 'STAGES-GSBB', 'STAGES-GSLH', 
            'STAGES-GSSA', 'STAGES-GSSW', 'STAGES-MSMI', 'STAGES-MSNF', 'STAGES-MSQW', 'STAGES-MSTH', 'STAGES-STLK', 
            'ABC', 'NCHSDB', 'HOMEPAP', 'SVUH', 'CHAT', 'CCSHS', 'CFS', 'MROS'
        }
    )
    parser.add_argument('--target_domain', type=str_to_set, default={'HANG7', 'SYSU'})
    parser.add_argument('--cache_root', type=str, default=None)
    parser.add_argument('--random_shift_len', type=int, default=100)

    """ test """
    parser.add_argument('--save_pred', action='store_true')
    parser.add_argument('--save_dir', type=str, default=None)

    args = parser.parse_args()

    args.run_name = os.path.join(root, "run", current_time)
    args.model_dir = os.path.join(args.run_name, "model_dir")
    args.log_dir = os.path.join(args.run_name, "log_dir")
    args.cache_root = os.path.join(args.cache_root, current_time)

    if args.save_pred and args.save_dir is None:
        args.save_dir = os.path.join(args.run_name, "predicts")

    run(args)


if __name__ == "__main__":
    main()
