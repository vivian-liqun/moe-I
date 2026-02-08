import sys

import torch
print("CUDA available:", torch.cuda.is_available())
sys.path.append('..')

from models.surf_baseline import teacher_base,student_base,TeacherWithPrompt
print("CUDA available:", torch.cuda.is_available())
from src.surf_baseline_multi_dataloader import surf_baseline_multi_dataloader
from configuration.config_baseline_multi_KD import args
import itertools
import os
print("CUDA available:", torch.cuda.is_available())
import torch.nn as nn
from lib.model_develop import student_train_base
import torch.optim as optim

import numpy as np
import datetime
import random


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def deeppix_main(args):
    train_loader = surf_baseline_multi_dataloader(train=True, args=args)
    test_loader = surf_baseline_multi_dataloader(train=False, args=args)


    seed_torch(args.version)
    args.log_name = args.name + '.csv'
    pre_teacher = args.pre_teacher_root
    args.model_name = args.name
    teacher_model = teacher_base(args)
    teacher_model.load_state_dict(
        torch.load(
            os.path.join(pre_teacher)))
    teacher_model.eval()
    teacher_model=TeacherWithPrompt(teacher_model)
    teacher_model =teacher_model()
    student_model = student_base(args)

    if torch.cuda.is_available():
        student_model.cuda()
        teacher_model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")
    criterion = nn.CrossEntropyLoss()
    all_params = itertools.chain(student_model.parameters(), teacher_model.parameters())
    trainable_params = filter(lambda p: p.requires_grad, all_params)

    optimizer = optim.SGD(trainable_params, lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)


    args.retrain = False
    student_train_base(model=student_model,teacher_model=teacher_model,cost=criterion, optimizer=optimizer, train_loader=train_loader,
                              test_loader=test_loader,
                              args=args)


if __name__ == '__main__':

    deeppix_main(args=args)
