import sys
sys.path.append('..')
import torch
print("CUDA available:", torch.cuda.is_available())
from models.surf_baseline import student_base
from datasets.surf_txt import SURF
from configuration.config_baseline_multi_KD import args
from lib.model_develop import calc_accuracy_multi_p
import torch
import os
from torch.utils.data import Subset
import numpy as np
from src.surf_baseline_multi_dataloader import surf_multi_transforms_test
def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = ".\data\CASIA-SURF-Challenge"
    txt_dir = root_dir + '/test_private_list.txt'
    surf_dataset = SURF(txt_dir=txt_dir,
                        root_dir=root_dir,
                        transform=surf_multi_transforms_test, miss_modal=args.miss_modal)


    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0)

    result, para ,x,y= calc_accuracy_multi_p(model=model, loader=test_loader, verbose=True, hter=True)
    print(result)

    return result



def performance_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

    result_model = []
    numbers = [1]

    for i in numbers:

        i = args.version
        result_list = []
        pretrain_dir = ".\\output\\models\\" + args.backbone   + "_acc_best.pth"
        args.gpu = 1
        args.modal = 'multi'
        args.miss_modal = 0
        args.backbone = "resnet18_se"
        args.inplace_new = 384
        print(pretrain_dir)

        for j in range(len(modality_combination)):
            args.p = modality_combination[j]
            print(args.p)
            model = student_base(args)
            test_para = torch.load(pretrain_dir)
            model.load_state_dict(torch.load(pretrain_dir))

            result = batch_test(model=model, args=args)

            result_list.append(result)

        result_arr = np.array(result_list)
        result_mean = np.mean(result_arr, axis=0)
        print(result_mean)
        result_model.append(result_mean)
    result_model = np.array(result_model)
    print(np.mean(result_model, axis=0))


if __name__ == '__main__':

    performance_test()
