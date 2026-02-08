'''模型训练相关的函数'''

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from pyexpat import features
from tqdm import tqdm
import time
import csv

import time

import os
import torch.nn as nn

from lib.model_develop_utils import GradualWarmupScheduler

from models.surf_baseline import teacher_base_valid,student_base
from configuration.config_baseline_multi_KD import args

def calc_accuracy_multi_p(model ,loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')
    model.train(False)
    pre_teacher = args.pre_teacher_root
    teacher_model = teacher_base_valid(args)
    teacher_model.load_state_dict(
        torch.load(
            os.path.join(pre_teacher)))
    teacher_model.eval()
    teacher_model.cuda()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    mul_full = []
    std_full = []
    total_loss_rgb = 0
    total_loss_ir = 0
    total_loss_depth = 0
    total_loss_rgb_ir = 0
    total_loss_rgb_depth = 0
    total_loss_ir_depth = 0
    total_loss_rgb_ir_depth = 0
    total_loss_rgb_t = 0
    total_loss_ir_t = 0
    total_loss_depth_t = 0
    total_loss_rgb_ir_t = 0
    total_loss_rgb_depth_t = 0
    total_loss_ir_depth_t = 0
    total_loss_rgb_ir_depth_t = 0

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
            batch_sample['image_depth'], batch_sample[
            'binary_label']

        if torch.cuda.is_available():
            img_rgb = img_rgb.cuda()
            img_ir = img_ir.cuda()
            img_depth = img_depth.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batchs = model(img_rgb, img_ir, img_depth)

            if isinstance(outputs_batchs, tuple):
                outputs_batch = outputs_batchs[0]
            # print(outputs_batch)
            p=outputs_batchs[4]
            outputs_batchs_t = teacher_model(img_rgb, img_ir, img_depth,p)
            t_out=outputs_batchs_t[0]
            # 预先计算所有模态组合的条件
            target = target.long()

            cond_rgb_all = (p[:, 0] * (1 - p[:, 1]) * (1 - p[:, 2])) == 1
            cond_ir_all = ((1 - p[:, 0]) * p[:, 1] * (1 - p[:, 2])) == 1
            cond_depth_all = ((1 - p[:, 0]) * (1 - p[:, 1]) * p[:, 2]) == 1
            cond_rgb_ir_all = (p[:, 0] * p[:, 1] * (1 - p[:, 2])) == 1
            cond_rgb_depth_all = (p[:, 0] * (1 - p[:, 1]) * p[:, 2]) == 1
            cond_ir_depth_all = ((1 - p[:, 0]) * p[:, 1] * p[:, 2]) == 1
            cond_rgb_ir_depth_all = (p[:, 0] * p[:, 1] * p[:, 2]) == 1

            auxi_loss_m = auxi_cross_entropy(outputs_batch, target)
            auxi_loss_t = auxi_cross_entropy(t_out, target)

            # 为分母加上 0.1 防止除以 0
            epsilon = 0.1

            # 计算各种模态的损失
            x_rgb = torch.sum(auxi_loss_m * (p[:, 0] * (1 - p[:, 1]) * (1 - p[:, 2]))) / (
                    cond_rgb_all.sum() + epsilon)
            x_ir = torch.sum(auxi_loss_m * ((1 - p[:, 0]) * p[:, 1] * (1 - p[:, 2]))) / (
                    cond_ir_all.sum() + epsilon)
            x_depth = torch.sum(auxi_loss_m * ((1 - p[:, 0]) * (1 - p[:, 1]) * p[:, 2])) / (
                    cond_depth_all.sum() + epsilon)
            x_rgb_ir = torch.sum(auxi_loss_m * (p[:, 0] * p[:, 1] * (1 - p[:, 2]))) / (
                    cond_rgb_ir_all.sum() + epsilon)
            x_rgb_depth = torch.sum(auxi_loss_m * (p[:, 0] * (1 - p[:, 1]) * p[:, 2])) / (
                    cond_rgb_depth_all.sum() + epsilon)
            x_ir_depth = torch.sum(auxi_loss_m * ((1 - p[:, 0]) * p[:, 1] * p[:, 2])) / (
                    cond_ir_depth_all.sum() + epsilon)
            x_rgb_ir_depth = torch.sum(auxi_loss_m * (p[:, 0] * p[:, 1] * p[:, 2])) / (
                    cond_rgb_ir_depth_all.sum() + epsilon)
            x_rgb_t = torch.sum(auxi_loss_t * (p[:, 0] * (1 - p[:, 1]) * (1 - p[:, 2]))) / (
                    cond_rgb_all.sum() + epsilon)
            x_ir_t = torch.sum(auxi_loss_t * ((1 - p[:, 0]) * p[:, 1] * (1 - p[:, 2]))) / (
                    cond_ir_all.sum() + epsilon)
            x_depth_t = torch.sum(auxi_loss_t * ((1 - p[:, 0]) * (1 - p[:, 1]) * p[:, 2])) / (
                    cond_depth_all.sum() + epsilon)
            x_rgb_ir_t = torch.sum(auxi_loss_t * (p[:, 0] * p[:, 1] * (1 - p[:, 2]))) / (
                    cond_rgb_ir_all.sum() + epsilon)
            x_rgb_depth_t = torch.sum(auxi_loss_t * (p[:, 0] * (1 - p[:, 1]) * p[:, 2])) / (
                    cond_rgb_depth_all.sum() + epsilon)
            x_ir_depth_t = torch.sum(auxi_loss_t * ((1 - p[:, 0]) * p[:, 1] * p[:, 2])) / (
                    cond_ir_depth_all.sum() + epsilon)
            x_rgb_ir_depth_t = torch.sum(auxi_loss_t * (p[:, 0] * p[:, 1] * p[:, 2])) / (
                    cond_rgb_ir_depth_all.sum() + epsilon)
            total_loss_rgb += x_rgb.item()
            total_loss_ir += x_ir.item()
            total_loss_depth += x_depth.item()
            total_loss_rgb_ir += x_rgb_ir.item()
            total_loss_rgb_depth += x_rgb_depth.item()
            total_loss_ir_depth += x_ir_depth.item()
            total_loss_rgb_ir_depth += x_rgb_ir_depth.item()

            total_loss_rgb_t += x_rgb_t.item()
            total_loss_ir_t += x_ir_t.item()
            total_loss_depth_t += x_depth_t.item()
            total_loss_rgb_ir_t += x_rgb_ir_t.item()
            total_loss_rgb_depth_t += x_rgb_depth_t.item()
            total_loss_ir_depth_t += x_ir_depth_t.item()
            total_loss_rgb_ir_depth_t += x_rgb_ir_depth_t.item()




        outputs_full.append(outputs_batch)
        labels_full.append(target)


    print(
        " total_loss_rgb={:.5f}, total_loss_ir={:.5f}, total_loss_depth={:.5f}, total_loss_rgb_ir={:.5f}, total_loss_rgb_depth={:.5f}, total_loss_ir_depth={:.5f}, total_loss_rgb_ir_depth={:.5f}".format(
            total_loss_rgb, total_loss_ir, total_loss_depth,
            total_loss_rgb_ir, total_loss_rgb_depth, total_loss_ir_depth,
            total_loss_rgb_ir_depth
        )
    )
    print(
        " total_loss_rgb_t={:.5f}, total_loss_ir_t={:.5f}, total_loss_depth_t={:.5f}, total_loss_rgb_ir_t={:.5f}, total_loss_rgb_depth_t={:.5f}, total_loss_ir_depth_t={:.5f}, total_loss_rgb_ir_depth_t={:.5f}".format(
            total_loss_rgb_t, total_loss_ir_t, total_loss_depth_t,
            total_loss_rgb_ir_t, total_loss_rgb_depth_t, total_loss_ir_depth_t,
            total_loss_rgb_ir_depth_t
        )
    )
    # 1. 计算七个差值（无t减有t）
    d_rgb = total_loss_rgb - total_loss_rgb_t
    d_ir = total_loss_ir - total_loss_ir_t
    d_depth = total_loss_depth - total_loss_depth_t
    d_rgb_ir = total_loss_rgb_ir - total_loss_rgb_ir_t
    d_rgb_depth = total_loss_rgb_depth - total_loss_rgb_depth_t
    d_ir_depth = total_loss_ir_depth - total_loss_ir_depth_t
    d_rgb_ir_depth = total_loss_rgb_ir_depth - total_loss_rgb_ir_depth_t

    # 将差值放入列表
    differences = [d_rgb, d_ir, d_depth, d_rgb_ir, d_rgb_depth, d_ir_depth, d_rgb_ir_depth]
    differences = np.array(differences)
    differences = differences
    exp_x = np.exp(differences)

    # 计算指数
    probabilities=exp_x / np.sum(exp_x)  # 归一化[1,6](@ref)
    # total_loss = sum(differences)
    # # 归一化损失值得到概率
    # probabilities = [loss / total_loss for loss in differences]

    prob = np.array(probabilities)
    print(f"modality_drop 函数接收到的 prob: {prob}")

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)

    # if std_concat:
    #     std_full = torch.cat(std_full, dim=0)
    #     # mul_full = torch.cat(mul_full, dim=0)

    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    # print((labels_full - labels_predicted))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1

        try:

            APCER = living_wrong / (living_wrong + living_right)
            NPCER = spoofing_wrong / (spoofing_wrong + spoofing_right)

            ACER = (APCER + NPCER) / 2

            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(living_right, living_wrong, spoofing_right, spoofing_wrong), (
                outputs_full, labels_full, mul_full, std_full)
            return [accuracy, 0, 0, 0, 0, 0], (outputs_full, labels_full, mul_full, std_full)

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, APCER, NPCER, ACER], (outputs_full, labels_full, mul_full, std_full),prob,differences
    else:
        return [accuracy], (outputs_full, labels_full, mul_full, std_full)

def normalize_tensor(tensor):
    """
    对输入张量进行归一化
    """
    # 检查张量的维度
    num_dims = len(tensor.shape)
    if num_dims < 4:
        # 如果维度小于 4，在所有非批量维度上计算均值和标准差
        dims = tuple(range(1, num_dims))
    else:
        dims = (1, 2, 3)
    # 计算均值和标准差
    mean = tensor.mean(dim=dims, keepdim=True)
    std = tensor.std(dim=dims, keepdim=True)
    # 归一化
    normalized_tensor = (tensor - mean) / (std + 1e-8)  # 避免除零
    return normalized_tensor

def samplewise_cosine_similarity(mul, std):
    """
    逐样本计算余弦相似度
    """
    # 归一化输入张量
    mul = normalize_tensor(mul)
    std = normalize_tensor(std)
    # 检查张量的维度
    num_dims = len(mul.shape)
    if num_dims < 4:
        dims = tuple(range(1, num_dims))
    else:
        dims = (1, 2, 3)
    # 计算点积
    dot_product = (mul * std).sum(dim=dims)
    # 计算 L2 范数
    mul_norm = torch.norm(mul, p=2, dim=dims)
    std_norm = torch.norm(std, p=2, dim=dims)
    # 计算余弦相似度
    cosine_sim = dot_product / (mul_norm * std_norm + 1e-10)
    return cosine_sim

def compute_samplewise_cosine_loss_p(mul, std):
    """
    计算逐样本余弦相似度损失
    """
    cosine_sim = samplewise_cosine_similarity(mul, std)
    # 将余弦相似度映射到 [0, 1] 范围
    positive_similarity = (1-cosine_sim ) / 2
    loss = positive_similarity.sum()  # 将所有样本的相似度相加
    return loss

def student_train_base(model, teacher_model,cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    auxi_cross_entropy = nn.CrossEntropyLoss(reduction='none')

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 70, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 1 / 6),
                                                                              np.int32(args.train_epoch * 2 / 6),
                                                                              np.int32(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    else:
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0

    epoch = 0
    accuracy_best = 0
    hter_best = 1
    acer_best = 1
    loss_kl_sum = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")


    # Train
    differences=[0,0,0,0,0,0,0]
    dif=[0,0,0,0,0,0,0]
    while epoch < epoch_num:



        train_loss_cl = 0
        train_loss=0
        train_loss_m = 0
        train_loss_kl = 0
        train_loss_d = 0


        if epoch > 20:
            differences = differences
            # differences=[1, 1, 1, 1, 1, 1, 1]
            differences = [0, 0, 0, 0, 0, 0, 0]
            w = [1, 1, 1, 1, 1, 1, 1]
            w = [0, 0, 0, 0, 0, 0, 0]
        else:
            differences = [0, 0, 0, 0, 0, 0, 0]
            # differences = [1, 1, 1, 1, 1, 1, 1]
            w = [0, 0, 0, 0, 0, 0, 0]
            # w = [1, 1, 1, 1, 1, 1, 1]
        print(f"梯度使用的权重为: {differences}")
        print(f"提示使用的权重为: {w}")



        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
                batch_sample['image_depth'], batch_sample[
                'binary_label']
            # if epoch == 0:
            #     continue
            if torch.cuda.is_available():
                img_rgb = img_rgb.cuda()
                img_ir = img_ir.cuda()
                img_depth = img_depth.cuda()
                target = target.cuda()
            # grad_capture.clear()
            optimizer.zero_grad()
            # for p in model.parameters():
            #     p.grad = None

            model.args.epoch = epoch
            output,  layer3, layer4,features,p= model(img_rgb, img_ir, img_depth)

            toutput, tlayer3, tlayer4, tfeatures = teacher_model(img_rgb, img_ir, img_depth,p,w)


            if isinstance(output, tuple):
                output = output[0]



            if args.dataset != 'surf1':
                target = target.long()
                fusion_loss = auxi_cross_entropy(output, target)

                fusion_loss = fusion_loss.mean()


            else:

               print(type(p[:, 0]))

            e_loss_p1 = compute_samplewise_cosine_loss_p(output, toutput)
            e_loss_p2 = compute_samplewise_cosine_loss_p(layer3, tlayer3)
            e_loss_p3 = compute_samplewise_cosine_loss_p(layer4, tlayer4)
            e_loss_p4 = compute_samplewise_cosine_loss_p(features, tfeatures)

            d_loss = e_loss_p4


            if epoch > 5:

                loss_cls = fusion_loss*18


            else:
                loss_cls = fusion_loss

            loss=loss_cls+d_loss
            train_loss_cl += loss_cls.item()
            train_loss_d += d_loss.item()
            train_loss += loss.item()

            loss.backward()
            if isinstance(differences, torch.Tensor):
                differences = differences.clone().detach()
            else:
                # 如果differences原本是Python列表，转换为张量
                differences = torch.tensor(differences)
            prompt_weights_tensor = differences.cuda()

            if teacher_model.prompt_layer3.grad is not None:
                with torch.no_grad():  # 确保梯度修改操作本身不记录梯度
                    # 遍历每个提示类型 (0到6)
                    for i in range(teacher_model.prompt_layer3.grad.size(0)):  # 第一个维度是提示数量
                        if teacher_model.prompt_layer3.grad is not None:
                            # 将第 i 个提示的梯度乘以其对应的权重
                            teacher_model.prompt_layer3.grad[i] = teacher_model.prompt_layer3.grad[i] * prompt_weights_tensor[i]
            if teacher_model.prompt_layer4.grad is not None:
                with torch.no_grad():  # 确保梯度修改操作本身不记录梯度
                    # 遍历每个提示类型 (0到6)
                    for i in range(teacher_model.prompt_layer4.grad.size(0)):  # 第一个维度是提示数量
                        if teacher_model.prompt_layer4.grad is not None:
                            # 将第 i 个提示的梯度乘以其对应的权重
                            teacher_model.prompt_layer4.grad[i] = teacher_model.prompt_layer4.grad[i] * prompt_weights_tensor[i]
            if teacher_model.prompt_features.grad is not None:
                with torch.no_grad():  # 确保梯度修改操作本身不记录梯度
                    # 遍历每个提示类型 (0到6)
                    for i in range(teacher_model.prompt_features.grad.size(0)):  # 第一个维度是提示数量
                        if teacher_model.prompt_features.grad is not None:
                            # 将第 i 个提示的梯度乘以其对应的权重
                            teacher_model.prompt_features.grad[i] = teacher_model.prompt_features.grad[i] * prompt_weights_tensor[i]

            # grad_stats.update(grad_capture.gradients, p, batch_size=img_rgb.size(0))
            optimizer.step()

            # 打印梯度信息
            # if batch_num % 10 == 0:
        # grad_stats.print_epoch_summary(epoch)

        result_test, _ ,differences,nd= calc_accuracy_multi_p(model, loader=test_loader, hter=True, verbose=True)
        acer_test = result_test[-1]
        accuracy_test = result_test[0]

        if acer_test < acer_best and epoch > 0:
            acer_best = acer_test

            save_path = os.path.join(args.model_root, args.name + '_acer_best_' + '.pth')
            torch.save(model.state_dict(), save_path, )

        if accuracy_test > accuracy_best and epoch > 0:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name +'_acc_best'+ '.pth')
            torch.save(model.state_dict(), save_path, )
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        log_list.append( acer_test)
        log_list.append(acer_best)
        print(
            "Epoch {}, loss_cl={:.5f},loss_m={:.5f},loss={:.5f},loss_d={:.5f},accuracy_test={:.5f},accuracy_best={:.5f},acer_test={:.3f},  acer_best={:.3f}".format(
                epoch, train_loss_cl / len(train_loader),
                train_loss_m / len(train_loader), train_loss / len(train_loader),
                train_loss_d / len(train_loader),
                accuracy_test, accuracy_best,acer_test,acer_best))






        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1

    train_duration_sec = int(time.time() - start)
    # grad_capture.remove_hooks()
    print("training is end", train_duration_sec)
