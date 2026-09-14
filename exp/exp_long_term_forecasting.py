from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import os
import csv 
import time
import warnings
import math 
import numpy as np
import shutil  
import json
from utils.dtw_metric import dtw, accelerated_dtw
import torch.nn.functional as F
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')

class DBLoss(nn.Module):
    """
    Decomposition-based Loss Function (DBLoss) 
    """
    def __init__(self, alpha=0.5, beta=0.5):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def _ema_decompose(self, x):
        B, T, C = x.shape
        device = x.device
        
        powers = torch.flip(torch.arange(T, dtype=torch.float64, device=device), dims=(0,)).unsqueeze(1) 
        
        weights = torch.pow((1.0 - self.alpha), powers) # [T, 1]
        divisor = weights.unsqueeze(0) # [1, T, 1]
        
        # ===========================================================
        # 🚀 使用 torch.cat 替代原地切片赋值，保护计算图！
        w_0 = weights[0:1, :]
        w_rest = weights[1:, :] * self.alpha
        weights_mod = torch.cat([w_0, w_rest], dim=0).unsqueeze(0) # [1, T, 1]
        # ===========================================================
        
        x_f64 = x.to(torch.float64)
        out = torch.cumsum(x_f64 * weights_mod, dim=1)
        out = out / divisor
        
        Trend = out.to(torch.float32)
        Seasonality = x - Trend
        return Seasonality, Trend

    def forward(self, pred, true):
        # 1. 对预测值和真实值在预测窗口内分别进行因果 EMA 分解
        pred_season, pred_trend = self._ema_decompose(pred)
        true_season, true_trend = self._ema_decompose(true)
        
        # 2. 分别计算季节项损失(L2)和趋势项损失(L1)
        L_S = F.mse_loss(pred_season, true_season)  
        L_T = F.l1_loss(pred_trend, true_trend)     
        
        # 3. Scale Alignment 机制 (Stop-gradient 阻断强行耦合)
        alignment_ratio = L_S / (L_T + 1e-8)
        L_T_aligned = L_T * alignment_ratio.detach()
        
        # 4. 加权最终 Loss
        loss = self.beta * L_S + (1.0 - self.beta) * L_T_aligned
        return loss

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.actual_train_epochs = 0 

        # 设置 logs 根目录
        checkpoints_abs_path = os.path.abspath(self.args.checkpoints)
        root_dir = os.path.dirname(checkpoints_abs_path) 
        self.initial_lr = args.learning_rate  # 在初始化时就锁定初始学习率
        self.log_root_path = os.path.join(root_dir, 'logs')
        
        if not os.path.exists(self.log_root_path):
            os.makedirs(self.log_root_path)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), 
            lr=self.args.learning_rate, 
            weight_decay=self.args.weight_decay
        )
        return model_optim


    def _select_criterion(self):
        # ==================== 【关键修改：根据参数自适应切换 DBLoss】 ====================
        use_dbloss = getattr(self.args, 'use_dbloss', True)  # 默认启用 DBLoss
        if use_dbloss:
            dbloss_alpha = getattr(self.args, 'dbloss_alpha', 0.5)
            dbloss_beta = getattr(self.args, 'dbloss_beta', 0.5)
            criterion = DBLoss(alpha=dbloss_alpha, beta=dbloss_beta)
            print(f"[Criterion] Loaded DBLoss (alpha={dbloss_alpha}, beta={dbloss_beta})")
        else:
            criterion = nn.MSELoss()
            print("[Criterion] Using standard MSELoss")
        return criterion

    def vali(self, vali_data, vali_loader, criterion=None):
        """
        与 Time-Series-Library 的 test() 保持完全相同的评估方式：

        1. 收集所有 batch 的预测和真实值；
        2. 拼接为完整 NumPy 数组；
        3. 调用 utils.metrics.metric()；
        4. 返回 MSE、MAE。

        注意：metric() 的返回顺序是：mae, mse, rmse, mape, mspe,因此本函数最终返回 mse, mae。
        """
        preds = []
        trues = []

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]
                ).float()

                dec_inp = torch.cat(
                    [
                        batch_y[:, :self.args.label_len, :],
                        dec_inp,
                    ],
                    dim=1,
                ).float().to(self.device)

                # Forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                        )
                else:
                    outputs = self.model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                    )

                f_dim = -1 if self.args.features == "MS" else 0

                # 与 test() 一样，先截取预测长度
                outputs = outputs[:, -self.args.pred_len:, :]
                targets = batch_y[:, -self.args.pred_len:, :]

                # 转成 NumPy
                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

                # 与 test() 保持相同的 inverse 逻辑
                if vali_data.scale and self.args.inverse:
                    shape = targets.shape

                    if outputs.shape[-1] != targets.shape[-1]:
                        repeat_times = int(
                            targets.shape[-1] / outputs.shape[-1]
                        )
                        outputs = np.tile(
                            outputs,
                            [1, 1, repeat_times],
                        )

                    outputs = vali_data.inverse_transform(
                        outputs.reshape(
                            shape[0] * shape[1],
                            -1,
                        )
                    ).reshape(shape)

                    targets = vali_data.inverse_transform(
                        targets.reshape(
                            shape[0] * shape[1],
                            -1,
                        )
                    ).reshape(shape)

                # 与 test() 一样，最后再按 M/MS 截取通道
                outputs = outputs[:, :, f_dim:]
                targets = targets[:, :, f_dim:]

                preds.append(outputs)
                trues.append(targets)

        if was_training:
            self.model.train()

        if len(preds) == 0:
            raise RuntimeError(
                "验证或测试 DataLoader 中没有有效样本。"
            )

        # 与 test() 完全相同的拼接方式
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1,preds.shape[-2],preds.shape[-1],)
        trues = trues.reshape(-1,trues.shape[-2],trues.shape[-1],)

        # 直接使用 Time-Series-Library 的 metric()
        mae, mse, rmse, mape, mspe = metric(preds,trues,)

        return float(mse), float(mae)

    def adjust_learning_rate(self, optimizer, epoch, args):
        if args.lradj == 'type1':
            # 每轮衰减 50%
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            # 官方经典硬编码衰减矩阵
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        elif args.lradj == 'type3':
            # 初始平稳后衰减
            lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}

        elif args.lradj in ['cos', 'cosine']:
            # 经典的余弦退火 (无预热，直接退火)，非常适合稳健收敛
            lr_adjust = {epoch: args.learning_rate * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / args.train_epochs))}
        elif args.lradj == 'cos_warmup':
            # 专属反超策略：前2轮缓慢热身，然后余弦退火
            warmup_epochs = 2
            if epoch <= warmup_epochs:
                lr = args.learning_rate * (epoch / warmup_epochs)
            else:
                lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.train_epochs - warmup_epochs)))
            lr_adjust = {epoch: lr}
        else:
            # 默认兜底
            lr_adjust = {epoch: args.learning_rate}

        # 执行更新
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('>>> Updating learning rate to {}'.format(lr))

    def train(self, setting, trial=None):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)


        train_start_time = time.time()
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.actual_train_epochs = epoch + 1
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                model_instance = self.model.module if hasattr(self.model, 'module') else self.model

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_cropped = batch_y[:, -self.args.pred_len:, f_dim:]
                        
                        #  🚀 这里的 criterion 将由 DBLoss 接管！
                        loss = criterion(outputs, batch_y_cropped)

                        train_loss.append(loss.item())

                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    # 梯度裁剪
                    if self.args.clip_strategy != 'None':
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_cropped = batch_y[:, -self.args.pred_len:, f_dim:]
                    # 🚀 这里的 criterion 将由 DBLoss 接管！
                    loss = criterion(outputs, batch_y_cropped)  
                    train_loss.append(loss.item())

                    loss.backward()
                    if self.args.clip_strategy != 'None':
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()


                    # ==================== 【核心：每 100 iter 的详细显示】 ====================
                if (i + 1) % 100 == 0:
                    # 1. 计算基础速度信息
                    speed = (time.time() - time_now) / 100
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    

            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            

            real_vali_mse, real_vali_mae = self.vali(vali_data, vali_loader, criterion)
            # 仅用于终端观察，不用于 EarlyStopping 和参数更新
            real_test_mse, real_test_mae = self.vali(test_data, test_loader, criterion)


            # 2. 【终极终端输出】：在一行里同时打印当前优化Loss以及最真实的 MSE, MAE 战报
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.8f} | Real Vali MSE: {3:.10f} MAE: {4:.10f} | Real Test MSE: {5:.4f} MAE: {6:.4f}".format(
                epoch + 1, train_steps, train_loss, real_vali_mse, real_vali_mae, real_test_mse, real_test_mae))

            # =========================================================================
            #  早停依然基于真实的物理 MSE 做出无偏判断
            # =========================================================================
            early_stopping(real_vali_mse, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            self.adjust_learning_rate(model_optim, epoch + 1, self.args)



        self.total_train_time = time.time() - train_start_time
        self.total_train_iters = self.actual_train_epochs * train_steps
        best_model_path = os.path.join(path, "checkpoint.pth")

        self.model.load_state_dict(torch.load(best_model_path,map_location=self.device,))

        final_val_mse, final_val_mae = self.vali(vali_data,vali_loader,)

        final_test_mse, final_test_mae = self.vali(test_data,test_loader,)

        print(
            "FINAL_BEST | "
            f"Val MSE: {final_val_mse:.10f} "
            f"Val MAE: {final_val_mae:.10f} | "
            f"Test MSE: {final_test_mse:.10f} "
            f"Test MAE: {final_test_mae:.10f}",
            flush=True,
        )

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            # 注意：这里路径可能需要根据你的实际保存路径调整，原代码是 './checkpoints/' + setting
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path))
            else:
                # 兼容旧路径写法，如果新路径不存在
                old_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
                if os.path.exists(old_path):
                    self.model.load_state_dict(torch.load(old_path))
                else:
                    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} or {old_path}")

        preds = []
        trues = []

        # 重置显存统计并记录开始时间
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        inference_start_time = time.time()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len:, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                # 注释掉可视化图表生成,每次测试都会生成大量的 .pdf 图片，实验阶段不需要看图。
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 计算推理总时间和显存峰值
        total_inference_time = time.time() - inference_start_time
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0  # 单位 MB
        peak_mem_gb = round(peak_mem_mb / 1024, 2)  # 转换为GB，保留2位小数

        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6  # 单位：百万 (M)
        
        #计算单迭代耗时
        # 确保这些属性在 train 中被正确初始化，否则需要默认值
        train_time_total = getattr(self, 'total_train_time', 0)
        total_train_iters = getattr(self, 'total_train_iters', 1) # 避免除以0
        
        train_time_per_iter_s = round(train_time_total / total_train_iters, 2) if total_train_iters > 0 else 0
        
        total_test_iters = len(test_loader)
        infer_time_per_iter_s = round(total_inference_time / total_test_iters, 2) if total_test_iters > 0 else 0
        
        #获取GPU型号
        gpu_type = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
   

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)


        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        

        # ==================== 🚀 全量参数与指标归档系统 ====================
        # 1. 获取基础参数字典 (并进行 JSON 序列化安全处理)
        args_dict = {}
        for key, value in vars(self.args).items():
            # 遇到 PyTorch 的 device 对象，强制转成字符串 (例如: 'cuda:0')
            if isinstance(value, torch.device):
                args_dict[key] = str(value)
            # 如果有其他不能直接转 JSON 的奇怪对象，也可以加在这里
            else:
                args_dict[key] = value
        
        # 2. 补充计算出的性能指标
        metrics_dict = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe)
        }
        
        # 3. 补充效率指标
        efficiency_dict = {
            "params_M": float(sum(p.numel() for p in self.model.parameters()) / 1e6),
            "train_time_total": float(getattr(self, 'total_train_time', 0)),
            "test_time_total": float(time.time() - inference_start_time),
            "gpu_mem_max_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0)
        }
        
        # 合并所有数据
        final_record = {
            "args": args_dict,
            "metrics": metrics_dict,
            "efficiency": efficiency_dict,
            "setting_name": setting,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        # 4. 将全量数据保存到实验专属文件夹的 JSON 中
        # 注意：这里我们存到 logs 文件夹，因为下面的逻辑会把 checkpoints 删掉！
        # log_sub_path = os.path.join(self.log_root_path, setting)
        # if not os.path.exists(log_sub_path):
        #     os.makedirs(log_sub_path)
            
        # json_file_path = os.path.join(log_sub_path, 'config_and_results.json')
        # with open(json_file_path, 'w', encoding='utf-8') as f:
        #     json.dump(final_record, f, indent=4, ensure_ascii=False)
            
        # print(f'>>> Full experiment record has been saved to {json_file_path}')
        # =================================================================
        # -----------------------------------------------------------------

        # 提取具体的数据集名称 (例如从 'electricity.csv' 提取 'electricity'，或者使用 model_id 的第一部分)
        dataset_specific_name = self.args.data_path.split('.')[0] # 得到 electricity 或 ETTh1
        safe_dataset = dataset_specific_name.replace('/', '_').replace('\\', '_')
        feature_mode = self.args.features
        # ==================== 【修改：定制 CSV 保存路径】 ====================
        # 判断是否为 EPF 数据集 (通过 args.data 或 data_path 灵活匹配)
        # 判定是否为 EPF 数据
        epf_keywords = ['np', 'be', 'de', 'fr', 'pjm']
        is_epf = False
        
        # 综合检查 data 和 data_path
        check_str = (str(self.args.data) + str(self.args.data_path)).lower()
        if any(k in check_str for k in epf_keywords):
            is_epf = True


        # 设置目标文件夹
        if is_epf:
            csv_target_dir = "/home/mlc8/fy/MySeriesModel/CSV文件/EPF"
        else:
            csv_target_dir = "/home/mlc8/fy/MySeriesModel/CSV文件"

        # 如果文件夹不存在，自动创建
        if not os.path.exists(csv_target_dir):
            os.makedirs(csv_target_dir)

        # 拼接最终的文件绝对路径
        res_file = os.path.join(csv_target_dir, f"long_term_forecast_{safe_dataset}_{feature_mode}.csv")
        # ===================================================================

        # 准备要记录的数据字典

        res_data = {
            # 1. 核心实验配置
            "model_id": self.args.model_id,
            "model": self.args.model,
            "data": dataset_specific_name,
            "features": feature_mode,
            "seq_len": self.args.seq_len,
            "pred_len": self.args.pred_len,        

            "dbloss_alpha": getattr(self.args, 'dbloss_alpha', 0.5),
            "dbloss_beta": getattr(self.args, 'dbloss_beta', 0.5),
            # ----------------------------------------------------

            # 2. 模型性能指标
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "mspe": round(mspe, 4),
            
            # 3. 效率与复杂度
            "params_M": round(total_params, 2),
            "train_time_total": round(train_time_total, 2),
            "train_time_per_iter_s": train_time_per_iter_s,
            "test_time_total": round(total_inference_time, 2),
            "infer_time_per_iter_s": infer_time_per_iter_s,
            "gpu_mem_max_mb": round(peak_mem_mb, 2),
            "gpu_mem_max_gb": peak_mem_gb,
            
            # 4. 其他信息
            "train_epochs": self.args.train_epochs,
            "actual_train_epochs": getattr(self, 'actual_train_epochs', self.args.train_epochs),
            "des": self.args.des,
            "setting": setting, 
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        }

        file_exists = os.path.isfile(res_file)
        with open(res_file, 'a', newline='', encoding='utf-8') as f:
            # 获取当前的字段名
            fieldnames = list(res_data.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writeheader()
            else:
                # 【重要防御逻辑】：防止因新增字段导致旧 CSV 文件报错
                # 如果文件已存在，我们检查现有的表头是否包含我们新增的字段
                with open(res_file, 'r', encoding='utf-8') as f_read:
                    reader = csv.reader(f_read)
                    existing_headers = next(reader, [])
                
                # 如果旧文件的表头里没有 patch_len，说明这是加字段后第一次追加
                # dictwriter 可能会因为 fieldnames 不匹配报错，所以我们做个容错
                if "patch_len" not in existing_headers:
                    # 重新以写模式打开（不覆盖，稍后用 pandas 或手动处理比较麻烦，最简单的是在旧文件里直接追加缺少的字段对应的空值，但DictWriter严格限制。
                    # 这里我们采取的策略是：只写存在于旧表头里的字段，或者如果你不在乎旧文件，直接删掉旧文件重新生成即可。
                    # 最优雅的办法是忽略警告强制写入：
                    writer = csv.DictWriter(f, fieldnames=existing_headers, extrasaction='ignore')
            
            writer.writerow(res_data)
        
        print(f'>>> Experiment results have been appended to {res_file}')
        # ---------------- 修改结束 ----------------
    
        # 保留原有的 txt 记录功能 (可选，如果需要也可以改成动态文件名)
        # f = open("result_exogenous_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw_val))
        # f.write('\n\n')
        # f.close()

        # ==================== 【自动清理空间】 ====================
        # 1. 删除 checkpoints 文件夹
        cp_path = os.path.join(self.args.checkpoints, setting)
        if os.path.exists(cp_path):
            shutil.rmtree(cp_path)
            print(f"[Cleanup] Deleted checkpoint folder: {cp_path}")
        
        # 2. 删除对应的 TensorBoard 日志文件夹
        # log_sub_path = os.path.join(self.log_root_path, setting)
        # if os.path.exists(log_sub_path):
        #     shutil.rmtree(log_sub_path)
        #     print(f"[Cleanup] Deleted log folder: {log_sub_path}")
        # ===================================================================

        # 注释掉预测张量保存（最省空间）,这是占用磁盘空间 90% 以上的元凶。
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return self.model
