import argparse
import os
import torch
import torch.backends.cudnn as cudnn # 🚀 引入 cudnn 控制确定性
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu (1=True, 0=False)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2024, help="Randomization seed") # 🚀 将 seed 设为接收参数
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    # 为了兼容打印工具，补全缺失参数
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')


    # --- PA-LED 架构参数 ---
    parser.add_argument('--num_states', type=int, default=64, help='Number of learnable dictionary states')
    parser.add_argument('--lag_window', type=int, default=3, help='Causal lag window size (e.g., 3, 5, 7)') 
    parser.add_argument('--use_vgm', type=int, default=1, help='是否使用VGM空间去噪模块 (1=是, 0=否)')
    parser.add_argument('--use_multiscale', type=int, default=1, help='是否使用多尺度因果膨胀卷积 (1=是, 0=否)')
    parser.add_argument('--use_led', type=int, default=1, help='是否使用LED环境字典 (1=是, 0=否)')
    parser.add_argument('--clip_strategy', type=str, default='norm_clip_1.0', help='梯度裁剪策略')

    # PA_LED_v3参数
    parser.add_argument('--use_var_gate', type=int, default=1, help='use variable gated projection; 0 or 1')
    parser.add_argument('--fusion_pos', type=str, default='late', help='fusion position: early or late')
    parser.add_argument('--weight_decay', type=float, default=0, help='optimizer weight decay')

    #PA_LED_v4参数
    parser.add_argument('--backbone', type=str, default='transformer', help='backbone type: transformer or linear')
    parser.add_argument('--use_prior_alpha', type=int, default=1, help='Use logit prior initialization (1=True, 0=False)')

   
    # ==============================================================================
    parser.add_argument('--stride', type=int, default=8, help='patch 步长 (XGate_Prime / SMA_PatchRouter 共用)')
    parser.add_argument('--sma_kernel', type=int, default=25, help='SMA 平滑核大小 (XGate_Prime / SMA_PatchRouter 共用)')   
    parser.add_argument('--use_exo', type=int, default=1, help='启用外生变量路由 (1=Yes, 0=No)')


    # XLinear hyperparameters
    parser.add_argument('--c_ff', type=int, default=32)

    parser.add_argument('--embed_dropout', type=float, default=0.1)
    parser.add_argument('--t_dropout', type=float, default=0.1)
    parser.add_argument('--c_dropout', type=float, default=0.1)
    parser.add_argument('--head_dropout', type=float, default=0.1)

    parser.add_argument('--usenorm', type=int, default=1)

    parser.add_argument('--t_ff', type=int, default=0,
                        help='FeatureGate / 趋势MLP 瓶颈维度'
                             ' 推荐 256~1024, t_ff=512@dm96 参数量从107M降至~8M)')

    parser.add_argument('--num_groups', type=int, default=0,help=' Low-rank bottleneck rank / group size')
    parser.add_argument(
        '--router_prior_rank',
        type=int,
        default=4,
        help='Rank of the shared template in ExoPathGlobalLocal'
    )
    parser.add_argument(
        '--router_branch_mode',
        type=str,
        default='both',
        choices=['both', 'shared_only', 'local_only'],
        help='Routing branches enabled in ExoPathGlobalLocal'
    )
    parser.add_argument(
        '--ablation_mode',
        type=str,
        default='full',
        choices=[
            'full',
            'fixed_ema',
            'wo_decomp',
            'wo_ebg',
            'wo_router',
            'trend_only_route',
            'symmetric_route',
            'wo_origin',
        ],
        help='Core ablation mode for ExoPath'
    )

    # ================= 🚀 DBLoss 新增核心接口 =================
    parser.add_argument('--use_dbloss', type=int, default=1, help='1 to use DBLoss, 0 to use MSELoss')
    parser.add_argument('--dbloss_alpha', type=float, default=0.5, help='Smoothing factor for DBLoss EMA (default 0.5)')
    parser.add_argument('--dbloss_beta', type=float, default=0.5, help='Weight balancing factor for Season/Trend in DBLoss (default 0.5)')
    # ==========================================================

    args = parser.parse_args()
    
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fix_seed)
        torch.cuda.manual_seed_all(fix_seed)
        cudnn.deterministic = True  # 强制使用确定性算法 
        cudnn.benchmark = False     # 禁用自动寻找最优算子，防止引入动态随机性
    # =========================================================================

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)


    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args) 
            # ==================== 🚀 彻底解决权重相互覆盖的问题 ====================
            # 放弃之前那长长一串没用的字符串，直接用包含全部物理参数的 model_id 来命名！
            setting = '{}_{}_sl{}_pl{}_{}'.format(
                args.task_name,
                args.model_id, # 注意！这个 model_id 必须是 Optuna 生成的带有 T[x] 编号的！
                args.seq_len,
                args.pred_len,
                ii
            )
            # =======================================================================

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        # 同理，非训练模式下也使用 model_id 保证命名独立
        setting = '{}_{}_sl{}_pl{}_{}'.format(
                args.task_name,
                args.model_id, 
                args.seq_len,
                args.pred_len,
                ii
        )

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
