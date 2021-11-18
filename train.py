'''
train model from mnist and cifar-10 ,and finally sample from latent space to generate realistic-looking pictures
and compare with the CNF model and the data.
'''

import argparse
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

import lib.utils as utils
import lib.layers as layers
import lib.odenvp as odenvp
import lib.multiscale_parallel as multiscale_parallel

from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import set_cnf_options, count_parameters, count_total_time, count_nfe
from train_misc import standard_normal_logprob

from my_rewrite import Adam_t


def get_train_loader(train_set, epoch):
    if args.batch_size_schedule != "":
        epochs = [0] + list(map(int, args.batch_size_schedule.split("-")))
        n_passed = sum(np.array(epochs) <= epoch)
        current_batch_size = int(args.batch_size * n_passed)
    else:
        current_batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=current_batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    logger.info("===> Using batch size {}. Total {} iterations/epoch.".format(current_batch_size, len(train_loader)))
    return train_loader


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x


def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="./data", train=False, transform=trans(im_size), download=True)

    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(
            root="./data", train=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ]), download=True
        )
        test_set = dset.CIFAR10(root="./data", train=False, transform=trans(im_size), download=True)

    elif args.data == 'cifar100':
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR100(
            root="./data", train=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ]), download=True
        )
        test_set = dset.CIFAR100(root="./data", train=False, transform=trans(im_size), download=True)

    elif args.data == "svhn":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.SVHN(root="./data", split="train", transform=trans(im_size), download=True)
        test_set = dset.SVHN(root="./data", split="test", transform=trans(im_size), download=True)

    elif args.data == 'lsun_church':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.LSUN(
            'data', ['church_outdoor_train'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )
        test_set = dset.LSUN(
            'data', ['church_outdoor_val'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )

    elif args.data == 'celeba':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.CelebA(
            root='data', download=True, split='train', transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ])
        )
        test_set = dset.CelebA(
            root='data', download=True, split='valid', transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )

    elif args.data == 'fashion_mnist':
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.FashionMNIST(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.FashionMNIST(root="./data", train=False, transform=trans(im_size), download=True)

    elif args.data == 'imagenet32':
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize

    data_shape = (im_dim, im_size, im_size)
    if not args.conv:
        data_shape = (im_dim * im_size * im_size,)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True
    )
    return train_set, test_loader, data_shape


def compute_bits_per_dim(x, model, integration_times=None, update_t=False):
    zero = torch.zeros(x.shape[0], 1).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module
    if not update_t:
        z, delta_logp = model(x, zero, integration_times=integration_times, update_t=update_t)
    else:
        z, delta_logp, z_diff, logpz_diff = model(x, zero, integration_times=integration_times, update_t=update_t)
    #    z, delta_logp = model(x, zero, integration_times=integration_times, update_t=update_t)  # run model forward

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)
    if not update_t:
        return bits_per_dim
    else:
        return (bits_per_dim, z, z_diff, logpz_diff)


#    return (bits_per_dim, z_diff, logpz_diff)


def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    if args.multiscale:  # 多尺度结构的CNF
        model = odenvp.ODENVP(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            nonlinearity=args.nonlinearity,
            alpha=args.alpha,
            cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns,
                        "solver": args.solver},

        )

    elif args.parallel:  # 并行计算的CNF
        model = multiscale_parallel.MultiscaleParallelCNF(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            alpha=args.alpha,
            time_length=args.time_length,
        )
    else:
        if args.autoencode:  # 采用编码结构的CNF
            def build_cnf():
                autoencoder_diffeq = layers.AutoencoderDiffEqNet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.AutoencoderODEfunc(
                    autoencoder_diffeq=autoencoder_diffeq,
                    divergence_fn=args.divergence_fn,  # 只能选择"approximate"使用
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf
        else:  # 正常堆叠的CNF
            def build_cnf():
                diffeq = layers.ODEnet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.ODEfunc(
                    diffeq=diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    train_T=args.train_T,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf

        chain = [layers.LogitTransform(alpha=args.alpha)] if args.alpha > 0 else [layers.ZeroMeanTransform()]
        chain = chain + [build_cnf() for _ in range(args.num_blocks)]
        if args.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        model = layers.SequentialFlow(chain)
    return model


# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']

    parser = argparse.ArgumentParser("Continuous Normalizing Flow with or without optimized t1")
    parser.add_argument("--data",
                        choices=["mnist", "cifar10", "svhn", 'lsun_church', 'celeba', 'fashion_mnist', 'cifar100',
                                 'imagenet32'],
                        type=str, default="imagenet32")
    parser.add_argument("--dims", type=str, default="64,64,64")  # 不采用multi scale的取值
    parser.add_argument("--strides", type=str, default="1,1,1,1")
    parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')

    parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
    parser.add_argument(
        "--layer_type", type=str, default="concat",
        choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
    )
    parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
    parser.add_argument(
        "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "swish"]
    )
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

    parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
    parser.add_argument('--test_atol', type=float, default=None)
    parser.add_argument('--test_rtol', type=float, default=None)

    parser.add_argument("--imagesize", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument('--time_length', type=float, default=1.0)
    parser.add_argument('--time_optim', type=eval, default=False, choices=[True, False])  # 选择是否使用优化t1的CNF
    parser.add_argument('--time_sample', type=eval, default=True, choices=[True, False])  # 选择是否采样t1的CNF
    parser.add_argument('--train_T', type=eval, default=False)

    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument(
        "--batch_size_schedule", type=str, default="",
        help="Increases the batchsize at every given epoch, dash separated."
    )
    parser.add_argument("--test_batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_iters", type=float, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--spectral_norm_niter", type=int, default=10)

    parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
    parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--autoencode', type=eval, default=True, choices=[True, False])
    parser.add_argument('--rademacher', type=eval, default=True, choices=[True, False])
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--multiscale', type=eval, default=True, choices=[True, False])
    parser.add_argument('--parallel', type=eval, default=False, choices=[True, False])

    # sample t1
    parser.add_argument('--normal', type=eval, default=False, choices=[True, False])
    parser.add_argument('--uniform', type=eval, default=True, choices=[True, False])
    parser.add_argument('--sigma', type=float, default=0.2)
    parser.add_argument('--range_u', type=float, default=0.5)

    # Regularizations
    parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
    parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
    parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
    parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
    parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
    parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

    parser.add_argument("--time_penalty", type=float, default=0.1, help="Regularization on the end_time t1.")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1e10,
        help="Max norm of graidents (default is just stupidly high to avoid any clipping)"
    )
    parser.add_argument('--clip', type=eval, default=True, choices=[True, False])
    parser.add_argument('--clip_u', type=float, default=0.1)

    parser.add_argument("--begin_epoch", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save", type=str, default="experiments/cnf")
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=10)

    args = parser.parse_args()

    args.resume = os.path.join(args.save, "checkpt.pth")
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

    if args.layer_type == "blend" and args.time_optim is False:
        logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
        args.time_length = 1.0

    logger.info(args)
    utils.makedirs(args.save)

    # get deivce
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_set, test_loader, data_shape = get_dataset(args)

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_shape, regularization_fns)

    if torch.cuda.is_available():
        model = model.to(device)

    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    if args.spectral_norm:
        add_spectral_norm(model, logger)
    set_cnf_options(args, model)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr = args.lr

    # restore parameters
    '''
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)
    '''

    # load trained parameters
    if os.path.exists(args.resume):
        print("continue...")
        # checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        checkpt = torch.load(args.resume)

        # 这里注意，一定要指定map_location参数，否则会导致第一块gpu占用更多资源
        model.load_state_dict(checkpt["state_dict"])

        args = checkpt["args"]

        # 定义时间上界
        t1 = checkpt["t1"]
        time_list = checkpt["time_list"]
        begin_epoch = checkpt["epoch"]
        itr = checkpt["itr"]
        best_loss = checkpt["best_loss"]

        time_meter = checkpt["time_meter"]
        bpd_meter = checkpt["bpd_meter"]
        loss_meter = checkpt["loss_meter"]
        steps_meter = checkpt["steps_meter"]
        grad_meter = checkpt["grad_meter"]
        tt_meter = checkpt["tt_meter"]

        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])

            # manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)

    else:
        print("Restart...")
        logger.info(model)
        logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

        # 定义时间上界
        t1 = torch.eye(1).to(device)
        time_list = [t1.item()]

        time_meter = utils.RunningAverageMeter(0.97)
        bpd_meter = utils.RunningAverageMeter(0.97)
        loss_meter = utils.RunningAverageMeter(0.97)
        steps_meter = utils.RunningAverageMeter(0.97)
        grad_meter = utils.RunningAverageMeter(0.97)
        tt_meter = utils.RunningAverageMeter(0.97)

        best_loss = float("inf")
        itr = 0
        begin_epoch = args.begin_epoch

    #    if torch.cuda.is_available():
    #        model = torch.nn.DataParallel(model).cuda()

    # For visualization.
    fixed_z = cvt(torch.randn(100, *data_shape))  # 从隐空间中采样噪声数据

    if args.spectral_norm and not args.resume:
        spectral_norm_power_iteration(model, 500)

    # 定义时间上界
    #    t1 = torch.eye(1).to(device)
    #    time_list = [t1.item()]
    adam_t = Adam_t(lr=args.lr)
    m = nn.ReLU()
    '''
    autoencoder_diffeq = layers.AutoencoderDiffEqNet(
        hidden_dims=hidden_dims,
        input_shape=data_shape,
        strides=strides,
        conv=args.conv,
        layer_type=args.layer_type,
        nonlinearity=args.nonlinearity
    )
    autoencoder_odefunc = layers.AutoencoderODEfunc(
        autoencoder_diffeq=autoencoder_diffeq,
        divergence_fn=args.divergence_fn,  # 只能选择"approximate"使用
        residual=args.residual,
        rademacher=args.rademacher,
    )

    if torch.cuda.is_available():
        autoencoder_odefunc = autoencoder_odefunc.cuda()
    '''

    for epoch in range(begin_epoch, args.num_epochs + 1):
        '''
        if epoch > args.begin_epoch + args.num_epochs / 2:  # 学习率衰减
            lr = lr / 10
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        '''

        model.train()
        train_loader = get_train_loader(train_set, epoch)

        for _, (x, y) in enumerate(train_loader):
            start = time.time()
            update_lr(optimizer, itr)
            optimizer.zero_grad()

            if not args.conv:
                x = x.view(x.shape[0], -1)

            # cast data and move to device
            x = cvt(x)

            # sample t1 or not
            if args.time_sample:
                if args.normal:
                    T = np.random.normal(1, args.sigma, 1)  # update t1 in a gauss-mode
                    t1 = torch.from_numpy(T)
                elif args.uniform:
                    T = np.random.uniform(1 - args.range_u, 1, 1)  # update t1 in a uniform-mode
                    t1 = torch.from_numpy(T)

            # 更新积分区间
            integration_times = torch.tensor([0.0, t1]).to(x)
            # compute loss
            # 此处加入t1的更新
            if args.time_optim:
                bpd, z, z_diff, logpz_diff = compute_bits_per_dim(x, model, integration_times, update_t=True)
            elif args.time_sample:
                bpd = compute_bits_per_dim(x, model, integration_times)
            else:
                bpd = compute_bits_per_dim(x, model)

            if regularization_coeffs:  # 引入正则化项
                reg_states = get_regularization(model, regularization_coeffs)
                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                )
                loss = bpd + reg_loss
            else:
                loss = bpd

            total_time = count_total_time(model)
            #            loss = loss + total_time * args.time_penalty  # 时间惩罚系数

            # 引入积分上界t1的正则化项
            t_upper = t1.item()
            loss = loss + t_upper * args.time_penalty

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪，防止过拟合

            # calculate NFE
            nfe = count_nfe(model)

            optimizer.step()

            # 此处选择是否优化t1版本的模型
            if args.time_optim:
                zero = torch.zeros(x.shape[0], 1).to(x)
                #                z, logpz_cal, z_diff, logpz_diff = model(x, zero, integration_times=integration_times,
                #                                                         update_t=True)  # run model forward
                logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)

                dlog = torch.autograd.grad(logpz.mean(), z, retain_graph=False)[0]
                '''
                z_diff, _ = autoencoder_odefunc(t1, (z, torch.zeros(x.shape[0], 1).to(device)))
                _, logp_diff = autoencoder_odefunc(t1, (z, torch.zeros(x.shape[0], 1).to(device)))  # 修改了求导部分
                '''

                if not args.multiscale:
                    dlogp = dlog.mul(z_diff)
                    # dlogp_dim = dlogp.size()
                    # dlogp = dlogp.view(dlogp_dim[0], -1)
                    # dlogp_dim2 = dlogp.size()
                    # dlogp = dlogp.sum(1, keepdim=True) / dlogp_dim2[1]
                    dlogp = dlogp.sum(2)
                    dlogp = dlogp.sum(2)
                    loss_diff = -(dlogp - logpz_diff).mean()
                else:
                    dlogp = dlog.mul(z_diff)
                    dlogp = dlogp.sum(1, keepdim=True)
                    loss_dif = dlogp - logpz_diff
                    loss_diff = -(dlogp - logpz_diff).mean()
                pre_t = t1.item()
                t1 = adam_t(t1.item(), loss_diff)
                time_list.append(t1.item())
                tmp = t1.item()
                del z_diff, dlog, logpz_diff, t1
                t1 = torch.tensor(tmp).to(device)
                t1 = m(t1 - args.clip_u) + args.clip_u

                if args.clip:  # [0.1, 1.9]
                    #                    t1 = m(t1 - args.clip_u) + args.clip_u
                    t1 = 2 - args.clip_u - m(2 - args.clip_u - t1)

            if args.spectral_norm:
                spectral_norm_power_iteration(model, args.spectral_norm_niter)

            time_meter.update(time.time() - start)
            bpd_meter.update(bpd.item())
            loss_meter.update(loss.item())
            steps_meter.update(nfe)
            grad_meter.update(grad_norm)
            tt_meter.update(total_time)

            if itr % args.log_freq == 0:
                log_message = (
                    "Iter {:04d} | Time {:.4f}({:.4f})({:.4f}) | Bit/dim {:.4f}({:.4f}) | Loss {:.4f}({:.4f}) | "
                    "NFE {:.0f}({:.2f}) | Grad Norm {:.4f}({:.4f}) | Total Time {:.2f}({:.2f}) | t1 {:.4f}".format(
                        itr, time_meter.val, time_meter.avg, time_meter.all / 3600, bpd_meter.val, bpd_meter.avg,
                        loss_meter.val,
                        loss_meter.avg, steps_meter.val,
                        steps_meter.avg, grad_meter.val, grad_meter.avg, tt_meter.val, tt_meter.avg, t1.item()
                    )
                )
                print("Iter {:04d} | Time {:.4f}({:.4f})({:.4f}) | Bit/dim {:.4f}({:.4f}) | Loss {:.4f}({:.4f}) | "
                      "NFE {:.0f}({:.2f}) | Grad Norm {:.4f}({:.4f}) | Total Time {:.2f}({:.2f}) | t1 {:.4f}".format(
                    itr, time_meter.val, time_meter.avg, time_meter.all / 3600, bpd_meter.val, bpd_meter.avg,
                    loss_meter.val, loss_meter.avg,
                    steps_meter.val,
                    steps_meter.avg, grad_meter.val, grad_meter.avg, tt_meter.val, tt_meter.avg, t1.item()
                ))
                if regularization_coeffs:
                    log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)
                logger.info(log_message)

            itr += 1
        else:
            # compute test loss
            model.eval()
            if epoch % args.val_freq == 0:
                with torch.no_grad():
                    start = time.time()
                    logger.info("validating...")
                    print("validating...")
                    losses = []
                    for (x, y) in test_loader:
                        if not args.conv:
                            x = x.view(x.shape[0], -1)
                        x = cvt(x)
                        if args.time_optim:
                            integration_times_eval = torch.tensor([0.0, t1]).to(x)
                            loss = compute_bits_per_dim(x, model, integration_times_eval).cpu()
                        else:
                            loss = compute_bits_per_dim(x, model).cpu()
                        losses.append(loss)

                    loss = np.mean(losses)
                    logger.info(
                        "Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}, t1 {:.4f}".format(epoch, time.time() - start, loss,
                                                                                       t1.item()))
                    print(
                        "Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}, t1 {:.4f}".format(epoch, time.time() - start, loss,
                                                                                       t1.item()))
                    if loss < best_loss:
                        best_loss = loss
                        utils.makedirs(args.save)
                        '''
                        torch.save({
                            "args": args,
                            "state_dict": model.state_dict() if torch.cuda.is_available() else model.state_dict(),
                            "optim_state_dict": optimizer.state_dict(),
                            "t1_dict": t1,
                        }, os.path.join(args.save, "checkpt.pth"))
                        '''
                        torch.save({
                            "args": args,
                            "state_dict": model.state_dict() if torch.cuda.is_available() else model.state_dict(),
                            "optim_state_dict": optimizer.state_dict(),
                            "t1": t1,
                            "time_list": time_list,
                            "best_loss": best_loss,
                            "epoch": epoch,
                            "itr": itr,
                            "time_meter": time_meter,
                            "bpd_meter": bpd_meter,
                            "loss_meter": loss_meter,
                            "steps_meter": steps_meter,
                            "grad_meter": grad_meter,
                            "tt_meter": tt_meter
                        }, args.resume)

            if epoch % args.save_freq == 0:
                torch.save(time_list, args.save + '_time.pkl')

            # visualize samples and density
            with torch.no_grad():
                fig_filename = os.path.join(args.save, "figs", "{:04d}.jpg".format(epoch))
                utils.makedirs(os.path.dirname(fig_filename))
                if args.time_optim:
                    integration_times_sample = torch.tensor([0.0, t1]).to(fixed_z)
                    generated_samples = model(fixed_z, reverse=True, integration_times=integration_times_sample).view(
                        -1,
                        *data_shape)
                else:
                    generated_samples = model(fixed_z, reverse=True).view(-1, *data_shape)
                save_image(generated_samples, fig_filename, nrow=10)

            continue
        break
