import math

import torchvision.transforms as transforms
from models import ResNet18
from torch.utils.data import TensorDataset
from tqdm import tqdm

from .poi_util import *
from .util import *

device = 'cuda'


def get_dataset(args):
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(), ])
    trainset = h5_dataset(args.dataset_root, True, None)
    args.num_classes = len(np.unique(np.array(trainset.targets)))
    train_poi_set, poi_idx = poi_dataset(trainset, poi_methond=args.corruption_type, transform=train_transform,
                                         poi_rates=args.corruption_ratio, random_seed=args.random_seed,
                                         tar_lab=args.tar_lab)

    return train_poi_set, poi_idx


def build_training(args):
    model = ResNet18(num_classes=args.num_classes).cuda()
    optimizer_a = torch.optim.AdamW(model.parameters(), lr=2.5e-5 * args.batch_size / 32, betas=(0.9, 0.95),
                                    weight_decay=0.05)

    vnet = nnVent(1, 100, 150, 1).cuda()


    optimizer_c = torch.optim.SGD(vnet.parameters(), args.v_lr)
    return model, optimizer_a, vnet, optimizer_c

import dimod
import neal
def createBQM(net, input, layersList, beta=4, target=None):
    # input 求平均
    input = input.mean()
    with torch.no_grad():
        bias_input = input* net.weights_0
        bias_lim = 1 # 限制bias的范围
        # h = {idx_loc: (bias + bias_input[idx_loc]).clip(-bias_lim, bias_lim).item() for idx_loc, bias in
        #      enumerate(net.bias_0)}
        h = {}
        # 遍历隐藏层神经元的索引
        for idx_loc in range(len(net.bias_0)):
            h[idx_loc] = (net.bias_0[idx_loc] + bias_input[0, idx_loc]).clip(-bias_lim, bias_lim).item()

        if target is not None:
            bias_nudge = -( beta * target ).mean()
            h.update({idx_loc + layersList[1]: (bias + bias_nudge).clip(-bias_lim, bias_lim).item() for
                      idx_loc, bias in enumerate(net.bias_1)})
        else:
            h.update({idx_loc + layersList[1]: bias.clip(-bias_lim, bias_lim).item() for idx_loc, bias in
                      enumerate(net.bias_1)})

        J = {}
        for k in range(layersList[1]):
            for j in range(layersList[2]):
                J.update({(k, j + layersList[1]): net.weights_1[k][j].clip(-1, 1)})

        # 默认为ising模型
        model = dimod.BinaryQuadraticModel.from_ising(h, J, 0)

        return model

def trainonceQA(net, data, cost):
    from dwave.system import DWaveSampler, EmbeddingComposite
    # data cost 到 cpu上
    data = data.cpu().detach(); cost = cost.cpu().detach()
    loss = data * cost
    store_seq = None
    store_s = None
    qpu_sampler = neal.SimulatedAnnealingSampler()
    # qpu_sampler = EmbeddingComposite(DWaveSampler())

    # 自由相：当前网络参数下前向传播
    model_free = createBQM(net, data, net.layersList, beta=net.beta)
    qpu_seq = qpu_sampler.sample(model_free, num_reads=10, num_sweeps=100, auto_scale=4)
    s_free = np.array([list(sample.values()) for sample in qpu_seq.samples()])

    # 推波相：引入损失的梯度进行微扰
    model_nudge = createBQM(net, data, net.layersList, beta=net.beta, target= loss)
    qpu_s = qpu_sampler.sample(model_nudge, num_reads=10, num_sweeps=100, reverse=True)

    # 提取推波相的采样结果
    store_s = np.array([list(sample.values()) for sample in qpu_s.samples()])


    # 使用EP算法更新VNet的参数
    if store_seq is None:
        store_seq = qpu_seq.record["sample"][0].reshape(1, qpu_seq.record["sample"][0].shape[0])  # qpu_seq
        store_s = qpu_s.record["sample"][0].reshape(1, qpu_s.record["sample"][0].shape[0])  # qpu_s
    else:
        store_seq = np.concatenate(
            (store_seq, qpu_seq.record["sample"][0].reshape(1, qpu_seq.record["sample"][0].shape[0])), 0)
        store_s = np.concatenate(
            (store_s, qpu_s.record["sample"][0].reshape(1, qpu_s.record["sample"][0].shape[0])), 0)
    seq = [store_seq[:, :net.layersList[1]], store_seq[:, net.layersList[1]:]] # 自由相的输出
    s = [store_s[:, :net.layersList[1]], store_s[:, net.layersList[1]:]] # 推波相的输出

    net.updateParams(data, s, seq, batch_size = len(data) )

def QA_train_sifter(args, dataset):
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                   pin_memory=True, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    mnet_list = []
    vnet_list = []
    for i in range(args.num_sifter):
        print("-----------Training sifter number: " + str(i) + "-----------")
        model, optimizer_a, vnet = build_training_with_imqa(args)
        grad_models, grad_optimizers = build_grad_models(args, model)
        model, optimizer_a = warmup(model, optimizer_a, train_dataloader, args)
        raw_meta_model = ResNet18(num_classes=args.num_classes).cuda()
        for i in range(args.res_epochs):
            train_iter = tqdm(enumerate(train_dataloader), total=int(len(dataset) / args.batch_size) + 1)
            for iteration, (input_train, target_train) in train_iter:
                input_var, target_var = input_train.cuda(), target_train.cuda()

                # virtual training 虚拟训练
                meta_model = copy.deepcopy(raw_meta_model)  # 复制
                meta_model.load_state_dict(model.state_dict())  # 复制
                y_f_hat = meta_model(input_var)  # 使用元模型进行前向传播，得到预测值
                cost = criterion(y_f_hat, target_var)
                cost_v = torch.reshape(cost, (len(cost), 1))  # 重塑形状

                v_lambda = vnet.forward(cost_v.data)  # 权重模型vnet进行前向传播，预测权重
                v_lambda = v_lambda.view(-1)  # 重塑形状
                v_lambda = norm_weight(v_lambda)  # 归一化
                v_lambda = v_lambda.to(cost.device)  # 将 v_lambda 移动到与 cost 相同的设备上

                l_f_meta = torch.sum(v_lambda * cost)

                # virtual backward & update
                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, (meta_model.parameters()), create_graph=True, allow_unused=True)
                # 执行反向传播，得到梯度

                # compute gradient gates and update the model
                new_grads, _ = compute_gated_grad(grads, grad_models, args.top_k, args.num_act)  # 梯度门控
                pseudo_optimizer = MetaSGD(meta_model, meta_model.parameters(), lr=args.meta_lr)  # 更新元模型
                pseudo_optimizer.load_state_dict(optimizer_a.state_dict())
                pseudo_optimizer.meta_step(new_grads)

                res_y_f_hat = meta_model(input_var)  # 使用元模型进行前向传播，得到预测值
                res_cost = criterion(res_y_f_hat, target_var)
                res_cost_v = torch.reshape(res_cost, (len(res_cost), 1))
                res_v_bf_lambda = vnet.forward(res_cost_v.data)
                res_v_bf_lambda = res_v_bf_lambda.view(-1)
                res_v_lambda = 1 - res_v_bf_lambda
                res_v_lambda = norm_weight(res_v_lambda)
                res_v_lambda = res_v_lambda.to(res_cost.device) # 将 res_v_lambda 移动到与 res_cost 相同的设备上
                valid_loss = -torch.sum((res_v_lambda) * res_cost)

            #   optimizer_c.zero_grad() 原训练过程代码，现在用EP平衡传播训练表述
                for go in grad_optimizers:
                    go.zero_grad()
                valid_loss.backward()
                # valid_loss 对vent进行平衡传播
                trainonceQA(vnet, -res_v_lambda, res_cost)

                for go in grad_optimizers:
                    go.step()
                del grads, new_grads  # 释放内存

                # actuall update，实际更新
                y_f = model(input_var)  # 主模型进行前向传播
                cost_w = (criterion(y_f, target_var) )
                cost_v = torch.reshape(cost_w, (len(cost_w), 1))

                with torch.no_grad():
                    w_new = vnet.forward(cost_v)  # 使用权重模型进行前向传播，得到预测权重

                w_new = w_new.view(-1)  # 重塑形状
                w_new = norm_weight(w_new)  # 归一化
                w_new = w_new.to(cost_w.device)  # 将 w_new 移动到与 cost 相同的设备上

                l_f = torch.sum(w_new * cost_w)  # 加权损失

                optimizer_a.zero_grad()
                l_f.backward()
                optimizer_a.step()
                # l_f 对vent进行平衡传播
                trainonceQA(vnet, w_new, cost_w)

        vnet_list.append(copy.deepcopy(vnet))
        mnet_list.append(copy.deepcopy(model))
    return vnet_list, mnet_list

import numpy as np
import kaiwu as kw


def create_kaiwu_qubo_model(net, input_data, layers_list, beta=4, target=None):
    """
    使用 Kaiwu SDK 构建 QUBO 模型

    参数：
    - net: 神经网络对象，包含 weights_0, weights_1, bias_0, bias_1 等属性
    - input_data: 输入数据
    - layers_list: 网络的层数列表，例如 [输入层大小, 隐藏层大小, 输出层大小]
    - beta: 调整参数
    - target: 目标输出（可选）

    返回：
    - qubo_model: Kaiwu 的 QUBO 模型
    """
    # 计算输入数据的平均值
    input_mean = input_data.mean()

    # 获取网络层大小
    num_input = layers_list[0]
    num_hidden = layers_list[1]
    num_output = layers_list[2]
    num_variables = num_hidden + num_output  # 总变量数量

    # 创建二元变量
    variables = [kw.qubo.Binary(f'x_{i}') for i in range(num_variables)]

    # 设置偏置限制
    bias_lim = 1  # 限制偏置的范围

    # 初始化 QUBO 表达式
    qubo_expr = 0

    # **隐藏层偏置计算**
    bias_input = input_mean * net.weights_0
    for idx_loc in range(num_hidden):
        bias = (net.bias_0[idx_loc] + bias_input[0, idx_loc])
        bias_clipped = np.clip(bias, -bias_lim, bias_lim).item()
        qubo_expr += bias_clipped * variables[idx_loc]

    # **输出层偏置计算**
    if target is not None:
        bias_nudge = -(beta * target).mean()
        for idx_loc, bias in enumerate(net.bias_1):
            adjusted_bias = np.clip(bias + bias_nudge, -bias_lim, bias_lim).item()
            qubo_expr += adjusted_bias * variables[idx_loc + num_hidden]
    else:
        for idx_loc, bias in enumerate(net.bias_1):
            adjusted_bias = np.clip(bias, -bias_lim, bias_lim).item()
            qubo_expr += adjusted_bias * variables[idx_loc + num_hidden]

    # **交互项（二次项）**
    for k in range(num_hidden):
        for j in range(num_output):
            idx_i = k  # 隐藏层变量索引
            idx_j = j + num_hidden  # 输出层变量索引
            weight = np.clip(net.weights_1[k][j], -1, 1).item()
            qubo_expr += weight * variables[idx_i] * variables[idx_j]

    # **构建 QUBO 模型**
    qubo_model = kw.qubo.make(qubo_expr)

    return qubo_model

class QUBONetwork(nn.Module):
    def __init__(self):
        super(QUBONetwork, self).__init__()
        self.layersList = [1, 500, 1]
        self.sign_beta = 1
        self.beta = 10

        with torch.no_grad():
            N_inputs, N_hidden, N_output = self.layersList[0], self.layersList[1], self.layersList[2]

            self.weights_0 = torch.tensor( 2 * (np.random.rand(N_inputs, N_hidden) - 0.5) * math.sqrt(1 / N_inputs) )
            self.weights_1 = torch.tensor( 2 * (np.random.rand(N_hidden, N_output) - 0.5) * math.sqrt(1 / N_hidden) )

            self.weights_0 = self.weights_0
            self.weights_1 = self.weights_1

            self.bias_0 = torch.zeros(N_hidden)
            self.bias_1 = torch.zeros(N_output)

    def computeGrads(self, data, s, seq, batch_size=1):
        # 计算梯度的过程，如果有需要记得传入batch_size
        data_mean = data.mean().item()
        with torch.no_grad():
            coef = self.sign_beta * 10 * batch_size
            gradsW, gradsB = [], []

            gradsW.append(-(np.matmul(s[0].T, s[1]) - np.matmul(seq[0].T, seq[1])) / coef)
            gradsW.append(-((data_mean * s[0]) - (data_mean * seq[0])) / coef)

            gradsB.append(-(s[1] - seq[1]).sum(0) / coef)
            gradsB.append(-(s[0] - seq[0]).sum(0) / coef)

            return gradsW, gradsB

    def updateParams(self, data, s, seq, batch_size=1):
        lr1 = 0.001
        lr2 = 0.0001
        with torch.no_grad():
            ## Compute gradients and update weights from simulated sampling
            gradsW, gradsB = self.computeGrads(data, s, seq, batch_size)

            # weights
            assert self.weights_1.shape == gradsW[0].shape
            self.weights_1 += lr1 * gradsW[0]
            self.weights_1 = self.weights_1.clip(-1, 1)

            assert self.weights_0.shape == gradsW[1].shape
            self.weights_0 += lr1 * gradsW[1]
            self.weights_0 = self.weights_0.clip(-1, 1)

            # biases
            assert self.bias_1.shape == gradsB[0].shape
            self.bias_1 += lr2 * gradsB[0]
            self.bias_1 = self.bias_1.clip(-1, 1)

            assert self.bias_0.shape == gradsB[1].shape
            self.bias_0 += lr2 * gradsB[1]
            self.bias_0 = self.bias_0.clip(-1, 1)

            del gradsW, gradsB

    def forward(self, x):
        """
        前向传播函数，输入为x，输出为预测概率，值在0-1之间，且与输入x形状一致
        """
        batch_size = x.shape[0]
        # 获取权重所在的设备
        device = self.weights_0.device
        # 将输入 x 移动到相同设备
        x = x.to(device).double()

        # 隐藏层前向传播
        hidden = torch.matmul(x, self.weights_0) + self.bias_0
        hidden = F.relu(hidden)  # ReLU激活
        # 输出层前向传播
        output = torch.matmul(hidden, self.weights_1) + self.bias_1
        output = torch.sigmoid(output)  # 使用Sigmoid函数将输出映射到0-1之间
        return output

def trainonceCIM(net, data, cost):
    # 将 data 和 cost 移到 CPU 并脱离计算图
    data = data.cpu().detach()
    cost = cost.cpu().detach()
    loss = data * cost

    # 初始化存储变量
    s_free_all = []
    s_nudge_all = []


    cim_sampler = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=1000,
        alpha=0.99,
        cutoff_temperature=0.001,
        iterations_per_t=10,
        size_limit=10
    )


    # **自由相：当前网络参数下前向传播**
    model_free = create_kaiwu_qubo_model(net, data, net.layersList, beta=net.beta)
    qubo_matrix_free = kw.qubo.qubo_model_to_qubo_matrix(model_free)['qubo_matrix']
    ising_matrix_free, _ = kw.qubo.qubo_matrix_to_ising_matrix(qubo_matrix_free)
    cim_seq = cim_sampler.solve(ising_matrix_free)
    opt_seq = kw.sampler.optimal_sampler(ising_matrix_free, cim_seq, 0)
    best_seq = opt_seq[0][0]

    s_free = best_seq.reshape(1, -1)  # 保持二维形状

    # **推波相：引入损失的梯度进行微扰**
    model_nudge = create_kaiwu_qubo_model(net, data, net.layersList, beta=net.beta, target=loss)
    qubo_matrix_nudge = kw.qubo.qubo_model_to_qubo_matrix(model_nudge)['qubo_matrix']
    ising_matrix_nudge, _ = kw.qubo.qubo_matrix_to_ising_matrix(qubo_matrix_nudge)
    cim_s = cim_sampler.solve(ising_matrix_nudge)
    opt_s = kw.sampler.optimal_sampler(ising_matrix_nudge, cim_s, 0)
    best_s = opt_s[0][0]

    s_nudge = best_s.reshape(1, -1)  # 保持二维形状

    # **累积采样结果**
    # s_free 和 s_nudge 的形状为 (1, num_variables)，截断第二个维度多出的部分
    # s_free = s_free[:, :net.layersList[1]]; s_nudge = s_nudge[:, :net.layersList[1]]
    s_free_all.append(s_free)
    s_nudge_all.append(s_nudge)

    # 将列表转换为 numpy 数组
    s_free_all = np.vstack(s_free_all)
    s_nudge_all = np.vstack(s_nudge_all)

    # **提取自由相和推波相的输出**

    seq = [
        s_free_all[:, :net.layersList[1]],             # 隐藏层输出
        s_free_all[:, net.layersList[1]:501]              # 输出层输出
    ]
    s = [
        s_nudge_all[:, :net.layersList[1]],            # 隐藏层输出
        s_nudge_all[:, net.layersList[1]:501]             # 输出层输出
    ]

    # **更新网络参数**
    net.updateParams(data, s, seq, batch_size=len(data))




def build_training_with_imqa(args):
    model = ResNet18(num_classes=args.num_classes).cuda()
    optimizer_a = torch.optim.AdamW(model.parameters(), lr=2.5e-5 * args.batch_size / 32, betas=(0.9, 0.95),
                                    weight_decay=0.05)
    vnet = QUBONetwork()

    return model, optimizer_a, vnet
############################################################################################################

def meta_sifter(args, dataset):
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                   pin_memory=True, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    mnet_list = []
    vnet_list = []
    for i in range(args.num_sifter):
        print("-----------Training sifter number: " + str(i) + "-----------")
        model, optimizer_a, vnet, optimizer_c = build_training(args)
        grad_models, grad_optimizers = build_grad_models(args, model)
        model, optimizer_a = warmup(model, optimizer_a, train_dataloader, args)
        raw_meta_model = ResNet18(num_classes=args.num_classes).cuda()
        for i in range(args.res_epochs):
            train_iter = tqdm(enumerate(train_dataloader), total=int(len(dataset) / args.batch_size) + 1)
            for iteration, (input_train, target_train) in train_iter:
                input_var, target_var = input_train.cuda(), target_train.cuda()

                # virtual training 虚拟训练
                meta_model = copy.deepcopy(raw_meta_model)  # 复制
                meta_model.load_state_dict(model.state_dict())  # 复制
                y_f_hat = meta_model(input_var)  # 使用元模型进行前向传播，得到预测值
                cost = criterion(y_f_hat, target_var)
                cost_v = torch.reshape(cost, (len(cost), 1))  # 重塑形状
                # print(cost_v.data)
                v_lambda = vnet(cost_v.data)  # 权重模型vnet进行前向传播，预测权重
                v_lambda = v_lambda.view(-1)  # 重塑形状
                v_lambda = norm_weight(v_lambda)  # 归一化
                l_f_meta = torch.sum(v_lambda * cost)

                # virtual backward & update
                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, (meta_model.parameters()), create_graph=True, allow_unused=True)
                # 执行反向传播，得到梯度

                # compute gradient gates and update the model
                new_grads, _ = compute_gated_grad(grads, grad_models, args.top_k, args.num_act)  # 梯度门控
                pseudo_optimizer = MetaSGD(meta_model, meta_model.parameters(), lr=args.meta_lr)  # 更新元模型
                pseudo_optimizer.load_state_dict(optimizer_a.state_dict())
                pseudo_optimizer.meta_step(new_grads)

                res_y_f_hat = meta_model(input_var)  # 使用元模型进行前向传播，得到预测值
                res_cost = criterion(res_y_f_hat, target_var)
                res_cost_v = torch.reshape(res_cost, (len(res_cost), 1))
                res_v_bf_lambda = vnet(res_cost_v.data)
                res_v_bf_lambda = res_v_bf_lambda.view(-1)
                res_v_lambda = 1 - res_v_bf_lambda
                res_v_lambda = norm_weight(res_v_lambda)

                valid_loss = -torch.sum((res_v_lambda) * res_cost)

                optimizer_c.zero_grad()
                for go in grad_optimizers:
                    go.zero_grad()
                valid_loss.backward()
                optimizer_c.step()
                for go in grad_optimizers:
                    go.step()
                del grads, new_grads  # 释放内存

                # actuall update，实际更新
                y_f = model(input_var)  # 主模型进行前向传播
                cost_w = (criterion(y_f, target_var) )
                cost_v = torch.reshape(cost_w, (len(cost_w), 1))

                with torch.no_grad():
                    w_new = vnet(cost_v)  # 使用权重模型进行前向传播，得到预测权重

                w_new = w_new.view(-1)  # 重塑形状
                w_new = norm_weight(w_new)  # 归一化
                l_f = torch.sum(w_new * cost_w)  # 加权损失


                optimizer_a.zero_grad()
                l_f.backward()
                optimizer_a.step()
        vnet_list.append(copy.deepcopy(vnet))
        mnet_list.append(copy.deepcopy(model))
    return vnet_list, mnet_list

def CIM_train_sifter(args, dataset):
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                   pin_memory=True, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    mnet_list = []
    vnet_list = []
    for i in range(args.num_sifter):
        print("-----------Training sifter number: " + str(i) + "-----------")
        model, optimizer_a, vnet = build_training_with_imqa(args)
        grad_models, grad_optimizers = build_grad_models(args, model)
        model, optimizer_a = warmup(model, optimizer_a, train_dataloader, args)
        raw_meta_model = ResNet18(num_classes=args.num_classes).cuda()
        for i in range(args.res_epochs):
            train_iter = tqdm(enumerate(train_dataloader), total=int(len(dataset) / args.batch_size) + 1)
            for iteration, (input_train, target_train) in train_iter:
                input_var, target_var = input_train.cuda(), target_train.cuda()

                # virtual training 虚拟训练
                meta_model = copy.deepcopy(raw_meta_model)  # 复制
                meta_model.load_state_dict(model.state_dict())  # 复制
                y_f_hat = meta_model(input_var)  # 使用元模型进行前向传播，得到预测值
                cost = criterion(y_f_hat, target_var)
                cost_v = torch.reshape(cost, (len(cost), 1))  # 重塑形状

                v_lambda = vnet.forward(cost_v.data)  # 权重模型vnet进行前向传播，预测权重, size == batch_size
                v_lambda = v_lambda.view(-1)  # 重塑形状
                v_lambda = norm_weight(v_lambda)  # 归一化
                v_lambda = v_lambda.to(cost.device)  # 将 v_lambda 移动到与 cost 相同的设备上

                l_f_meta = torch.sum(v_lambda * cost)

                # virtual backward & update
                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, (meta_model.parameters()), create_graph=True, allow_unused=True)

                # compute gradient gates and update the model
                new_grads, _ = compute_gated_grad(grads, grad_models, args.top_k, args.num_act)  # 梯度门控
                pseudo_optimizer = MetaSGD(meta_model, meta_model.parameters(), lr=args.meta_lr)  # 更新元模型
                pseudo_optimizer.load_state_dict(optimizer_a.state_dict())
                pseudo_optimizer.meta_step(new_grads)

                res_y_f_hat = meta_model(input_var)  # 使用元模型进行前向传播，得到预测值
                res_cost = criterion(res_y_f_hat, target_var)
                res_cost_v = torch.reshape(res_cost, (len(res_cost), 1))
                res_v_bf_lambda = vnet.forward(res_cost_v.data)
                res_v_bf_lambda = res_v_bf_lambda.view(-1)
                res_v_lambda = 1 - res_v_bf_lambda
                res_v_lambda = norm_weight(res_v_lambda)
                res_v_lambda = res_v_lambda.to(res_cost.device) # 将 res_v_lambda 移动到与 res_cost 相同的设备上
                valid_loss = -torch.sum((res_v_lambda) * res_cost)

            #   optimizer_c.zero_grad()
                for go in grad_optimizers:
                    go.zero_grad()
                valid_loss.backward()
                # valid_loss 对vent进行平衡传播
                trainonceCIM(vnet, -res_v_lambda, res_cost)

                for go in grad_optimizers:
                    go.step()
                del grads, new_grads

                # actuall update，实际更新
                y_f = model(input_var)  # 主模型进行前向传播
                cost_w = (criterion(y_f, target_var) )
                cost_v = torch.reshape(cost_w, (len(cost_w), 1))

                with torch.no_grad():
                    w_new = vnet.forward(cost_v)  # 使用权重模型进行前向传播，得到预测权重

                w_new = w_new.view(-1)  # 重塑形状
                w_new = norm_weight(w_new)  # 归一化
                w_new = w_new.to(cost_w.device)  # 将 w_new 移动到与 cost 相同的设备上

                l_f = torch.sum(w_new * cost_w)  # 加权损失

                optimizer_a.zero_grad()
                l_f.backward()
                optimizer_a.step()
                # l_f 对vent进行平衡传播
                trainonceCIM(vnet, w_new, cost_w)

        vnet_list.append(copy.deepcopy(vnet))
        mnet_list.append(copy.deepcopy(model))
    return vnet_list, mnet_list



def test_sifter(args, dataset, vnet_list, mnet_list):
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    v_res = np.zeros((args.num_sifter, len(dataset)), dtype=np.float32)
    for i in range(args.num_sifter):
        v = np.zeros((len(dataset)), dtype=np.float32)
        meta_model = mnet_list[i]
        meta_model.eval()
        vnet = vnet_list[i]
        # meta_model.train()
        for b, (images, labels) in tqdm(enumerate(test_dataloader), total=int(len(dataset) / args.batch_size)):
            input_var, target_var = images.cuda(), labels.cuda()
            y_f_hat = meta_model(input_var)
            cost = criterion(y_f_hat, target_var)
            cost_v = torch.reshape(cost, (len(cost), 1))

            v_lambda = vnet(cost_v.data)
            batch_size = v_lambda.size()[0]
            v_lambda = v_lambda.view(-1)

            zero_idx = b * batch_size
            v[zero_idx:zero_idx + batch_size] = v_lambda.detach().cpu().numpy()

        v_res[i, :] = copy.deepcopy(v)
    return v_res


def get_sifter_result(args, dataset, v_res, total_pick=1000):
    class_per = []
    for i in np.unique(dataset.targets):
        percent = len(np.where(np.array(dataset.targets) == i)[0]) / len(dataset)
        class_per.append(math.ceil(total_pick * percent))

    new_mat = np.mean(v_res, axis=0)
    new_idx = []
    for i in range(args.num_classes):
        pick_p = class_per[i]
        tar_idx = np.where(np.array(dataset.targets) == i)[0]
        p_tail = (len(tar_idx) - pick_p) / len(tar_idx) * 100
        cutting = np.percentile(new_mat[tar_idx], p_tail)
        tar_new_idx = np.where(new_mat[tar_idx] >= cutting)[0]
        if tar_new_idx.shape[0] > pick_p:
            tar_new_idx = tar_new_idx[:pick_p]
        new_idx.append(tar_idx[tar_new_idx])
    new_idx = [i for item in new_idx for i in item]
    new_idx = np.array(new_idx)
    return new_idx

def Meta_Sift(args, dataset, total_pick=1000):
    set_seed(args.random_seed)
    test_poi_set = copy.deepcopy(dataset)
    test_poi_set.transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
    ])
    vnet_list, mnet_list = meta_sifter(args, dataset)
    v_res = test_sifter(args, test_poi_set, vnet_list, mnet_list)
    return get_sifter_result(args, test_poi_set, v_res, total_pick)

def QDetection_IM(args, dataset, total_pick=1000):
    set_seed(args.random_seed)
    test_poi_set = copy.deepcopy(dataset)
    test_poi_set.transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
    ])
    # vnet_list, mnet_list = train_sifter(args, dataset)
    vnet_list, mnet_list = CIM_train_sifter(args, dataset)
    v_res = test_sifter(args, test_poi_set, vnet_list, mnet_list)
    return get_sifter_result(args, test_poi_set, v_res, total_pick)

def QDetection_QA(args, dataset, total_pick=1000):
    set_seed(args.random_seed)
    test_poi_set = copy.deepcopy(dataset)
    test_poi_set.transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
    ])
    vnet_list, mnet_list = QA_train_sifter(args, dataset)
    v_res = test_sifter(args, test_poi_set, vnet_list, mnet_list)
    return get_sifter_result(args, test_poi_set, v_res, total_pick)



