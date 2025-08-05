import math

import torchvision.transforms as transforms
from models import ResNet18
from torch.utils.data import TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import copy

from .poi_util import h5_dataset, poi_dataset, set_seed
from .util import (
    nnVent,
    build_grad_models,
    warmup,
    norm_weight,
    compute_gated_grad,
    MetaSGD,
)




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
    model = ResNet18(num_classes=args.num_classes).to(args.device)
    optimizer_a = torch.optim.AdamW(model.parameters(), lr=2.5e-5 * args.batch_size / 32, betas=(0.9, 0.95),
                                    weight_decay=0.05)

    vnet = nnVent(1, 100, 150, 1).to(args.device)


    optimizer_c = torch.optim.SGD(vnet.parameters(), args.v_lr)
    return model, optimizer_a, vnet, optimizer_c

import dimod
import neal
def createBQM(net, input, layersList, beta=4, target=None):
    input = input.mean()
    with torch.no_grad():
        bias_input = input* net.weights_0
        bias_lim = 1 
        # h = {idx_loc: (bias + bias_input[idx_loc]).clip(-bias_lim, bias_lim).item() for idx_loc, bias in
        #      enumerate(net.bias_0)}
        h = {}
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


        model = dimod.BinaryQuadraticModel.from_ising(h, J, 0)
        return model

def trainonceQA(net, data, cost):
    from dwave.system import DWaveSampler, EmbeddingComposite

    data = data.cpu().detach(); cost = cost.cpu().detach()
    loss = data * cost
    store_seq = None
    store_s = None
    qpu_sampler = neal.SimulatedAnnealingSampler()
    # qpu_sampler = EmbeddingComposite(DWaveSampler())

    model_free = createBQM(net, data, net.layersList, beta=net.beta)
    qpu_seq = qpu_sampler.sample(model_free, num_reads=10, num_sweeps=100, auto_scale=4)
    s_free = np.array([list(sample.values()) for sample in qpu_seq.samples()])

    model_nudge = createBQM(net, data, net.layersList, beta=net.beta, target= loss)
    qpu_s = qpu_sampler.sample(model_nudge, num_reads=10, num_sweeps=100, reverse=True)

    store_s = np.array([list(sample.values()) for sample in qpu_s.samples()])

    if store_seq is None:
        store_seq = qpu_seq.record["sample"][0].reshape(1, qpu_seq.record["sample"][0].shape[0])  # qpu_seq
        store_s = qpu_s.record["sample"][0].reshape(1, qpu_s.record["sample"][0].shape[0])  # qpu_s
    else:
        store_seq = np.concatenate(
            (store_seq, qpu_seq.record["sample"][0].reshape(1, qpu_seq.record["sample"][0].shape[0])), 0)
        store_s = np.concatenate(
            (store_s, qpu_s.record["sample"][0].reshape(1, qpu_s.record["sample"][0].shape[0])), 0)
    seq = [store_seq[:, :net.layersList[1]], store_seq[:, net.layersList[1]:]]
    s = [store_s[:, :net.layersList[1]], store_s[:, net.layersList[1]:]] 

    net.updateParams(data, s, seq, batch_size = len(data) )

def QA_train_sifter(args, dataset):
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                   pin_memory=True, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    mnet_list = []
    vnet_list = []
    for i in range(args.num_sifter):
        print("-----------Training sifter number: " + str(i) + "-----------")
        model, optimizer_a, vnet = build_training_with_imqa(args)
        grad_models, grad_optimizers = build_grad_models(args, model)
        model, optimizer_a = warmup(model, optimizer_a, train_dataloader, args)
        raw_meta_model = ResNet18(num_classes=args.num_classes).to(args.device)
        for i in range(args.res_epochs):
            train_iter = tqdm(enumerate(train_dataloader), total=int(len(dataset) / args.batch_size) + 1)
            for iteration, (input_train, target_train) in train_iter:
                input_var, target_var = input_train.to(args.device), target_train.to(args.device)

             
                meta_model = copy.deepcopy(raw_meta_model) 
                meta_model.load_state_dict(model.state_dict()) 
                y_f_hat = meta_model(input_var) 
                cost = criterion(y_f_hat, target_var)
                cost_v = torch.reshape(cost, (len(cost), 1)) 

                v_lambda = vnet.forward(cost_v.data) 
                v_lambda = v_lambda.view(-1) 
                v_lambda = norm_weight(v_lambda)  
                v_lambda = v_lambda.to(cost.device)  

                l_f_meta = torch.sum(v_lambda * cost)

                # virtual backward & update
                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, (meta_model.parameters()), create_graph=True, allow_unused=True)
               

                # compute gradient gates and update the model
                new_grads, _ = compute_gated_grad(grads, grad_models, args.top_k, args.num_act)  
                pseudo_optimizer = MetaSGD(meta_model, meta_model.parameters(), lr=args.meta_lr) 
                pseudo_optimizer.load_state_dict(optimizer_a.state_dict())
                pseudo_optimizer.meta_step(new_grads)

                res_y_f_hat = meta_model(input_var)  
                res_cost = criterion(res_y_f_hat, target_var)
                res_cost_v = torch.reshape(res_cost, (len(res_cost), 1))
                res_v_bf_lambda = vnet.forward(res_cost_v.data)
                res_v_bf_lambda = res_v_bf_lambda.view(-1)
                res_v_lambda = 1 - res_v_bf_lambda
                res_v_lambda = norm_weight(res_v_lambda)
                res_v_lambda = res_v_lambda.to(res_cost.device) 
                valid_loss = -torch.sum((res_v_lambda) * res_cost)

        
                for go in grad_optimizers:
                    go.zero_grad()
                valid_loss.backward()
               
                trainonceQA(vnet, -res_v_lambda, res_cost)

                for go in grad_optimizers:
                    go.step()
                del grads, new_grads  

                # actuall update
                y_f = model(input_var) 
                cost_w = (criterion(y_f, target_var) )
                cost_v = torch.reshape(cost_w, (len(cost_w), 1))

                with torch.no_grad():
                    w_new = vnet.forward(cost_v) 

                w_new = w_new.view(-1) 
                w_new = norm_weight(w_new)  
                w_new = w_new.to(cost_w.device) 

                l_f = torch.sum(w_new * cost_w) 

                optimizer_a.zero_grad()
                l_f.backward()
                optimizer_a.step()
               
                trainonceQA(vnet, w_new, cost_w)

        vnet_list.append(copy.deepcopy(vnet))
        mnet_list.append(copy.deepcopy(model))
    return vnet_list, mnet_list

import numpy as np



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
        batch_size = x.shape[0]
        device = self.weights_0.device
        x = x.to(device).double()

        hidden = torch.matmul(x, self.weights_0) + self.bias_0
        hidden = F.relu(hidden)  # ReLU

        output = torch.matmul(hidden, self.weights_1) + self.bias_1
        output = torch.sigmoid(output)  
        return output





def build_training_with_imqa(args):
    model = ResNet18(num_classes=args.num_classes).to(args.device)
    optimizer_a = torch.optim.AdamW(model.parameters(), lr=2.5e-5 * args.batch_size / 32, betas=(0.9, 0.95),
                                    weight_decay=0.05)
    vnet = QUBONetwork()

    return model, optimizer_a, vnet


def meta_sifter(args, dataset):
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                   pin_memory=True, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    mnet_list = []
    vnet_list = []
    for i in range(args.num_sifter):
        print("-----------Training sifter number: " + str(i) + "-----------")
        model, optimizer_a, vnet, optimizer_c = build_training(args)
        grad_models, grad_optimizers = build_grad_models(args, model)
        model, optimizer_a = warmup(model, optimizer_a, train_dataloader, args)
        raw_meta_model = ResNet18(num_classes=args.num_classes).to(args.device)
        for i in range(args.res_epochs):
            train_iter = tqdm(enumerate(train_dataloader), total=int(len(dataset) / args.batch_size) + 1)
            for iteration, (input_train, target_train) in train_iter:
                input_var, target_var = input_train.to(args.device), target_train.to(args.device)

                
                meta_model = copy.deepcopy(raw_meta_model)  
                meta_model.load_state_dict(model.state_dict()) 
                y_f_hat = meta_model(input_var)  
                cost = criterion(y_f_hat, target_var)
                cost_v = torch.reshape(cost, (len(cost), 1)) 
       
                v_lambda = vnet(cost_v.data) 
                v_lambda = v_lambda.view(-1) 
                v_lambda = norm_weight(v_lambda) 
                l_f_meta = torch.sum(v_lambda * cost)

                # virtual backward & update
                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, (meta_model.parameters()), create_graph=True, allow_unused=True)
               

                # compute gradient gates and update the model
                new_grads, _ = compute_gated_grad(grads, grad_models, args.top_k, args.num_act)  
                pseudo_optimizer = MetaSGD(meta_model, meta_model.parameters(), lr=args.meta_lr)  
                pseudo_optimizer.load_state_dict(optimizer_a.state_dict())
                pseudo_optimizer.meta_step(new_grads)

                res_y_f_hat = meta_model(input_var) 
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
                del grads, new_grads  

                # actuall update
                y_f = model(input_var)  
                cost_w = (criterion(y_f, target_var) )
                cost_v = torch.reshape(cost_w, (len(cost_w), 1))

                with torch.no_grad():
                    w_new = vnet(cost_v)  

                w_new = w_new.view(-1)  
                w_new = norm_weight(w_new)  
                l_f = torch.sum(w_new * cost_w)  


                optimizer_a.zero_grad()
                l_f.backward()
                optimizer_a.step()
        vnet_list.append(copy.deepcopy(vnet))
        mnet_list.append(copy.deepcopy(model))
    return vnet_list, mnet_list



def test_sifter(args, dataset, vnet_list, mnet_list):
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    v_res = np.zeros((args.num_sifter, len(dataset)), dtype=np.float32)
    for i in range(args.num_sifter):
        v = np.zeros((len(dataset)), dtype=np.float32)
        meta_model = mnet_list[i]
        meta_model.eval()
        vnet = vnet_list[i]
        # meta_model.train()
        for b, (images, labels) in tqdm(enumerate(test_dataloader), total=int(len(dataset) / args.batch_size)):
            input_var, target_var = images.to(args.device), labels.to(args.device)
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



