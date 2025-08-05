import os
from torch import autocast, optim

# from QDetection import *
from torch.utils.data import DataLoader, Subset
from torchvision import models


# methodname = targeted_label_filpping  narcissus  badnets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.metrics import accuracy_score
import numpy as np
import random
from torch.autograd import Variable


def test_resnet18_accuracy(trainset, testset, filtered_indices, num_epochs=10, batch_size=32, learning_rate=0.001):

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    trainset.transform = data_transform
    testset.transform = data_transform


    filtered_trainset = Subset(trainset, filtered_indices)


    train_loader = DataLoader(filtered_trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)


    model = models.resnet18() # weights="pretrained"
    num_features = model.fc.in_features
    num_classes = trainset.num_classes
    model.fc = nn.Linear(num_features, num_classes)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f'Overall Test Accuracy: {test_accuracy * 100:.2f}%')


    target_label = 38
    target_indices = [i for i, label in enumerate(all_labels) if label == target_label]
    if len(target_indices) > 0:
        target_preds = [all_preds[i] for i in target_indices]
        target_labels = [all_labels[i] for i in target_indices]
        target_accuracy = accuracy_score(target_labels, target_preds)
        print(f'Accuracy for Label {target_label}: {target_accuracy * 100:.2f}%')
    else:
        target_accuracy = None
        print(f'No samples with Label {target_label} in the test set.')


    del model
    torch.cuda.empty_cache()

    return test_accuracy, target_accuracy

def create_baseline_dataset(trainset, num_samples=4000):

    labels = np.array(trainset.targets)
    num_classes = trainset.num_classes

    samples_per_class = num_samples // num_classes
    class_indices = []
    for c in range(num_classes):
        indices = np.where(labels == c)[0]

        if len(indices) < samples_per_class:
            selected_indices = indices
        else:
            selected_indices = np.random.choice(indices, samples_per_class, replace=False)
        class_indices.extend(selected_indices)


    if len(class_indices) < num_samples:
        remaining_indices = list(set(range(len(trainset))) - set(class_indices))
        additional_indices = np.random.choice(remaining_indices, num_samples - len(class_indices), replace=False)
        class_indices.extend(additional_indices)


    random.shuffle(class_indices)

    return class_indices

def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_results(model, data_set, args):
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=128, num_workers=4, shuffle=False)
    model = model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data) in enumerate(data_loader):
            inputs, targets = data[0].to(args.device), data[1].to(args.device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total * 100


def get_NCR(dataset, poi_idx, result_idx):
    return len(set(poi_idx) & set(result_idx)) / len(result_idx) / (len(poi_idx) / len(dataset)) * 100


def norm_weight(weights):
    norm = torch.sum(weights)
    if norm >= 0.0001:
        normed_weights = weights / norm
    else:
        normed_weights = weights
    return normed_weights


class nnVent(nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(nnVent, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu1(x)
        out = self.linear3(x)
        return torch.sigmoid(out)


def warmup(model, optimizer, data_loader, args):
    for w_i in range(args.warmup_epochs):
        for iters, (input_train, target_train) in enumerate(data_loader):
            model.train()
            input_var, target_var = input_train.to(args.device), target_train.to(args.device)
            optimizer.zero_grad()

            with autocast(device_type=args.device):
                outputs = model(input_var)
                loss = F.cross_entropy(outputs, target_var)
            loss.backward()
            optimizer.step()
        print('Warmup Epoch {} '.format(w_i))
    return model, optimizer


def grad_function(grad, grad_model):
    grad_size = grad.size()
    if len(grad_size) == 4:
        reduced_grad = torch.sum(grad, dim=[1, 2, 3]).view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    elif len(grad_size) == 2:
        reduced_grad = torch.sum(grad, dim=[1]).view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    else:
        reduced_grad = grad.view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    return grad_act


def compute_gated_grad(grads, grad_models, num_opt, num_act):
    new_grads = []
    acts = []
    gates = []
    for grad in grads[0:-num_opt]:
        new_grads.append(grad.detach())
    for g_id, grad in enumerate(grads[-num_opt:-2]):
        grad_act = grad_function(grad, grad_models[g_id]) 
        if grad_act > 0.5:
            new_grads.append(grad_act * grad)
        else:
            new_grads.append((1 - grad_act) * grad.detach())
    acts.append(grad_act)
    for grad in grads[-2::]:
        new_grads.append(grad)
    act_loss = (torch.sum(torch.cat(acts)) - num_act) ** 2
    return new_grads, act_loss


def to_var(x, args, requires_grad=True):
    if torch.cuda.is_available():
        x = x.to(args.device)
    return Variable(x, requires_grad=requires_grad)


class nnGradGumbelSoftmax(nn.Module):
    def __init__(self, input, hidden, args, input_norm=False):
        super(nnGradGumbelSoftmax, self).__init__()
        # self.bn = MetaBatchNorm1d(input)
        self.linear1 = nn.Linear(input, hidden)
        self.relu1 = nn.PReLU()
        self.linear2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.PReLU()

        self.act = nn.Linear(hidden, 2)
        self.register_buffer('weight_act', to_var(self.act.weight.data, args=args, requires_grad=True))
        self.register_buffer('bias_act', to_var(self.act.bias.data, args=args, requires_grad=True))
        self.input_norm = input_norm

    def forward(self, x):
        if self.input_norm:
            x_mean, x_std = x.mean(), x.std()
            x = (x - x_mean) / (x_std + 1e-9)
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = F.linear(x, self.weight_act, self.bias_act)
        y = F.gumbel_softmax(x, tau=5)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return y_hard


def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def build_thres_model(args, weight_shape):
    hidden_dim = 128
    model = nnGradGumbelSoftmax(weight_shape[0], hidden_dim, args, input_norm=True)
    model.to(args.device)
    return model


def build_grad_models(args, model):
    grad_models = []
    grad_optimizers = []
    for param in list(model.parameters())[-args.top_k:-2]:
        param_shape = param.size()
        _grad_model = build_thres_model(args, param_shape)
        _optimizer = torch.optim.SGD(_grad_model.parameters(), args.go_lr,
                                     momentum=args.momentum, nesterov=args.nesterov,
                                     weight_decay=0)
        grad_models.append(_grad_model)
        grad_optimizers.append(_optimizer)
    return grad_models, grad_optimizers


from torch.optim.sgd import SGD


class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters
            # current_module._parameters.__setitem__(name, parameters)

    def meta_step(self, grads):
        group = self.param_groups[0]
        lr = group['lr']

        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            parameter.detach_()
            self.set_parameter(self.net, name, parameter.add(grad, alpha=-lr))


def named_params(model, curr_module=None, memo=None, prefix=''):
    if memo is None:
        memo = set()

    if hasattr(curr_module, 'named_leaves'):
        for name, p in curr_module.named_leaves():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
    else:
        for name, p in curr_module._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p

    for mname, module in curr_module.named_children():
        submodule_prefix = prefix + ('.' if prefix else '') + mname
        for name, p in named_params(model, module, memo, submodule_prefix):
            yield name, p


def update_params(model, lr_inner, first_order=False, source_params=None, detach=False):
    '''
        official implementation
    '''
    if source_params is not None:
        for tgt, src in zip(model.named_parameters(), source_params):
            name_t, param_t = tgt
            # name_s, param_s = src
            # grad = param_s.grad
            # name_s, param_s = src
            if src is None:
                print('skip param')
                continue
            # grad = src
            tmp = param_t - lr_inner * src
            set_param(model, model, name_t, tmp)
    return model


def set_param(model, curr_mod, name, param):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(model, mod, rest, param)
                break
    else:
        setattr(getattr(curr_mod, name), 'data', param)
