import os
import pickle
import random
import torch
import torch.nn.functional as F
from Config import sgdConfig, device
from LoadData import getData_mnist
from MainModel import MLP, flatten_list, unflatten_vector, caL_loss_acc, mean, gm
from Attack import same_value, sign_flipping, zero_gradient, sample_duplicating

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("code_neural_network")+len("code_neural_network")]


def centralized_sgd(setting, attack=None, save_model=False, save_results=False):
    """

    :param setting: 'iid' or 'noniid'
    :param aggregate: mean or gm(geometric median)
    :param attack: Byzantine attacks, like sample-duplicating attacks
    :param save_model: whether to save the global model
    :param save_results: whether to save the experiment results
    """
    # initialize the global model and data loader
    lambda0 = 0.01
    u = 0.004
    epsilon = 0.2
    Gmax = 0.01
    du = 2 * sgdConfig['lr'] * Gmax
    gamma = 0.6
    model = [MLP(784, 50, 50, 10).to(device)
             for _ in range(sgdConfig['nodeSize'])]
    model0 = MLP(784, 50, 50, 10).to(device)
    train_dataset_subset, train_loader, test_loader = getData_mnist(setting, sgdConfig)
    train_loaders_splited = [
        torch.utils.data.DataLoader(dataset=subset, batch_size=sgdConfig['batchSize'], shuffle=True, pin_memory=True)
        for subset in train_dataset_subset
    ]
    train_loaders_splited_iter = [iter(loader) for loader in train_loaders_splited]

    train_loss_list = []
    test_acc_list = []
    byzantine = []
    regular = []

    # initialize the set of gradients
    worker_grad = [
        [torch.zeros_like(para, requires_grad=False) for para in model0.parameters()]
        for _ in range(sgdConfig['nodeSize'])
    ]
    worker_model = [
        [torch.zeros_like(para, requires_grad=False) for para in model0.parameters()]
        for _ in range(sgdConfig['nodeSize'])
    ]
    master_grad = [torch.zeros_like(para, requires_grad=False) for para in model0.parameters()]

    signinfo = [
        [torch.zeros_like(para, requires_grad=False) for para in model0.parameters()]
        for _ in range(sgdConfig['nodeSize'])
    ]

    # start training
    for iteration in range(1, sgdConfig['iterations'] + 1):
        print('Train iteration: {}'.format(iteration))

        # randomly generate the set of Byzantine workers and regular workers.
        if iteration == 1:
            byzantine = random.sample(range (sgdConfig['nodeSize']), sgdConfig['byzantineSize'])
            regular = list(set(range(sgdConfig['nodeSize'])).difference(byzantine))
            print(byzantine)
            print(regular)

        count = 0

        model0.train()

        for id in regular:

            model[id].train()
            if setting == 'iid':
                try:
                    batch_iterator = train_loaders_splited_iter[id]
                    data, target = next(batch_iterator)
                except StopIteration:
                    train_loaders_splited_iter = [iter(loader) for loader in train_loaders_splited]
                    batch_iterator = train_loaders_splited_iter[id]
                    data, target = next(batch_iterator)
            elif setting == 'noniid':
                try:
                    batch_iterator = train_loaders_splited_iter[count]
                    data, target = next(batch_iterator)
                except StopIteration:
                    train_loaders_splited_iter = [iter(loader) for loader in train_loaders_splited]
                    batch_iterator = train_loaders_splited_iter[count]
                    data, target = next(batch_iterator)
                count += 1

            data, target = data.to(device), target.to(device)
            output = model[id](data)
            loss = F.nll_loss(output, target)

            # autograd
            model[id].zero_grad()
            loss.backward()

            # obtain the sign information
            for index, (para, para0) in enumerate(zip(model[id].parameters(), model0.parameters())):
                signinfo[id][index].data.zero_()

                # flip the sign information
                # random_flip = torch.rand_like(para)
                # flip = torch.ones_like(para)
                # flip[random_flip > gamma] = -1
                # signinfo[id][index].data.add_(torch.sign(para0.data - para.data) * flip, alpha=1)

                # add the Gaussian noise to the sign information
                sigma = torch.maximum(2/3*(para0.data - para.data), 4*du/epsilon * torch.ones_like(para))
                gauss = torch.randn_like(para) * sigma
                signinfo[id][index].data.add_(torch.sign(para0.data - para.data + gauss), alpha=1)

                # signinfo[id][index].data.add_(torch.sign(para0.data - para.data), alpha=1)

            for index, (para, signvalue) in enumerate(zip(model[id].parameters(), signinfo[id])):
                worker_grad[id][index].data.zero_()
                worker_grad[id][index].data.add_(para.grad.data, alpha=1)
                worker_grad[id][index].data.add_(para, alpha=sgdConfig['decayWeight'])
                worker_grad[id][index].data.add_(signvalue.data, alpha=-lambda0)

            for index, para in enumerate(model[id].parameters()):
                worker_model[id][index].data.zero_()
                worker_model[id][index].data.add_(para.data, alpha=1)

            for para, grad in zip(model[id].parameters(), worker_grad[id]):
                para.data.add_(grad, alpha=-sgdConfig['lr'])


        # the master node aggregate the stochastic gradients under Byzantine attacks
        worker_model_flat = flatten_list(worker_model)

        if attack != None:
            worker_model_flat = attack(worker_model_flat, regular, byzantine)

        # for id in range(sgdConfig['nodeSize']):
        #     worker_model[id] = unflatten_vector(worker_model_flat[id], model0)

        for id in byzantine:
            worker_model[id] = unflatten_vector(worker_model_flat[id], model0)
            for index, (para0, paraby) in enumerate(zip(model0.parameters(), worker_model[id])):
                signinfo[id][index].data.zero_()
                signinfo[id][index].data.add_(torch.sign(para0.data - paraby.data), alpha=1)

        signinfo_flat = flatten_list(signinfo)
        signinfo_flat = torch.sum(signinfo_flat, dim=0)
        signmaster = unflatten_vector(signinfo_flat, model0)

        for index, (para, signvalue) in enumerate(zip(model0.parameters(), signmaster)):
            master_grad[index].data.zero_()
            master_grad[index].data.add_(para.data, alpha=u)
            master_grad[index].data.add_(signvalue.data, alpha=lambda0)
            para.data.add_(master_grad[index].data, alpha=-sgdConfig['lr'])

        # the master node update the global model
        # for (para, grad) in zip(model0.parameters(), master_grad):
        #     para.data.add_(grad, alpha=-sgdConfig['lr'])

        # calculate loss and accuracy of the testing data.
        test_loss, test_acc = caL_loss_acc(model0, device, test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_acc))
        test_acc_list.append(test_acc)

    # save model
    if save_model:
        torch.save(model0.state_dict(), "RSA.pt")

    # save experiment results
    if save_results:
        output = open(rootPath+"/results_"+setting+"/rsa_same_value_gauss_0.2.pkl", "wb")
        pickle.dump((sgdConfig, train_loss_list, test_acc_list), output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    centralized_sgd(setting='iid', attack=same_value,
                    save_model=False, save_results=True)








