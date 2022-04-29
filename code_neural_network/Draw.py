import os
import matplotlib.pyplot as plt
import pickle
from Config import optConfig

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("code_neural_network")+len("code_neural_network")]


def draw_mnist(setting):
    iter = [i for i in range(1, optConfig['iterations'] + 1)]
    algorithms = ['rsa']
    color = ['skyblue', 'gold', 'green', 'red']
    marker = ['v', '^', 's', 'o']

    plt.figure(1)
    plt.locator_params('x', nbins=6)

    with open(rootPath + '/results_' + setting + '/rsa_gauss_attack' + '.pkl', 'rb') as f:
        config, train_loss, test_acc = pickle.load(f)
        plt.plot(iter, test_acc, color='skyblue', marker='v', ls='-', markevery=200,
                 label='RSA')

    with open(rootPath + '/results_' + setting + '/rsa_gauss_attack_gauss_0.2_by4' + '.pkl', 'rb') as f:
        config, train_loss, test_acc = pickle.load(f)
        plt.plot(iter, test_acc, color='gold', marker='*', ls='-', markevery=200,
                 label='DP-RSA(F) $\epsilon$=0.2')

    with open(rootPath + '/results_' + setting + '/rsa_gauss_attack_gauss_0.4_by4' + '.pkl', 'rb') as f:
        config, train_loss, test_acc = pickle.load(f)
        plt.plot(iter, test_acc, color='orange', marker='^', ls='-', markevery=200,
                 label='DP-RSA(F) $\epsilon$=0.4')

    with open(rootPath + '/results_' + setting + '/rsa_gauss_attack_gauss_1.38_by4' + '.pkl', 'rb') as f:
        config, train_loss, test_acc = pickle.load(f)
        plt.plot(iter, test_acc, color='red', marker='>', ls='-', markevery=200,
                 label='DP-RSA(F) $\epsilon$=1.38')

    # with open(rootPath + '/results_' + setting + '/rsa_sample_duplicating_gauss_0.4_na0.05' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     plt.plot(iter, test_acc, color='darkred', marker='o', ls='-', markevery=200,
    #              label='DP-RSA(G) $\lambda$=0.05')




    # with open(rootPath + '/results_' + setting + '/mnist_sgd' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     plt.plot(iter, test_acc, color='skyblue', marker='v', ls='-', markevery=200,
    #              label='SGD')
    #
    # with open(rootPath + '/results_' + setting + '/rsa_none' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     plt.plot(iter, test_acc, color='steelblue', marker='>', ls='-', markevery=200,
    #              label='RSA')
    #
    # with open(rootPath + '/results_' + setting + '/rsa_none_gauss_0.2' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     plt.plot(iter, test_acc, color='red', marker='*', ls='-', markevery=200,
    #              label='DP-RSA(G) $\epsilon$=0.2')
    #
    # with open(rootPath + '/results_' + setting + '/rsa_none_flip_1.38' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     plt.plot(iter, test_acc, color='darkred', marker='<', ls='-', markevery=200,
    #              label='DP-RSA(F) $\epsilon$=1.38')
    #
    # with open(rootPath + '/results_' + setting + '/mnist_signsgd' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     plt.plot(iter, test_acc, color='orange', marker='^', ls='-', markevery=200,
    #              label='SignSGD')
    #
    # with open(rootPath + '/results_' + setting + '/mnist_sgd_gm' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     plt.plot(iter, test_acc, color='royalblue', marker='o', ls='-', markevery=200,
    #              label='SGD with GM')

    plt.xticks(fontsize=18)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=18)
    plt.ylabel('Classification Accuracy', fontsize=18)
    plt.xlabel('Number of iterations', fontsize=18)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(linestyle=':')
    plt.subplots_adjust(bottom=0.135)
    plt.savefig(rootPath + '/picture/MNIST_epsilon_sign_flipping_flip.eps', dpi=300)
    plt.show()


if __name__ == '__main__':
    draw_mnist(setting='iid')



