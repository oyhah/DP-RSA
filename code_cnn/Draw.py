import os
import matplotlib.pyplot as plt
import pickle
from Config import optConfig

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("code_cnn")+len("code_cnn")]


def draw_cifar(setting):
    iter = [i for i in range(1, optConfig['iterations'] + 1)]
    step = 312.5
    epochs = [i for i in range(0, int(optConfig['iterations']/step) + 1, 4)]

    plt.figure(1)
    plt.locator_params('x', nbins=6)

    # Compare
    # with open(rootPath + '/results_' + setting + '/rsa_gauss_attack' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='steelblue', marker='>', ls='-', markevery=1,
    #              label='RSA')
    #
    # with open(rootPath + '/results_' + setting + '/rsa_gauss_attack_gauss_0.2' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='red', marker='*', ls='-', markevery=1,
    #              label='RSA (G) $\epsilon$=0.2')
    #
    # with open(rootPath + '/results_' + setting + '/rsa_gauss_attack_flip_1.38' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='darkred', marker='<', ls='-', markevery=1,
    #              label='RSA (F) $\epsilon$=1.38')
    #
    # with open(rootPath + '/results_' + setting + '/sgd_gauss_attack' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='skyblue', marker='>', ls='-', markevery=1,
    #              label='SGD')
    #
    # with open(rootPath + '/results_' + setting + '/signsgd_gauss_attack' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='orange', marker='^', ls='-', markevery=1,
    #              label='SignSGD')
    #
    # with open(rootPath + '/results_' + setting + '/sgdgm_gauss_attack' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='royalblue', marker='o', ls='-', markevery=1,
    #              label='SGD with GM')

    # Epsilon
    # with open(rootPath + '/results_' + setting + '/rsa_gauss_attack' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='steelblue', marker='v', ls='-', markevery=1,
    #              label='RSA')
    #
    # with open(rootPath + '/results_' + setting + '/rsa_gauss_attack_gauss_0.2_by4' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='gold', marker='*', ls='-', markevery=1,
    #              label='DP-RSA (G) $\epsilon$=0.2')
    #
    # with open(rootPath + '/results_' + setting + '/rsa_gauss_attack_gauss_0.4_by4' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='orange', marker='^', ls='-', markevery=1,
    #              label='DP-RSA (G) $\epsilon$=0.4')
    #
    # with open(rootPath + '/results_' + setting + '/rsa_gauss_attack_gauss_1.38_by4' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='red', marker='>', ls='-', markevery=1,
    #              label='DP-RSA (G) $\epsilon$=1.38')

    # Byzantine size
    # with open(rootPath + '/results_' + setting + '/rsa_sample_duplicating_gauss_0.2' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='steelblue', marker='v', ls='-', markevery=1,
    #              label='DP-RSA (G) b=2')
    #
    # with open(rootPath + '/results_' + setting + '/rsa_sample_duplicating_gauss_0.2_by4' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='gold', marker='*', ls='-', markevery=1,
    #              label='DP-RSA (G) b=4')
    #
    # with open(rootPath + '/results_' + setting + '/rsa_sample_duplicating_gauss_0.2_by6' + '.pkl', 'rb') as f:
    #     config, train_loss, test_acc = pickle.load(f)
    #     test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
    #     test_acc_plot.append(test_acc[optConfig['iterations']-1])
    #     plt.plot(epochs, test_acc_plot, color='orange', marker='^', ls='-', markevery=1,
    #              label='DP-RSA (G) b=6')

    # Lambda
    with open(rootPath + '/results_' + setting + '/rsa_sample_duplicating_gauss_0.2_lambda0.0001' + '.pkl', 'rb') as f:
        config, train_loss, test_acc = pickle.load(f)
        test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
        test_acc_plot.append(test_acc[optConfig['iterations']-1])
        plt.plot(epochs, test_acc_plot, color='skyblue', marker='v', ls='-', markevery=1,
                 label='DP-RSA (G) $\lambda$=0.0001')

    with open(rootPath + '/results_' + setting + '/rsa_sample_duplicating_gauss_0.2_lambda0.001' + '.pkl', 'rb') as f:
        config, train_loss, test_acc = pickle.load(f)
        test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
        test_acc_plot.append(test_acc[optConfig['iterations']-1])
        plt.plot(epochs, test_acc_plot, color='steelblue', marker='v', ls='-', markevery=1,
                 label='DP-RSA (G) $\lambda$=0.001')

    with open(rootPath + '/results_' + setting + '/rsa_sample_duplicating_gauss_0.2' + '.pkl', 'rb') as f:
        config, train_loss, test_acc = pickle.load(f)
        test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
        test_acc_plot.append(test_acc[optConfig['iterations']-1])
        plt.plot(epochs, test_acc_plot, color='gold', marker='*', ls='-', markevery=1,
                 label='DP-RSA (G) $\lambda$=0.002')

    with open(rootPath + '/results_' + setting + '/rsa_sample_duplicating_gauss_0.2_lambda0.01' + '.pkl', 'rb') as f:
        config, train_loss, test_acc = pickle.load(f)
        test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
        test_acc_plot.append(test_acc[optConfig['iterations']-1])
        plt.plot(epochs, test_acc_plot, color='orange', marker='^', ls='-', markevery=1,
                 label='DP-RSA (G) $\lambda$=0.01')

    with open(rootPath + '/results_' + setting + '/rsa_sample_duplicating_gauss_0.2_lambda0.1' + '.pkl', 'rb') as f:
        config, train_loss, test_acc = pickle.load(f)
        test_acc_plot = [test_acc[i] for i in range(0, optConfig['iterations'], int(step*4))]
        test_acc_plot.append(test_acc[optConfig['iterations']-1])
        plt.plot(epochs, test_acc_plot, color='red', marker='>', ls='-', markevery=1,
                 label='DP-RSA (G) $\lambda$=0.1')


    plt.xticks(fontsize=18)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], fontsize=18)
    plt.ylabel('Classification Accuracy', fontsize=18)
    plt.xlabel('Number of Epochs', fontsize=18)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(linestyle=':')
    plt.subplots_adjust(bottom=0.135)
    plt.savefig(rootPath + '/picture/CNN_lambda_sample_duplicating.eps', dpi=300)
    plt.show()


if __name__ == '__main__':
    draw_cifar(setting='noniid')



