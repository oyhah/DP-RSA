import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optConfig = {
    'nodeSize': 20,
    'byzantineSize': 0,

    'iterations': 25000,
    'decayWeight': 0.00,

    'fixSeed': True,
    'seed': 100,

    'batchSize': 1,
    'test_batchSize': 1000,
}


sgdConfig = optConfig.copy()
sgdConfig['lr'] = 0.01
sgdConfig['batchSize'] = 8


