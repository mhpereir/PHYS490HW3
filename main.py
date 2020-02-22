import json, argparse, torch
import matplotlib.pyplot as plt
import numpy as np

from time import time
from bm import BM
from data_utils import Data

if __name__ == '__main__':
    start_time = time()
    parser = argparse.ArgumentParser(description='Assignment 3: Boltzmann Machine script')
    
    parser.add_argument('input_path', metavar='inputs',
                        help='input data file name (csv)')
    parser.add_argument('params_path', metavar='params',
                        help='hyper params file name (json)')
    parser.add_argument('--output_path', metavar='results', default='',
                        help='path to results')
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    args = parser.parse_args()
    
    input_file_path  = args.input_path
    params_file_path = args.params_path
    output_file_path = args.output_path
    
    verbosity        = args.v
    
    with open(params_file_path) as paramfile:
        param_file = json.load(paramfile)
    
    n         = 4                               # number of spins in a chains
    N         = int(param_file['batch_size'])
    eta       = param_file['eta']
    n_epochs  = int(param_file['n_epoch'])
    n_epoch_v = int(param_file['n_epoch_v'])
    
    data = Data(input_file_path)
    bm   = BM(eta, n)
    
    loss = torch.nn.KLDivLoss(reduction='batchmean')
        
    p_in_array    = torch.from_numpy(np.array([data.p_in[key] for key in data.p_in.keys()]))
    exp_val_data  = bm.expected_val(data.input_data, len(data.input_data))
    
    train_loss = []
    fig,ax = plt.subplots()
    for epoch in range(1,n_epochs+1):
        bm.gen_model_data(N)
        bm.probs()
        
        if len(bm.p_model) < len(data.p_in):
            for key in data.p_in.keys():
                try:
                    bm.p_model[key]
                except:
                    bm.p_model[key] = 1e-8
        
        p_mod_array = torch.from_numpy(np.array([bm.p_model[key] for key in data.p_in.keys()]))
        #train_loss.append(loss(np.log(p_mod_array), p_in_array).item())
        train_loss.append(bm.KL_div(p_mod_array, p_in_array))
        exp_val_model = bm.expected_val(bm.model_data, N)
                
        bm.update(exp_val_data, exp_val_model)
        
        if verbosity > 1 and epoch%n_epoch_v == 0:
            print('[{}/{}] weights: {}; KLDivLoss: {:4f}'.format(epoch, n_epochs, bm.W, train_loss[-1]))
        
    print('Final results')
    for i in range(0,n):
        print('({},{}):{}'.format(i,i+1,int(round(bm.W[i]))))
    
    with open(str(output_file_path)+'output.txt', 'w') as output:
        output.write('{(0,1) : %i, (1,2) : %i, (2,3) : %i, (3,0) : %i}' % (round(bm.W[0]),
                                                                           round(bm.W[1]),
                                                                           round(bm.W[2]),
                                                                           round(bm.W[3])))
    
    print('Ellapsed time: {:2f} sec'.format(time() - start_time))
    
    ax.plot(range(0,n_epochs), train_loss, color='k')
    ax.set_xlabel('Train epochs')
    ax.set_ylabel('KLDivLoss')
    fig.savefig(str(output_file_path)+'loss_function.png')