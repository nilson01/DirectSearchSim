import sys
import json
from itertools import product
from utils import *

# Generate Data
def generate_and_preprocess_data(params, replication_seed, run='train'):

    # torch.manual_seed(replication_seed)
    sample_size = params['sample_size'] 
    device = params['device']

    # Simulate baseline covariates
    O1 = torch.randn(5, sample_size, device=device)
    Z1 = torch.randn(sample_size, device=device)
    Z2 = torch.randn(sample_size, device=device)

    if params['noiseless']:
        Z1.fill_(0)
        Z2.fill_(0)

    # Stage 1 data simulation
    x1, x2, x3, x4, x5 = O1[0], O1[1], O1[2], O1[3], O1[4]
    pi_10 = torch.ones(sample_size, device=device)
    pi_11 = torch.exp(0.5 - 0.5 * x3)
    pi_12 = torch.exp(0.5 * x4)
    matrix_pi1 = torch.stack((pi_10, pi_11, pi_12), dim=0).t()

    result1 = A_sim(matrix_pi1, stage=1)
    A1, probs1 = result1['A'], result1['probs']
    A1 += 1

    g1_opt = ((x1 > -1).float() * ((x2 > -0.5).float() + (x2 > 0.5).float())) + 1
    Y1 = torch.exp(1.5 - torch.abs(1.5 * x1 + 2) * (A1 - g1_opt).pow(2)) + Z1

    # Stage 2 data simulation
    pi_20 = torch.ones(sample_size, device=device)
    pi_21 = torch.exp(0.2 * Y1 - 1)
    pi_22 = torch.exp(0.5 * x4)
    matrix_pi2 = torch.stack((pi_20, pi_21, pi_22), dim=0).t()

    result2 = A_sim(matrix_pi2, stage=2)
    A2, probs2 = result2['A'], result2['probs']
    A2 += 1

    Y1_opt = torch.exp(torch.tensor(1.5, device=device)) + Z1
    g2_opt = (x3 > -1).float() * ((Y1_opt > 0.5).float() + (Y1_opt > 3).float()) + 1

    Y2 = torch.exp(1.26 - torch.abs(1.5 * x3 - 2) * (A2 - g2_opt).pow(2)) + Z2

    if run != 'test':
      # transform Y for direct search 
      Y1, Y2 = transform_Y(Y1, Y2)

    # Propensity score stack
    pi_tensor_stack = torch.stack([probs1['pi_10'], probs1['pi_11'], probs1['pi_12'], probs2['pi_20'], probs2['pi_21'], probs2['pi_22']])

    # Adjusting A1 and A2 indices
    A1_indices = (A1 - 1).long().unsqueeze(0)  # A1 actions, Subtract 1 to match index values (0, 1, 2)
    A2_indices = (A2 - 1 + 3).long().unsqueeze(0)   # A2 actions, Add +3 to match index values (3, 4, 5) for A2, with added dimension

    # Gathering probabilities based on actions
    P_A1_given_H1_tensor = torch.gather(pi_tensor_stack, dim=0, index=A1_indices).squeeze(0)  # Remove the added dimension after gathering
    P_A2_given_H2_tensor = torch.gather(pi_tensor_stack, dim=0, index=A2_indices).squeeze(0)  # Remove the added dimension after gathering

    # Calculate Ci tensor
    Ci = (Y1 + Y2) / (P_A1_given_H1_tensor * P_A2_given_H2_tensor)

    # Input preparation
    input_stage1 = O1.t()
    input_stage2 = torch.cat([O1.t(), A1.unsqueeze(1), Y1.unsqueeze(1)], dim=1)

    if run == 'test':
        return input_stage1, input_stage2, Ci, Y1, Y2, A1, A2, P_A1_given_H1_tensor, P_A2_given_H2_tensor, g1_opt, g2_opt, Z1, Z2

    # Splitting data into training and validation sets
    train_size = int(params['training_validation_prop'] * sample_size)
    train_tensors = [tensor[:train_size] for tensor in [input_stage1, input_stage2, Ci, Y1, Y2, A1, A2]]
    val_tensors = [tensor[train_size:] for tensor in [input_stage1, input_stage2, Ci, Y1, Y2, A1, A2]]

    return tuple(train_tensors), tuple(val_tensors)




# def surr_opt(tuple_train, tuple_val, params):
    
#     sample_size = params['sample_size'] 
#     best_val_loss, best_model_stage1_params, best_model_stage2_params = float('inf'), None, None

#     nn_stage1 = initialize_and_prepare_model(1, params, sample_size)
#     nn_stage2 = initialize_and_prepare_model(2, params, sample_size)

#     optimizer, scheduler = initialize_optimizer_and_scheduler(nn_stage1, nn_stage2, params)

#     #  Training and Validation data
#     train_data = {'input1': tuple_train[0], 'input2': tuple_train[1], 'Ci': tuple_train[2], 'A1': tuple_train[5], 'A2': tuple_train[6]}
#     val_data = {'input1': tuple_val[0], 'input2': tuple_val[1], 'Ci': tuple_val[2], 'A1': tuple_val[5], 'A2': tuple_val[6]}


#     # Training and Validation loop for both stages
#     for epoch in range(params['n_epoch']):

#         train_loss = process_batches(nn_stage1, nn_stage2, train_data, params, optimizer, is_train=True)
#         val_loss = process_batches(nn_stage1, nn_stage2, val_data, params, optimizer, is_train=False)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_stage1_params = nn_stage1.state_dict()
#             best_model_stage2_params = nn_stage2.state_dict()

#         # Update the scheduler with the current epoch's validation loss
#         update_scheduler(scheduler, params, val_loss)

#     # Save the best model parameters for Stage 1 and Stage 2 during validation
#     torch.save(best_model_stage1_params, f'best_model_stage_surr_1_{sample_size}.pt')
#     torch.save(best_model_stage2_params, f'best_model_stage_surr_2_{sample_size}.pt')
#     return (nn_stage1, nn_stage2)


def surr_opt(tuple_train, tuple_val, params):
    
    sample_size = params['sample_size'] 
    
    train_losses, val_losses = [], []
    best_val_loss, best_model_stage1_params, best_model_stage2_params, epoch_num_model = float('inf'), None, None, 0

    nn_stage1 = initialize_and_prepare_model(1, params, sample_size)
    nn_stage2 = initialize_and_prepare_model(2, params, sample_size)

    optimizer, scheduler = initialize_optimizer_and_scheduler(nn_stage1, nn_stage2, params)

    #  Training and Validation data
    train_data = {'input1': tuple_train[0], 'input2': tuple_train[1], 'Ci': tuple_train[2], 'A1': tuple_train[5], 'A2': tuple_train[6]}
    val_data = {'input1': tuple_val[0], 'input2': tuple_val[1], 'Ci': tuple_val[2], 'A1': tuple_val[5], 'A2': tuple_val[6]}


    # Training and Validation loop for both stages
    for epoch in range(params['n_epoch']):

        train_loss = process_batches(nn_stage1, nn_stage2, train_data, params, optimizer, is_train=True)
        train_losses.append(train_loss)

        val_loss = process_batches(nn_stage1, nn_stage2, val_data, params, optimizer, is_train=False)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            epoch_num_model = epoch
            best_val_loss = val_loss
            best_model_stage1_params = nn_stage1.state_dict()
            best_model_stage2_params = nn_stage2.state_dict()

        # Update the scheduler with the current epoch's validation loss
        update_scheduler(scheduler, params, val_loss)

    # # Save the best model parameters for Stage 1 and Stage 2 during validation
    # torch.save(best_model_stage1_params, f'best_model_stage_surr_1_{sample_size}.pt')
    # torch.save(best_model_stage2_params, f'best_model_stage_surr_2_{sample_size}.pt')

    model_dir = 'models'
    # Check if the directory exists, if not, create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define file paths for saving models
    model_path_stage1 = os.path.join(model_dir, f'best_model_stage_surr_1_{sample_size}.pt')
    model_path_stage2 = os.path.join(model_dir, f'best_model_stage_surr_2_{sample_size}.pt')
    
    # Save the models
    torch.save(best_model_stage1_params, model_path_stage1)
    torch.save(best_model_stage2_params, model_path_stage2)
    
    return (nn_stage1, nn_stage2, (train_losses, val_losses), epoch_num_model)




def eval_DTR(V_replications, num_replications, nn_stage1, nn_stage2, df, params):

    sample_size = params['sample_size'] 

    # Generate and preprocess data for evaluation
    processed_result = generate_and_preprocess_data(params, replication_seed=num_replications, run='test')
    test_input_stage1, test_input_stage2, Ci_tensor, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test, P_A1_g_H1, P_A2_g_H2, d1_star, d2_star, Z1, Z2  = processed_result

    nn_stage1 = initialize_and_load_model(1, sample_size, params)
    nn_stage2 = initialize_and_load_model(2, sample_size, params)

    # Calculate test outputs for all networks in stage 1
    A1, test_outputs_stage1 = compute_test_outputs(nn = nn_stage1, test_input = test_input_stage1, A_tensor = A1_tensor_test, params=params, is_stage1=True)
    test_input_stage2, Y1_pred = prepare_stage2_test_input(O1_tensor_test = test_input_stage1 , A1 = A1, 
                                                           g1_opt_conditions = d1_star, Z1_tensor_test = Z1)

    # Calculate test outputs for all networks in stage 2
    A2, test_outputs_stage2 = compute_test_outputs(nn = nn_stage2, test_input = test_input_stage2, A_tensor = A2_tensor_test, params=params, is_stage1=False)
    Y2_pred =  prepare_Y2_pred(O1_tensor_test = test_input_stage1, A1 = A1, A2 = A2, g2_opt_conditions = d2_star, 
                               Z1_tensor_test = Z1, Z2_tensor_test = Z2)


    # Append to DataFrame
    new_row = {
        'Behavioral_A1': A1_tensor_test.cpu().numpy().tolist(),
        'Behavioral_A2': A2_tensor_test.cpu().numpy().tolist(),
        'Predicted_A1': A1.cpu().numpy().tolist(),
        'Predicted_A2':  A2.cpu().numpy().tolist(),
        'Optimal_A1': d1_star.cpu().numpy().tolist(),
        'Optimal_A2': d2_star.cpu().numpy().tolist()
        }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


    message = f'Y1_pred mean: {torch.mean(Y1_pred)}, Y2_pred mean:  {torch.mean(Y2_pred)}, Y1_pred+Y2_pred mean: {torch.mean(Y1_pred + Y2_pred)} \n'
    tqdm.write(message)


    V_replications = calculate_policy_values(Y1_tensor=Y1_tensor, Y2_tensor=Y2_tensor, 
                                             d1_star=d1_star, d2_star=d2_star, 
                                             Y1_pred=Y1_pred, Y2_pred=Y2_pred, 
                                             V_replications=V_replications, 
                                             Z1_tensor_test=Z1, Z2_tensor_test=Z2)

    return V_replications, df



def simulations(num_replications, V_replications, params):
    columns = ['Behavioral_A1', 'Behavioral_A2', 'Predicted_A1', 'Predicted_A2', 'Optimal_A1', 'Optimal_A2']
    df = pd.DataFrame(columns=columns)
    losses_dict = {}
    epoch_num_model_lst = []

    for replication in tqdm(range(num_replications), desc="Replications_M1"):

        tqdm.write(f"Replication # -------------->>>>>  {replication+1}")

        # Generate and preprocess data for training
        tuple_train, tuple_val = generate_and_preprocess_data(params, replication_seed=replication, run='train')

        #  Estimate treatment regime : model --> surr_opt
        tqdm.write("Training started!")
        nn_stage1, nn_stage2, trn_val_loss_tpl, epoch_num_model = surr_opt(tuple_train, tuple_val, params)
        epoch_num_model_lst.append(epoch_num_model)
        losses_dict[replication] = trn_val_loss_tpl
        
        # eval_DTR
        tqdm.write("Evaluation started")
        V_replications, df = eval_DTR(V_replications, replication, nn_stage1, nn_stage2, df, params )
        
    return V_replications, df, losses_dict, epoch_num_model_lst





def run_training(config, config_updates, V_replications, replication_seed):
    torch.manual_seed(replication_seed)
    local_config = {**config, **config_updates}  # Create a local config that includes both global settings and updates
    
    # Execute the simulation function using updated settings
    V_replications, df, losses_dict, epoch_num_model_lst = simulations(local_config['num_replications'], V_replications, local_config)
    accuracy_df = calculate_accuracies(df, V_replications)
    return accuracy_df, df, losses_dict, epoch_num_model_lst
    

    
    
   


def main():
    # Load configuration and set up the device
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    training_validation_prop = config['training_validation_prop']
    train_size = int(training_validation_prop * config['sample_size'])
    print("Training size:", train_size)

    # Define parameter grid for grid search
    # param_grid = {
    #     'activation_function': ['RELU', 'ELU'],
    #     'batch_size': [64, 128, 256, 384],
    #     'learning_rate': [0.0007, 0.007, 0.07],
    #     'num_layers': [4, 5, 6, 7]
    # }    

    param_grid = {
        'activation_function': ['ELU'],
        'batch_size': [1024, 3072],
        'learning_rate': [0.007],
        'num_layers': [4]
    }
    # Perform operations whose output should go to the file
    run_grid_search(config, param_grid)


def run_grid_search(config, param_grid):
    
    # Initialize for storing results and performance metrics
    results = {}
    all_dfs = pd.DataFrame()  # DataFrames from each run
    all_losses_dicts = []  # Losses from each run
    all_epoch_num_lists = []  # Epoch numbers from each run

    for params in product(*param_grid.values()):
        current_config = dict(zip(param_grid.keys(), params))
        performances = pd.DataFrame()

        for i in range(2):  # Assume 2 replications
            V_replications = {
                "V_replications_M1_pred": [],
                "V_replications_M1_behavioral": [],
                "V_replications_M1_optimal": []
            }
            performance, df, losses_dict, epoch_num_model_lst = run_training(config, current_config, V_replications, replication_seed=i)
            performances = pd.concat([performances, performance], axis=0)
            all_dfs = pd.concat([all_dfs, df], axis=0)
            all_losses_dicts.append(losses_dict)
            all_epoch_num_lists.append(epoch_num_model_lst)

        # Store and print average performance across replications for each configuration
        # Convert the configuration dictionary to a hashable json # config_key = tuple(sorted(current_config.items()))
        config_key = json.dumps(current_config, sort_keys=True)
        results[config_key] = performances.mean()
        print("Performances for configuration:", config_key)
        print(performances)
        print("\n\n")

    save_simulation_data(all_dfs, all_losses_dicts, all_epoch_num_lists, results)
    load_and_process_data(config)

class FlushFile:
    """File-like wrapper that flushes on every write."""
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()  # Flush output after write

    def flush(self):
        self.f.flush()
        
if __name__ == '__main__':
    # Setup to write output to a file instead of the console
    original_stdout = sys.stdout  
    with open('output.txt', 'w') as f:
        sys.stdout = FlushFile(f)   # Replace stdout with an instance that flushes on every write to the file we created.
        main()  
    sys.stdout = original_stdout  



# if __name__ == '__main__':
#     main()








# setting = 'tao'
# f_model = 'surr_opt'
# print("\n")

# sample_size = 15000  # 500, 1000 are the cases to check
# num_replications = 3
# n_epoch = 150 # 150

# training_validation_prop = 0.5 #0.95 #0.01
# train_size = int(training_validation_prop * sample_size)
# print("Training size: ", train_size)

# # batch_prop = 0.2 #0.07, 0.2
# batch_size = 512 # 64, 128, 256, 512, 3000
# print("Mini-batch size: ", batch_size)


# noiseless = True # True False. # no noise
# tree_type =  True # True False

# surrogate_num = 1 # 1 - old multiplicative one  2- new one
# option_sur = 2 # 2, 4 # if surrogate_num = 1 then from 1-5 options, if surrogate_num = 2 then 1-> assymetric, 2 -> symmetric


# # Set the device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# network_parameters_surogate = {
#   'setting': setting,
#   'device':device,
#   'noiseless':noiseless,   # Boolean flag to indicate if the noise
#   'sample_size':sample_size,
#   'batch_size': batch_size, # math.ceil(batch_prop*sample_size), #int(0.038*sample_size),
#   'training_validation_prop':training_validation_prop,
#   'n_epoch': n_epoch,
#   'num_networks': 2,
#   'input_dim_stage1': 5, # [O1] --> [x1,...x5]
#   'output_dim_stage1': 1,
#   'input_dim_stage2': 7, # [O1, A1, Y1, O2]

#   'output_dim_stage2': 1,
#   'hidden_dim_stage1': 20, #20
#   'hidden_dim_stage2': 20, #20
#   'dropout_rate': 0.4, #0.3, 0.43

#   'optimizer_type': 'adam',  # Can be 'adam' or 'rmsprop'
#   'optimizer_lr': 0.07, # 0.07, 0.007
#   'optimizer_weight_decay': 0.001,  #1e-4,  Default: 0. Weight decay (L2 regularization) helps prevent overfitting by penalizing large weights.

#   'use_scheduler': True, # True False
#   'scheduler_type': 'reducelronplateau',  # Can be 'reducelronplateau', 'steplr', or 'cosineannealing'
#   'scheduler_step_size': 30, # optim.lr_scheduler.StepLR
#   'scheduler_gamma': 0.8,

#   'initializer': 'he', # he, custon # He initialization (aka Kaiming initialization)

#   # 'f_model': 'surr_opt',
#   'surrogate_num': surrogate_num,
#   'option_sur': option_sur, # if surrogate_num = 1 then 5 options, if surrogate_num = 2 then 1-> assymetric, 2 -> symmetric
# }

# # Initialize V_replications dictionary
# V_replications = {"V_replications_M1_pred": [], "V_replications_M1_behavioral": [], "V_replications_M1_optimal": []}

# print('DGP Setting: ' , setting)
# print("f_model: ", f_model)

# # Run the simulation
# V_replications, df, losses_dict, epoch_num_model_lst = simulations(num_replications, V_replications, network_parameters_surogate)
# summarize_v_values(V_replications, num_replications)
# accuracy_df = calculate_accuracies(df, V_replications)
# print(accuracy_df)

# selected_indices = [i for i in range(num_replications)]
# plot_simulation_surLoss_losses_in_grid(selected_indices, losses_dict, n_epoch)

