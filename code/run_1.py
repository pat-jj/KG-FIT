import argparse
import wandb

from torch.utils.data   import DataLoader
from utils              import *
from model_1            import *
from dataloader         import TrainDataset
from dataloader         import BidirectionalOneShotIterator



def construct_args():
    parser = argparse.ArgumentParser(description='KG-FIT')
    # Data paths
    parser.add_argument('--data_path', type=str, default='data', help='Path to the dataset')
    parser.add_argument('--process_path', type=str, default='/shared/pj20/lamake_data', help='Path to the entity hierarchy')
    parser.add_argument('--dataset', type=str, default='FB15K-237', help='Dataset name')
    parser.add_argument('--hierarchy_type', type=str, default='seed', choices=['seed', 'llm'],  help='Type of hierarchy to use')
    
    # train, valid, test
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test',  action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    # Model settings
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')

    # Model hyperparameters
    parser.add_argument('--model', type=str, default='TransE', help='Knowledge graph embedding model')
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['euclidean', 'cosine', 'complex', 'pi'],help='Distance metric for link prediction')
    
    # Hyperparameters
    parser.add_argument('--rho', type=float, default=0.5, help='Weight for the randomly initialized component')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='Weight for the inter-level cluster separation')
    parser.add_argument('--lambda_2', type=float, default=0.5, help='Weight for the hierarchical distance maintenance')
    parser.add_argument('--lambda_3', type=float, default=0.5, help='Weight for the cluster cohesion')
    parser.add_argument('--zeta_1', type=float, default=0.5, help='Weight for the entire hierarchical constraint')
    parser.add_argument('--zeta_2', type=float, default=0.5, help='Weight for the text embedding deviation')
    parser.add_argument('--zeta_3', type=float, default=0.5, help='Weight for the link prediction score')
    
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--early_stop', type=int, default=10, help='Number of epochs for early stopping')
    parser.add_argument('--cuda', action='store_true', help='Use GPU for training')
    parser.add_argument('--uni_weight', action='store_true', help='Use uniform weight for positive and negative samples')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=10, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    args = parser.parse_args()
    
    args.data_path = f'{args.data_path}/{args.dataset}'
    args.save_path = f'{args.process_path}/{args.dataset}/checkpoints/{args.model}_{args.hierarchy_type}_batch_{args.batch_size}_hidden_{args.hidden_dim}_dist_{args.distance_metric}'
    
    return args

args = construct_args()

def log_metrics(mode, step, metrics):
    wandb.log({f"{mode}_{k}": v for k, v in metrics.items()}, step=step)
    print(f"{mode} step {step}: {metrics}")

def main(args):
    wandb.init(project="kgfit", config=args, name=f"{args.dataset}-{args.model}-{args.hierarchy_type}-{args.hidden_dim}")
    loss_table = wandb.Table(columns=["text_dist_n", "self_cluster_dist_n", "neighbor_cluster_dist_n", "hier_dist_n", "negative_sample_loss", "positive_sample_loss", "loss"])
    
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')
    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
        id2entity = {v: k for k, v in entity2id.items()}
            
    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
            
    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Base Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples  = read_triple(os.path.join(args.data_path, 'test.txt'),  entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    entity_info_train = read_entity_info(os.path.join(f'{args.process_path}/{args.dataset}',\
        f'entity_info_{args.hierarchy_type}_hier.json'), train_triples, id2entity)
    entity_info_valid = read_entity_info(os.path.join(f'{args.process_path}/{args.dataset}',\
        f'entity_info_{args.hierarchy_type}_hier.json'), valid_triples, id2entity)
    entity_info_test = read_entity_info(os.path.join(f'{args.process_path}/{args.dataset}',\
        f'entity_info_{args.hierarchy_type}_hier.json'), test_triples, id2entity)
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    # Load the entity hierarchy and text embeddings
    entity_text_embeddings = read_entity_initial_embedding(args)
    # Load the cluster embeddings
    cluster_embeddings = read_cluster_embeddings(args)
    
    ###### KG-FIT Model ######
    kgfit_model = KGFIT(
        base_model=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        entity_text_embeddings=entity_text_embeddings,
        cluster_embeddings=cluster_embeddings,
        rho=args.rho,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        zeta_1=args.zeta_1,
        zeta_2=args.zeta_2,
        zeta_3=args.zeta_3,
        distance_metric=args.distance_metric,
    )
    wandb.watch(kgfit_model)
    ##########################
    
    logging.info('Model Parameter Configuration:')
    for name, param in kgfit_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    
    
    if args.cuda:
        kgfit_model = kgfit_model.cuda()
        
    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, entity_info_train, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, entity_info_train, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kgfit_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2
            
    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kgfit_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
    else:
        logging.info('Ramdomly Initializing %s Base Model...' % args.model)
        init_step = 0
        
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
       
       
    ###### Training ######
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        
        #Training Loop
        for step in range(init_step, args.max_steps):
            
            log = kgfit_model.train_step(kgfit_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            loss_table.add_data(
                log['loss'],
                log['text_dist_n'],
                log['self_cluster_dist_n'],
                log['neighbor_cluster_dist_n'],
                log['hier_dist_n'],
                log['negative_sample_loss'],
                log['positive_sample_loss']
            )
        
            wandb.log({"loss_details": loss_table}, step=step)
    
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kgfit_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kgfit_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kgfit_model.test_step(kgfit_model, valid_triples, all_true_triples, entity_info_valid, args)
                log_metrics('Valid', step, metrics)
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kgfit_model, optimizer, save_variable_list, args)
        wandb.save(args.save_path)
    
    ######################
    
    ###### Evaluation ######
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kgfit_model.test_step(kgfit_model, valid_triples, all_true_triples, entity_info_valid, args)
        log_metrics('Valid', step, metrics)
        wandb.log({f"Valid_{k}": v for k, v in metrics.items()}, step=step)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kgfit_model.test_step(kgfit_model, test_triples, all_true_triples, entity_info_test, args)
        log_metrics('Test', step, metrics)
        wandb.log({f"Test_{k}": v for k, v in metrics.items()}, step=step)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kgfit_model.test_step(kgfit_model, train_triples, all_true_triples, entity_info_train, args)
        log_metrics('Test', step, metrics)
        wandb.log({f"Train_{k}": v for k, v in metrics.items()}, step=step)
    
    ######################
        
        
if __name__ == '__main__':
    args = construct_args()
    main(args)