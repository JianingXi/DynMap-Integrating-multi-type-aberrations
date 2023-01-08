import os

import torch
import torch.autograd as autograd
import torch.optim as optim

import time
import random

import utils



USE_CUDA = torch.cuda.is_available()

class Config(object):
    def __init__(self):
        self.dataset = None
        self.learning_rate1 = 0.001
        self.learning_rate2 = 0.0005
        self.early_stopping_round = 0
        self.L1_flag = True
        self.embedding_size = 100
        self.num_batches = 100
        self.train_times = 100
        self.margin = 1.0
        self.filter = True
        self.momentum = 0.9
        self.optimizer = optim.Adam
        self.loss_function = loss.marginLoss
        self.entity_total = 0
        self.relation_total = 0
        self.batch_size = 0


if __name__ == "__main__":

    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--dataset', type=str)
    argparser.add_argument('-l1', '--learning_rate1', type=float, default=0.001)
    argparser.add_argument('-l2', '--learning_rate2', type=float, default=0.0005)
    argparser.add_argument('-es', '--early_stopping_round', type=int, default=1)
    argparser.add_argument('-L', '--L1_flag', type=int, default=1)
    argparser.add_argument('-em', '--embedding_size', type=int, default=100)
    argparser.add_argument('-nb', '--num_batches', type=int, default=100)
    argparser.add_argument('-n', '--train_times', type=int, default=100)
    argparser.add_argument('-m', '--margin', type=float, default=1.0)
    argparser.add_argument('-f', '--filter', type=int, default=1)
    argparser.add_argument('-mo', '--momentum', type=float, default=0.9)
    argparser.add_argument('-s', '--seed', type=int, default=0)
    argparser.add_argument('-op', '--optimizer', type=int, default=1)
    argparser.add_argument('-lo', '--loss_type', type=int, default=0)
    argparser.add_argument('-p', '--port', type=int, default=5000)
    argparser.add_argument('-np', '--num_processes', type=int, default=4)

    args = argparser.parse_args()

    dir_cur_name = './Output/' + args.dataset + '/'
    file_train_loss = open(dir_cur_name + "Loss_train.txt", 'w')
    file_valid_loss = open(dir_cur_name + "Loss_valid.txt", 'w')

    if args.seed != 0:
        torch.manual_seed(args.seed)

    trainTotal, trainList, trainDict = loadTriple('./Data_KG/' + args.dataset, 'train2id.txt')
    validTotal, validList, validDict = loadTriple('./Data_KG/' + args.dataset, 'valid2id.txt')
    tripleTotal, tripleList, tripleDict = loadTriple('./Data_KG/' + args.dataset, 'triple2id.txt')
    config = Config()
    config.dataset = args.dataset
    config.learning_rate = args.learning_rate1

    config.early_stopping_round = args.early_stopping_round

    if args.L1_flag == 1:
        config.L1_flag = True
    else:
        config.L1_flag = False

    config.embedding_size = args.embedding_size
    config.num_batches = args.num_batches
    config.train_times = args.train_times
    config.margin = args.margin

    if args.filter == 1:
        config.filter = True
    else:
        config.filter = False

    config.momentum = args.momentum

    if args.optimizer == 0:
        config.optimizer = optim.SGD
    elif args.optimizer == 1:
        config.optimizer = optim.Adam
    elif args.optimizer == 2:
        config.optimizer = optim.RMSprop

    if args.loss_type == 0:
        config.loss_function = loss.marginLoss

    config.entity_total = getAnythingTotal('./Data_KG/' + config.dataset, 'entity2id.txt')
    config.relation_total = getAnythingTotal('./Data_KG/' + config.dataset, 'relation2id.txt')
    config.batch_size = trainTotal // config.num_batches

    shareHyperparameters = {'dataset': args.dataset,
                            'learning_rate1': args.learning_rate1,
                            'learning_rate2': args.learning_rate2,
                            'early_stopping_round': args.early_stopping_round,
                            'L1_flag': args.L1_flag,
                            'embedding_size': args.embedding_size,
                            'margin': args.margin,
                            'filter': args.filter,
                            'momentum': args.momentum,
                            'seed': args.seed,
                            'optimizer': args.optimizer,
                            'loss_type': args.loss_type,
                            }

    trainHyperparameters = shareHyperparameters.copy()
    trainHyperparameters.update({'type': 'train_loss'})

    validHyperparameters = shareHyperparameters.copy()
    validHyperparameters.update({'type': 'valid_loss'})

    hit10Hyperparameters = shareHyperparameters.copy()
    hit10Hyperparameters.update({'type': 'hit10'})

    meanrankHyperparameters = shareHyperparameters.copy()
    meanrankHyperparameters.update({'type': 'mean_rank'})
    loss_function = config.loss_function()
    model = model.DynMapPretrainModelSameSize(config)

    if USE_CUDA:
        model.cuda()
        loss_function.cuda()
        longTensor = torch.cuda.LongTensor
        floatTensor = torch.cuda.FloatTensor

    else:
        longTensor = torch.LongTensor
        floatTensor = torch.FloatTensor

    optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
    margin = autograd.Variable(floatTensor([config.margin]))

    start_time = time.time()

    filename = '_'.join(
        ['l1', str(args.learning_rate1),
         'l2', str(args.learning_rate2),
         'es', str(args.early_stopping_round),
         'L', str(args.L1_flag),
         'em', str(args.embedding_size),
         'nb', str(args.num_batches),
         'n', str(args.train_times),
         'm', str(args.margin),
         'f', str(args.filter),
         'mo', str(args.momentum),
         's', str(args.seed),
         'op', str(args.optimizer),
         'lo', str(args.loss_type), ]) + '_model.ckpt'

    trainBatchList = getBatchList(trainList, config.num_batches)

    phase = 0

    for epoch in range(config.train_times):
        total_loss = floatTensor([0.0])
        random.shuffle(trainBatchList)
        for batchList in trainBatchList:
            if config.filter == True:
                pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_filter_all(
                    batchList,
                    config.entity_total, tripleDict)
            else:
                pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_raw_all(
                    batchList,
                    config.entity_total)

            batch_entity_set = set(pos_h_batch + pos_t_batch + neg_h_batch + neg_t_batch)
            batch_relation_set = set(pos_r_batch + neg_r_batch)
            batch_entity_list = list(batch_entity_set)
            batch_relation_list = list(batch_relation_set)

            pos_h_batch = autograd.Variable(longTensor(pos_h_batch))
            pos_t_batch = autograd.Variable(longTensor(pos_t_batch))
            pos_r_batch = autograd.Variable(longTensor(pos_r_batch))
            neg_h_batch = autograd.Variable(longTensor(neg_h_batch))
            neg_t_batch = autograd.Variable(longTensor(neg_t_batch))
            neg_r_batch = autograd.Variable(longTensor(neg_r_batch))

            model.zero_grad()
            pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e = model(pos_h_batch,
                                                                 pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch,
                                                                 neg_r_batch)

            if args.loss_type == 0:
                losses = loss_function(pos, neg, margin)
            else:
                losses = loss_function(pos, neg)
            ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
            rel_embeddings = model.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
            losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings) + loss.normLoss(
                pos_h_e) + loss.normLoss(pos_t_e) + loss.normLoss(neg_h_e) + loss.normLoss(neg_t_e)

            losses.backward()
            optimizer.step()
            total_loss += losses.data

        # agent.append(trainCurve, epoch, total_loss[0])

        if epoch % 10 == 0:
            now_time = time.time()
            print(now_time - start_time)

            # save train loss
            log_train_loss = "Train total loss: %d %f" % (epoch, total_loss[0])
            print(log_train_loss)
            file_train_loss.write("%d\t%f\n" % (epoch, total_loss[0]))
            # print("Train total loss: %d %f" % (epoch, total_loss[0]))

        if epoch % 10 == 0:
            if config.filter == True:
                pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_filter_random(
                    validList,
                    config.batch_size, config.entity_total, tripleDict)
            else:
                pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_raw_random(
                    validList,
                    config.batch_size, config.entity_total)
            pos_h_batch = autograd.Variable(longTensor(pos_h_batch))
            pos_t_batch = autograd.Variable(longTensor(pos_t_batch))
            pos_r_batch = autograd.Variable(longTensor(pos_r_batch))
            neg_h_batch = autograd.Variable(longTensor(neg_h_batch))
            neg_t_batch = autograd.Variable(longTensor(neg_t_batch))
            neg_r_batch = autograd.Variable(longTensor(neg_r_batch))

            pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e = model(pos_h_batch,
                                                                 pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch,
                                                                 neg_r_batch)

            if args.loss_type == 0:
                losses = loss_function(pos, neg, margin)
            else:
                losses = loss_function(pos, neg)
            ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
            rel_embeddings = model.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
            losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings) + loss.normLoss(
                pos_h_e) + loss.normLoss(pos_t_e) + loss.normLoss(neg_h_e) + loss.normLoss(neg_t_e)

            # save valid loss
            log_valid_loss = "Valid batch loss: %d %f" % (epoch, losses.data.item())
            print(log_valid_loss)
            file_valid_loss.write("%d\t%f\n" % (epoch, losses.data.item()))


        torch.save(model, os.path.join(dir_cur_name, filename).replace('\\', '/'))

    # save embedding final
    ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()
    rel_embeddings = model.rel_embeddings.weight.data.cpu().numpy()
    ent_proj_embeddings = model.ent_proj_embeddings.weight.data.cpu().numpy()
    rel_proj_embeddings = model.rel_proj_embeddings.weight.data.cpu().numpy()
    L1_flag = model.L1_flag
    filter = model.filter

    file_embed_t = open(dir_cur_name + "embedding_ent_final.txt", 'w')
    file_embed_t.write(str(ent_embeddings.tolist()))
    file_embed_t.close()

    file_embed_t = open(dir_cur_name + "embedding_rel_final.txt", 'w')
    file_embed_t.write(str(rel_embeddings.tolist()))
    file_embed_t.close()

    file_embed_t = open(dir_cur_name + "embedding_ent_proj_final.txt", 'w')
    file_embed_t.write(str(ent_proj_embeddings.tolist()))
    file_embed_t.close()

    file_embed_t = open(dir_cur_name + "embedding_rel_proj_final.txt", 'w')
    file_embed_t.write(str(rel_proj_embeddings.tolist()))
    file_embed_t.close()

    # my addition
    with open(dir_cur_name + 'model_para_%s_%s.pkl' % (config.dataset, str(config.embedding_size)), 'wb') as f_t:
        pickle.dump(model, f_t)


    file_train_loss.close()
    file_valid_loss.close()

    print('Done!')