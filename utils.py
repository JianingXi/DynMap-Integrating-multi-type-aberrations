import os
import torch
import torch.autograd as autograd
import torch.nn as nn

import pickle
import random


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor
else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor

class Triple(object):
    def __init__(self, head, tail, relation):
        self.h = head
        self.t = tail
        self.r = relation

# Compare two Triples in the order of head, relation and tail
def cmp_head(a, b):
    return (a.h < b.h or (a.h == b.h and a.r < b.r) or (a.h == b.h and a.r == b.r and a.t < b.t))

# Compare two Triples in the order of tail, relation and head
def cmp_tail(a, b):
    return (a.t < b.t or (a.t == b.t and a.r < b.r) or (a.t == b.t and a.r == b.r and a.h < b.h))

# Compare two Triples in the order of relation, head and tail
def cmp_rel(a, b):
    return (a.r < b.r or (a.r == b.r and a.h < b.h) or (a.r == b.r and a.h == b.h and a.t < b.t))

def minimal(a, b):
    if a > b:
        return b
    return a

def cmp_list(a, b):
    return (minimal(a.h, a.t) > minimal(b.h, b.t))

# Write a list of Triples into a file, with three numbers (head tail relation) per line
def process_list(tripleList, dataset, filename):
    with open(os.path.join('./datasets/', dataset, filename).replace('\\', '/'), 'w') as fw:
        fw.write(str(len(tripleList)) + '\n')
        for triple in tripleList:
            fw.write(str(triple.h) + '\t' + str(triple.t) + '\t' + str(triple.r) + '\n')

emptyTriple = Triple(0, 0, 0)

def getRel(triple):
    return triple.r

# Gets the number of entities/relations/triples
def getAnythingTotal(inPath, fileName):  # read the total number in line 1
    with open(os.path.join(inPath, fileName).replace('\\', '/'), 'r') as fr:
        for line in fr:
            return int(line)

def loadTriple(inPath, fileName):
    with open(os.path.join(inPath, fileName).replace('\\', '/'), 'r') as fr:
        i = 0
        tripleList = []
        for line in fr:
            if i == 0:
                tripleTotal = int(line)
                i += 1
            else:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[1])
                rel = int(line_split[2])
                tripleList.append(Triple(head, tail, rel))

    tripleDict = {}
    for triple in tripleList:
        tripleDict[(triple.h, triple.t, triple.r)] = True

    return tripleTotal, tripleList, tripleDict


def which_loss_type(num):
    if num == 0:
        return loss.marginLoss
    elif num == 1:
        return loss.EMLoss
    elif num == 2:
        return loss.WGANLoss
    elif num == 3:
        return nn.MSELoss

# Split the tripleList into #num_batches batches
def getBatchList(tripleList, num_batches):
	batchSize = len(tripleList) // num_batches
	batchList = [0] * num_batches
	for i in range(num_batches - 1):
		batchList[i] = tripleList[i * batchSize : (i + 1) * batchSize]
	batchList[num_batches - 1] = tripleList[(num_batches - 1) * batchSize : ]
	return batchList


# Sample a batch of #batchSize triples from tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# without checking whether false negative samples exist.
def getBatch_raw_random(tripleList, batchSize, entityTotal):
	oldTripleList = random.sample(tripleList, batchSize)
	newTripleList = [corrupt_head_raw(triple, entityTotal) if random.random() < 0.5
		else corrupt_tail_raw(triple, entityTotal) for triple in oldTripleList]
	ph, pt ,pr = getThreeElements(oldTripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr

# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# without checking whether false negative samples exist.
def getBatch_raw_all(tripleList, entityTotal):
	newTripleList = [corrupt_head_raw(triple, entityTotal) if random.random() < 0.5
		else corrupt_tail_raw(triple, entityTotal) for triple in tripleList]
	ph, pt ,pr = getThreeElements(tripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr

# Sample a batch of #batchSize triples from tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# with checking whether false negative samples exist.
def getBatch_filter_random(tripleList, batchSize, entityTotal, tripleDict):
	if len(tripleList) < batchSize:
		batchSize = len(tripleList)
	oldTripleList = random.sample(tripleList, batchSize)
	newTripleList = [corrupt_head_filter(triple, entityTotal, tripleDict) if random.random() < 0.5
		else corrupt_tail_filter(triple, entityTotal, tripleDict) for triple in oldTripleList]
	ph, pt ,pr = getThreeElements(oldTripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr

# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# with checking whether false negative samples exist.
def getBatch_filter_all(tripleList, entityTotal, tripleDict):
	newTripleList = [corrupt_head_filter(triple, entityTotal, tripleDict) if random.random() < 0.5
		else corrupt_tail_filter(triple, entityTotal, tripleDict) for triple in tripleList]
	ph, pt ,pr = getThreeElements(tripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr


class marginLoss(nn.Module):
    def __init__(self):
        super(marginLoss, self).__init__()

    def forward(self, pos, neg, margin):
        zero_tensor = floatTensor(pos.size())
        zero_tensor.zero_()
        zero_tensor = autograd.Variable(zero_tensor)
        return torch.sum(torch.max(pos - neg + margin, zero_tensor))


def normLoss(embeddings, dim=1):
    norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
    return torch.sum(torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0]))))


def projection_DynMap_pytorch_samesize(entity_embedding, entity_projection, relation_projection):
	return entity_embedding + torch.sum(entity_embedding * entity_projection, dim=1, keepdim=True) * relation_projection

class DynMapPretrainModelSameSize(nn.Module):
    def __init__(self, config):
        super(DynMapPretrainModelSameSize, self).__init__()
        self.dataset = config.dataset
        self.learning_rate = config.learning_rate
        self.early_stopping_round = config.early_stopping_round
        self.L1_flag = config.L1_flag
        self.filter = config.filter
        self.embedding_size = config.embedding_size
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.batch_size = config.batch_size

        # with open('./transE_%s_%s_best.pkl' % (config.dataset, str(config.embedding_size)), 'rb') as fr:
        #     ent_embeddings_list = pickle.load(fr)
        #     rel_embeddings_list = pickle.load(fr)
        with open('./model/' + config.dataset + '_transE_pytorch/' + 'transE_pytorch_%s_%s_best.pkl' % (config.dataset, str(config.embedding_size)), 'rb+') as fr:
            T_out = pickle.load(fr)
            ent_embeddings_list = T_out.ent_embeddings
            rel_embeddings_list = T_out.rel_embeddings
            input_ent = torch.cuda.LongTensor(range(ent_embeddings_list.num_embeddings)) # change embedding to torch
            input_rel = torch.cuda.LongTensor(range(rel_embeddings_list.num_embeddings))

        ent_weight = floatTensor(ent_embeddings_list(input_ent))
        rel_weight = floatTensor(rel_embeddings_list(input_rel))
        # ent_weight = floatTensor(ent_embeddings_list)
        # rel_weight = floatTensor(rel_embeddings_list)
        ent_proj_weight = floatTensor(self.entity_total, self.embedding_size)
        rel_proj_weight = floatTensor(self.relation_total, self.embedding_size)
        ent_proj_weight.zero_()
        rel_proj_weight.zero_()

        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_proj_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_proj_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.ent_proj_embeddings.weight = nn.Parameter(ent_proj_weight)
        self.rel_proj_embeddings.weight = nn.Parameter(rel_proj_weight)

    def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_h_proj = self.ent_proj_embeddings(pos_h)
        pos_t_proj = self.ent_proj_embeddings(pos_t)
        pos_r_proj = self.rel_proj_embeddings(pos_r)

        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)
        neg_r_e = self.rel_embeddings(neg_r)
        neg_h_proj = self.ent_proj_embeddings(neg_h)
        neg_t_proj = self.ent_proj_embeddings(neg_t)
        neg_r_proj = self.rel_proj_embeddings(neg_r)

        pos_h_e = projection_DynMap_pytorch_samesize(pos_h_e, pos_h_proj, pos_r_proj)
        pos_t_e = projection_DynMap_pytorch_samesize(pos_t_e, pos_t_proj, pos_r_proj)
        neg_h_e = projection_DynMap_pytorch_samesize(neg_h_e, neg_h_proj, neg_r_proj)
        neg_t_e = projection_DynMap_pytorch_samesize(neg_t_e, neg_t_proj, neg_r_proj)

        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
        return pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e
