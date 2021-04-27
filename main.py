from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import os


def compute_scores(score_instances):
    """
    Given a list of scored instances [(stuff, label, score)], this method computes Average Precision, Reciprocal Rank,
    and Accuracy.
    AP is none if no positive instance is in scored instances.

    :param score_instances:
    :return:
    """
    # sort score instances based on score from highest to lowest
    sorted_score_instances = sorted(score_instances, key=lambda score_instance: score_instance[2])[::-1]
    total_predictions = 0.0
    total_correct_pos = 0.0
    total_precisions = []
    first_correct = -1
    total_correct = 0.0
    for stuff, label, score in sorted_score_instances:
        # print(stuff, label, score)
        if abs(score - label) < 0.5:
            total_correct += 1
        total_predictions += 1
        # debug
        if label > 0:
        # if label == 1:
            total_correct_pos += 1
            if first_correct == -1:
                first_correct = total_predictions
            total_precisions.append(total_correct_pos/total_predictions)
    ap = sum(total_precisions) * 1.0 / len(total_precisions) if len(total_precisions) > 0 else None
    rr = 0.0 if first_correct == -1 else 1.0 / first_correct
    acc = total_correct / len(score_instances)
    return ap, rr, acc

    
class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        
    def get_data_idxs(self, data):
        """
        Vectorize data
        :param data:
        :return:
        """
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        """
        Build a mapping from (source entity, relation) to target entity
        :param data:
        :return:
        """
        er_vocab = defaultdict(list)
        # print("get_er_vocab data size:", len(data))
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        """
        Create training batches and also sample negative examples (all triples that are not in training are
        considered as negative)

        label for each data point in the batch is a vector of dimension (1, number of entities). Each label
        corresponds to a pair of (source entity, relation). All target entities that share the (source entity, relation)
        have value 1 in the vector. Other entities are 0.

        :param er_vocab:
        :param er_vocab_pairs: list(er_vocab.keys())
        :param idx:
        :return:
        """
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate(self, model, data, data_pn=None, print_to_file=False):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        # print("Number of data points: %d" % len(test_data_idxs))
        
        for i in range(0, len(test_data_idxs), self.batch_size):

            # data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            data_batch = np.array(test_data_idxs[i:i+self.batch_size])

            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            # print(data_batch.shape)
            # predictions: [batch_size, num_entities]
            # er_vocab: {(e1, r): [e2 if (e1, r, e2)=True] }
            # data_batch: [batch_size, 3]

            # removing all positive examples from train/test/val that is not the query triple
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                # print("filt size", len(filt))
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            # print(sort_idxs[0])
            # print(e2_idx[0])
            # print(np.where(sort_idxs[j]==e2_idx[j].item()))

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        # Important: added calculation of relation-wise AP score
        if data_pn:

            if print_to_file:
                results_dir = os.path.join(os.getcwd(), "results")
                if not os.path.exists(results_dir):
                    os.mkdir(results_dir)

            for rel in data_pn:
                rel_data_pn = data_pn[rel]

                # vectorize data
                rel_data_pn = [(self.entity_idxs[rel_data_pn[i][0]], self.relation_idxs[rel_data_pn[i][1]],
                                self.entity_idxs[rel_data_pn[i][2]], rel_data_pn[i][3]) for i in range(len(rel_data_pn))]

                rel_score_instances = []
                for i in range(0, len(rel_data_pn), self.batch_size):
                    data_batch_raw = rel_data_pn[i:i + self.batch_size]
                    data_batch = np.array(data_batch_raw)
                    e1_idx = torch.tensor(data_batch[:, 0])
                    r_idx = torch.tensor(data_batch[:, 1])
                    e2_idx = torch.tensor(data_batch[:, 2])
                    row_indices = torch.arange(e2_idx.shape[0])
                    if self.cuda:
                        e1_idx = e1_idx.cuda()
                        r_idx = r_idx.cuda()
                        e2_idx = e2_idx.cuda()
                        row_indices = row_indices.cuda()
                    # [batch_size, num_entities]
                    predictions = model.forward(e1_idx, r_idx)
                    # [batch_size, 1]
                    predictions = predictions[row_indices, e2_idx]
                    predictions = predictions.cpu().numpy()
                    batch_score_instances = [(triple, triple[3], score) for triple, score in zip(data_batch_raw, predictions)]
                    rel_score_instances.extend(batch_score_instances)

                # for si in rel_score_instances:
                #     print(si)

                ap, rr, acc = compute_scores(rel_score_instances)
                print("Rel {} AP: {}".format(rel, ap))
                # print("Rel {} ACC: {}".format(rel, acc))

                if print_to_file:
                    rel_filename = os.path.join(results_dir, "{}.txt".format(rel))
                    with open(rel_filename, "w") as fh:
                        fh.write("Rel {} AP: {}\n\n".format(rel, ap))
                        fh.write("Subject\tRelation\tObject\tLabel\tPrediction\n")
                        for rsi in sorted(rel_score_instances, key=lambda score_instance: score_instance[2])[::-1]:
                            fh.write("{: <30}\t{: <10}\t{: <30}\t{: <5}\t{: <5}\n".format(self.idx2entity[rsi[0][0]],
                                                              self.idx2relation[rsi[0][1]],
                                                              self.idx2entity[rsi[0][2]],
                                                              rsi[1], rsi[2]))

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))

    def train_and_eval(self):
        print("\nTraining the TuckER model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        self.idx2entity = {self.entity_idxs[e]: e for e in self.entity_idxs}
        self.idx2relation = {self.relation_idxs[r]: r for r in self.relation_idxs}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("\nStarting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])  
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print("*"*30)
            print("epoch: {}".format(it))
            print("takes {} seconds".format(time.time()-start_train))
            print("avg loss: {}".format(np.mean(losses)))

            # evaluation
            model.eval()
            with torch.no_grad():
                print("Validation:")
                self.evaluate(model, d.valid_data, d.valid_data_pn)
                if not it%2:
                    print("Test:")
                    start_test = time.time()
                    if it == self.num_iterations:
                        self.evaluate(model, d.test_data, d.test_data_pn, print_to_file=True)
                    else:
                        self.evaluate(model, d.test_data, d.test_data_pn)
                    print(time.time()-start_test)

        self.model = model
           

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18, WN18RR or Robot.")
    parser.add_argument("--path_dataset", type=bool, default=False, nargs="?",
                        help="Whether the dataset is in path-based format")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True, path_dataset=args.path_dataset)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing)
    experiment.train_and_eval()

    print("Play around with the trained model")

    entities = d.entities
    relations = d.relations
    entity_idxs = experiment.entity_idxs
    relation_idxs = experiment.relation_idxs
    idx2entity = experiment.idx2entity
    model = experiment.model
    use_cuda = experiment.cuda
    
    # sort entities
    type2entities = {}
    for e in entities:
        t = e[0]
        if t == "a":
            typ = "action"
        elif t == "l":
            typ = "location"
        elif t == "m":
            typ = "material"
        elif t == "o":
            typ = "object"
        elif t == "r":
            typ = "room"
        elif t == "s":
            typ == "state" 
        if typ not in type2entities:
            type2entities[typ] = []
        type2entities[typ].append(e)

    while True:
        print("")
        print("="*100)
        print("")
        print("Choose a query relation from the following list:")
        print(relations)
        print("Note that the name of the relation suggests the correct type of the source and target entities")
        print("e.g., for relation ObjInLoc, a type-correct triple will be (o:apple.n.01, ObjInLoc, l:fridge.n.01)")
        while True:
            r = input("-->type the name of chosen relation: ")
            if r in relations:
                break
            else:
                print("input relation {} is not defined".format(r))
        print("")
        
        print("Choose a source entity from the following lists:")
        for typ in type2entities:
            print("{}: {}".format(typ, type2entities[typ]))
        while True:
            e1 = input("-->type the name of chosen entity: ")
            if e1 in entities:
                break
            else:
                print("input entity {} is not defined".format(r))
        print("")

        # e1 = 'r:bathroom.n.01'
        # r2 = 'r:kitchen.n.01'

        model.eval()
        with torch.no_grad():
            e1_idx = entity_idxs[e1]
            r_idx = relation_idxs[r]
            e1_idx = torch.tensor([e1_idx])
            r_idx = torch.tensor([r_idx])

            if use_cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()

            predictions = model.forward(e1_idx, r_idx)
            predictions = predictions.cpu().numpy()[0]
            score_instances = []
            for i, s in enumerate(predictions):
                score_instances.append((idx2entity[i], s))

            score_instances = sorted(score_instances, key=lambda x: x[1])[::-1]
            print("The model predicts the following 10 top answers:")
            for k in range(10):
                print("No.{} answer: {} (score: {})".format(k+1, score_instances[k][0], score_instances[k][1]))


        f = input("-->continue? Y/N ")
        if "n" in f.lower():
            break
            # batch_score_instances = [(triple, triple[3], score) for triple, score in zip(data_batch_raw, predictions)]
            # rel_score_instances.extend(batch_score_instances)
