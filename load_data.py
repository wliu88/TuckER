import os
from collections import defaultdict

class Data:

    def __init__(self, data_dir="data/FB15k-237/", reverse=False, path_dataset=False):
        self.train_data_pn = None
        self.test_data_pn = None
        self.valid_data_pn = None
        if not path_dataset:
            self.train_data = self.load_data(data_dir, "train", reverse=reverse)
            self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
            self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        else:
            self.train_data, self.test_data, self.valid_data, \
            self.train_data_pn, self.test_data_pn, self.valid_data_pn = self.load_path_data(data_dir, reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]
        print("Number of entities: {}".format(len(self.entities)))

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

    def load_path_data(self, data_dir, reverse):
        """
        This function loads data in the path-based format
        :param data_dir:
        :return:
        """
        test_rel_filename = os.path.join(data_dir, "split/relations_to_run.tsv")
        assert os.path.exists(test_rel_filename)

        relations = []
        with open(test_rel_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    relations.append(line)

        def get_triples(rel, filename):
            pos_triples = []
            labeled_triples = []
            with open(filename, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        s, t, label = line.split("\t")
                        label = int(label)
                        labeled_triples.append([s, rel, t, label])
                        if label == 1:
                            pos_triples.append([s, rel, t])
            return labeled_triples, pos_triples

        train_data = []
        valid_data = []
        test_data = []
        train_data_pn = defaultdict(list)
        valid_data_pn = defaultdict(list)
        test_data_pn = defaultdict(list)
        for rel in relations:
            rel_folder = os.path.join(data_dir, "split", rel)
            train_filename = os.path.join(rel_folder, "training.tsv")
            test_filename = os.path.join(rel_folder, "testing.tsv")
            valid_filename = os.path.join(rel_folder, "development.tsv")
            rel_train_data_pn, rel_train_data = get_triples(rel, train_filename)
            rel_test_data_pn, rel_test_data = get_triples(rel, test_filename)
            rel_valid_data_pn, rel_valid_data = get_triples(rel, valid_filename)
            train_data.extend(rel_train_data)
            test_data.extend(rel_test_data)
            valid_data.extend(rel_valid_data)
            train_data_pn[rel] = rel_train_data_pn
            test_data_pn[rel] = rel_test_data_pn
            valid_data_pn[rel] = rel_valid_data_pn

        if reverse:
            train_data += [[i[2], i[1]+"_reverse", i[0]] for i in train_data]
            test_data += [[i[2], i[1] + "_reverse", i[0]] for i in test_data]
            valid_data += [[i[2], i[1] + "_reverse", i[0]] for i in valid_data]

        return train_data, test_data, valid_data, train_data_pn, test_data_pn, valid_data_pn
