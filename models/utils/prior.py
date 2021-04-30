import torch


class PriorQueue(object):
    def __init__(self, max_size=5):
        self.depths = None
        self.confs = None
        self.proj_matrices = None
        self.max_size = max_size
        self.length = 0

    def size(self):
        return self.length

    def update(self, depth, conf, proj_matrix):
        if self.depths is None:
            self.depths = depth.unsqueeze(1)
            self.confs = conf.unsqueeze(1)
            self.proj_matrices = proj_matrix.unsqueeze(1)
        else:
            cur_size = self.max_size - 1
            self.depths = torch.cat((depth.unsqueeze(1), self.depths[:, :cur_size, ...]), dim=1)
            self.confs = torch.cat((conf.unsqueeze(1), self.confs[:, :cur_size, ...]), dim=1)
            self.proj_matrices = torch.cat((proj_matrix.unsqueeze(1), self.proj_matrices[:, :cur_size, ...]), dim=1)

        self.length = self.depths.size(1)

    def get(self):
        return self.depths, self.confs, self.proj_matrices

    def reset(self):
        self.depths, self.confs, self.proj_matrices = None, None, None

    def is_full(self):
        return self.length == self.max_size
