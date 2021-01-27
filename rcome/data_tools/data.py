import torch
import io
import math
class RawDataloader(object):
    def __init__(self, dataset, batch_size=200, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __iter__(self):
        if(self.shuffle):
            self.indexed = torch.randperm(len(self.dataset))
        self.indexed = torch.arange(len(self.dataset))
        self.ci = 0
        return self

    def __len__(self):
        return len(self.dataset)//self.batch_size +  (0 if(len(self.dataset)%self.batch_size == 0) else 1)

    def _next_index(self):
        if(self.ci >= (len(self))):
            raise StopIteration
        value = self.dataset[self.indexed[self.ci*self.batch_size: min((self.ci+1)*self.batch_size, len(self.dataset)) ]]

        self.ci += 1
        return value
    def __next__(self):
        return self._next_index()

class DeviceMapper(object):
    def __init__(self, device):
        self.device = device
    def __call__(self, tensor):
        if(self.device == "cuda"):
            return tensor.cuda()
        elif(self.device == "cpu"):
            return tensor.cpu()
        else:
            return tensor.new(tensor, device=self.device)

class Normalizer(object):
    def __init__(self, index=1):
        self.index = 1

    def __call__(self, tensor):
        tensor[:,self.index]/=tensor[:,self.index].sum()
        return tensor

class PadCollate(object):
    def __init__(self, dim=0):
        self.dim = dim
    @staticmethod
    def pad_tensor(vec, pad, dim):
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, vec.new(*pad_size).zero_()], dim=dim)

    def pad_collate(self, batch):
        mbatch = []
        for i in range(len(batch[0])):
            try:
                x_max_len = max([x[i].shape[self.dim] for x in batch])
                cbatch = [PadCollate.pad_tensor(x[i], pad=x_max_len, dim=self.dim) for x in batch]
                mbatch.append(torch.stack([x for x in cbatch], dim=0))
            except:
                print(batch)
                print("Padding error for tuple "+str(i))
        return tuple(mbatch)

    def __call__(self, batch):
        return self.pad_collate(batch)

def save_dataset_xml(file_path, dataset):
    with io.open(file_path,"w") as m_file:
        X, Y = dataset.X,dataset.Y
        for (x, y) in zip(X,Y):
            c_line = ""
            for l in y:
                c_line+=str(int(l))+","
            c_line = c_line[:-1]+" "
            for c in x:
                c_line+=str(int(c[0].item()))+":"+str(int(c[1].item()))+" "
            c_line = c_line[:-1]+"\n"
            m_file.write(c_line)



def collate_fn_simple(my_list):
    return torch.cat(my_list)