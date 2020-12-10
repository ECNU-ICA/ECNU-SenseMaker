import torch.utils.data


# %%
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        lenx = len(self.x)
        leny = len(self.y)
        assert lenx == leny
        return lenx


# %%
class MyDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.idx = 0
        self.dataset_len = len(self.dataset)

        # if shuffle:
        #     self.dataset = list(self.dataset)
        #     np.random.shuffle(self.dataset)

    def __getitem__(self, item):
        # 总长度
        with_batch_len = (self.dataset_len - 1) // self.batch_size + 1
        if isinstance(item, slice):
            # 是否是切片操作
            res = []
            start = item.start if item.start else 0
            stop = item.stop if item.stop else with_batch_len
            if start > with_batch_len or stop > with_batch_len or start < 0 or stop < 0:
                raise Exception('start >= with_batch_len or stop >= with_batch_len or start < 0 or stop < 0 ', start,
                                stop)
            for i in range(start, stop):
                res.append(self.dataset[i * self.batch_size:(i + 1) * self.batch_size])
            return res
        else:
            if item >= with_batch_len:
                raise Exception('item >= self.dataset_len')
            return self.dataset[item * self.batch_size:(item + 1) * self.batch_size]

    def __iter__(self):
        return self

    def __len__(self):
        return (self.dataset_len - 1) // self.batch_size + 1

    def __next__(self):
        if self.idx >= self.dataset_len:
            self.idx = 0
            raise StopIteration
        res = self.dataset[self.idx:self.idx + self.batch_size]
        self.idx += self.batch_size
        return res
