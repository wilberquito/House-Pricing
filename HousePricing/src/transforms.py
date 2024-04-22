import torch


class ToTensor(object):
    """Transform sample into Tensor"""

    def __call__(self, sample):

        inputs = sample["inputs"]
        ids = sample["id"]

        inputs = torch.tensor(inputs, dtype=torch.float32)
        ids = torch.tensor(ids, dtype=torch.int32)

        if "target" in sample:
            target = sample["target"]
            target = torch.tensor(target, dtype=torch.float32)
            return {"id": ids, "inputs": inputs, "target": target}
        else:
            return {"id": ids, "inputs": inputs}
