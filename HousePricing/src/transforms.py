import torch


class ToTensor(object):
    """Transform sample into Tensor"""

    def __call__(self, sample):
        if "target" in sample:
            inputs, target = sample["inputs"], sample["target"]
            inputs = torch.tensor(inputs, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
            return {"inputs": inputs, "target": target}
        else:
            inputs = sample["inputs"]
            inputs = torch.tensor(inputs, dtype=torch.float32)
            return {"inputs": inputs}
