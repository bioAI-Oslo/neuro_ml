import torch_geometric.data.batch


def batch_to_device(batch, device):
    if isinstance(batch, torch_geometric.data.batch.Batch):
        return batch.x.to(device), batch.edge_index.to(device), batch.y.to(device)
    else:
        x, inputs, y = batch
        x = x.to(device)
        inputs = tuple(inp.to(device) for inp in inputs)
        y = y.to(device)
        return x, inputs, y
