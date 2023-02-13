import torch
from tqdm import tqdm
from neuro_ml.fit.batch_to_device import batch_to_device
import sklearn.metrics


def val(model, data_loader, criterion, device):
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        # For each batch in the data loader calculate the validation loss
        for batch_idx, batch in enumerate(
            (t := tqdm(data_loader, leave=False, colour="#FF5666"))
        ):
            x, other_inputs, y = batch_to_device(batch, device)

            y_hat = model(x, other_inputs)

            loss = criterion(y_hat, y)

            avg_loss += loss.item()
            t.set_description(f"Val loss: {loss:.4f}/({avg_loss/(batch_idx + 1):.4f})")

    return avg_loss / len(data_loader)
