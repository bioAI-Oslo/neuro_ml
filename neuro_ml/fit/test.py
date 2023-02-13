from neuro_ml.dataset import create_test_dataloader
import torch
import torch
from tqdm import tqdm
from neuro_ml.fit.batch_to_device import batch_to_device
from zenlog import log


def test_model(model, epoch, dataset_params, model_params, model_is_classifier, device):
    test_loader = create_test_dataloader(
        model.DATASET,
        dataset_params,
        model_is_classifier,
    )

    model = model(model_params)
    model.load_state_dict(torch.load(f"models/{model.NAME}/{epoch}.pt"))
    model.to(device)
    criterion = torch.nn.MSELoss()

    model.eval()
    avg_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            (t := tqdm(test_loader, leave=False, colour="#FF5666"))
        ):
            x, other_inputs, y = batch_to_device(batch, device)

            y_hat = model(x, other_inputs).flatten()

            loss = criterion(y_hat, y)

            avg_loss += loss.item()
    avg_test_loss = avg_loss / len(test_loader)
    log.info(f"Avg test loss: {avg_test_loss}")
