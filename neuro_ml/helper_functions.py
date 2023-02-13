def find_mean_variance(data_loader):
    mean = 0
    variance = 0
    for batch in data_loader:
        y = batch[-1]
        mean += y.mean().sum()

    mean /= len(data_loader)

    for batch in data_loader:
        y = batch[-1]
        variance += ((y - mean).pow(2)).sum() / len(y)

    variance /= len(data_loader)

    return mean, variance

def _train_val_test_split_filenames(all_filenames, train_val_test_size, seed):
    assert sum(train_val_test_size) == 1, "Sum of the train/val/test size must be 0"
    assert all(
        0 <= split_size for split_size in train_val_test_size
    ), "All objects of train/val/test size must be non-zero"

    train_filenames, val_filenames, test_filenames = random_split(
        all_filenames,
        [
            int(len(all_filenames) * train_val_test_size[0]),
            int(len(all_filenames) * train_val_test_size[1]),
            len(all_filenames)
            - int(len(all_filenames) * train_val_test_size[0])
            - int(len(all_filenames) * train_val_test_size[1]),
        ],
        generator=torch.Generator().manual_seed(seed),
    )

    return train_filenames, val_filenames, test_filenames

