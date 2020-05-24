import pickle
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


# Loss
def get_batched_loss(data_loader, model, loss_func, prior_only=None, loss_triples=True):
    """
    Gets loss in a batched fashion.
    Input is data loader, model to produce output and loss function
    Assuming loss output is VLB, reconstruct_loss, KL
    """
    losses = [[], [], []] if loss_triples else []  # [VLB, Reconstruction Loss, KL] or [Loss]
    for batch in data_loader:
        out = model(batch) if prior_only is None else model(batch, prior_only)
        loss = loss_func(batch, out)

        if loss_triples:
            losses[0].append(loss[0].cpu().item())
            losses[1].append(loss[1].cpu().item())
            losses[2].append(loss[2].cpu().item())
        else:
            losses.append(loss.cpu().item())

    losses = np.array(losses)

    if not loss_triples:
        return np.mean(losses)
    return np.mean(losses[0]), np.mean(losses[1]), np.mean(losses[2])


# L2 Distance squared
l2_dist = lambda x, y: (x - y) ** 2


# Helper functions for 2D datasets
def plot_vae_training_plot(train_losses, test_losses, title):
    elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label='-elbo_train')
    plt.plot(x_train, recon_train, label='recon_loss_train')
    plt.plot(x_train, kl_train, label='kl_loss_train')
    plt.plot(x_test, elbo_test, label='-elbo_test')
    plt.plot(x_test, recon_test, label='recon_loss_test')
    plt.plot(x_test, kl_test, label='kl_loss_test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


def sample_data_1_a(count):
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + (rand.randn(count, 2) * [[5.0, 1.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


def sample_data_2_a(count):
    rand = np.random.RandomState(0)
    return [[-1.0, 2.0]] + (rand.randn(count, 2) * [[1.0, 5.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


def sample_data_1_b(count):
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + rand.randn(count, 2) * [[5.0, 1.0]]


def sample_data_2_b(count):
    rand = np.random.RandomState(0)
    return [[-1.0, 2.0]] + rand.randn(count, 2) * [[1.0, 5.0]]


def sample_2d_data(dset_id):
    assert dset_id in [1, 2]

    if dset_id == 1:
        dset_fn = sample_data_1_a
    else:
        dset_fn = sample_data_2_a

    train_data, test_data = dset_fn(10000), dset_fn(2500)
    return train_data.astype('float32'), test_data.astype('float32')


def show_results_2d_data(dset_id, fn):
    train_data, test_data = sample_2d_data(dset_id)
    train_losses, test_losses, samples_noise, samples_nonoise = fn(train_data, test_data)

    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')

    plot_vae_training_plot(train_losses, test_losses, f'Dataset {dset_id} Train Plot')
    save_scatter_2d(samples_noise, title='Samples with Decoder Noise')
    save_scatter_2d(samples_nonoise, title='Samples without Decoder Noise')


# Helper functions for multidimensional datasets
def show_results_images_vae(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = "data"
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    train_losses, test_losses, samples, reconstructions, interpolations = fn(train_data, test_data)
    samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype(
        'float32'), interpolations.astype('float32')
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'Dataset {dset_id} Train Plot')
    show_samples(samples, title=f'Dataset {dset_id} Samples')
    show_samples(reconstructions, title=f'Dataset {dset_id} Reconstructions')
    show_samples(interpolations, title=f'Dataset {dset_id} Interpolations')


def show_results_images_vqvae(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = "data"
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(
        train_data, test_data, dset_id)
    samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')

    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')

    show_training_plot(vqvae_train_losses, vqvae_test_losses, f'Dataset {dset_id} VQ-VAE Train Plot')
    show_training_plot(pixelcnn_train_losses, pixelcnn_test_losses, f'Dataset {dset_id} PixelCNN Prior Train Plot')
    show_samples(samples, title=f'Dataset {dset_id} Samples')
    show_samples(reconstructions, title=f'Dataset {dset_id} Reconstructions')


def show_results_images_vqvae2(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = "data"
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(
        train_data, test_data, dset_id)
    samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
    show_training_plot(vqvae_train_losses, vqvae_test_losses, f'Dataset {dset_id} VQ-VAE Train Plot')
    show_training_plot(pixelcnn_train_losses, pixelcnn_test_losses,
                       f'Dataset {dset_id} PixelCNN Prior Train Plot')
    show_samples(samples, title=f'Dataset {dset_id} Samples')
    show_samples(reconstructions, title=f'Dataset {dset_id} Reconstructions')


def save_scatter_2d(data, title):
    plt.figure()
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1])


# General utils
def show_training_plot(train_losses, test_losses, title):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.show()


def load_pickled_data(fname, include_labels=False):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    train_data, test_data = data['train'], data['test']
    if 'mnist.pkl' in fname or 'shapes.pkl' in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype('uint8')
        test_data = (test_data > 127.5).astype('uint8')
    if 'celeb.pkl' in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data['train_labels'], data['test_labels']
    return train_data, test_data


def show_samples(samples, nrow=10, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
