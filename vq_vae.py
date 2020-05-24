import time

import torch.nn as nn

from utils import *
from vae import Encoder, Decoder


class MaskedConv2d(nn.Conv2d):
    """
    Class extending nn.Conv2d to use masks.
    """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding=0):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding)
        self.register_buffer('mask', torch.ones(out_channels, in_channels, kernel_size, kernel_size).float())

        # _, depth, height, width = self.weight.size()
        h, w = kernel_size, kernel_size

        if mask_type == 'A':
            self.mask[:, :, h // 2, w // 2:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0
        else:
            self.mask[:, :, h // 2, w // 2 + 1:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ResidualMaskedConv2d(nn.Module):
    """
    Residual Links between MaskedConv2d-layers
    As described in Figure 5 in "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            MaskedConv2d('B', in_dim, in_dim // 2, kernel_size=1, padding=0),
            nn.ReLU(),
            MaskedConv2d('B', in_dim // 2, in_dim // 2, kernel_size=7, padding=3),
            nn.ReLU(),
            MaskedConv2d('B', in_dim // 2, in_dim, kernel_size=1, padding=0),
            nn.ReLU())

    def forward(self, x):
        return self.net(x) + x


class PixelRCNN(nn.Module):
    """
    Pixel R-CNN-class using residual blocks from "Pixel Recurrent Neural Networks" by Aaron van den Oord et. al.
    """

    def __init__(self, in_channels=64, K=128, conv_filters=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=K, embedding_dim=in_channels)

        self.net = nn.Sequential(
            # A 7x7 A-type convolution with batch norm
            MaskedConv2d('A', in_channels, conv_filters, kernel_size=7, padding=3),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            # 10 Residual masked convolutons with batch norms
            ResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            ResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            ResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            ResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            ResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            ResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            ResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            ResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            ResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            ResidualMaskedConv2d(conv_filters), nn.BatchNorm2d(conv_filters), nn.ReLU(),
            nn.Conv2d(conv_filters, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, K, kernel_size=1)).cuda()

    def forward(self, x):
        x = self.embedding(x).permute(0, 3, 1, 2).contiguous()
        return self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.K = K
        self.D = D
        self.codebook = nn.Embedding(num_embeddings=K, embedding_dim=D)
        self.codebook.weight.data.uniform_(-1 / K, 1 / K)

    def forward(self, z_e):
        N, D, H, W = z_e.size()
        z_e = z_e.permute(0, 2, 3, 1).contiguous()  # (N, D, H/4, W/4) --> (N, H/4, W/4, D)
        z_e = z_e.view(-1, self.D)

        weights = self.codebook.weight

        # Sampling nearest embeddings
        distances = l2_dist(z_e[:, None], weights[None, :])
        q = distances.sum(dim=2).min(dim=1)[1]  # Using [1] to get indices instead of values after min-function
        z_q = weights[q]

        # (N, H/4, W/4, D) -> (N, D, H/4, W/4)
        z_q = z_q.view(N, H, W, D)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        q = q.long().view(N, H, W)

        # Class vector q, and code vector z_q
        return q, z_q


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim), nn.ReLU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x) + x


class Encoder(nn.Module):
    def __init__(self, D=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=D, kernel_size=4, stride=2, padding=1),  # 16 x 16
            nn.BatchNorm2d(D), nn.ReLU(),
            nn.Conv2d(in_channels=D, out_channels=D, kernel_size=4, stride=2, padding=1),  # 8 x 8
            ResidualBlock(D),
            ResidualBlock(D)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, D=256):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(D),
            ResidualBlock(D),
            nn.BatchNorm2d(D), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=D, out_channels=D, kernel_size=4, stride=2, padding=1),  # 16 x 16
            nn.BatchNorm2d(D), nn.ReLU(),
            nn.ConvTranspose2d(in_channels=D, out_channels=3, kernel_size=4, stride=2, padding=1),  # 32 x 32
        )

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):
    def __init__(self, K=128, D=256):
        super().__init__()
        self.K = K
        self.D = D

        self.codebook = VectorQuantizer(K=K, D=D)
        self.encoder = Encoder(D=D)
        self.decoder = Decoder(D=D)
        self.pixelcnn_prior = PixelRCNN(K=K)
        self.pixelcnn_loss_fct = nn.CrossEntropyLoss()

    def forward(self, x, prior_only=False):
        z_e = self.encoder(x)
        q, z_q = self.codebook(z_e)
        if prior_only: return q, self.pixelcnn_prior(q)

        z_e_altered = (z_q - z_e).detach() + z_e
        x_reconstructed = self.decoder(z_e_altered)

        return x_reconstructed, z_e, z_q

    def get_pixelcnn_prior_loss(self, x, output):
        q, logit_probs = output
        return self.pixelcnn_loss_fct(logit_probs, q)

    def get_vae_loss(self, x, output):
        N, C, H, W = x.shape
        x_reconstructed, z_e, z_q = output

        reconstruction_loss = l2_dist(x, x_reconstructed).sum() / (N * H * W * C)
        vq_loss = l2_dist(z_e.detach(), z_q).sum() / (N * H * W * C)
        commitment_loss = l2_dist(z_e, z_q.detach()).sum() / (N * H * W * C)

        return reconstruction_loss + vq_loss + commitment_loss


def train_vq_vae(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
                used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of VQ-VAE train losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of VQ-VAE train losses evaluated once at initialization and after each epoch
    - a (# of training iterations,) numpy array of PixelCNN prior train losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of PixelCNN prior train losses evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples (an equal number from each class) with values in {0, ... 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
        FROM THE TEST SET with values in [0, 255]
    """

    start_time = time.time()
    N, H, W, C = train_data.shape

    def dequantize(x, dequantize=True, reverse=False, alpha=.1):
        with torch.no_grad():
            if reverse:
                return torch.ceil_(torch.sigmoid(x) * 255)

            if dequantize:
                x += torch.zeros_like(x).uniform_(0, 1)

            p = alpha / 2 + (1 - alpha) * x / 256
            return torch.log(p) - torch.log(1 - p)

    batch_size = 128
    dataset_params = {
        'batch_size': batch_size,
        'shuffle': True
    }

    print("[INFO] Creating model and data loaders")
    train_data = torch.from_numpy(np.transpose(train_data, [0, 3, 1, 2])).float().cuda()
    test_data = torch.from_numpy(np.transpose(test_data, [0, 3, 1, 2])).float().cuda()
    train_loader = torch.utils.data.DataLoader(dequantize(train_data), **dataset_params)
    test_loader = torch.utils.data.DataLoader(dequantize(test_data), **dataset_params)

    # Model
    n_epochs_vae = 20 if dset_id == 1 else 40
    n_epochs_cnn = 15 if dset_id == 1 else 30
    lr = 1e-3
    K, D = 128, 256
    vq_vae = VQVAE(K=K, D=D).cuda()

    optimizer = torch.optim.Adam(vq_vae.parameters(), lr=lr)

    # Training
    def train(model, no_epochs, prior_only=False):
        """
        Trains model and returns training and test losses
        """
        model_name = "VQ-VAE" if not prior_only else "Pixel-RCNN"
        print(f"[INFO] Training {model_name}")

        loss_fct = vq_vae.get_vae_loss if not prior_only else vq_vae.get_pixelcnn_prior_loss

        train_losses = []
        test_losses = [get_batched_loss(test_loader, vq_vae, loss_fct, prior_only, loss_triples=False)]

        for epoch in range(no_epochs):
            epoch_start = time.time()
            for batch in train_loader:
                optimizer.zero_grad()
                output = vq_vae(batch, prior_only)
                loss = loss_fct(batch, output)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.cpu().item())

            test_loss = get_batched_loss(test_loader, vq_vae, loss_fct, prior_only, loss_triples=False)
            test_losses.append(test_loss)

            print(
                f"[{100*(epoch+1)/no_epochs:.2f}%] Epoch {epoch + 1} - Test loss: {test_loss:.2f} - Time elapsed: {time.time() - epoch_start:.2f}")

        return np.array(train_losses), np.array(test_losses)

    vq_vae_train_losses, vq_vae_test_losses = train(vq_vae, n_epochs_vae)
    pixel_cnn_train_losses, pixel_cnn_test_losses = train(vq_vae, n_epochs_cnn, prior_only=True)

    # Utility methods
    def sample(noise=False, no_samples=100):
        shape = (no_samples, H // 4, W // 4)
        q_samples = torch.zeros(size=shape).long().cuda()

        for i in range(H // 4):
            for j in range(W // 4):
                out = vq_vae.pixelcnn_prior(q_samples)
                proba = F.softmax(out, dim=1)
                q_samples[:, i, j] = torch.multinomial(proba[:, :, i, j], 1).squeeze().float()

        z_q_samples = vq_vae.codebook.codebook.weight[q_samples.view(-1, D)].float()

        # Shape (N, W, H, D) -> (N, D, W, H)
        z_q_samples = z_q_samples.view(shape + (D,))
        z_q_samples = z_q_samples.permute(0, 3, 1, 2).contiguous()

        x_samples = vq_vae.decoder(z_q_samples)
        samples = dequantize(x_samples, reverse=True).detach().cpu().numpy()
        return np.transpose(samples, [0, 2, 3, 1])  # Get it in (N, H, W, C)

    def reconstruction_pairs(no_reconstructions=50):
        """
        Creating reconstruction pairs (x, x') where x is the original image and x' is the decoder-output
        """
        x_original = test_data[:no_reconstructions]  # .float().cuda()
        x_dequantized = dequantize(x_original, dequantize=False)

        x_reconstructed = vq_vae(x_dequantized)[0]
        x_reconstructed = dequantize(x_reconstructed.float(), reverse=True)

        pairs = torch.zeros_like(torch.cat((x_original, x_reconstructed), dim=0)).detach().cpu().numpy()
        pairs[::2] = x_original.detach().cpu().numpy()
        pairs[1::2] = x_reconstructed.detach().cpu().numpy()

        pairs = np.clip(pairs, 0, 255)
        return np.transpose(pairs, [0, 2, 3, 1])  # Get it in (N, H, W, C)

    torch.cuda.empty_cache()
    vq_vae.eval()
    with torch.no_grad():
        print("[INFO] Sampling images")
        samples = sample(noise=False)

        print("[INFO] Creating reconstructing pairs")
        pairs = reconstruction_pairs()

    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")

    print("Samples", samples.shape)
    print("Pairs", pairs.shape)
    return vq_vae_train_losses, vq_vae_test_losses, pixel_cnn_train_losses, pixel_cnn_test_losses, samples, pairs


def train_and_show_results_svhn():
    """
    Trains VQ-VAE for images and display samples, reconstructions and interpolations, and training plot for SVHN Dataset
    """
    show_results_images_vqvae(1, train_vq_vae)


def train_and_show_results_cifar():
    """
    Trains VQ-VAE for images and display samples, reconstructions and interpolations, and training plot for CIFAR-10 Dataset
    """
    show_results_images_vqvae(2, train_vq_vae)
