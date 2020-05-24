import time

import torch.nn as nn

from utils import *
from vae import Encoder, Decoder
from vq_vae import MaskedConv2d, VQVAE


class LayerNorm(nn.LayerNorm):
    def __init__(self, color_conditioning, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_conditioning = color_conditioning

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        if self.color_conditioning:
            x = x.contiguous().view(*(x_shape[:-1] + (3, -1)))
        x = super().forward(x)
        if self.color_conditioning:
            x = x.view(*x_shape)
        return x.permute(0, 3, 1, 2).contiguous()


class StackLayerNorm(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.h_layer_norm = LayerNorm(False, n_filters)
        self.v_layer_norm = LayerNorm(False, n_filters)

    def forward(self, x):
        vx, hx = x.chunk(2, dim=1)
        vx, hx = self.v_layer_norm(vx), self.h_layer_norm(hx)
        return torch.cat((vx, hx), dim=1)


class GatedConv2d(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, k=7, padding=3):
        super().__init__()

        self.vertical = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=k,
                                  padding=padding, bias=False)
        self.horizontal = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, k),
                                    padding=(0, padding), bias=False)
        self.vtoh = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1,
                              bias=False)
        self.htoh = nn.Conv2d(out_channels, out_channels, kernel_size=1,
                              bias=False)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        # No need for special color condition masking here since we get to see everything
        self.vmask[:, :, k // 2 + 1:, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2 + 1:] = 0
        if mask_type == 'A':
            self.hmask[:, :, :, k // 2] = 0

    def down_shift(self, x):
        x = x[:, :, :-1, :]
        pad = nn.ZeroPad2d((0, 0, 1, 0))
        return pad(x)

    def forward(self, x):
        vx, hx = x.chunk(2, dim=1)

        self.vertical.weight.data *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx = self.vertical(vx)
        hx_new = self.horizontal(hx)
        # Allow horizontal stack to see information from vertical stack
        hx_new = hx_new + self.vtoh(self.down_shift(vx))

        # Gates
        vx_1, vx_2 = vx.chunk(2, dim=1)
        vx = torch.tanh(vx_1) * torch.sigmoid(vx_2)

        hx_1, hx_2 = hx_new.chunk(2, dim=1)
        hx_new = torch.tanh(hx_1) * torch.sigmoid(hx_2)
        hx_new = self.htoh(hx_new)
        hx = hx + hx_new

        return torch.cat((vx, hx), dim=1)


class GatedPixelCNN(nn.Module):
    """
    The following Gated PixelCNN is taken from class material given on Piazza
    """

    def __init__(self, K, in_channels=64, n_layers=15, n_filters=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=K, embedding_dim=in_channels)

        self.in_conv = MaskedConv2d('A', in_channels, n_filters, kernel_size=7, padding=3)
        model = []
        for _ in range(n_layers - 2):
            model.extend([nn.ReLU(), GatedConv2d('B', n_filters, n_filters, 7, padding=3)])
            model.append(StackLayerNorm(n_filters))

        self.out_conv = MaskedConv2d('B', n_filters, K, kernel_size=7, padding=3)
        self.net = nn.Sequential(*model)

    def forward(self, x):
        z = self.embedding(x).permute(0, 3, 1, 2).contiguous()

        out = self.in_conv(z)
        out = self.net(torch.cat((out, out), dim=1)).chunk(2, dim=1)[1]
        out = self.out_conv(out)
        return out


class Improved_VQVAE(VQVAE):
    """
    VQ-VAE using Gated PixelCNN
    """

    def __init__(self, K=128, D=256):
        super().__init__()
        self.pixelcnn_prior = GatedPixelCNN(K=K)


def train_vq_vae_with_gated_pixelcnn_prior(train_data, test_data):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]

    Returns
    - a (# of training iterations,) numpy array of VQ-VAE train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of VQ-VAE train losses evaluated once at initialization and after each epoch
    - a (# of training iterations,) numpy array of PixelCNN prior train losess evaluated every minibatch
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

    batch_size = 64
    dataset_params = {
        'batch_size': batch_size,
        'shuffle': True
    }

    print("[INFO] Creating model and data loaders")
    train_data = torch.from_numpy(np.transpose(train_data, [0, 3, 1, 2])).float()
    test_data = torch.from_numpy(np.transpose(test_data, [0, 3, 1, 2])).float().cuda()
    train_loader = torch.utils.data.DataLoader(dequantize(train_data), **dataset_params)
    test_loader = torch.utils.data.DataLoader(dequantize(test_data), **dataset_params)

    # Model
    n_epochs_vae, n_epochs_cnn = 30, 30
    lr = 8e-4
    K, D = 128, 256
    vq_vae = Improved_VQVAE(K=K, D=D).cuda()

    optimizer = torch.optim.Adam(vq_vae.parameters(), lr=lr)

    # Training
    def train(model, no_epochs, prior_only=False):
        """
        Trains model and returns training and test losses
        """
        model_name = "VQ-VAE" if not prior_only else "Gated Pixel-CNN"
        print(f"[INFO] Training {model_name}")

        loss_fct = vq_vae.get_vae_loss if not prior_only else vq_vae.get_pixelcnn_prior_loss

        train_losses = []
        test_losses = [get_batched_loss(test_loader, vq_vae, loss_fct, prior_only, loss_triples=False)]

        for epoch in range(no_epochs):
            epoch_start = time.time()
            for batch in train_loader:
                optimizer.zero_grad()
                batch = batch.cuda()
                output = vq_vae(batch, prior_only)
                loss = loss_fct(batch, output)
                loss.backward()
                optimizer.step()
                batch = batch.cpu()

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


def train_and_show_results_cifar():
    """
    Trains VQ-VAE with Gated PixelCNN prior for images and display samples, reconstructions and training plot for CIFAR-10 Dataset
    """
    show_results_images_vqvae2(2, train_vq_vae_with_gated_pixelcnn_prior)
