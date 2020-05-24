import time

import torch.distributions as dist
import torch.nn as nn

from utils import *
from vae import Encoder, Decoder


class LinearMasked(nn.Linear):
    """
    Class implementing nn.Linear with mask
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            LinearMasked(input_dim, hidden_dim),
            nn.ReLU(),
            LinearMasked(hidden_dim, hidden_dim),
            nn.ReLU(),
            LinearMasked(hidden_dim, 2 * input_dim)
        )
        self.apply_masks()

    def forward(self, x):
        return self.net(x)

    def apply_masks(self):
        # Set order of masks, i.e. who can make which edges
        order_in = np.arange(self.input_dim)
        order_hidden_1 = np.random.randint(low=0, high=self.input_dim, size=(self.hidden_dim,))
        order_hidden_2 = np.random.randint(low=0, high=self.input_dim, size=(self.hidden_dim,))
        order_out = np.concatenate((np.arange(self.input_dim), np.arange(self.input_dim)))

        # Create masks
        masks = []
        masks.append(order_in[:, None] <= order_hidden_1[None, :])
        masks.append(order_hidden_1[:, None] <= order_hidden_2[None, :])
        masks.append(order_hidden_2[:, None] < order_out[None, :])

        # Set the masks in all LinearMasked layers
        layers = [l for l in self.net.modules() if isinstance(l, LinearMasked)]
        for i in range(len(layers)):
            layers[i].set_mask(masks[i])


class VLAE(nn.Module):
    """
    Variational Lossy Autoencoder from "Variational Lossy Autoencoder" Xi Chen et. al.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.made = MADE(latent_dim)

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z_mu, z_log_sigma = torch.chunk((self.encoder(x)), 2, dim=1)
        z_sigma = torch.exp(z_log_sigma)
        z = self.reparameterize(z_mu, z_sigma)

        made_mu, made_log_sigma = torch.chunk((self.made(z)), 2, dim=1)
        eps = z * torch.exp(made_log_sigma) + made_mu

        x_mu = self.decoder(z)
        x_sigma = torch.ones_like(x_mu)

        return x_mu, x_sigma, z, z_mu, z_sigma, eps, made_log_sigma

    def reparameterize(self, mu, sigma, noise=True):
        normal = dist.normal.Normal(0, 1)
        eps = normal.sample(sigma.shape).cuda()

        return mu + eps * sigma if noise else mu


def train_vlae(train_data, test_data):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
        and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
        and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
        FROM THE TEST SET with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
        pairs of test images. The output should be those 100 images flattened into
        the specified shape with values in {0, ..., 255}
    """
    start_time = time.time()
    N, H, W, C = train_data.shape

    prior_eps = dist.normal.Normal(0, 1)

    def dequantize(x, dequantize=True, reverse=False, alpha=.1):
        with torch.no_grad():
            if reverse:
                return torch.ceil_(torch.sigmoid(x) * 255)

            if dequantize:
                x += torch.zeros_like(x).uniform_(0, 1)

            p = alpha / 2 + (1 - alpha) * x / 256
            return torch.log(p) - torch.log(1 - p)

    def loss_function(batch, output):
        def repeat(tensor, K=50):
            shape = (K,) + tuple(tensor.shape)
            return torch.cat(K * [tensor]).reshape(shape)

        K = 50
        x_mu, x_sigma, z, z_mu, z_sigma, eps, made_log_sigma = output
        x_mu, x_sigma, z, z_mu, z_sigma, eps, made_log_sigma = repeat(x_mu), repeat(x_sigma), repeat(z), repeat(
            z_mu), repeat(z_sigma), repeat(eps), repeat(made_log_sigma)
        k_batch = repeat(batch)

        # Find VLB, reconstruction loss and KL-term
        log_p_z = prior_eps.log_prob(eps) + made_log_sigma  # log p(z) = log p(𝜖) + log det∣𝑑𝜖/𝑑𝑧∣

        z_normal = dist.normal.Normal(z_mu, z_sigma)
        posterior_log_prob = z_normal.log_prob(z)  # q(z | x) -  Posterior probability

        x_normal = dist.normal.Normal(x_mu, x_sigma)
        x_reconstruct_log_prob = x_normal.log_prob(k_batch)  # p(x | z) -  Probability of batch given z

        VLB = -torch.mean(
            log_p_z.sum(dim=(0, 2)) + x_reconstruct_log_prob.sum(dim=(0, 2, 3, 4)) - posterior_log_prob.sum(
                dim=(0, 2))) / K
        reconstruction_loss = -torch.mean(x_reconstruct_log_prob.sum(dim=(0, 2, 3, 4))) / K
        KL = torch.mean(posterior_log_prob.sum(dim=(0, 2)) - log_p_z.sum(
            dim=(0, 2))) / K  # KL(q(z|x) | p(z)) = log q(z | x) - log p(z)

        return VLB, reconstruction_loss, KL

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
    n_epochs = 30  # if dset_id == 1 else 50
    lr = 1e-3  # if dset_id == 1 else 5e-4
    latent_dim = 16  # if dset_id == 1 else 32
    vlae = VLAE(latent_dim=latent_dim).cuda()

    optimizer = torch.optim.Adam(vlae.parameters(), lr=lr)

    # Training
    train_losses = []
    init_test_loss = get_batched_loss(test_loader, vlae, loss_function)
    test_losses = [[*init_test_loss]]
    print(
        f"Initial loss-variables: VLB: {test_losses[0][0]:.2f}, Reconstruct loss: {test_losses[0][1]:.2f}, KL: {test_losses[0][2]:.2f}")

    print("[INFO] Training")
    for epoch in range(n_epochs):
        epoch_start = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            output = vlae(batch)
            loss = loss_function(batch, output)
            loss[0].backward()
            optimizer.step()

            train_losses.append([loss[0].cpu().item(), loss[1].cpu().item(), loss[2].cpu().item()])

        test_loss = get_batched_loss(test_loader, vlae, loss_function)
        test_losses.append([*test_loss])

        print(
            f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1} - Test loss (VLB/RL/KL): {test_loss[0]:.2f}/{test_loss[1]:.2f}/{test_loss[2]:.2f} - Time elapsed: {time.time() - epoch_start:.2f}")

    # Utility methods
    def sample(noise=False, no_samples=100):
        """
        Sample z ~ p(z) and x ~ p(x|z) if noise, x = mu(z) otherwise
        """
        sample_shape = (no_samples, latent_dim)
        z = prior_eps.sample(sample_shape).cuda()

        for i in range(latent_dim):
            made_mu, made_log_sigma = torch.chunk((vlae.made(z)), 2, dim=1)
            eps = prior_eps.sample(sample_shape).cuda()

            # With 𝜖 = 𝑧 ⊙ 𝜎(𝑧) + 𝜇(𝑧) --> z = (𝜖 - 𝜇(𝑧))/𝜎(𝑧)
            z[:, i] = (eps[:, i] - made_mu[:, i]) / torch.exp(made_log_sigma[:, i])

        x_mu = vlae.decoder(z)  # Shape is (N, 2)
        x_sigma = torch.ones_like(x_mu)

        x = vlae.reparameterize(x_mu, x_sigma, noise)  # Shape is (no_samples, H, W, C)

        samples = dequantize(x, reverse=True).detach().cpu().numpy()
        return np.transpose(samples, [0, 2, 3, 1])  # Get it in (N, H, W, C)

    def reconstruction_pairs(no_reconstructions=50):
        """
        Creating reconstruction pairs (x, x') where x is the original image and x' is the decoder-output
        """
        x_original = test_data[:no_reconstructions]
        x_dequantized = dequantize(x_original, dequantize=False)

        x_reconstructed = vlae(x_dequantized)[0]
        x_reconstructed = dequantize(x_reconstructed, reverse=True)

        pairs = torch.zeros_like(torch.cat((x_original, x_reconstructed), dim=0)).detach().cpu().numpy()
        pairs[::2] = x_original.detach().cpu().numpy()
        pairs[1::2] = x_reconstructed.detach().cpu().numpy()

        pairs = np.clip(pairs, 0, 255)
        return np.transpose(pairs, [0, 2, 3, 1])  # Get it in (N, H, W, C)

    def interpolate_images(no_interpolations=10):
        interpolations = torch.zeros(size=(no_interpolations * 10, C, H, W)).float().cuda()
        counter = 0
        weights = np.linspace(0, 1, 10)[1:-1]

        for i in range(no_interpolations):
            x_a, x_b = test_data[i].unsqueeze(0), test_data[i + no_interpolations].unsqueeze(0)
            x_a_dequantized, x_b_dequantized = dequantize(x_a, dequantize=False), dequantize(x_b, dequantize=False)

            z_a, z_b = vlae(x_a_dequantized)[2], vlae(x_b_dequantized)[2]

            interpolations[counter] = x_a
            counter += 1

            for weight in weights:
                z_interpolated = (1 - weight) * z_a + weight * z_b

                x_interpolated = vlae.decoder(z_interpolated)
                x_interpolated = dequantize(x_interpolated, reverse=True)

                interpolations[counter] = x_interpolated[0]
                counter += 1

            interpolations[counter] = x_b
            counter += 1

        interpolations = interpolations.detach().cpu().numpy()
        interpolations = np.clip(interpolations, 0, 255)
        return np.transpose(interpolations, [0, 2, 3, 1])  # Get it in (N, H, W, C)

    torch.cuda.empty_cache()
    vlae.eval()
    with torch.no_grad():
        # Do samples
        print("[INFO] Sampling images")
        samples = sample(noise=False)

        print("[INFO] Creating reconstructing pairs")
        pairs = reconstruction_pairs()

        print("[INFO] Interpolating")
        interpolations = interpolate_images()

    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")

    train_losses, test_losses = np.array(train_losses), np.array(test_losses)
    print("Samples", samples.shape)
    print("Pairs", pairs.shape)
    print("Interpolations", interpolations.shape)

    return np.array(train_losses), np.array(test_losses), samples, pairs, interpolations


def train_and_show_results_svhn():
    """
    Trains VLAE for images and display samples, reconstructions and interpolations, and training plot for SVHN Dataset
    """
    show_results_images_vae(1, train_vlae)


def train_and_show_results_cifar():
    """
    Trains VLAE for images and display samples, reconstructions and interpolations, and training plot for CIFAR-10 Dataset
    """
    show_results_images_vae(2, train_vlae)
