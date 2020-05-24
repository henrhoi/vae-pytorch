import time

import torch.distributions as dist
import torch.nn as nn

from utils import *


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class VAE2D(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = MLP(in_dim, hidden_dim=32, out_dim=in_dim * 2)
        self.decoder = MLP(in_dim, hidden_dim=32, out_dim=in_dim * 2)

    def forward(self, x):
        z_mu, z_log_sigma = torch.chunk((self.encoder(x)), 2, dim=1)  # z_sigma is in log(std)
        z_sigma = torch.exp(z_log_sigma)
        z = self.reparameterize(z_mu, z_sigma)

        x_mu, x_log_sigma = torch.chunk((self.decoder(z)), 2, dim=1)  # z_sigma is in log(std)
        x_sigma = torch.exp(x_log_sigma)

        return x_mu, x_sigma, z, z_mu, z_sigma

    def reparameterize(self, mu, sigma, noise=True):
        normal = dist.normal.Normal(0, 1)
        eps = normal.sample(sigma.shape).cuda()

        return mu + eps * sigma if noise else mu


def train_vae_2d(train_data, test_data):
    """
    train_data: An (n_train, 2) numpy array of floats
    test_data: An (n_test, 2) numpy array of floats

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
        and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch on train data
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
        and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch on test data
    - a numpy array of size (1000, 2) of 1000 samples WITH decoder noise, i.e. sample z ~ p(z), x ~ p(x|z)
    - a numpy array of size (1000, 2) of 1000 samples WITHOUT decoder noise, i.e. sample z ~ p(z), x = mu(z)
    """
    start_time = time.time()
    prior_p_z = dist.normal.Normal(0, 1)

    def loss_function(batch, output):
        def repeat(tensor, K=50):
            N, d = tensor.shape
            return torch.cat(K * [tensor]).reshape(K, N, d)

        x_mu, x_sigma, z, z_mu, z_sigma = output
        x_mu, x_sigma, z, z_mu, z_sigma = repeat(x_mu), repeat(x_sigma), repeat(z), repeat(z_mu), repeat(z_sigma)
        k_batch = repeat(batch)

        # Find VLB, reconstruction loss and KL-term
        log_p_z = prior_p_z.log_prob(z)  # p(z) from N(0, I)

        z_normal = dist.normal.Normal(z_mu, z_sigma)
        posterior_log_prob = z_normal.log_prob(z)  # q(z | x) -  Posterior probability

        x_normal = dist.normal.Normal(x_mu, x_sigma)
        x_reconstruct_log_prob = x_normal.log_prob(k_batch)  # p(x | z) -  Probability of batch given z

        VLB = -torch.mean(log_p_z + x_reconstruct_log_prob - posterior_log_prob) * 2
        reconstruction_loss = -torch.mean(x_reconstruct_log_prob) * 2
        KL = torch.mean(posterior_log_prob - log_p_z) * 2  # KL(q(z|x) | p(z)) = log q(z | x) - log p(z)

        return VLB, reconstruction_loss, KL

    batch_size = 64
    dataset_params = {
        'batch_size': batch_size,
        'shuffle': True
    }

    print("[INFO] Creating model and data loaders")
    train_loader = torch.utils.data.DataLoader(torch.from_numpy(train_data).float().cuda(), **dataset_params)
    test_loader = torch.utils.data.DataLoader(torch.from_numpy(test_data).float().cuda(), **dataset_params)

    # Model
    n_epochs = 10
    input_shape = train_data.shape[1]
    vae = VAE2D(in_dim=input_shape).cuda()

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # Training
    train_losses = []
    init_test_loss = get_batched_loss(test_loader, vae, loss_function)
    test_losses = [[*init_test_loss]]
    print(test_losses)
    print(
        f"Initial loss-variables: VLB:{test_losses[0][0]:.2f}, Reconstruct loss:{test_losses[0][1]:.2f}, KL:{test_losses[0][2]:.2f}")

    print("[INFO] Training")
    for epoch in range(n_epochs):
        epoch_start = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            output = vae(batch)
            loss = loss_function(batch, output)
            loss[0].backward()
            optimizer.step()

            train_losses.append([loss[0].cpu().item(), loss[1].cpu().item(), loss[2].cpu().item()])

        test_loss = get_batched_loss(test_loader, vae, loss_function)
        test_losses.append([*test_loss])

        print(
            f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1} - Test loss (VLB/RL/KL): {test_loss[0]:.2f}/{test_loss[1]:.2f}/{test_loss[2]:.2f} - Time elapsed: {time.time() - epoch_start:.2f}")

    def sample(noise, no_samples=1000):
        """
        Sample z ~ p(z) and x ~ p(x|z) if noise, x = mu(z) otherwise
        """
        sample_shape = (no_samples, 2)
        z = prior_p_z.sample(sample_shape).cuda()

        x_mu, x_log_sigma = torch.chunk((vae.decoder(z)), 2, dim=1)  # Shape is (N, 2), (N, 2)
        x = vae.reparameterize(x_mu, torch.exp(x_log_sigma), noise)  # Shape is (N, 2)

        return x.cpu().numpy()

    torch.cuda.empty_cache()
    vae.eval()
    with torch.no_grad():
        # Do samples
        print("[INFO] Sampling with noise")
        noised_samples = sample(noise=True)

        print("[INFO] Sampling without noise")
        unnoised_samples = sample(noise=False)

    print(f"[DONE] Time elapsed: {time.time() - start_time:.2f} s")

    train_losses, test_losses = np.array(train_losses), np.array(test_losses)
    return np.array(train_losses), np.array(test_losses), noised_samples, unnoised_samples


def train_and_show_results_left_diagonal():
    """
    Trains VAE for 2D data and displays samples, with and without decoder noise, and training plot for Gaussian with a full covariance matrix, left diagonal
    """
    show_results_2d_data(1, train_vae_2d)


def train_and_show_results_right_diagonal():
    """
    Trains VAE for 2D data and displays samples, with and without decoder noise, and training plot for Gaussian with a full covariance matrix, right diagonal
    """
    show_results_2d_data(2, train_vae_2d)
