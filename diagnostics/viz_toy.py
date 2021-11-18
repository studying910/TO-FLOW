import os
import math
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_trajectory(args, model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu', t1=0.5):
    model.eval()

    #  Sample from prior
    z_samples = torch.randn(2000, 2).to(device)

    # sample from a grid
    npts = 800
    side = np.linspace(-4, 4, npts)
    xx, yy = np.meshgrid(side, side)
    xx = torch.from_numpy(xx).type(torch.float32).to(device)
    yy = torch.from_numpy(yy).type(torch.float32).to(device)
    z_grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)

    with torch.no_grad():
        # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = torch.sum(standard_normal_logprob(z_samples), 1, keepdim=True)
        logp_grid = torch.sum(standard_normal_logprob(z_grid), 1, keepdim=True)
        t = 0
        for cnf in model.chain:
            if args.time_optim:
                end_time = t1
            elif args.time_sample:
                end_time = t1
            else:
                end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
            integration_times = torch.linspace(0, end_time, ntimes)

            z_traj, _ = cnf(z_samples, logp_samples, integration_times=integration_times, reverse=True)
            z_traj = z_traj.cpu().numpy()

            grid_z_traj, grid_logpz_traj = [], []
            inds = torch.arange(0, z_grid.shape[0]).to(torch.int64)
            for ii in torch.split(inds, int(z_grid.shape[0] * memory)):
                _grid_z_traj, _grid_logpz_traj = cnf(
                    z_grid[ii], logp_grid[ii], integration_times=integration_times, reverse=True
                )
                _grid_z_traj, _grid_logpz_traj = _grid_z_traj.cpu().numpy(), _grid_logpz_traj.cpu().numpy()
                grid_z_traj.append(_grid_z_traj)
                grid_logpz_traj.append(_grid_logpz_traj)
            grid_z_traj = np.concatenate(grid_z_traj, axis=1)
            grid_logpz_traj = np.concatenate(grid_logpz_traj, axis=1)

            plt.figure(figsize=(8, 8))
            for _ in range(z_traj.shape[0]):
                plt.clf()

                # plot target potential function
                ax = plt.subplot(2, 2, 1, aspect="equal")

                ax.hist2d(data_samples[:, 0], data_samples[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Target", fontsize=32)

                # plot the density
                ax = plt.subplot(2, 2, 2, aspect="equal")

                z, logqz = grid_z_traj[t], grid_logpz_traj[t]

                xx = z[:, 0].reshape(npts, npts)
                yy = z[:, 1].reshape(npts, npts)
                qz = np.exp(logqz).reshape(npts, npts)

                plt.pcolormesh(xx, yy, qz)
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                cmap = matplotlib.cm.get_cmap(None)
                ax.set_facecolor(cmap(0.))
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Density", fontsize=32)

                # plot the samples
                ax = plt.subplot(2, 2, 3, aspect="equal")

                zk = z_traj[t]
                ax.hist2d(zk[:, 0], zk[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Samples", fontsize=32)

                # plot vector field
                ax = plt.subplot(2, 2, 4, aspect="equal")

                K = 13j
                y, x = np.mgrid[-4:4:K, -4:4:K]
                K = int(K.imag)
                zs = torch.from_numpy(np.stack([x, y], -1).reshape(K * K, 2)).to(device, torch.float32)
                logps = torch.zeros(zs.shape[0], 1).to(device, torch.float32)
                dydt = cnf.odefunc(integration_times[t], (zs, logps))[0]
                dydt = -dydt.cpu().detach().numpy()
                dydt = dydt.reshape(K, K, 2)

                logmag = 2 * np.log(np.hypot(dydt[:, :, 0], dydt[:, :, 1]))
                ax.quiver(
                    x, y, dydt[:, :, 0], dydt[:, :, 1],
                    np.exp(logmag), cmap="coolwarm", scale=20., width=0.015, pivot="mid"
                )
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                ax.axis("off")
                ax.set_title("Vector Field", fontsize=32)

                makedirs(savedir)
                plt.savefig(os.path.join(savedir, f"viz-{t:05d}.jpg"))
                t += 1


def trajectory_to_video(savedir):
    import subprocess
    bashCommand = 'ffmpeg -y -i {} {}'.format(os.path.join(savedir, 'viz-%05d.jpg'), os.path.join(savedir, 'traj.mp4'))
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
