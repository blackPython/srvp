import os
import random 
import torch
import sys

import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributions as distrib
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

import args
import helper
import data.base as data
import module.srvp as srvp
import module.utils as utils
from train import train

def train_worker(rank,opt,world_size):
    if rank != 0:
        sys.stdout = open(os.devnull, 'w')

    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["MASTER_PORT"] = '8080'

    torch.distributed.init_process_group("nccl",rank=rank,world_size=world_size)
    device = torch.device(rank)
    
    if not opt.seed:
        opt.seed = 42
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed+rank)
    random.seed(opt.seed)

    batch_size = opt.batch_size//world_size

    dataset = data.load_dataset(opt,train=True)
    trainset = dataset.get_fold('train')

    def worker_init_fn(worker_id):
        np.random.seed((opt.seed + itr + opt.local_rank*10 + worker_id)%(2**32-1))
    #Dataloader
    sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas = world_size, rank = rank)
    train_loader = DataLoader(trainset, batch_size = batch_size, collate_fn = data.collate_fn, num_workers = opt.num_workers,sampler = sampler, drop_last = True, pin_memory = True, worker_init_fn = worker_init_fn)

    model = srvp.StochasticLatentResidualVideoPredictor(opt.nx, opt.nc, opt.nf, opt.nhx, opt.ny, opt.nz, opt.skipco, opt.nt_inf, opt.nlayers_inf, opt.nh_res, opt.nlayers_res, opt.archi).to(device)
    model.init(res_gain=opt.res_gain)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
    opt.niter = opt.lr_scheduling_burnin + opt.lr_scheduling_niter
    niter = opt.lr_scheduling_niter
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda i: max(0,(niter-i)/niter))

    forward_fn = torch.nn.paralle.DistributedDataParallel(model)

    if rank == 0:
        pb = tqdm(total=opt.niter,ncols = 0)
    itr = 0
    finished = False
    try:
        while not finished:
            for batch in train_loader:
                if itr>= opt.niter:
                    finished = True
                    status_code = 0
                    break
                itr += 1
                model.train()
                loss, disto, rate_y_0, rate_z = train(forward_fn, optimizer, batch, device, opt)
                if itr >= opt.lr_scheduling_burnin:
                    lr_scheduler.step()

                if rank == 0:
                    pd.set_postfix(loss = loss, disto = disto, rate_y_0 = rate_y_0, rate_z = rate_z, refresh = False)
                    pd.update()
    except KeyboardInterrupt:
            status_code = 130

    if rank == 0:
        pb.close()

    print('Done')
    print('Saving...')
    torch.save(model.state_dict(), os.path.join(opt.save_path, 'model.pt'))
    return status_code

if __name__ == "__main__":
    p = args.create_args()
    temp = p.parse_args()
    opt = helper.DotDict(vars(temp))
    assert torch.cuda.is_available()

    num_gpus = torch.cuda.device_count()
    mp.spawn(train_worker, args=(opt,num_gpus),nprocs=num_gpus,join = True)
    
