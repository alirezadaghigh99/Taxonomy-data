import torch
import os

def device():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        backend = torch.distributed.get_backend()
        if backend == 'nccl':
            local_rank = int(os.getenv('LOCAL_RANK', '0'))
            return torch.device(f'cuda:{local_rank}')
        elif backend == 'gloo':
            return torch.device('cpu')
    elif 'HOROVOD_RANK' in os.environ:
        local_rank = int(os.getenv('HOROVOD_LOCAL_RANK', '0'))
        return torch.device(f'cuda:{local_rank}')
    elif 'XRT_SHARD_LOCAL_ORDINAL' in os.environ:
        index = int(os.getenv('XRT_SHARD_LOCAL_ORDINAL', '0'))
        return torch.device(f'xla:{index}')
    else:
        return torch.device('cpu')

