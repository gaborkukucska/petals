import torch

PUBLIC_INITIAL_PEERS = [
    "/dns/ai.gettingitalldone.com/tcp/31337/p2p/QmTbqDrBxCioZMYCjTUHu5GLVERw369VkY7fBTMiKFFDXu",
]

# The reachability API is currently used only when connecting to the public swarm
REACHABILITY_API_URL = "http://health.gettingitalldone.com"

DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")
