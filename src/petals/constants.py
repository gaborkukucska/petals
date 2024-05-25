import torch

PUBLIC_INITIAL_PEERS = [
    "/ip4/192.168.1.14/tcp/31337/p2p/QmcTB9tM37HrKWDaCKEDGtbiGYEpN8mqNcBiNnoF929wWM",
]

# The reachability API is currently used only when connecting to the public swarm
REACHABILITY_API_URL = "http://192.168.1.14:1111"

DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")
