from argparse import ArgumentParser
from reward.deploy_reward import serve_reward_model


argumentParser = ArgumentParser()
argumentParser.add_argument("--num_replicas", type=int, default=8)
args = argumentParser.parse_args()

if __name__ == "__main__":
    serve_reward_model(args.num_replicas, gpu_ids=[i for i in range(args.num_replicas)])
