import argparse, numpy as np, pandas as pd
from pathlib import Path

def run(args):
    # Toy causal prioritization: pretend each road segment has pre/post 'complaints'
    rng = np.random.default_rng(0)
    n = 50
    segments = [f"segment_{i:04d}" for i in range(n)]
    pre = rng.poisson(5, n) + rng.integers(0,3,n)  # before fix complaints
    effect = rng.normal(1.0, 0.5, n)               # predicted benefit
    post = np.clip(pre - effect, 0, None)          # after hypothetical fix
    uplift = pre - post
    df = pd.DataFrame({'segment': segments, 'pre_rate': pre, 'post_pred': post, 'estimated_uplift': uplift})
    df['priority_score'] = 0.7*df['estimated_uplift'] + 0.3*df['pre_rate']
    df = df.sort_values('priority_score', ascending=False)
    out = Path('out'); out.mkdir(exist_ok=True, parents=True)
    df.to_csv(out/'prioritized_repairs.csv', index=False)
    print(f"Exported prioritized repairs to {out/'prioritized_repairs.csv'}")

if __name__ == "__main__":
    p=argparse.ArgumentParser(); args=p.parse_args(); run(args)
