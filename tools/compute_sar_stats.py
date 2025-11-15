import sys
import os
from pathlib import Path

# dodaj korzeń repo do sys.path
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

from source import transforms as T

p = Path(repo_root) / 'dataset' / 'train' / 'labels'
files = sorted([str(x) for x in p.glob('*.tif')])
print(f'Found {len(files)} label files, using first 100 (or less)')
use = files[:100]
mean, std = T.compute_sar_stats(use, load_fn=None, verbose=True)
print('SAR mean:', mean)
print('SAR std :', std)

