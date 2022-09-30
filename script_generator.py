from itertools import permutations
from argparse import Namespace
from pathlib import Path
from mdh import GlobalHandler as GH
from numpy.random import default_rng
mmseed = 1126
for yo in range(0, 6):
    mother_seed_list = default_rng(seed=mmseed).integers(1e4, size=yo)
    dir_ = Path('script')
    dir_.mkdir(parents=True,exist_ok=True)
    device = [0]*12
    for mother_seed in mother_seed_list[yo-1:]:
        rng = default_rng(seed=mother_seed)
        seed_list = rng.integers(1e4, size=12)

        args = Namespace()
        args.num_iters = 6000
        # args.mode = 'uda'
        args.method = 'CDAC_LC'
        args.alpha = 0.3
        args.T = 0.6
        args.lr = 0.01
        args.update_interval = 500
        args.note = f'mother_{mother_seed}_separate_lr'
        gh = GH()

        l = [[] for _ in range(len(device))]
        for i, (s, t) in enumerate(permutations(range(4), 2)):
            idx = i % len(device)
            args.source, args.target, args.seed = s, t, seed_list[i]
            args.init = gh.regSearch(f':CDAC/.*_lr_5.*seed:{seed_list[i]}.*{s}.target.{t}')[0]
            cmd = 'python main.py ' + ' '.join([f'--{k} {v}' for k, v in args.__dict__.items()]) + f' --device {device[idx]}'
            l[idx].append(cmd)
        for i in range(len(device)):
            with (dir_ / f'scriptMME_LC{i}.sh').open('a') as f:
                f.write('\n'.join(l[i if yo % 2 == 1 else 11 - i])+'\n')
