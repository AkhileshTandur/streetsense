import argparse, subprocess, sys

def main():
    p=argparse.ArgumentParser(description='StreetSense CLI')
    sub=p.add_subparsers(dest='cmd')

    p_synth=sub.add_parser('synth'); p_synth.add_argument('--n_trips', type=int, default=30)
    p_self=sub.add_parser('selfsup'); p_self.add_argument('--epochs', type=int, default=2)
    p_sup=sub.add_parser('supervised'); p_sup.add_argument('--epochs', type=int, default=2)
    p_al=sub.add_parser('active'); p_al.add_argument('--budget', type=int, default=50)
    p_pr=sub.add_parser('prioritize')

    args=p.parse_args()
    if args.cmd=='synth': subprocess.check_call([sys.executable, 'streetsense/data/make_synthetic.py', '--n_trips', str(args.n_trips)])
    elif args.cmd=='selfsup': subprocess.check_call([sys.executable, 'streetsense/train_selfsup.py', '--epochs', str(args.epochs)])
    elif args.cmd=='supervised': subprocess.check_call([sys.executable, 'streetsense/train_supervised.py', '--epochs', str(args.epochs)])
    elif args.cmd=='active': subprocess.check_call([sys.executable, 'streetsense/active_learning.py', '--budget', str(args.budget)])
    elif args.cmd=='prioritize': subprocess.check_call([sys.executable, 'streetsense/prioritize_causal.py'])
    else: p.print_help()

if __name__ == '__main__':
    main()
