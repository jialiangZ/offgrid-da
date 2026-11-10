"""Performance-only data prefetch override. Config, optimizer and sampling unchanged."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_full_update_v1 as core
Original=core.trainer.DataLoader

def loader(*args,**kwargs):
 kwargs.update(num_workers=2,multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=True)
 return Original(*args,**kwargs)

def main():
 core.trainer.DataLoader=loader
 if '--train' in sys.argv:sys.argv.remove('--train')
 core.run_train()
if __name__=='__main__':main()
