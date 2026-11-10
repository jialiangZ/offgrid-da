"""Fresh matched training with fixed training-only error-balanced loss."""
import json,sys,hashlib,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import error_balanced_trainer as trainer
from run_full_update_v1 import Data
from offgrid_da.data.multivar_dataset import multivar_collate
from offgrid_da.data.full_state_data import CHANNELS,digest
from offgrid_da.models import CrossAttentionAnalysisModel
from error_balanced import balanced_loss,latitude_weights,metrics

def main():
    stats_path=ROOT/'outputs/single_da_09_full82_loss_balanced/loss_statistics.json'
    stats=json.loads(stats_path.read_text());assert stats['complete'] and stats['cycles']==4384
    settings=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
    assert settings['loss_statistics_sha256']==digest(stats_path)
    trainer.LazyArchesDataset=Data;trainer.CachedSamples=lambda d:d;trainer.single_cycle_collate=multivar_collate
    original=trainer.DataLoader
    def loader(*a,**kw):
        kw.update(num_workers=2,multiprocessing_context='spawn',persistent_workers=True,prefetch_factor=2)
        return original(*a,**kw)
    trainer.DataLoader=loader
    initial={}
    def factory(kind,cfg):
        torch_model=CrossAttentionAnalysisModel(channels=82,**cfg)
        model=torch_model if kind=='da' else trainer.BackgroundCorrection(torch_model)
        if not initial:
            h=hashlib.sha256()
            for prefix,module in [('input',torch_model.input),('blocks',torch_model.blocks),('output',torch_model.output)]:
                for name,tensor in module.state_dict().items():h.update((prefix+name).encode());h.update(tensor.numpy().tobytes())
            initial['convolution_sha256']=h.hexdigest()
        return model
    trainer.make_model=factory
    trainer.training_loss=lambda p,y:balanced_loss(p,y,stats['loss_weights'])
    def evaluate(model,loader,device,std):
        if model:model.eval()
        sse=np.zeros(82);baseline=np.zeros(82);den=0.;cycles=0
        with torch.inference_mode():
            for batch in loader:
                x,o,y=trainer.pack(batch,device);prediction=x if model is None else model(x,o)['analysis']
                if not torch.isfinite(prediction).all():raise ValueError('Nonfinite evaluation')
                lat=latitude_weights(x).double()
                sse+=((prediction.double()-y.double()).square()*lat).sum((0,2,3)).cpu().numpy()
                baseline+=((x.double()-y.double()).square()*lat).sum((0,2,3)).cpu().numpy()
                den+=float(lat.expand(len(x),1,*x.shape[-2:]).sum());cycles+=len(x)
        result=metrics(sse,den,stats['loss_weights'],np.array(stats['physical_std']),CHANNELS)
        bg=metrics(baseline,den,stats['loss_weights'],np.array(stats['physical_std']),CHANNELS)
        return {**result,'baseline0_score':bg['score'],'baseline':bg,'cycles':cycles,
                'selection_metric':'dimensionless training-error-balanced RMSE'}
    trainer.evaluate=evaluate
    trainer.main()
    out=Path(sys.argv[sys.argv.index('--output')+1]);result=json.loads((out/'result.json').read_text())
    result.update(initial,loss_statistics_sha256=digest(stats_path))
    from offgrid_da.data.arches_training import atomic_json
    atomic_json(out/'result.json',result)
if __name__=='__main__':main()
