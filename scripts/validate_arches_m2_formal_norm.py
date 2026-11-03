import json,sys,time,torch
from pathlib import Path
root=Path('/home/jlzhang/Documents/offgrid-da'); sys.path.insert(0,str(root/'src'))
from offgrid_da.data.arches_dataset import ArchesSingleCycleDADataset
from offgrid_da.data.collate import single_cycle_collate
from offgrid_da.models import CrossAttentionAnalysisModel
torch.manual_seed(42); torch.set_num_threads(4)
ds=ArchesSingleCycleDADataset(root/'outputs/bg_t2m_train_2018_2020/manifest.json',normalization_path=root/'outputs/normalization/arches_train_2018_2020.json',start='2018-01-01',end='2018-01-02'); s=ds[0]; b=single_cycle_collate([s]); mean=ds.norm_stats.mean('background','2m_temperature'); std=ds.norm_stats.std('background','2m_temperature'); bg=b['background_normalized']; target=(b['reference']-mean)/std
obs={k:b[k] for k in ('obs_values','obs_locations','obs_time_offsets','obs_variables','obs_sources','obs_quality','observation_mask')}; obs['obs_values']=(obs['obs_values']-mean)/std
model=CrossAttentionAnalysisModel(dim=8,heads=2,depth=1,max_neighbors=8,query_chunk=512); opt=torch.optim.Adam(model.parameters(),lr=0.0003); losses=[]
for i in range(30):
 opt.zero_grad(); pred=model(bg,obs)['analysis']; loss=(pred-target).square().mean(); loss.backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()); opt.step(); losses.append(float(loss)); print('step',i+1,'loss',losses[-1],flush=True)
assert losses[-1] < losses[0]
with torch.no_grad():
 model.eval(); expected=model(bg,obs)['analysis']; empty={k:v.clone() for k,v in obs.items()}; empty['observation_mask'][:]=False; assert torch.equal(model(bg,empty)['analysis'],bg)
out=root/'outputs/bg_t2m_train_2018_2020'; path=out/'m2_diagnostic_checkpoint.pt'; torch.save(model.state_dict(),path); loaded=CrossAttentionAnalysisModel(dim=8,heads=2,depth=1,max_neighbors=8,query_chunk=512); loaded.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)); loaded.eval()
with torch.no_grad(): torch.testing.assert_close(loaded(bg,obs)['analysis'],expected,rtol=0,atol=0)
report={'scope':'one real batch overfit diagnostic with complete training-period statistics','losses':losses,'empty_exact_identity':True,'checkpoint_reload_exact':True,'context_obs':int(obs['observation_mask'].sum()),'device':'cpu'}
(out/'m2_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)
