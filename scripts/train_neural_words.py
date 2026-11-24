import torch,numpy as np,json
from pathlib import Path
torch.set_num_threads(4);torch.manual_seed(42)
d=np.load('/tmp/echo80-neural-data.npz');mean=d['train_x'].mean((0,1));scale=d['train_x'].std((0,1)).clip(.1)
def tensors(s):return torch.tensor(((d[s+'_x']-mean)/scale).transpose(0,2,1),dtype=torch.float32),torch.tensor(d[s+'_y'],dtype=torch.long)
x,y=tensors('train');v,vy=tensors('val');tx,ty=tensors('test')
class Net(torch.nn.Module):
 def __init__(self):
  super().__init__();self.layers=torch.nn.Sequential(torch.nn.Conv1d(154,96,3,padding=1),torch.nn.ReLU(),torch.nn.Dropout(.25),torch.nn.Conv1d(96,96,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool1d(4),torch.nn.Flatten(),torch.nn.Linear(384,80))
 def forward(self,x):return self.layers(x)
m=Net();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);best=-1
for epoch in range(120):
 m.train()
 for ids in torch.randperm(len(y)).split(64):
  bx=x[ids]+torch.randn_like(x[ids])*.025
  loss=torch.nn.functional.cross_entropy(m(bx),y[ids]);opt.zero_grad();loss.backward();opt.step()
 m.eval()
 with torch.no_grad():score=int((m(v).argmax(1)==vy).sum())
 if score>best:
  best=score;state={k:t.clone() for k,t in m.state_dict().items()}
m.load_state_dict(state)
with torch.no_grad():pred=m(tx).argmax(1)
report=dict(validation_correct=best,validation_total=len(vy),test_correct=int((pred==ty).sum()),test_usable=len(ty),per_word={str(w):dict(correct=int(((pred==ty)&(ty==i)).sum()),total=int((ty==i).sum())) for i,w in enumerate(d['labels'])})
Path('reports/word80-neural-comparison.json').write_text(json.dumps(report,indent=2))
torch.save(dict(state=state,mean=mean.tolist(),scale=scale.tolist(),labels=d['labels'].tolist()),'models/word80-neural-candidate.pt')
print(best,len(vy),report['test_correct'],len(ty))
