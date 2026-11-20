import io,os,zipfile,urllib.request,csv,json
from pathlib import Path
url='https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip'
class Remote(io.RawIOBase):
 def __init__(self):self.pos=0;self.size=45924134223
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,n,w=0):
  self.pos=[0,self.pos,self.size][w]+n;return self.pos
 def read(self,n=-1):
  n=min(n if n>=0 else self.size,self.size-self.pos)
  if n<=0:return b''
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'}),timeout=60) as r:
   assert r.status==206
   b=r.read(n);self.pos+=len(b);return b
z=zipfile.ZipFile(Remote()); names=z.namelist()
root=Path('datasets/raw/asl-citizen-selected');root.mkdir(exist_ok=True)
for name in names:
 if '/splits/' in name and name.endswith('.csv') and '__MACOSX' not in name:
  data=z.read(name);(root/Path(name).name).write_bytes(data);print(name,data[:180],flush=True)
(root/'entries.json').write_text(json.dumps({n.split('/')[-1]:[i.header_offset,i.compress_size,i.file_size,i.CRC,i.compress_type] for n in names if n.endswith('.mp4') for i in [z.getinfo(n)]}))
