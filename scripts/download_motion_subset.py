"""Extract selected members by HTTPS ranges; verify each member CRC/size."""
import json,struct,zlib,subprocess,tempfile,concurrent.futures,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URL='https://data.mendeley.com/public-files/datasets/8fmvr9m98w/files/9dcdeaa8-4df1-4eb9-98df-3231f66a10c7/file_downloaded'

def main():
    index_path=ROOT/'data/raw/signalpha/video-index.json'
    if not index_path.exists():
        index_path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:
            tail=Path(temp)/'tail'
            subprocess.run(['curl','-sSL','--fail','--max-time','60','--range','-65536',URL,'-o',str(tail)],check=True)
            data=tail.read_bytes();position=data.rfind(b'PK\x05\x06')
            if position<0:raise ValueError('ZIP directory footer missing')
            footer=struct.unpack_from('<4s4H2LH',data,position);offset=footer[6]
            central=Path(temp)/'central'
            subprocess.run(['curl','-sSL','--fail','--max-time','120','--range',f'{offset}-1247063490',URL,'-o',str(central)],check=True)
            sparse=Path(temp)/'index.zip'
            with sparse.open('wb') as file:file.seek(offset);file.write(central.read_bytes())
            with zipfile.ZipFile(sparse) as archive:
                index=[{'name':i.filename,'offset':i.header_offset,'compressed':i.compress_size,'size':i.file_size,'crc':i.CRC,'method':i.compress_type} for i in archive.infolist() if i.filename.lower().endswith('.avi')]
            index_path.write_text(json.dumps(index,indent=2))
    index=json.loads(index_path.read_text())
    chosen=[r for r in index if r['name'].split('/')[1] in ['J','Z','A','I','D','K','S','U']]
    def fetch(r):
        member=Path(r['name'])
        if member.is_absolute() or '..' in member.parts:raise ValueError('Unsafe archive member path')
        target=ROOT/'data/raw/signalpha'/member
        if target.exists() and target.stat().st_size==r['size'] and zlib.crc32(target.read_bytes())==r['crc']:return
        with tempfile.TemporaryDirectory() as temp:
            part=Path(temp)/'member.bin'
            end=min(1247063490,r['offset']+r['compressed']+4095)
            subprocess.run(['curl','-sSL','--fail','--retry','2','--max-time','120','--range',f"{r['offset']}-{end}",URL,'-o',str(part)],check=True)
            raw=part.read_bytes()
            if raw[:4]!=b'PK\x03\x04':raise ValueError('Unexpected local header')
            header=struct.unpack_from('<4s5H3L2H',raw,0);start=30+header[-2]+header[-1]
            data=raw[start:start+r['compressed']]
            if r['method']==8:data=zlib.decompress(data,-15)
            elif r['method']!=0:raise ValueError('Unsupported ZIP compression')
            if len(data)!=r['size'] or zlib.crc32(data)!=r['crc']:raise ValueError('Member size/CRC mismatch')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for n,_ in enumerate(pool.map(fetch,chosen),1):
            if n%10==0:print(f'{n}/{len(chosen)} clips verified',flush=True)
    print('Finished selected clip download; signer metadata remains unavailable.')

if __name__=='__main__':main()
