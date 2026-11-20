"""Reserve fresh dictionary examples for evaluation, never for training."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,re,subprocess
from pathlib import Path
from urllib.parse import urlparse,unquote
from scripts.acquire_wlasl import inspect_video


def main():
    root=Path('datasets/raw/fresh-dictionary');root.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text())
    fitted=json.loads(Path('datasets/preprocessed/word-dense/extraction.json').read_text())['rows']
    ids={r['video_id'] for r in fitted}
    basenames={unquote(urlparse(i['url']).path).rsplit('/',1)[-1].lower() for e in metadata for i in e['instances'] if str(i['video_id']) in ids}
    known_hashes={r['video_sha256'] for r in fitted}
    for p in Path('datasets/raw/web-evaluation').glob('*.mp4'):known_hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
    choices=[];log=[]
    for label in ['computer','who','deaf','year','cool','hello','food']:
        page='https://www.signasl.org/sign/'+label
        response=subprocess.run(['curl','-fL','--silent','--show-error','--max-time','20',page],capture_output=True,text=True)
        if response.returncode:log.append(dict(label=label,status='page_failed'));continue
        urls=list(dict.fromkeys(re.findall(r'<source\s+src="(https://[^\"]+\.mp4)"',response.stdout)))
        selected=[]
        for url in urls:
            basename=unquote(urlparse(url).path).rsplit('/',1)[-1].lower()
            stem=basename[:-4]
            if not re.fullmatch(re.escape(label)+r'(?:[-_][a-z0-9-]+|\d*)?',stem):continue
            if basename in basenames:log.append(dict(label=label,url=url,status='known_source_name_excluded'));continue
            selected.append(dict(label=label,url=url,source_page=page,role='fresh_evaluation_only'))
            if len(selected)==2:break
        choices.extend(selected)
    for provider in ['signlanguagestudent','elementalaslconcepts']:
        choices.append(dict(label='who',url='https://media.signbsl.com/videos/asl/'+provider+'/mp4/who.mp4',source_page='https://www.signasl.org/sign/who',role='fresh_evaluation_only'))
    def download(row):
        key=hashlib.sha256(row['url'].encode()).hexdigest()[:12];out=root/(row['label']+'-'+key+'.mp4')
        row.update(path=str(out.resolve()),video_id=key)
        try:
            if not out.exists():
                subprocess.run(['curl','-fL','--silent','--show-error','--max-time','20','--max-filesize','15000000','--proto','=https','--proto-redir','=https',row['url'],'-o',str(out)],check=True,capture_output=True,timeout=25)
            info=inspect_video(out);sha=hashlib.sha256(out.read_bytes()).hexdigest()
            if sha in known_hashes:raise ValueError('Exact content already used or evaluated')
            if info['duration_s']>10:raise ValueError('Needs reviewed segment')
            row.update(status='downloaded',sha256=sha,decode=info)
        except (ValueError,OSError,subprocess.SubprocessError) as exc:row.update(status='excluded_or_failed',error=str(exc))
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(download,choices))
    (root/'sources.json').write_text(json.dumps(dict(sources=rows,selection_log=log,
        policy='First two matching dictionary media names per label after excluding known source basenames. '
        'Exact hashes checked against dense dataset and previous web clips. Labels from dictionary, no inference-based selection. '
        'Re-encoded duplicates and signer overlap cannot be ruled out. hello/food are unsupported-word probes.'),indent=2)+'\n')
    for row in rows:print(row['label'],row['status'],row['url'],row.get('error',''),flush=True)

if __name__=='__main__':main()
