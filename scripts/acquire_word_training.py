"""Acquire additional official training-only word clips; never imports evaluation data.

Requires yt-dlp in the chosen downloader Python and an ffmpeg executable.
Outputs separate metadata and media; existing manifests and models untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit, unquote
from scripts.acquire_wlasl import inspect_video


def dictionary_mirror_ids(metadata, sources):
    """Reserve WLASL originals that share a dictionary mirror's provider and filename."""
    mirrors = set()
    for source in sources:
        url = urlsplit(source['url'])
        parts = unquote(url.path).lower().strip('/').split('/')
        if url.hostname == 'media.signbsl.com' and len(parts) >= 5 and parts[:2] == ['videos', 'asl']:
            mirrors.add((parts[2], parts[-1]))
    reserved = set()
    for entry in metadata:
        for item in entry['instances']:
            url = urlsplit(item['url'])
            filename = unquote(url.path).lower().split('/')[-1]
            for domain_part in (url.hostname or '').lower().split('.'):
                if (domain_part, filename) in mirrors:
                    reserved.add(str(item['video_id']))
    return reserved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--downloader-python',required=True)
    p.add_argument('--downloader-packages',required=True)
    p.add_argument('--ffmpeg',required=True)
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--labels',type=Path,default=Path('config/word_prototype.json'))
    p.add_argument('--output',type=Path,default=Path('datasets/raw/word-training-extension'))
    a=p.parse_args()
    root=a.output;root.mkdir(parents=True,exist_ok=True)
    videos=root/'videos';videos.mkdir(exist_ok=True)
    metadata=json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text())
    labels=json.loads(a.labels.read_text())['labels']
    external=json.loads(Path('datasets/raw/web-evaluation/sources-segmented.json').read_text())
    forbidden={r['url'] for r in external}
    fresh=Path('datasets/raw/fresh-dictionary/sources.json')
    weak = Path('datasets/raw/weak80-evaluation/sources.json')
    weak_sources = json.loads(weak.read_text()) if weak.exists() else []
    forbidden.update(r['url'] for r in weak_sources)
    dictionary_sources = json.loads(fresh.read_text())['sources'] if fresh.exists() else []
    forbidden.update(r['url'] for r in dictionary_sources)
    reserved_ids = dictionary_mirror_ids(metadata, dictionary_sources + weak_sources)
    existing=json.loads(Path('datasets/preprocessed/words/extraction.json').read_text())
    existing_ids={r['video_id'] for r in existing['rows']}
    selected=[]
    queues=[]
    for entry in metadata:
        if entry['gloss'] not in labels:continue
        queues.append([(entry['gloss'],i) for i in entry['instances'] if i['split']=='train'
            and ('youtube.com/' in i['url'] or 'youtu.be/' in i['url'])
            and i['url'] not in forbidden and str(i['video_id']) not in existing_ids | reserved_ids])
    while any(queues) and len(selected)<a.limit:
        for q in queues:
            if q and len(selected)<a.limit:selected.append(q.pop(0))
    env={**os.environ,'PYTHONPATH':a.downloader_packages};results=[]
    for label,item in selected:
        identity=str(item['video_id'])
        if not identity.isdigit():raise ValueError('Invalid video ID')
        raw=root/(identity+'-source.mp4');out=videos/(identity+'.mp4')
        row={**item,'label':label,'role':'training_only','path':str(out.resolve())}
        try:
            if not raw.exists():
                subprocess.run([a.downloader_python,'-m','yt_dlp','--no-playlist','--socket-timeout','15',
                    '--retries','0','--fragment-retries','0','--max-filesize','30M','--no-warnings',
                    '-f','bestvideo[height<=480][ext=mp4]','-S','vcodec:h264','-o',str(raw),item['url']],
                    env=env,check=True,capture_output=True,timeout=90)
            command=[a.ffmpeg,'-v','error','-y','-i',str(raw)]
            if item['frame_end']!=-1:
                command+=['-ss',str((item['frame_start']-1)/item['fps']),'-t',str((item['frame_end']-item['frame_start']+1)/item['fps'])]
            elif item['frame_start']!=1:raise ValueError('Ambiguous open-ended trim')
            subprocess.run(command+['-an','-c:v','libx264','-crf','18',str(out)],check=True,capture_output=True)
            info=inspect_video(out)
            if info['duration_s']>10:raise ValueError('More than ten seconds; review segmentation')
            row.update(status='acquired',sha256=hashlib.sha256(out.read_bytes()).hexdigest(),decode=info)
        except (ValueError,OSError,subprocess.SubprocessError) as exc:
            row.update(status='failed',error=str(exc))
            if out.exists():out.unlink()
        results.append(row)
        (root/'acquisition.json').write_text(json.dumps(results,indent=2)+'\n')
        print(label,identity,row['status'],flush=True)
    print('Acquired',sum(r['status']=='acquired' for r in results),'of',len(results),'training-only clips')

if __name__=='__main__':main()
