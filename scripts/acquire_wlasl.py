"""Acquire a bounded batch of official direct HTTPS MP4 clips, with decode checks.

YouTube, SWF, HTTP-only and nontrivial trim entries are logged as skipped.
No host substitutions, authentication, TLS bypasses or third-party mirrors.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlparse


def inspect_video(path, max_frames=4500):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise ValueError('Cannot open video')
        fps=float(cap.get(cv2.CAP_PROP_FPS));declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not 0 < fps <= 120:raise ValueError('Invalid or unsupported FPS')
        count=0;shape=None
        while True:
            ok,frame=cap.read()
            if not ok:break
            if frame is None or frame.ndim!=3:raise ValueError('Invalid frame')
            if shape is not None and frame.shape!=shape:raise ValueError('Dimensions change')
            shape=frame.shape;count+=1
            if count>max_frames:raise ValueError('Video exceeds decode frame budget')
        if count<8:raise ValueError('Fewer than eight decoded frames')
        if count/fps>90:raise ValueError('Video exceeds 90 second isolated-clip budget')
        if declared>0 and abs(count-declared)>1:raise ValueError('Decoded count disagrees with container')
        return dict(decoded_frames=count,fps=fps,duration_s=count/fps,width=shape[1],height=shape[0])
    finally:cap.release()


def eligibility(item):
    url=urlparse(item.get('url',''))
    if url.scheme!='https':return 'requires_http_or_unsupported_scheme'
    if url.hostname in ('youtube.com','www.youtube.com','youtu.be'):return 'requires_youtube_downloader'
    if not url.path.lower().endswith('.mp4'):return 'requires_format_conversion'
    if item['frame_start']!=1 or item['frame_end']!=-1:return 'requires_explicit_source_trimming'
    return None


def acquire(item, directory):
    video_id=item['video_id'];destination=directory/(video_id+'.mp4')
    row={k:item[k] for k in ('video_id','label','split','signer_id','url')}
    temp=directory/(video_id+'.part.mp4')
    try:
        if destination.exists():
            info=inspect_video(destination);row.update(status='cached',**info)
        else:
            proc=subprocess.run(['curl','--fail','--location','--silent','--show-error',
                '--max-time','15','--max-filesize','30000000','--proto','=https',
                '--proto-redir','=https',item['url'],'--output',str(temp)],
                capture_output=True,text=True,timeout=20)
            if proc.returncode:raise ValueError(proc.stderr.strip()[-400:] or 'Download failed')
            info=inspect_video(temp);temp.replace(destination);row.update(status='downloaded',**info)
        row['sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
        row['trim_policy']='direct_isolated_full_clip_as_official_preprocess'
        row['path']=str(destination.resolve())
    except (ValueError,OSError,subprocess.TimeoutExpired) as exc:
        row.update(status='failed',error=str(exc))
    finally:
        if temp.exists():temp.unlink()
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata',type=Path)
    parser.add_argument('--classes',type=int,default=30)
    parser.add_argument('--limit',type=int,default=90,help='Max eligible entries attempted in this batch')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path('datasets/raw/wlasl'))
    args=parser.parse_args()
    data=json.loads(args.metadata.read_text())
    if not 2<=args.classes<=len(data) or args.limit<1 or not 1<=args.workers<=4:parser.error('Invalid bounds')
    args.output.mkdir(parents=True,exist_ok=True);videos=args.output/'videos';videos.mkdir(exist_ok=True)
    skipped=[];eligible=[]
    # Interleave labels to cover vocabulary before taking additional examples.
    queues=[]
    for entry in data[:args.classes]:
        queue=[]
        for instance in entry['instances']:
            item={**instance,'label':entry['gloss']}
            if not str(item['video_id']).isdigit():raise ValueError('Invalid video ID')
            reason=eligibility(item)
            if reason:skipped.append(dict(video_id=item['video_id'],label=item['label'],status='skipped',reason=reason))
            else:queue.append(item)
        queues.append(queue)
    while any(queues):
        for queue in queues:
            if queue:eligible.append(queue.pop(0))
    selected=eligible[:args.limit];results=[]
    # Bounded workers; avoid unbounded downloader retries or full-corpus acquisition.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(acquire,item,videos) for item in selected]
        for future in as_completed(jobs):
            results.append(future.result())
            if len(results)%10==0:print(f'Completed {len(results)}/{len(jobs)}; '+str(dict(Counter(r['status'] for r in results))),flush=True)
    report=dict(metadata_sha256=hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        classes=args.classes,eligible_total=len(eligible),attempted=len(selected),
        not_attempted=len(eligible)-len(selected),status_counts=dict(Counter(r['status'] for r in results)),
        skipped_counts=dict(Counter(r['reason'] for r in skipped)),results=sorted(results,key=lambda r:r['video_id']),skipped=skipped,
        note='Decode-valid does not establish correct signing, bounding boxes, signer identity or training adequacy. Official splits preserved. No training.')
    name=args.output/'acquisition.json';name.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','skipped')},indent=2))


if __name__=='__main__':main()
