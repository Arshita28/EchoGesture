"""Download publisher archive using bounded parallel HTTP ranges; verify SHA-256."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata_url = 'https://data.mendeley.com/public-api/datasets/xs6mvhx6rh'
    metadata = json.loads(subprocess.check_output(
        ['curl', '-sSL', '--fail', '--max-time', '30', metadata_url]))
    if metadata['version'] != 1 or metadata['data_licence']['short_name'] != 'CC BY 4.0':
        raise ValueError('Publisher version/license changed; review before importing')
    file = metadata['files'][0]['content_details']
    target = ROOT / 'data/raw/ASLYset.rar'
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = file['sha256_hash']
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        print('Archive already verified')
        return
    size, chunk = file['size'], 4 * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix='aslyset-') as temporary:
        def fetch(index):
            start, end = index * chunk, min((index + 1) * chunk, size) - 1
            part = Path(temporary) / str(index)
            subprocess.run(['curl', '-sSL', '--fail', '--retry', '3', '--max-time', '180',
                            '--range', f'{start}-{end}', file['download_url'], '-o', str(part)], check=True)
            if part.stat().st_size != end - start + 1:
                raise ValueError('Server returned unexpected range size')
            return index
        count = (size + chunk - 1) // chunk
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            futures = [pool.submit(fetch, i) for i in range(count)]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                future.result()
                if completed % 20 == 0:
                    print(f'{completed}/{count} chunks downloaded', flush=True)
        digest = hashlib.sha256()
        pending = target.with_suffix('.partial')
        with pending.open('wb') as output:
            for i in range(count):
                content = (Path(temporary) / str(i)).read_bytes()
                digest.update(content)
                output.write(content)
        if digest.hexdigest() != expected:
            raise ValueError('Archive checksum mismatch')
        pending.replace(target)
    (ROOT / 'docs/aslyset-metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Verified {target}', flush=True)


if __name__ == '__main__':
    main()
