"""Read-only FineWeb Parquet text search, preserving complete matching rows.

Stage at most one Parquet per worker; validate its declared SHA before scanning.
Independent workers stop when the complete terminal text is found.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import shlex
import shutil
import subprocess
import time
import unicodedata
import urllib.request

REV = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
EOS = '<|endoftext|>'
ANCHORS = ('CDCnews', 'As AIDS spreads', 'Sarah Breen')
BRACKETS = {'-LRB-':'(', '-RRB-':')', '-LSB-':'[', '-RSB-':']', '-LCB-':'{', '-RCB-':'}'}

def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
    os.replace(tmp,path)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):
            h.update(b)
    return h.hexdigest()

def normalise(text):
    text = unicodedata.normalize('NFC', text)
    for old,new in BRACKETS.items():
        text = text.replace(old,new)
    text = text.replace('``','"').replace("''",'"')
    return re.sub(r'\s+','',text)

def install_dns(path):
    addresses = json.loads(path.read_text())
    original = socket.getaddrinfo
    def resolve(host,port,*args,**kwargs):
        if host in addresses:
            return [item for ip in addresses[host] for item in original(ip,port,*args,**kwargs)]
        return original(host,port,*args,**kwargs)
    socket.getaddrinfo = resolve

def url(entry):
    return f'https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/{REV}/'+entry['path']

def download(entry, path, opener, stop, via_ssh=None):
    size = path.stat().st_size if path.exists() else 0
    if size > entry['size']:
        raise ValueError('oversized staged file')
    if size < entry['size']:
        headers = {'User-Agent':'Mozilla/5.0'}
        if size:
            headers['Range'] = f'bytes={size}-{entry["size"]-1}'
        if via_ssh:
            assert re.fullmatch(r'sample/100BT/\d+_\d+\.parquet',entry['path'])
            command = ['env','-u','http_proxy','-u','https_proxy','-u','all_proxy','-u','HTTP_PROXY','-u','HTTPS_PROXY','-u','ALL_PROXY',
                       'curl','--fail','--location','--silent','--show-error','--connect-timeout','20','--max-time','3600',
                       '--retry','2','--user-agent','Mozilla/5.0']
            if size:
                command += ['--range',f'{size}-']
            command.append(url(entry))
            stderr = path.with_suffix('.download.err').open('ab')
            child = subprocess.Popen(['ssh',via_ssh,shlex.join(command)],stdout=subprocess.PIPE,stderr=stderr)
            response = child.stdout
        else:
            req = urllib.request.Request(url(entry),headers=headers)
            response = opener.open(req,timeout=45)
            if size:
                assert response.status==206 and response.headers.get('Content-Range')==f'bytes {size}-{entry["size"]-1}/{entry["size"]}'
            else:
                assert response.status==200
        try:
            with path.open('ab') as out:
                for block in iter(lambda:response.read(4<<20),b''):
                    if stop.exists():
                        if via_ssh:
                            child.terminate()
                        return False
                    if out.tell()+len(block)>entry['size']:
                        raise ValueError('oversized network response')
                    out.write(block)
            if via_ssh and child.wait()!=0:
                raise RuntimeError(f'network download failed; see {path.with_suffix(".download.err")}')
        finally:
            response.close()
            if via_ssh:
                if child.poll() is None:
                    child.terminate();child.wait()
                stderr.close()
    assert path.stat().st_size==entry['size']
    assert sha(path)==entry['lfs']['oid'], 'upstream Parquet SHA mismatch'
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--dns',type=Path)
    parser.add_argument('--download-via-ssh',help='Network I/O on a reachable login host; CPU scan remains local')
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--priority-file',action='append',default=[],help='Scan these files first within the worker assignment')
    parser.add_argument('--probe',action='store_true')
    args = parser.parse_args()
    assert 0<=args.worker<args.workers<=4
    args.campaign = args.campaign.resolve()
    args.campaign.mkdir(parents=True,exist_ok=True)
    if args.dns:
        install_dns(args.dns)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    entries = json.loads(args.manifest.read_text())
    # Visit a file from each upload rank early, while examining every file exactly once.
    entries.sort(key=lambda e:tuple(reversed([int(x) for x in Path(e['path']).stem.split('_')])))
    stop = args.campaign/'found.json'
    if args.probe:
        entry = entries[0]
        lo,hi = entry['size']-8,entry['size']-1
        request = urllib.request.Request(url(entry),headers={'User-Agent':'Mozilla/5.0','Range':f'bytes={lo}-{hi}'})
        with opener.open(request,timeout=25) as response:
            assert response.status==206 and response.headers.get('Content-Range')==f'bytes {lo}-{hi}/{entry["size"]}'
            raw = response.read(9)
        assert len(raw)==8 and raw[-4:]==b'PAR1'
        atomic(args.campaign/'network_probe.json',{'status':'passed','job_id':os.getenv('SLURM_JOB_ID'),'source':entry['path'],'tls_verification':True,'proxies':False})
        print('NETWORK_PROBE_PASSED',flush=True)
        return
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    target = args.target.read_text()
    assert target.endswith(EOS)
    expected = normalise(target[:-len(EOS)])
    assigned = entries[args.worker::args.workers]
    assigned.sort(key=lambda entry:entry['path'] not in args.priority_file)
    stage = args.campaign/'staging'/f'worker_{args.worker}'
    stage.mkdir(parents=True,exist_ok=True)
    status_path = args.campaign/f'worker_{args.worker}.json'
    status = {'worker':args.worker,'job_id':os.getenv('SLURM_JOB_ID'),'status':'running',
              'assigned_files':len(assigned),'completed_files':0,'scanned_rows':0,'candidates':0,
              'manifest_sha256':sha(args.manifest),'target_sha256':sha(args.target),'files':[]}
    atomic(status_path,status)
    try:
        for entry in assigned:
            if stop.exists():
                status['status']='stopped_after_match';break
            name = Path(entry['path']).name
            status['current_file']=entry['path'];status['phase']='downloading';atomic(status_path,status)
            file = stage/name
            receipt_path = args.campaign/f'scan_{name}.json'
            if receipt_path.exists():
                old = json.loads(receipt_path.read_text())
                assert old['source_sha256']==entry['lfs']['oid'] and old['target_sha256']==status['target_sha256']
                status['files'].append(old);status['completed_files']+=1;status['scanned_rows']+=old['rows'];continue
            print('DOWNLOAD',entry['path'],entry['size'],flush=True)
            if shutil.disk_usage(stage).free < entry['size'] + (4<<30):
                raise OSError('insufficient local scratch space')
            if not download(entry,file,opener,stop,args.download_via_ssh):
                file.unlink(missing_ok=True);status['status']='stopped_after_match';break
            status['phase']='scanning';atomic(status_path,status)
            parquet = pq.ParquetFile(file)
            offset = 0
            candidates = []
            group_ends = []
            cursor = 0
            for gi in range(parquet.metadata.num_row_groups):
                cursor+=parquet.metadata.row_group(gi).num_rows;group_ends.append(cursor)
            for batch in parquet.iter_batches(batch_size=4096,columns=['text'],use_threads=False):
                if stop.exists():
                    status['status']='stopped_after_match';break
                texts = batch.column(0)
                mask = pc.or_(pc.match_substring(texts,'CDCnews'),pc.or_(
                    pc.match_substring_regex(texts,r'As\s+AIDS\s+spreads'),
                    pc.match_substring_regex(texts,r'Sarah\s+Breen')))
                indices = pc.indices_nonzero(mask).to_pylist()
                for local in indices:
                    at = offset+local
                    gi = next(i for i,end in enumerate(group_ends) if at<end)
                    before = group_ends[gi-1] if gi else 0
                    table = parquet.read_row_group(gi,use_threads=False).slice(at-before,1)
                    row = table.to_pylist()[0]
                    text = row['text'];norm = normalise(text)
                    exact = norm==expected
                    contains = expected in norm
                    cid = f'{Path(name).stem}_row{at}'
                    artifact = args.campaign/f'candidate_{cid}.json'
                    atomic(artifact,row)
                    match = {'source_file':entry['path'],'source_sha256':entry['lfs']['oid'],
                             'row_0_based':at,'row_group':gi,'artifact':str(artifact),
                             'anchors':[s for s in ANCHORS if s in text],
                             'normalized_complete_terminal_equal':exact,
                             'normalized_complete_terminal_contained':contains,
                             'upstream_literal_eos_count':text.count(EOS)}
                    candidates.append(match);status['candidates']+=1
                    print('CANDIDATE',cid,'exact',exact,'contains',contains,flush=True)
                    if exact or contains:
                        with (args.campaign/'match.lock').open('a') as lock:
                            fcntl.flock(lock,fcntl.LOCK_EX)
                            if not stop.exists():
                                arrow = args.campaign/'original_matched_record.arrow'
                                with pa.OSFile(str(arrow),'wb') as sink:
                                    with pa.ipc.new_stream(sink,table.schema) as writer:writer.write_table(table)
                                (args.campaign/'original_matched_text.txt').write_text(text)
                                atomic(args.campaign/'original_matched_record.json',row)
                                match.update(status='matched',job_id=os.getenv('SLURM_JOB_ID'),
                                             record_metadata={k:v for k,v in row.items() if k!='text'},
                                             arrow_sha256=sha(arrow),recovered_arrow_rows=1,
                                             original_parquet_retained=str(file),
                                             matching_policy='remove final terminal EOS, NFC, PTB bracket/quote mapping, ignore whitespace; retain embedded EOS',
                                             source_modified=False)
                                atomic(stop,match);status['status']='matched'
                                print('FOUND',json.dumps(match,ensure_ascii=False,default=str),flush=True)
                            else:
                                status['status']='stopped_after_match';file.unlink(missing_ok=True)
                        atomic(status_path,status)
                        return
                offset+=len(batch)
            if status['status']=='stopped_after_match':
                file.unlink(missing_ok=True);break
            receipt = {'source_file':entry['path'],'source_sha256':entry['lfs']['oid'],
                       'target_sha256':status['target_sha256'],'rows':offset,'candidates':candidates,
                       'source_modified':False}
            atomic(receipt_path,receipt)
            status['files'].append(receipt);status['completed_files']+=1;status['scanned_rows']+=offset
            file.unlink();atomic(status_path,status)
            print('SCANNED',entry['path'],offset,flush=True)
        else:status['status']='complete_assigned'
        atomic(status_path,status)
    except BaseException as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}');atomic(status_path,status)
        raise

if __name__=='__main__':
    main()
