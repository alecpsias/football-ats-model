"""Append-only, SHA-checked archives of CFB frozen pregame forecasts.

A one-run .zip retains the exact original JSON bytes from Build 5 and later.
Never substitute a later rerun for an earlier pregame forecast. The ZIP files
are small enough to commit to a Git repository. Git commit/push authenticity
is separate from the in-file checksum verification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from cfb.market import _date

ZIP_TIME=(2026,1,1,0,0,0)
ID_RE=re.compile(r'^[0-9]{6,20}$')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _zipwrite(z, name, data):
    info=zipfile.ZipInfo(name, ZIP_TIME)
    info.compress_type=zipfile.ZIP_DEFLATED
    info.external_attr=0o600 << 16
    z.writestr(info,data,compresslevel=9)


def _check_index(index, get_bytes):
    if index.get('status') != 'CFB_PROSPECTIVE_RESEARCH_RECORDS_FROZEN':
        raise ValueError('Not frozen CFB research forecasts')
    run=str(index.get('github_run_id',''))
    if not ID_RE.fullmatch(run):
        raise ValueError('Missing numeric GitHub run ID')
    if not isinstance(index.get('predictions'),list):
        raise ValueError('Missing forecast index')
    if index.get('game_records') != len(index['predictions']):
        raise ValueError('Index count mismatch')
    generate=_date(index['generated_at_utc'],'forecast generated')
    source=_date(index['snapshot_at_utc'],'snapshot')
    if source>generate:
        raise ValueError('Forecast generated before source capture')
    seen=set()
    records={}
    for entry in index['predictions']:
        gid=str(entry['game_id'])
        if not ID_RE.fullmatch(gid) or gid in seen or entry.get('file')!=gid+'.json':
            raise ValueError('Duplicate or unsafe CFB game ID')
        seen.add(gid)
        raw=get_bytes(gid+'.json')
        if digest(raw)!=entry['sha256']:
            raise ValueError('Forecast checksum mismatch '+gid)
        p=json.loads(raw)
        if (str(p.get('game_id'))!=gid or p.get('status')!=entry['status'] or
                str(p.get('github_run_id'))!=run):
            raise ValueError('Forecast/index mismatch '+gid)
        kickoff=_date(p['kickoff_utc'],'kickoff')
        pred_at=_date(p['prediction_generated_at_utc'],'prediction time')
        if not source<=pred_at<kickoff or pred_at!=generate:
            raise ValueError('Forecast timing invalid '+gid)
        if p.get('market_quote') is not None or p.get('ats_prediction') is not None:
            raise ValueError('Forecast includes a non-frozen market or ATS pick')
        if p['status']=='FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE':
            if p.get('simulated_margin_home') is None:
                raise ValueError('No scoring margin despite projection status')
        records[gid]=p
    return run,records


def pack_forecasts(source_dir, archive_root):
    """Bundle the *original bytes* and require an empty destination run slot."""
    source_dir=Path(source_dir)
    idx_raw=(source_dir/'index.json').read_bytes()
    idx=json.loads(idx_raw)
    run,records=_check_index(idx,lambda x:(source_dir/x).read_bytes())
    archive_root=Path(archive_root)
    archive_root.mkdir(parents=True,exist_ok=True)
    target=archive_root/(run+'.zip')
    if target.exists():
        raise FileExistsError('Refusing overwrite of archived CFB run '+run)
    with zipfile.ZipFile(target,'x') as z:
        _zipwrite(z,'index.json',idx_raw)
        for gid in sorted(records):
            _zipwrite(z,gid+'.json',(source_dir/(gid+'.json')).read_bytes())
    verify_archive(target)
    return {'run_id':run,'archive':str(target),'forecast_records':len(records),
            'zip_sha256':digest(target.read_bytes())}


def verify_archive(file):
    file=Path(file)
    if not ID_RE.fullmatch(file.stem) or file.suffix!='.zip':
        raise ValueError('Forecast archive filename must be numeric GitHub run ID.zip')
    with zipfile.ZipFile(file) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or not names or names[0]!='index.json':
            raise ValueError('Unsafe or duplicate forecast archive entries')
        if len(names)>1000 or any(x!='index.json' and
                (not x.endswith('.json') or not ID_RE.fullmatch(x[:-5])) for x in names):
            raise ValueError('Unsafe forecast archive entries')
        index=json.loads(z.read('index.json'))
        run,records=_check_index(index,z.read)
        if run!=file.stem or set(names)!={'index.json',*(g+'.json' for g in records)}:
            raise ValueError('Unexpected forecast archive content')
    return index,records



def seed_archive(source, root):
    """Restore first historical forward run unchanged from permanent repo seed."""
    source=Path(source)
    index,records=verify_archive(source)
    folder=Path(root);folder.mkdir(parents=True,exist_ok=True)
    target=folder/source.name
    if target.exists():
        if digest(target.read_bytes())!=digest(source.read_bytes()):
            raise ValueError('Historic first forecast archive differs: refusing overwrite')
        return {'seeded':False,'existing_verified':True,'games':len(records)}
    shutil.copyfile(source,target)
    verify_archive(target)
    return {'seeded':True,'games':len(records),'archive':str(target)}


def load_all_archives(root):
    """Earliest frozen forecast per game; no results-dependent selection."""
    all_p={}
    archives=sorted(Path(root).glob('*.zip')) if Path(root).is_dir() else []
    for file in archives:
        idx,records=verify_archive(file)
        for gid,p in records.items():
            if p['status']!='FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE':
                continue
            row={**p,'original_forecast_run_id':idx['github_run_id'],
                 'original_forecast_zip_sha256':digest(file.read_bytes())}
            prev=all_p.get(gid)
            if prev is None or (_date(p['prediction_generated_at_utc'],'generated'),idx['github_run_id']) < (
                    _date(prev['prediction_generated_at_utc'],'generated'),prev['original_forecast_run_id']):
                all_p[gid]=row
    return all_p


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('pack');a.add_argument('--frozen',required=True);a.add_argument('--out',required=True)
    b=sub.add_parser('verify');b.add_argument('file')
    c=sub.add_parser('summarize');c.add_argument('--archive-root',required=True)
    d=sub.add_parser('seed');d.add_argument('--source',required=True);d.add_argument('--out',required=True)
    args=p.parse_args()
    if args.command=='pack':
        result=pack_forecasts(args.frozen,args.out)
    elif args.command=='verify':
        i,records=verify_archive(args.file);result={'run_id':i['github_run_id'],'records':len(records),'verified':True}
    elif args.command=='seed':result=seed_archive(args.source,args.out)
    else:
        preds=load_all_archives(args.archive_root)
        result={'games':len(preds),'first_issued_forecasts_only':True}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
