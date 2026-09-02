from __future__ import annotations
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from .data_cleaner import clean_market_data
from .strategy import StrategyConfig, prepare_features, signal_mask

HORIZONS=(30,60,120)
@dataclass
class RetestDiscoveryResult:
    events: pd.DataFrame; funnel: pd.DataFrame; summary: pd.DataFrame; buckets: pd.DataFrame; examples: pd.DataFrame

def _split(s,ds,de,vs,ve):
    d=pd.Timestamp(s).date()
    if pd.Timestamp(ds).date()<=d<=pd.Timestamp(de).date(): return 'DEV'
    if pd.Timestamp(vs).date()<=d<=pd.Timestamp(ve).date(): return 'VALIDATION'

def _fwd(day,i,m,entry,atr):
    ids=day.index[pd.to_datetime(day.date)>=pd.Timestamp(day.loc[i,'date'])+pd.Timedelta(minutes=m)]
    return np.nan if len(ids)==0 else (float(day.loc[int(ids[0]),'close'])-entry)/atr

def _event(day,i,symbol,split):
    s=day.loc[i]; ts=pd.Timestamp(s.date); atr=float(s.atr); orh=float(s.or_high)
    if not(np.isfinite(atr) and atr>0 and np.isfinite(orh)): return None
    future=day[(pd.to_datetime(day.date)>ts)&(pd.to_datetime(day.date)<=ts+pd.Timedelta(minutes=60))]
    # Discovery definition: pullback reaches ORH +0.25 ATR or closer, then candle closes at/above ORH.
    c=future[(future.low<=orh+.25*atr)&(future.close>=orh)]
    if c.empty:return None
    j=int(c.index[0]); r=day.loc[j]; rt=pd.Timestamp(r.date); pre=day.loc[i:j]; entry=float(r.close)
    vwap=float(r.vwap) if pd.notna(r.get('vwap')) else np.nan
    out=dict(split=split,symbol=symbol,session=pd.Timestamp(s.session),signal_time=ts,retest_time=rt,
      minutes_to_retest=(rt-ts).total_seconds()/60,signal_rvol=float(s.rvol),gap_pct=float(s.gap_pct),
      breakout_extension_atr=(float(s.close)-orh)/atr,max_extension_before_retest_atr=(float(pre.high.max())-orh)/atr,
      deepest_pullback_vs_or_atr=(float(pre.low.min())-orh)/atr,retest_close_vs_or_atr=(entry-orh)/atr,
      retest_close_vs_vwap_pct=((entry/vwap)-1)*100 if np.isfinite(vwap) and vwap>0 else np.nan,
      retest_rvol=float(r.rvol) if pd.notna(r.get('rvol')) else np.nan,retest_above_vwap=bool(np.isfinite(vwap) and entry>=vwap),
      retest_bullish=bool(float(r.close)>float(r.open)),entry=entry,atr=atr)
    for h in HORIZONS:out[f'fwd_{h}m_r']=_fwd(day,j,h,entry,atr)
    return out

def run_retest_discovery(files:Iterable[Path],cfg:StrategyConfig,dev_start,dev_end,validation_start,validation_end):
    rows=[]; funnel=[]; files=list(files)
    for n,path in enumerate(files,1):
        symbol=path.name.split('_')[1] if '_' in path.name else path.stem
        clean=clean_market_data(pd.read_parquet(path),symbol).data
        if clean.empty:continue
        f=prepare_features(clean,cfg); ns=nr=0
        for session,d0 in f.groupby('session',sort=True):
            split=_split(session,dev_start,dev_end,validation_start,validation_end)
            if not split:continue
            d=d0.sort_values('date').reset_index(drop=True); ids=d.index[signal_mask(d,cfg)]
            if len(ids)==0:continue
            ns+=1;e=_event(d,int(ids[0]),symbol,split)
            if e is not None:rows.append(e);nr+=1
        funnel.append(dict(symbol=symbol,signals=ns,retests=nr))
        print(f'\rV14 discovery {n}/{len(files)} {symbol:14s} events={len(rows)}',end='',flush=True)
    print();ev=pd.DataFrame(rows);fun=pd.DataFrame(funnel)
    if ev.empty:return RetestDiscoveryResult(ev,fun,pd.DataFrame(),pd.DataFrame(),pd.DataFrame())
    sums=[]
    for split,g in ev.groupby('split'):
        z=dict(split=split,retests=len(g),above_vwap_rate=g.retest_above_vwap.mean(),bullish_retest_rate=g.retest_bullish.mean())
        for h in HORIZONS:
            x=g[f'fwd_{h}m_r'].dropna();z[f'avg_{h}m_r']=x.mean();z[f'median_{h}m_r']=x.median();z[f'positive_{h}m_rate']=(x>0).mean()
        sums.append(z)
    specs=[('time_to_retest','minutes_to_retest',[-np.inf,10,20,30,45,np.inf],['<=10','10-20','20-30','30-45','>45']),('pullback_depth_atr','deepest_pullback_vs_or_atr',[-np.inf,-.5,-.25,0,.25,np.inf],['<=-.5','-.5..-.25','-.25..0','0..0.25','>0.25']),('retest_vwap_pct','retest_close_vs_vwap_pct',[-np.inf,0,.25,.5,1,np.inf],['<=0','0-.25','.25-.5','.5-1','>1']),('retest_rvol','retest_rvol',[-np.inf,1.5,3,5,np.inf],['<1.5','1.5-3','3-5','>5'])]
    br=[]
    for name,col,bins,labels in specs:
        q=ev.copy();q['bucket']=pd.cut(q[col],bins=bins,labels=labels)
        for (split,b),g in q.dropna(subset=['bucket']).groupby(['split','bucket'],observed=True):br.append(dict(feature=name,split=split,bucket=str(b),events=len(g),avg_60m_r=g.fwd_60m_r.mean(),avg_120m_r=g.fwd_120m_r.mean(),positive_60m_rate=(g.fwd_60m_r>0).mean()))
    ex=[]
    for split,g in ev.dropna(subset=['fwd_120m_r']).groupby('split'):
        for typ,gg in [('SUCCESS',g.nlargest(3,'fwd_120m_r')),('FAIL',g.nsmallest(3,'fwd_120m_r'))]:z=gg.copy();z['example_type']=typ;ex.append(z)
    return RetestDiscoveryResult(ev,fun,pd.DataFrame(sums),pd.DataFrame(br),pd.concat(ex,ignore_index=True))

def _chart(g,title):
    if g.empty:return ''
    vals=g.avg_60m_r.fillna(0).tolist();mx=max(.1,max(abs(v) for v in vals));w=760/max(1,len(vals));p=[f'<h3>{escape(title)}</h3><svg viewBox="0 0 900 280">']
    for i,(_,r) in enumerate(g.reset_index(drop=True).iterrows()):
        x=80+i*w;v=float(r.avg_60m_r);hh=100*abs(v)/mx;y=130-hh if v>=0 else 130;p.append(f'<rect x="{x}" y="{y}" width="{max(8,w-6)}" height="{hh}" fill="currentColor" opacity=".65"/><text x="{x}" y="260" font-size="11">{escape(str(r.bucket))}</text>')
    return ''.join(p)+'<line x1="60" y1="130" x2="880" y2="130" stroke="currentColor" opacity=".4"/></svg>'

def write_retest_reports(r:RetestDiscoveryResult,report_dir:Path,cfg:StrategyConfig):
    report_dir.mkdir(parents=True,exist_ok=True);paths={}
    for n,d in [('events',r.events),('funnel',r.funnel),('summary',r.summary),('feature_buckets',r.buckets),('examples',r.examples)]:
        p=report_dir/(n+'.parquet' if n=='events' else n+'.csv');d.to_parquet(p,index=False) if n=='events' else d.to_csv(p,index=False);paths[n]=p
    cards=''.join(f'<div class="card"><b>{x.split}</b><br>Retests {int(x.retests)}<br>60m {x.avg_60m_r:+.3f}R<br>120m {x.avg_120m_r:+.3f}R</div>' for _,x in r.summary.iterrows())
    charts=''.join(_chart(g,f'{a} — {b}') for (a,b),g in r.buckets.groupby(['feature','split'],sort=False))
    cols=['split','example_type','symbol','session','signal_time','retest_time','fwd_60m_r','fwd_120m_r'];table=r.examples[cols].to_html(index=False)
    html='<!doctype html><meta charset="utf-8"><title>V14 Retest Discovery</title><style>body{font-family:system-ui;margin:30px;max-width:1200px}.grid{display:flex;gap:15px}.card{padding:16px;border:1px solid #bbb;border-radius:12px}svg{width:100%;height:280px}table{border-collapse:collapse;width:100%}td,th{padding:6px;border-bottom:1px solid #ddd;font-size:12px}</style><h1>V14 Retest Discovery + Opportunity Characteristics</h1><p><b>Discovery only.</b> No capital allocation or future-informed live ranking. 2026 remains locked.</p><div class="grid">'+cards+'</div><h2>Feature diagnostics</h2>'+charts+'<h2>Successful / failed historical examples</h2><p>Examples use future outcome only for visual research, never as a live ranking rule.</p>'+table
    hp=report_dir/'retest_discovery_dashboard.html';hp.write_text(html,encoding='utf-8');paths['dashboard']=hp;return paths
