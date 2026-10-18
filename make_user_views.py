"""Expose only rule-qualified hints in the main user view; keep diagnostics."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from render_results import render_markdown, summarize_result
from tex_sources import project_result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    directory=Path(args.run_dir).resolve()
    source=directory/'results.jsonl'
    total=0
    states=Counter()
    qualified=[]
    observations={}
    with source.open(encoding='utf-8') as handle:
        for line in handle:
            row=project_result(json.loads(line))
            total+=1
            states[row['algorithm'].get('candidate_state','unknown')]+=1
            support=row.get('support',{})
            key=support.get('group_key')
            if key:
                group=observations.setdefault(key,{'group_key':key,'old_paper':row['old'].get('paper_id'),
                    'old_statement_id':row['old'].get('statement_id'),
                    'pattern_labels':[s.get('label','') for s in row['algorithm'].get('signals',[])[:3]],
                    'raw_observed_paper_ids':set(),'qualified_paper_ids':set(),'member_pair_ids':[],
                    'target_statement_adoption_status':'unknown','mathematical_weakening':'unknown'})
                group['member_pair_ids'].append(row['pair_id'])
                raw_ids=support.get('raw_observed_later_paper_ids',support.get('raw_observed_paper_ids',[])) or []
                group['raw_observed_paper_ids'].update(raw_ids)
                group['qualified_paper_ids'].update(support.get('later_paper_ids',[]) or [])
                old_date=row['old'].get('date','')
                new_date=row['new'].get('date','')
                if (row['old'].get('paper_id')!=row['new'].get('paper_id') and old_date and new_date and old_date<new_date):
                    group['raw_observed_paper_ids'].add(row['new']['paper_id'])
            if row['algorithm'].get('eligible_for_strength_queue') is True:
                qualified.append(row)
    qualified.sort(key=lambda r:(r['algorithm'].get('investigation_queue',{}).get('rank',999),
                                 not r['algorithm'].get('eligible_for_self_revision_queue',False),
                                 -r['algorithm']['priority_score'],r['pair_id']))
    primary=[r for r in qualified if r['algorithm'].get('eligible_for_primary_investigation_queue') is True]
    alert_path=directory/'alerts.jsonl'
    with alert_path.open('w',encoding='utf-8') as handle:
        for row in qualified:
            handle.write(json.dumps(summarize_result(row),ensure_ascii=False,sort_keys=True)+'\n')
    primary_alert_path=directory/'primary_alerts.jsonl'
    with primary_alert_path.open('w',encoding='utf-8') as handle:
        for row in primary:
            handle.write(json.dumps(summarize_result(row),ensure_ascii=False,sort_keys=True)+'\n')
    primary_report_path=directory/'优先调查.md'
    primary_scope=('优先项由核心eligible_for_primary_investigation_queue确定：同篇/作者姓名关联的新增前提文字，'
                   '已扫描材料中未找到对应修订说明。没有认证人物身份、数学弱化或全渠道未说明。')
    primary_report_path.write_text(render_markdown(primary,metadata={'max_display':30,'scope':primary_scope,
                                       'preserve_input_order':True}),encoding='utf-8')
    scope=(f'已有候选共{total}对；当前确定性规则允许{len(qualified)}对进入文字强度变化线索队列。'
           f'其余{total-len(qualified)}对保留在results.jsonl及用户结果.md诊断视图，不填充主清单。'
           '主清单资格不代表数学弱化成立；无入列项也不代表没有变化。'
           '作者自身关联、第三方、引用归属和强度方向仍按条目状态区分。')
    header=('# 强度变化线索清单\n\n'+scope+'\n\n'
            f'其中核心标记的优先调查项：{len(primary)}对；[只看优先项]({primary_report_path.as_posix()})。'
            f'作者自身关联队列：{sum(bool(r["algorithm"].get("eligible_for_self_revision_queue")) for r in qualified)}对。'
            f'机器读取请用 [alerts.jsonl]({alert_path.as_posix()})；全量诊断另存。\n\n')
    body=render_markdown(qualified,metadata={'max_display':30,'scope':scope,'preserve_input_order':True})
    if body.startswith('# '):body=body.split('\n',1)[1].lstrip('\n')
    report=header+body
    report_path=directory/'线索清单.md'
    report_path.write_text(report,encoding='utf-8')
    repeated=[]
    for group in observations.values():
        if len(group['raw_observed_paper_ids'])<2:
            continue
        group['raw_observed_paper_ids']=sorted(group['raw_observed_paper_ids'])
        group['qualified_paper_ids']=sorted(group['qualified_paper_ids'])
        group['raw_observed_distinct_later_papers']=len(group['raw_observed_paper_ids'])
        group['qualified_distinct_later_papers']=len(group['qualified_paper_ids'])
        group['scope']='Current input only; same old text and type of text difference. Repeated retrieval is not verified use of the old result or population prevalence.'
        repeated.append(group)
    repeated.sort(key=lambda g:(-g['raw_observed_distinct_later_papers'],g['group_key']))
    repeated_path=directory/'repeated_observations.jsonl'
    with repeated_path.open('w',encoding='utf-8') as handle:
        for group in repeated:handle.write(json.dumps(group,ensure_ascii=False,sort_keys=True)+'\n')
    repeated_report=directory/'后文重复观察.md'
    overview=['# 本批后文重复文字观察','',
        '这是辅助队列：同一旧文本与同类文字差异在至少两篇后文候选中出现。保留这类线索供用户查看；未确认后文实际沿用同一结果，也不表示原结论错误或全库广泛使用。','',
        '|旧文|本批观察后文|通过强度门控的后文|变化类型|组号|','|---|---:|---:|---|---|']
    for group in repeated[:20]:
        label='；'.join(group['pattern_labels']).replace('|','\\|').replace('\n',' ')
        overview.append(f"|{group['old_paper']}|{group['raw_observed_distinct_later_papers']}|{group['qualified_distinct_later_papers']}|{label}|{group['group_key']}|")
    if not repeated:overview.append('\n本批没有达到两篇不同后文的文字重复观察组；不能据此推断全库不存在。')
    overview.extend(['',f'全量组和论文编号：[repeated_observations.jsonl]({repeated_path.as_posix()})。计数不能跨组或跨行直接相加。'])
    repeated_report.write_text('\n'.join(overview)+'\n',encoding='utf-8')
    digest=hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda:stream.read(65536),b''):digest.update(chunk)
    manifest={'source_results':str(source),'source_sha256':digest.hexdigest(),
              'input_pairs':total,'qualified_strength_hints':len(qualified),
              'qualified_self_revision_hints':sum(bool(r['algorithm'].get('eligible_for_self_revision_queue')) for r in qualified),
              'primary_investigation_hints':len(primary),
              'candidate_states':dict(states),'filter':'algorithm.eligible_for_strength_queue is True',
              'repeated_observation_groups':len(repeated),
              'human_or_model_decision_dependencies':False,'mathematical_weakening_certified':False,
              'empty_result_means_no_change':False,
              'outputs':{p.name:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
                         for p in (alert_path,report_path,repeated_path,repeated_report,primary_alert_path,primary_report_path)}}
    (directory/'user_views_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'input_pairs':total,'qualified_strength_hints':len(qualified),'output':str(report_path)},ensure_ascii=False))


if __name__=='__main__':
    main()
