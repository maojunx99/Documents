"""Refine inspected V2 false positives without changing token quotas or sources.

Function examples labelled Input/Output are not automatically stdio tasks.
The census remains provisional until the required full-row audit.
"""
import re
from pathlib import Path
import prepare_taxonomy_v2 as base

STRONG_STDIO = re.compile(r'\b(?:standard input|standard output|input specification|output specification|time limit|memory limit|codeforces|codechef|atcoder|stdin|stdout|flush the output)\b|输入格式|输出格式', re.I)
IO_HEADING = re.compile(r'(?:^|\n)\s*[-# ]*(?:input|output)(?:\s+format)?\s*[-:# ]*(?:\n|$)', re.I)
FUNCTION_SIGNATURE = re.compile(r'function signature|def\s+\w+\s*\(|complete the function|function problem|do not need to read input|don.t (?:have to|need to) (?:take|read) input', re.I)
REPAIR_EXPLICIT = re.compile(r'\b(?:debug|fix|repair|refactor|code review|unit tests?)\b|\b(?:modify|change|optimi[sz]e)\b.{0,80}\b(?:following|given|existing|provided|accompanying)\b|修复|调试', re.I | re.S)
EXPLAIN_EXISTING = re.compile(r'\b(?:explain|analy[sz]e|describe|understand)\b.{0,100}\b(?:following|given|existing|provided|this)\b.{0,50}\b(?:code|function|program|implementation)\b|解释.{0,15}代码', re.I | re.S)


def classify(q, source, flags, complete):
    body=re.sub(r'^\s*Tool available[^\n]*(?:\n|$)', '', q, flags=re.I)
    if source in base.MATH_SOURCE:return 'math','math_source',False
    if source=='ultrainteract' and base.MATH_TASK.search(body) and not STRONG_STDIO.search(body):
        explicit=bool(REPAIR_EXPLICIT.search(body) or re.search(r'\b(?:write|implement|create|modify)\b.{0,100}\b(?:code|function|program|method)\b',body,re.I|re.S))
        if not explicit:return 'math','ultrainteract_math_task_not_tool_language',False
    function_contract=bool(FUNCTION_SIGNATURE.search(body))
    contest=bool(STRONG_STDIO.search(body) or (IO_HEADING.search(body) and not function_contract))
    if source in base.REPAIR_SOURCE and contest:return 'code_reasoning_and_repair','explicit_stdio_or_contest',True
    if source in base.REPAIR_SOURCE and (base.CODE.search(body) or flags['code']['python_function_ast_ok']) and (REPAIR_EXPLICIT.search(body) or EXPLAIN_EXISTING.search(body)):
        return 'code_reasoning_and_repair','explicit_repair_or_existing_code_explanation',False
    if source in base.SHORT_SOURCE:
        function_task=bool(function_contract or base.FUNCTION.search(body) or flags['code']['python_function_ast_ok'])
        if function_task:
            if not complete:return 'excluded_short_function_truncated','primary_code_implementation_must_be_uncut',False
            if flags['code']['python_function_ast_ok']:return 'short_function','allowed_source_complete_python_function_task_review_required',False
            return 'review_other_language_function','no_verified_complete_python_function',False
    if source in base.GENERAL and not (base.CODE.search(body) and (base.FUNCTION.search(body) or contest or base.REPAIR.search(body))):
        return 'general_instruction','allowed_general_source',False
    return 'review_ambiguous','not_confidently_classified',contest


if __name__=='__main__':
    # Reuse the already reviewed census plumbing, versioning every output name.
    source=Path(base.__file__).read_text().replace('taxonomy_v2','taxonomy_v3')
    namespace={'__name__':'taxonomy_v3_census','__file__':__file__}
    exec(compile(source,__file__,'exec'),namespace)
    namespace['classify']=classify
    import argparse,time,traceback
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--storage',type=Path,required=True);x=a.parse_args()
    try:namespace['main'](x.run,x.storage)
    except BaseException as e:
        base.write(x.run/'taxonomy_v3_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time(),identity=base.identity()));raise
