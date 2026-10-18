"""Versioned source-only structural filter; no model scoring or dataset execution."""
import re,json

EXTERNAL=re.compile(r'\b(django|flask|tensorflow|pytorch|selenium|docker|kubernetes|database|sql|web server|web application|api key|rest api|https?|endpoint|external api|event loop|http requests?|file ?paths?|filepath|pickl\w*|pandas|numpy|scipy|opencv|image file|file system|filesystem)\b',re.I)

OTHERLANG=re.compile(r'\b(java|javascript|typescript|golang|c\+\+|c#|rust|ruby|php|swift|kotlin)\b',re.I)

MISSING=re.compile(r'\b(the above|given above|shown above|previous (?:code|function|question)|attached file|following (?:code|implementation))\b',re.I)

CONTEST=re.compile(r'\b(standard input|stdin|stdout|test cases? follow|first line contains|codeforces|atcoder|leetcode)\b',re.I)

def reason(prompt,row,isopen):
    if not 40<=len(prompt)<=4000:return 'prompt_chars'
    if OTHERLANG.search(prompt):return 'other_language'
    if EXTERNAL.search(prompt):return 'external_dependency_or_application'
    if CONTEST.search(prompt):return 'contest_or_stdio_format'
    if MISSING.search(prompt):return 'possible_missing_context'
    if re.search(r'(?m)^\s*(?:def |class |import |from \w+ import )',prompt):return 'implementation_or_stub_needs_separate_review'
    if not re.search(r'\bfunction\b',prompt,re.I):return 'no_function_contract'
    if not re.search(r'\b(return\w*|input|takes?|arguments?|parameters?)\b',prompt,re.I):return 'no_input_output_contract'
    if isopen:
        if row['domain']!='generic':return 'not_generic'
        try:
            j=json.loads(row['llm_judgement'])
            if min(float(j[k]['score']) for k in ['requirement_conformance','logical_correctness','edge_case_consideration'])<4:return 'source_judge_below4'
            if float(row['average_test_score'])!=1:return 'source_test_score_below1'
            if any(x!='pass' for x in json.loads(row['tests_execution_status'])):return 'source_test_failure'
        except (KeyError,ValueError,TypeError):return 'missing_quality_metadata'
    return None
