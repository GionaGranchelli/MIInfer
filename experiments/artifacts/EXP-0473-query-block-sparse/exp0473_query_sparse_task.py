import ast, hashlib, json, pathlib, subprocess, time, urllib.request

ROOT=pathlib.Path('/tmp/m31-gqa6-control-screen')
URL='http://127.0.0.1:8089/v1/chat/completions'
OUT=pathlib.Path('/tmp/exp0473_query_sparse_task.json')
SRC=pathlib.Path('/tmp/exp0469_agent_coding_screen.py')
tree=ast.parse(SRC.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='repository_context')
ns={'ROOT':pathlib.Path('/tmp/m31-gqa6-control-screen'),'pathlib':pathlib}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SRC),'exec'),ns)
context,files=ns['repository_context']()
TOOLS=[
 {'type':'function','function':{'name':'read_file','description':'Read a file inside the MIInfer checkout.','parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path']}}},
 {'type':'function','function':{'name':'replace_once','description':'Replace one unique exact span in tests/gqa6_sparse_window_oracle.cpp only.','parameters':{'type':'object','properties':{'path':{'type':'string'},'old':{'type':'string'},'new':{'type':'string'}},'required':['path','old','new']}}},
 {'type':'function','function':{'name':'run_check','description':'Build and run the sparse-window oracle; require adjacent cases 65535U, 65536U, and 65537U.','parameters':{'type':'object','properties':{},'required':[]}}},
]
def metrics():
 with urllib.request.urlopen('http://127.0.0.1:8089/metrics',timeout=10) as r: body=r.read().decode()
 d={}
 for line in body.splitlines():
  if line.startswith(('miinfer_prompt_tokens_total ','miinfer_generated_tokens_total ','miinfer_inference_requests_total ','miinfer_prefill_tokens_total ')):
   k,v=line.split(); d[k]=int(v)
 return d
def call(messages,fn_name,max_tokens=256):
 before=metrics(); start=time.time()
 body=json.dumps({'model':'qwen3.8-27b','messages':messages,'tools':TOOLS,'tool_choice':{'type':'function','function':{'name':fn_name}},'max_tokens':max_tokens,'temperature':0,'top_p':1,'top_k':1,'stream':True}).encode()
 req=urllib.request.Request(URL,data=body,headers={'Content-Type':'application/json','Accept':'text/event-stream'})
 content=[]; calls={}; first=None
 with urllib.request.urlopen(req,timeout=3600) as res:
  for raw in res:
   line=raw.decode('utf8','replace').strip()
   if not line.startswith('data: ') or line=='data: [DONE]': continue
   obj=json.loads(line[6:]); choices=obj.get('choices') or []
   if not choices: continue
   delta=choices[0].get('delta') or {}
   if delta.get('content') or delta.get('tool_calls'):
    if first is None: first=(time.time()-start)*1000
   if delta.get('content'): content.append(delta['content'])
   for item in delta.get('tool_calls',[]):
    c=calls.setdefault(item['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
    if item.get('id'): c['id']=item['id']
    f=item.get('function') or {}; c['function']['name']+=f.get('name',''); c['function']['arguments']+=f.get('arguments','')
 after=metrics()
 return {'content':''.join(content),'tool_calls':[calls[i] for i in sorted(calls)],'ttft_ms':first,'wall_ms':(time.time()-start)*1000,'before':before,'after':after}
def tool_result(call):
 f=call['function']; a=json.loads(f.get('arguments') or '{}'); name=f['name']
 if name=='read_file':
  if a.get('path')!='tests/gqa6_sparse_window_oracle.cpp': raise ValueError('unexpected read path')
  return (ROOT/a['path']).read_text()
 if name=='replace_once':
  if a.get('path')!='tests/gqa6_sparse_window_oracle.cpp': raise ValueError('unexpected write path')
  p=ROOT/a['path']; s=p.read_text(); count=s.count(a['old'])
  if count!=1: raise ValueError(f'expected unique old span; found {count}')
  p.write_text(s.replace(a['old'],a['new'],1)); return 'updated one unique span'
 if name=='run_check':
  p=ROOT/'tests/gqa6_sparse_window_oracle.cpp'; s=p.read_text()
  if '{2049U, 2560U, 2561U, 65535U, 65536U, 65537U, 73099U}' not in s: raise ValueError('missing exact adjacent 64K cases')
  z=subprocess.run(['podman','run','--rm','-v',f'{ROOT}:/workspace:Z','-w','/workspace','localhost/miinfer-dev:rocm-7.2.1','bash','-lc','cmake --build build/mi50-release --target miinfer-gqa6-sparse-window-oracle -j4 && build/mi50-release/miinfer-gqa6-sparse-window-oracle'],capture_output=True,text=True,timeout=600)
  return json.dumps({'exit_code':z.returncode,'stdout':z.stdout[-4000:],'stderr':z.stderr[-2000:]})
 raise ValueError('unsupported tool '+name)

messages=[
 {'role':'system','content':'You are a careful coding agent working in the MIInfer repository. Use only the provided tools. Inspect before editing. Make the exact scoped change and run its check.'},
 {'role':'user','content':context},
 {'role':'user','content':'Follow up on the completed 65536U boundary case. In tests/gqa6_sparse_window_oracle.cpp, extend the existing totals initializer to include 65535U, 65536U, and 65537U while preserving every other value. First call read_file, then replace_once using the exact old and new initializer, then call run_check. The check requires the adjacent 64K cases and builds/runs the oracle.'}
]
record={'context_bytes':len(context.encode()),'context_sha256':hashlib.sha256(context.encode()).hexdigest(),'source_file_count':len(files),'turns':[]}
try:
 r=call(messages,'read_file'); record['turns'].append(r)
 if len(r['tool_calls'])!=1: raise RuntimeError('required read_file did not yield exactly one call')
 messages.append({'role':'assistant','content':r['content'] or None,'tool_calls':r['tool_calls']})
 out=tool_result(r['tool_calls'][0]); record.setdefault('tool_results',[]).append({'name':'read_file','result':out})
 messages.append({'role':'tool','tool_call_id':r['tool_calls'][0]['id'],'content':out})
 messages.append({'role':'user','content':'Now make the exact requested replacement with replace_once.'})
 r=call(messages,'replace_once'); record['turns'].append(r)
 if len(r['tool_calls'])!=1: raise RuntimeError('required replace_once did not yield exactly one call')
 messages.append({'role':'assistant','content':r['content'] or None,'tool_calls':r['tool_calls']})
 out=tool_result(r['tool_calls'][0]); record.setdefault('tool_results',[]).append({'name':'replace_once','arguments':r['tool_calls'][0]['function']['arguments'],'result':out})
 messages.append({'role':'tool','tool_call_id':r['tool_calls'][0]['id'],'content':out})
 messages.append({'role':'user','content':'Run the requested check now.'})
 r=call(messages,'run_check'); record['turns'].append(r)
 if len(r['tool_calls'])!=1: raise RuntimeError('required run_check did not yield exactly one call')
 messages.append({'role':'assistant','content':r['content'] or None,'tool_calls':r['tool_calls']})
 out=tool_result(r['tool_calls'][0]); record.setdefault('tool_results',[]).append({'name':'run_check','arguments':r['tool_calls'][0]['function']['arguments'],'result':out})
except Exception as e: record['error']=repr(e)
record['source_sha256']=hashlib.sha256((ROOT/'tests/gqa6_sparse_window_oracle.cpp').read_bytes()).hexdigest()
record['finished_unix']=time.time()
OUT.write_text(json.dumps(record,indent=2)+'\n'); print(json.dumps(record,indent=2))
