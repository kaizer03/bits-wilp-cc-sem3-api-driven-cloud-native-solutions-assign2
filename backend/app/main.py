import hashlib
import json
import logging
import statistics
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime,timezone
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI,HTTPException,Request,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from sqlalchemy import select,func,update

from .config import ROOT,RESULTS,PROMPT_VERSION,MODEL_IDS
from .db import Session,Ticket,Trace,TicketEvent,initialise,now
from .knowledge import POLICIES,retrieve,grounding_flags
from .models import service,ModelUnavailable

log=logging.getLogger('northstar')


@asynccontextmanager
async def lifespan(app):
    initialise()
    with Session.begin() as s:
        for ticket in s.scalars(select(Ticket).where(Ticket.status=='processing')).all():
            ticket.status='review' if ticket.analysis_json!='{}' else 'new'
            s.add(TicketEvent(ticket_id=ticket.id,action='analysis_interrupted',detail='The previous analysis was interrupted by a service restart. Retry analysis.'))
    yield


app=FastAPI(title='Northstar Retail Support API',version='1.0.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5173','http://localhost:5173'],allow_methods=['GET','POST','PATCH'],allow_headers=['Content-Type'])


@app.exception_handler(ModelUnavailable)
async def model_error(request,exc):
    return JSONResponse(status_code=503,content={'detail':str(exc)})


class NewTicket(BaseModel):
    subject:str=Field(min_length=3,max_length=160)
    customer:str=Field(default='Guest customer',min_length=1,max_length=100)
    message:str=Field(min_length=8,max_length=6000)
    channel:Literal['text','voice']='text'


class Reply(BaseModel):
    reply:str=Field(min_length=12,max_length=6000)


class Question(BaseModel):
    question:str=Field(min_length=5,max_length=1000)


def record(ticket_id,task,model,callback,request_id=None):
    start=time.perf_counter();success=False;details={}
    try:
        value=callback();success=True
        if isinstance(value,tuple):value,details=value
        return value
    except Exception as e:
        details={'error_type':type(e).__name__}
        raise
    finally:
        with Session.begin() as s:
            s.add(Trace(request_id=request_id or uuid.uuid4().hex[:12],ticket_id=ticket_id,task=task,model=model,duration_ms=round((time.perf_counter()-start)*1000,2),success=success,details_json=json.dumps({'prompt_version':PROMPT_VERSION,**details})))


def get_ticket(s,ticket_id):
    t=s.get(Ticket,ticket_id)
    if not t:raise HTTPException(404,'Case not found')
    return t


@app.get('/api/health')
def health():
    with Session() as s:s.execute(select(func.count(Ticket.id)))
    return {'status':'ok','version':'1.0.0','model_device':service.device,'prompt_version':PROMPT_VERSION,'time':now()}


@app.get('/api/models')
def models():return service.status()


@app.get('/api/tickets')
def tickets():
    with Session() as s:return [t.as_dict() for t in s.scalars(select(Ticket).order_by(Ticket.id.desc())).all()]


@app.post('/api/tickets',status_code=201)
def create_ticket(payload:NewTicket):
    with Session.begin() as s:
        t=Ticket(**payload.model_dump());s.add(t);s.flush()
        s.add(TicketEvent(ticket_id=t.id,action='created',detail='Case received through '+t.channel))
        return t.as_dict()


@app.get('/api/tickets/{ticket_id}')
def detail(ticket_id:int):
    with Session() as s:
        t=get_ticket(s,ticket_id).as_dict()
        t['events']=[{'action':e.action,'detail':e.detail,'created_at':e.created_at} for e in s.scalars(select(TicketEvent).where(TicketEvent.ticket_id==ticket_id).order_by(TicketEvent.id)).all()]
        return t


def _analyse(ticket_id:int):
    with Session() as s:
        ticket=get_ticket(s,ticket_id)
        if ticket.status=='resolved':raise HTTPException(409,'Reopen the case before analysing it again.')
        message=ticket.message
    request_id=uuid.uuid4().hex[:12]
    intent=record(ticket_id,'intent',service.intent_version,lambda:service.intent(message),request_id)
    sentiment=record(ticket_id,'sentiment',MODEL_IDS['sentiment'],lambda:service.sentiment(message),request_id)
    evidence=record(ticket_id,'retrieval',MODEL_IDS['embedding'],lambda:retrieve(message,intent['label']),request_id)
    instruction='Return JSON with exactly two string fields: "summary" (one sentence about the customer issue) and "reply" (a polite response under 110 words). Use the supplied policy evidence only. Explain any relevant eligibility conditions. Do not claim you know the order state. If evidence is empty, ask for human review rather than making policy claims.'
    generated=record(ticket_id,'summary_and_reply',service.generation_model_id,lambda:service.generate(instruction,{'customer_message':message,'policy':evidence[:1]},350,required_fields=('summary','reply')),request_id)
    if not isinstance(generated.get('summary'),str) or not isinstance(generated.get('reply'),str):raise HTTPException(502,'The model output is missing a summary or reply. Please retry.')
    flags=grounding_flags(generated['reply'],message+' '+json.dumps(evidence[:1]))
    if not evidence:flags.append('No matching policy evidence. Human review is required.')
    if not intent['valid']:flags.append('The intent model produced an unrecognised label.')
    analysis={'intent':intent,'sentiment':sentiment,'summary':generated['summary'],'reply':generated['reply'],'evidence':evidence,'flags':flags,'request_id':request_id,'analysed_at':now(),'prompt_version':PROMPT_VERSION,'policy_version':POLICIES['version']}
    with Session.begin() as s:
        t=get_ticket(s,ticket_id)
        if t.status=='resolved':raise HTTPException(409,'The case was resolved while analysis was running.')
        t.analysis_json=json.dumps(analysis);t.status='review';t.updated_at=now();t.approved_reply='';t.approved_at=None
        s.add(TicketEvent(ticket_id=t.id,action='analysed',detail='Intent, sentiment, summary and draft generated; review required.'))
        return t.as_dict()


@app.post('/api/tickets/{ticket_id}/analyse')
def analyse(ticket_id:int):
    with Session.begin() as s:
        t=get_ticket(s,ticket_id)
        previous=t.status
        if previous in ('resolved','processing'):
            raise HTTPException(409,'This case is resolved or an analysis is already in progress.')
        changed=s.execute(update(Ticket).where(Ticket.id==ticket_id,Ticket.status.not_in(['resolved','processing'])).values(status='processing'))
        if changed.rowcount!=1:raise HTTPException(409,'An analysis is already in progress.')
    try:
        return _analyse(ticket_id)
    except ModelUnavailable:
        raise
    except RuntimeError as e:
        log.exception('Analysis failed for case %s',ticket_id)
        raise HTTPException(502,str(e)) from e
    finally:
        with Session.begin() as s:
            t=get_ticket(s,ticket_id)
            if t.status=='processing':t.status=previous


@app.post('/api/tickets/{ticket_id}/approve')
def approve(ticket_id:int,payload:Reply):
    with Session.begin() as s:
        t=get_ticket(s,ticket_id)
        if t.status!='review':raise HTTPException(409,'Analyse this case before approving a response.')
        t.approved_reply=payload.reply;t.approved_at=now();t.updated_at=now();t.status='approved'
        s.add(TicketEvent(ticket_id=t.id,action='approved',detail='Response reviewed and approved. No message was sent.'))
        return t.as_dict()


@app.patch('/api/tickets/{ticket_id}/status')
def change_status(ticket_id:int,status:Literal['resolved','review']):
    with Session.begin() as s:
        t=get_ticket(s,ticket_id)
        if status=='resolved' and not t.approved_reply:raise HTTPException(409,'Approve a response before resolving this case.')
        t.status=status;t.updated_at=now()
        if status=='review':t.approved_reply='';t.approved_at=None
        s.add(TicketEvent(ticket_id=t.id,action=status,detail='Case status updated.'))
        return t.as_dict()


@app.get('/api/policies')
def policies():return POLICIES


@app.post('/api/knowledge/ask')
def ask(payload:Question):
    rid=uuid.uuid4().hex[:12]
    evidence=record(None,'retrieval',MODEL_IDS['embedding'],lambda:retrieve(payload.question),rid)
    if not evidence:
        return {'answer':'The published store policies do not contain enough information to answer this question. Please request human review.','evidence':[],'abstained':True,'flags':[],'request_id':rid}
    mode='generated';flags=[]
    try:
        result=record(None,'policy_answer',service.generation_model_id,lambda:service.generate('Answer the question using only the supplied policy. Return JSON with one string field "answer". Keep it under 100 words. If the policy does not answer it, say human review is needed.',{'question':payload.question,'policy':evidence[:1]},220,required_fields=('answer',)),rid)
        answer=result.get('answer')
        if not isinstance(answer,str):raise RuntimeError('The model returned an invalid answer.')
        flags=grounding_flags(answer,evidence[0]['content'])
    except ModelUnavailable:
        raise
    except RuntimeError:
        flags=['The generated answer did not pass the output check. The published policy is shown verbatim.']
        answer=''
    if flags:
        answer=evidence[0]['content'];mode='verbatim_policy'
    return {'answer':answer,'evidence':evidence[:1],'abstained':False,'flags':flags,'answer_mode':mode,'request_id':rid}


@app.post('/api/transcribe')
async def transcribe(file:UploadFile):
    data=await file.read(25*1024*1024+1)
    if len(data)>25*1024*1024:raise HTTPException(413,'Audio files must be smaller than 25 MB.')
    from starlette.concurrency import run_in_threadpool
    try:return await run_in_threadpool(record,None,'transcription',MODEL_IDS['speech'],lambda:service.transcribe(data))
    except ValueError as e:raise HTTPException(422,str(e)) from e


@app.post('/api/models/compare')
def compare(payload:Question):
    return {'original':record(None,'intent_baseline',MODEL_IDS['intent_base'],lambda:service.intent(payload.question,base=True)),'fine_tuned':record(None,'intent',service.intent_version,lambda:service.intent(payload.question))}


@app.get('/api/operations')
def operations():
    with Session() as s:
        traces=s.scalars(select(Trace).order_by(Trace.id.desc()).limit(500)).all()
        counts=dict(s.execute(select(Ticket.status,func.count(Ticket.id)).group_by(Ticket.status)).all())
        events=s.scalars(select(TicketEvent).order_by(TicketEvent.id.desc()).limit(12)).all()
    durations=[t.duration_ms for t in traces]
    experiment=RESULTS/'intent_experiment.json';quality=RESULTS/'quality_evaluation.json'
    return {'case_counts':counts,'requests':len(traces),'success_rate':sum(t.success for t in traces)/len(traces) if traces else None,'p95_latency_ms':float(np.percentile(durations,95)) if durations else None,'latency_scope':'Individual model/retrieval operations; latest 500 traces, not concurrent load testing.','traces':[t.as_dict() for t in traces],'events':[{'ticket_id':e.ticket_id,'action':e.action,'detail':e.detail,'created_at':e.created_at} for e in events],'intent_experiment':json.loads(experiment.read_text()) if experiment.exists() else None,'quality_evaluation':json.loads(quality.read_text()) if quality.exists() else None,'policy_version':POLICIES['version'],'prompt_version':PROMPT_VERSION}


dist=ROOT/'frontend/dist'
if dist.is_dir():
    app.mount('/assets',StaticFiles(directory=dist/'assets'),name='assets')
    @app.get('/')
    def index():return FileResponse(dist/'index.html')
