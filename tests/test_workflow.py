"""State transitions, failure visibility and evidence boundaries."""
import os
import tempfile

os.environ['RUNTIME_DIR']=tempfile.mkdtemp(prefix='northstar-test-')
os.environ['DATABASE_URL']='sqlite:///'+os.environ['RUNTIME_DIR']+'/cases.db'

import pytest
from fastapi.testclient import TestClient
from backend.app import main
from backend.app.db import Session,Trace
from backend.app.models import ModelUnavailable
from backend.app.knowledge import grounding_flags
from sqlalchemy import select


@pytest.fixture
def client():
    with TestClient(main.app) as c:yield c


def new(client):
    r=client.post('/api/tickets',json={'subject':'Cancel order','customer':'Test Customer','message':'Please cancel my order before dispatch.'})
    assert r.status_code==201
    return r.json()['id']


def test_approval_requires_analysis(client):
    i=new(client)
    assert client.post(f'/api/tickets/{i}/approve',json={'reply':'Your enquiry is being reviewed.'}).status_code==409
    assert client.patch(f'/api/tickets/{i}/status?status=resolved').status_code==409


def test_unknown_case(client):
    assert client.get('/api/tickets/999999').status_code==404


def test_invalid_ticket_rejected(client):
    assert client.post('/api/tickets',json={'subject':'x','message':'x'}).status_code==422


def test_end_to_end_review_and_reopen(client,monkeypatch):
    i=new(client)
    monkeypatch.setattr(main.service,'intent',lambda *a,**kw:{'label':'cancel_order','valid':True})
    monkeypatch.setattr(main.service,'sentiment',lambda *a:{'label':'neutral','score':.7})
    monkeypatch.setattr(main,'retrieve',lambda *a:[{'id':'POL-02','content':'Cancellation before dispatch.'}])
    monkeypatch.setattr(main.service,'generate',lambda instruction,*a,**kw:('The customer requests cancellation.' if instruction.startswith('Summarise') else 'Please check the dispatch state before requesting cancellation.',{'model':'test'}))
    r=client.post(f'/api/tickets/{i}/analyse');assert r.status_code==200
    assert r.json()['status']=='review'
    edited='The request can be reviewed after checking the dispatch state.'
    r=client.post(f'/api/tickets/{i}/approve',json={'reply':edited})
    assert r.status_code==200 and r.json()['approved_reply']==edited
    assert client.patch(f'/api/tickets/{i}/status?status=resolved').json()['status']=='resolved'
    assert client.post(f'/api/tickets/{i}/analyse').status_code==409
    r=client.patch(f'/api/tickets/{i}/status?status=review')
    assert r.json()['approved_reply']=='' and r.json()['approved_at'] is None
    actions=[e['action'] for e in client.get(f'/api/tickets/{i}').json()['events']]
    assert actions==['created','analysed','approved','resolved','review']


def test_missing_model_returns_failure_and_trace(client,monkeypatch):
    i=new(client)
    def missing(*args):raise ModelUnavailable('Intent model is not available')
    monkeypatch.setattr(main.service,'intent',missing)
    assert client.post(f'/api/tickets/{i}/analyse').status_code==503
    assert client.get(f'/api/tickets/{i}').json()['status']=='new'
    with Session() as s:
        trace=s.scalars(select(Trace).where(Trace.ticket_id==i)).first()
        assert trace.success is False
        assert 'ModelUnavailable' in trace.details_json


def test_no_evidence_does_not_call_generator(client,monkeypatch):
    monkeypatch.setattr(main,'retrieve',lambda *a:[])
    def prohibited(*a,**kw):raise AssertionError('No answer should be generated without evidence')
    monkeypatch.setattr(main.service,'generate',prohibited)
    r=client.post('/api/knowledge/ask',json={'question':'Do you offer international delivery?'})
    assert r.status_code==200 and r.json()['abstained'] and not r.json()['evidence']


def test_numeric_and_action_claims_flagged():
    flags=grounding_flags('We have issued a refund of 999.','Refunds take 5 to 7 days.')
    assert len(flags)==2


def test_invalid_audio_is_rejected(client,monkeypatch):
    def invalid(*a):raise ValueError('The file could not be decoded as audio.')
    monkeypatch.setattr(main.service,'transcribe',invalid)
    r=client.post('/api/transcribe',files={'file':('bad.wav',b'bad','audio/wav')})
    assert r.status_code==422


def test_oversize_audio_rejected_before_inference(client):
    r=client.post('/api/transcribe',files={'file':('large.wav',b'0'*(25*1024*1024+1),'audio/wav')})
    assert r.status_code==413


def test_policy_scope_and_version(client):
    r=client.get('/api/policies').json()
    assert r['version']=='2026.11.01' and len(r['articles'])==13
    assert 'Fictional' in r['scope']


def test_concurrent_analysis_and_approval_are_blocked(client,monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    started=threading.Event();release=threading.Event();i=new(client)
    def slow_intent(*args):
        started.set();assert release.wait(10)
        raise ModelUnavailable('Controlled model failure')
    monkeypatch.setattr(main.service,'intent',slow_intent)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(client.post,f'/api/tickets/{i}/analyse')
        assert started.wait(10)
        assert client.post(f'/api/tickets/{i}/analyse').status_code==409
        assert client.post(f'/api/tickets/{i}/approve',json={'reply':'This response must not be approved.'}).status_code==409
        release.set();assert future.result().status_code==503
    assert client.get(f'/api/tickets/{i}').json()['status']=='new'


def test_completed_action_contraction_flagged():
    assert grounding_flags("We've provided a tracking link.",'Tracking can be found in My Orders.')


def test_promised_cancellation_and_unsupported_fee_are_flagged():
    flags=grounding_flags('We will process the cancellation and deduct any fees.','Cancellation is eligible before dispatch.')
    assert len(flags)==2


def test_unsupported_numeric_answer_is_replaced_with_policy(client,monkeypatch):
    monkeypatch.setattr(main,'retrieve',lambda *a:[{'id':'POL-09','content':'Raise a complaint through Help.'}])
    monkeypatch.setattr(main.service,'generate',lambda *a,**kw:('Raise your complaint within 7 days.',{}))
    r=client.post('/api/knowledge/ask',json={'question':'How can I raise a complaint?'})
    assert r.status_code==200
    assert r.json()['answer']=='Raise a complaint through Help.'
    assert r.json()['answer_mode']=='verbatim_policy'


def test_failed_generation_preserves_access_to_policy(client,monkeypatch):
    monkeypatch.setattr(main,'retrieve',lambda *a:[{'id':'POL-13','content':'Choose a product and complete checkout.'}])
    def invalid(*a,**kw):raise RuntimeError('Output was not valid JSON')
    monkeypatch.setattr(main.service,'generate',invalid)
    r=client.post('/api/knowledge/ask',json={'question':'How can I place an order?'})
    assert r.status_code==200 and r.json()['answer_mode']=='verbatim_policy'
    assert 'checkout' in r.json()['answer']
