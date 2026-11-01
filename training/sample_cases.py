"""Load clearly labeled fictional enquiries for a repeatable demonstration."""
import argparse
from pathlib import Path

import httpx
from sqlalchemy import select
from backend.app.config import ROOT
from backend.app.db import Session,Ticket

EXAMPLES=[
 {'subject':'Parcel tracking update','customer':'Priya Sharma','message':'My parcel has not arrived. Where can I track my order?','audio':'delivery','channel':'voice'},
 {'subject':'Cancel before dispatch','customer':'Arjun Mehta','message':'I would like to cancel my order before it is dispatched.','audio':'cancel','channel':'voice'},
 {'subject':'Return a damaged item','customer':'Neha Rao','message':'The item arrived damaged. I am very disappointed and would like to request a refund.','channel':'text'},
 {'subject':'Invoice for reimbursement','customer':'Rohan Das','message':'Please tell me where I can download the invoice for my last confirmed order.','channel':'text'},
 {'subject':'Update delivery address','customer':'Sara Khan','message':'I entered the wrong flat number in my order. How can I change the shipping address before dispatch?','channel':'text'},
 {'subject':'A great shopping experience','customer':'Aditi Sen','message':'Thank you for the excellent product and quick delivery. I am delighted and want to leave a review.','channel':'text'},
 {'subject':'Payment debited at checkout','customer':'Vikram Iyer','message':'My bank account was debited but the store did not confirm an order. This is frustrating. Please help with this payment issue.','channel':'text'},
 {'subject':'Newsletter preferences','customer':'Ishaan Roy','message':'How can I unsubscribe from your marketing newsletter while keeping order emails?','channel':'text'}
]

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8092');p.add_argument('--analyse',action='store_true');a=p.parse_args()
 with httpx.Client(base_url=a.url,timeout=240) as client:
  existing=client.get('/api/tickets').json()
  for example in EXAMPLES:
   match=next((t for t in existing if t['subject']==example['subject'] and t['customer']==example['customer']),None)
   if match:ticket=match
   else:
    payload={k:v for k,v in example.items() if k!='audio'}
    if example.get('audio'):
     path=ROOT/'data/evaluation/audio'/(example['audio']+'.wav')
     with path.open('rb') as f:r=client.post('/api/transcribe',files={'file':(path.name,f,'audio/wav')})
     r.raise_for_status();payload['message']=r.json()['text']
    r=client.post('/api/tickets',json=payload);r.raise_for_status();ticket=r.json()
    with Session.begin() as s:s.get(Ticket,ticket['id']).sample=True
   if a.analyse and ticket['status']=='new':
    r=client.post(f"/api/tickets/{ticket['id']}/analyse");r.raise_for_status();ticket=r.json()
   print(ticket['reference'],ticket['subject'],ticket['status'],flush=True)
