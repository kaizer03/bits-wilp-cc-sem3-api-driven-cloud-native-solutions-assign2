import json
import re
import threading

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import ROOT
from .models import service

POLICIES=json.loads((ROOT/'data/policies/store_policies.json').read_text())
ARTICLES=POLICIES['articles']
_vectors=None
_lock=threading.Lock()
vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english')
lexical=vectorizer.fit_transform([a['title']+' '+a['content'] for a in ARTICLES])
UNSUPPORTED=re.compile(r'\b(warrant\w*|international\w*|price match\w*|competitor\w*|gift cards?|loyalty|compensation|weather|president|bitcoin|cricket)\b',re.I)


def retrieve(query,intent=None,limit=2):
    global _vectors
    if UNSUPPORTED.search(query):return []
    with _lock:
        if _vectors is None:_vectors=service.embed([a['title']+'. '+a['content'] for a in ARTICLES])
    semantic=(_vectors@service.embed([query])[0])
    lex=(lexical@vectorizer.transform([query]).T).toarray().ravel()
    scored=[]
    for i,a in enumerate(ARTICLES):
        if a['id']=='POL-12':continue
        mapped=bool(intent and intent in a['intents'])
        score=.75*float(semantic[i])+.25*float(lex[i])+(.15 if mapped else 0)
        if mapped or (semantic[i]>=.32 and lex[i]>.015):
            scored.append({**a,'score':round(score,4),'policy_version':POLICIES['version']})
    return sorted(scored,key=lambda a:a['score'],reverse=True)[:limit]


def grounding_flags(text,source):
    flags=[]
    if re.search(r"\b(i have|we have|we've|i've|has been|have been)\s+(?:already\s+)?(processed|issued|cancelled|canceled|approved|sent|refunded|escalated|provided|updated|changed)",text,re.I):
        flags.append('The draft may imply an action was completed. Verify and edit before approval.')
    values=set(re.findall(r'\b\d+(?:\.\d+)?\b',text))
    allowed=set(re.findall(r'\b\d+(?:\.\d+)?\b',source))
    if values-allowed:flags.append('The draft includes numbers absent from the case and retrieved evidence.')
    if re.search(r"\b(?:we will|we'll|i will|i'll)\s+(?:process|approve|issue|refund|cancel|send|change|update)",text,re.I):
        flags.append('The draft promises an operational action. Confirm eligibility and avoid promising an outcome.')
    if re.search(r'\bfees?\b',text,re.I) and not re.search(r'\bfees?\b',source,re.I):
        flags.append('The draft mentions a fee outside the selected evidence. Verify the fee policy before approval.')
    return flags
