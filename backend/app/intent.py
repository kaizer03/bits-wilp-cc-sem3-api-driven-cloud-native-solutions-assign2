"""Score the same legal intent labels for original and adapted language models."""
import torch
from transformers.modeling_outputs import BaseModelOutput


def intent_prompt(text):
    return f'Identify the customer support intent. Message: {text}\nIntent:'


def classify_candidates(model,tokenizer,texts,labels,device):
    x=tokenizer([intent_prompt(t) for t in texts],padding=True,truncation=True,max_length=96,return_tensors='pt').to(device)
    targets=tokenizer(text_target=[l.replace('_',' ') for l in labels],padding=True,return_tensors='pt').input_ids.to(device)
    targets=targets.masked_fill(targets==tokenizer.pad_token_id,-100)
    batch=len(texts);classes=len(labels)
    with torch.inference_mode():
        hidden=model.encoder(**x).last_hidden_state
        expanded=hidden.repeat_interleave(classes,dim=0)
        mask=x['attention_mask'].repeat_interleave(classes,dim=0)
        expected=targets.repeat(batch,1)
        logits=model(encoder_outputs=BaseModelOutput(last_hidden_state=expanded),attention_mask=mask,labels=expected).logits
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),expected.reshape(-1),reduction='none',ignore_index=-100).reshape(expected.shape)
        scores=(loss.sum(1)/(expected!=-100).sum(1)).reshape(batch,classes)
        best=scores.argmin(1).cpu().tolist()
        chosen=scores.min(1).values.cpu().tolist()
    return [{'label':labels[i],'negative_mean_log_likelihood':float(value)} for i,value in zip(best,chosen)]
