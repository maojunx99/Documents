"""Hard reservation budget: count unresolved requests at their maximum output."""
class GenerationBudget:
    def __init__(self,rows=10000,tokens=20000000,max_tokens=8192):
        self.rows=rows;self.tokens=tokens;self.max_tokens=max_tokens
        self.started=0;self.completed=0;self.used=0;self.pending={}
    def reserve(self,key):
        if key in self.pending:raise ValueError('Duplicate request')
        if self.started>=self.rows or self.used+sum(self.pending.values())+self.max_tokens>self.tokens:return False
        self.pending[key]=self.max_tokens;self.started+=1;return True
    def finish(self,key,tokens):
        bound=self.pending[key]
        if not isinstance(tokens,int) or not 0<tokens<=bound:raise ValueError('Invalid actual completion count')
        self.pending.pop(key);self.used+=tokens;self.completed+=1
        assert self.used+sum(self.pending.values())<=self.tokens
    def snapshot(self):
        return dict(started=self.started,completed=self.completed,completion_tokens=self.used,unresolved_reserved_tokens=sum(self.pending.values()),maximum_total_tokens=self.used+sum(self.pending.values()))
