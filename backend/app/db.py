import json
import os
from datetime import datetime, timezone

from sqlalchemy import create_engine, event, ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import RUNTIME


def now():
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


url = os.getenv('DATABASE_URL', 'sqlite:///' + str(RUNTIME / 'support.db'))
engine = create_engine(url, connect_args={'check_same_thread': False} if url.startswith('sqlite') else {}, pool_pre_ping=True)
if url.startswith('sqlite'):
    @event.listens_for(engine, 'connect')
    def sqlite_settings(conn, _):
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA busy_timeout=15000')

Session = sessionmaker(engine, expire_on_commit=False)


class Ticket(Base):
    __tablename__ = 'tickets'
    id: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[str]
    customer: Mapped[str]
    message: Mapped[str] = mapped_column(Text)
    channel: Mapped[str] = mapped_column(default='text')
    status: Mapped[str] = mapped_column(default='new')
    created_at: Mapped[str] = mapped_column(default=now)
    updated_at: Mapped[str] = mapped_column(default=now)
    analysis_json: Mapped[str] = mapped_column(Text, default='{}')
    approved_reply: Mapped[str] = mapped_column(Text, default='')
    approved_at: Mapped[str | None] = mapped_column(default=None)
    sample: Mapped[bool] = mapped_column(default=False)

    def as_dict(self):
        return {k: getattr(self, k) for k in ('id','subject','customer','message','channel','status','created_at','updated_at','approved_reply','approved_at','sample')} | {'reference': f'NS-{self.id:04}', 'analysis': json.loads(self.analysis_json)}


class Trace(Base):
    __tablename__ = 'traces'
    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str]
    ticket_id: Mapped[int | None] = mapped_column(ForeignKey('tickets.id'), default=None)
    task: Mapped[str]
    model: Mapped[str]
    duration_ms: Mapped[float]
    success: Mapped[bool]
    created_at: Mapped[str] = mapped_column(default=now)
    details_json: Mapped[str] = mapped_column(Text, default='{}')

    def as_dict(self):
        return {k: getattr(self,k) for k in ('id','request_id','ticket_id','task','model','duration_ms','success','created_at')} | {'details':json.loads(self.details_json)}


class TicketEvent(Base):
    __tablename__ = 'ticket_events'
    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey('tickets.id'))
    action: Mapped[str]
    created_at: Mapped[str] = mapped_column(default=now)
    detail: Mapped[str] = mapped_column(Text,default='')


def initialise():
    Base.metadata.create_all(engine)
