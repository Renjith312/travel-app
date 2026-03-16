import React, { useState, useEffect, useRef, KeyboardEvent } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../api/client';
import ReactMarkdown from 'react-markdown';
import { Trip, ChatMessage, Itinerary, DayPlan, RawActivity } from '../types';
import { Send, ArrowLeft, MapPin, Calendar, Users, Wallet,
         CheckCircle, Car, Clock, Hotel, Compass } from 'lucide-react';
import './ChatPage.css';

const ACT_ICON: Record<string, string> = {
  FOOD:'🍽️', ACCOMMODATION:'🏨', TRANSPORT:'🚗',
  SIGHTSEEING:'🏛️', ACTIVITY:'🎯', SHOPPING:'🛍️',
  RELAXATION:'🌅', NIGHTLIFE:'🌃', OTHER:'📍',
};

export default function ChatPage() {
  const { tripId }  = useParams<{ tripId: string }>();
  const navigate    = useNavigate();
  const [trip,      setTrip]      = useState<Trip | null>(null);
  const [messages,  setMessages]  = useState<ChatMessage[]>([]);
  const [input,     setInput]     = useState<string>('');
  const [sending,   setSending]   = useState<boolean>(false);
  const [itinerary, setItinerary] = useState<Itinerary | null>(null);
  const [openDay,   setOpenDay]   = useState<number>(1);
  const [accepted,  setAccepted]  = useState<boolean>(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { loadTrip(); loadHistory(); }, [tripId]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:'smooth' }); }, [messages]);

  const loadTrip = async () => {
    try {
      const { data } = await api.get<Trip>(`/trips/${tripId}`);
      setTrip(data);
      if (data.itinerary) { setItinerary(data.itinerary); setOpenDay(1); }
    } catch(e) { console.error(e); }
  };

  const loadHistory = async () => {
    try {
      const { data } = await api.get<ChatMessage[]>(`/chat/history/${tripId}`);
      setMessages(data.length===0
        ? [{ role:'assistant', content:"Hi! I'm your travel planning assistant. Tell me where you'd like to go and I'll help plan the perfect trip! 🗺️" }]
        : data);
    } catch(e) { console.error(e); }
  };

  /** Strip backend-only graph status lines from assistant messages */
  const _stripGraphInfo = (text: string): string =>
    text.split('\n')
        .filter(line => !line.match(/^🗺️.*(?:places|graph|loaded|graph DB|API)/i))
        .join('\n')
        .trim();

  const send = async (text?: string) => {
    const msg = (text||input).trim();
    if (!msg||sending) return;
    setInput('');
    setMessages(m=>[...m,{ role:'user', content:msg }]);
    setSending(true);
    try {
      const { data } = await api.post('/chat/message',{ message:msg, tripId });
      setMessages(m=>[...m,{ role:'assistant', content:data.message }]);
      if (data.itinerary) { setItinerary(data.itinerary); setOpenDay(1); setAccepted(false); }
      if (data.tripStatus?.complete) loadTrip();
    } catch {
      setMessages(m=>[...m,{ role:'assistant', content:'Sorry, something went wrong. Please try again.' }]);
    } finally { setSending(false); inputRef.current?.focus(); }
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key==='Enter'&&!e.shiftKey) { e.preventDefault(); send(); }
  };

  const acceptItinerary = async () => {
    try {
      await api.patch(`/trips/${tripId}/status`,{ status:'PLANNED' });
      setAccepted(true);
      setTimeout(()=>navigate(`/trip/${tripId}`),1200);
    } catch(e) { console.error(e); }
  };

  const quickReplies = !itinerary
    ? ['Yes, generate my itinerary!','Add more adventure activities','Keep it budget-friendly']
    : ['Add a spa day','Make day 1 more relaxed','Suggest cheaper accommodation','Add local food experiences'];

  return (
    <div className="chat-root">
      <header className="chat-header">
        <button className="btn-ghost back-btn" onClick={()=>navigate('/')}><ArrowLeft size={16}/> Home</button>
        <div className="chat-trip-info">
          <div className="chat-trip-dot"/>
          <span>{trip?.destination||'New Trip'}</span>
          {trip?.startDate&&<span className="chat-trip-meta">· {new Date(trip.startDate).toLocaleDateString('en-IN',{month:'short',day:'numeric'})}</span>}
          {trip?.duration&&<span className="chat-trip-meta">· {trip.duration} days</span>}
        </div>
        {itinerary&&!accepted&&(
          <button className="btn-primary accept-btn" onClick={acceptItinerary}>
            <CheckCircle size={16}/> Accept Itinerary
          </button>
        )}
        {accepted&&<div className="accepted-badge"><CheckCircle size={16}/> Saved!</div>}
      </header>

      <div className="chat-body">
        <div className="chat-panel">
          <div className="chat-messages">
            {messages.map((m,i)=>(
              <div key={i} className={`msg msg-${m.role}`}>
                {m.role==='assistant'&&<div className="msg-avatar"><Compass size={14}/></div>}
                <div className="msg-bubble">
                  <ReactMarkdown>{_stripGraphInfo(m.content)}</ReactMarkdown>
                </div>
              </div>
            ))}
            {sending&&(
              <div className="msg msg-assistant">
                <div className="msg-avatar"><Compass size={14}/></div>
                <div className="msg-bubble typing"><span/><span/><span/></div>
              </div>
            )}
            <div ref={bottomRef}/>
          </div>
          <div className="quick-replies">
            {quickReplies.map((q,i)=>(
              <button key={i} className="quick-reply" onClick={()=>send(q)} disabled={sending}>{q}</button>
            ))}
          </div>
          <div className="chat-input-row">
            <textarea ref={inputRef} value={input} onChange={e=>setInput(e.target.value)}
              onKeyDown={onKey} placeholder="Type a message… (Enter to send)"
              rows={1} disabled={sending}/>
            <button className="send-btn" onClick={()=>send()} disabled={sending||!input.trim()}>
              <Send size={18}/>
            </button>
          </div>
        </div>

        <div className="itin-panel">
          {!itinerary ? (
            <div className="itin-empty">
              <div className="itin-empty-icon"><MapPin size={36} color="#6c8eff"/></div>
              <h3>Your itinerary will appear here</h3>
              <p>Chat with the assistant to generate your personalised day-by-day plan</p>
            </div>
          ) : (
            <ItineraryView itinerary={itinerary} trip={trip} openDay={openDay}
              setOpenDay={setOpenDay} accepted={accepted} onAccept={acceptItinerary}/>
          )}
        </div>
      </div>
    </div>
  );
}

interface ItinViewProps {
  itinerary: Itinerary;
  trip: Trip | null;
  openDay: number;
  setOpenDay: (d: number) => void;
  accepted: boolean;
  onAccept: () => void;
}
function ItineraryView({ itinerary, trip, openDay, setOpenDay, accepted, onAccept }: ItinViewProps) {
  const plans = itinerary.fullItinerary?.daily_plans || [];
  const notes = itinerary.fullItinerary?.notes || {};

  const dayMap: Record<number, { activities: RawActivity[]; stay?: string; theme?: string }> = {};
  if (plans.length>0) {
    plans.forEach(dp=>{ dayMap[dp.day_number]={ activities:dp.activities, stay:dp.stay_name, theme:dp.theme }; });
  } else {
    (itinerary.activities||[]).forEach(a=>{
      if (!dayMap[a.dayNumber]) dayMap[a.dayNumber]={ activities:[] };
      dayMap[a.dayNumber].activities.push({ name:a.title, details:a.description, start_time:a.startTime, end_time:a.endTime, estimatedCost:a.estimatedCost, type:a.type });
    });
  }
  const days = Object.keys(dayMap).map(Number).sort((a,b)=>a-b);

  return (
    <div className="itin-view">
      <div className="itin-header">
        <div>
          <h2>{trip?.destination||'Your Trip'}</h2>
          {trip?.startDate&&<p>{new Date(trip.startDate).toLocaleDateString('en-IN',{month:'long',day:'numeric',year:'numeric'})}</p>}
        </div>
        {!accepted
          ? <button className="btn-primary accept-btn-sm" onClick={onAccept}><CheckCircle size={14}/> Accept</button>
          : <div className="accepted-sm"><CheckCircle size={14}/> Saved</div>}
      </div>

      <div className="itin-summary">
        {trip?.duration&&<div className="sum-item"><Calendar size={14}/><span>{trip.duration} days</span></div>}
        {trip?.numberOfTravelers&&<div className="sum-item"><Users size={14}/><span>{trip.numberOfTravelers} travelers</span></div>}
        {trip?.totalBudget&&<div className="sum-item"><Wallet size={14}/><span>₹{Number(trip.totalBudget).toLocaleString()}</span></div>}
      </div>

      <div className="day-tabs">
        {days.map(d=>(
          <button key={d} className={`day-tab${openDay===d?' active':''}`} onClick={()=>setOpenDay(d)}>Day {d}</button>
        ))}
      </div>

      {days.map(d=>{
        if (openDay!==d) return null;
        const dp   = dayMap[d];
        const acts = dp?.activities||[];
        const stay = dp?.stay;
        const stayAct = acts.find(a=>(a.type||'').includes('ACCOMMODATION')||(a.name||'').toLowerCase().match(/hotel|resort|stay/));
        return (
          <div key={d} className="day-content">
            {dp?.theme&&<div className="day-theme">✨ {dp.theme}</div>}
            {(stay||stayAct)&&(
              <div className="stay-card">
                <Hotel size={16} color="#6c8eff"/>
                <div>
                  <div className="stay-label">Tonight's Stay</div>
                  <div className="stay-name">{stay||stayAct?.name}</div>
                  {stayAct?.estimatedCost&&<div className="stay-cost">≈ ₹{Number(stayAct.estimatedCost).toLocaleString()} / night</div>}
                </div>
              </div>
            )}
            <div className="activities-list">
              {acts.map((act,i)=>{
                const title = act.name||act.title||'Activity';
                const desc  = act.details||act.description||'';
                const emoji = ACT_ICON[act.type||'OTHER']||'📍';
                return (
                  <div key={i} className="act-item">
                    <div className="act-time">{act.start_time||act.startTime||act.time||''}</div>
                    <div className="act-line">
                      <div className="act-dot">{emoji}</div>
                      {i<acts.length-1&&<div className="act-connector"/>}
                    </div>
                    <div className="act-body">
                      <div className="act-title">{title}</div>
                      {desc&&<div className="act-desc">{desc}</div>}
                      <div className="act-meta">
                        {(act.end_time||act.endTime)&&<span className="act-duration"><Clock size={11}/> until {act.end_time||act.endTime}</span>}
                        {act.travel_from_previous&&<span className="act-travel"><Car size={11}/> {act.travel_from_previous}</span>}
                        {act.estimatedCost?<span className="act-cost">₹{Number(act.estimatedCost).toLocaleString()}</span>:null}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            {d<Math.max(...days)&&(
              <div className="next-day-note"><Hotel size={13}/><span>Tomorrow starts from your stay — {stay||'your hotel'}</span></div>
            )}
          </div>
        );
      })}

      {(notes.packing||notes.tips)&&(
        <div className="itin-notes">
          {notes.tips&&<div className="note-item"><span>💡</span><p>{notes.tips}</p></div>}
          {notes.packing&&<div className="note-item"><span>🎒</span><p>{notes.packing}</p></div>}
        </div>
      )}
    </div>
  );
}