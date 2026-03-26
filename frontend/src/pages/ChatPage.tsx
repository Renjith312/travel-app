import React, { useState, useEffect, useRef, KeyboardEvent, ChangeEvent } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../api/client';
import ReactMarkdown from 'react-markdown';
import { Trip, ChatMessage, Itinerary, DayPlan, RawActivity } from '../types';
import { Send, ArrowLeft, MapPin, Calendar, Users, Wallet,
         CheckCircle, Car, Clock, Hotel, Compass, Edit2, X, Zap } from 'lucide-react';
import './ChatPage.css';

const ACT_ICON: Record<string, string> = {
  FOOD:'🍽️', ACCOMMODATION:'🏨', TRANSPORT:'🚗',
  SIGHTSEEING:'🏛️', ACTIVITY:'🎯', SHOPPING:'🛍️',
  RELAXATION:'🌅', NIGHTLIFE:'🌃', OTHER:'📍',
};

export default function ChatPage() {
  const { tripId }  = useParams<{ tripId: string }>();
  const navigate    = useNavigate();
  const [trip,      setTrip]        = useState<Trip | null>(null);
  const [messages,  setMessages]    = useState<ChatMessage[]>([]);
  const [input,     setInput]       = useState<string>('');
  const [sending,   setSending]     = useState<boolean>(false);
  const [itinerary, setItinerary]   = useState<Itinerary | null>(null);
  const [openDay,   setOpenDay]     = useState<number>(1);
  const [accepted,  setAccepted]    = useState<boolean>(false);
  const [infoExpanded, setInfoExpanded] = useState<boolean>(false);
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
      setMessages(data);
    } catch(e) { console.error(e); }
  };

  const _stripGraphInfo = (text: string): string =>
    text.split('\n')
        .filter(line => !line.match(/^🗺️.*(?:places|graph|loaded|graph DB|API)/i))
        .join('\n').trim();

  const send = async (text?: string) => {
    const msg = (text || input).trim();
    if (!msg || sending) return;
    setInput('');
    setMessages(m => [...m, { role:'user', content:msg }]);
    setSending(true);
    try {
      const { data } = await api.post('/chat/message', { message:msg, tripId });
      setMessages(m => [...m, { role:'assistant', content:data.message }]);
      if (data.itinerary) {
        setItinerary(data.itinerary);
        setOpenDay(1);
        setAccepted(false);
        loadTrip(); // refresh trip metadata and full itinerary
      }
    } catch {
      setMessages(m => [...m, { role:'assistant', content:'Sorry, something went wrong. Please try again.' }]);
    } finally { setSending(false); inputRef.current?.focus(); }
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const acceptItinerary = async () => {
    try {
      await api.patch(`/trips/${tripId}/status`, { status:'PLANNED' });
      setAccepted(true);
      setTimeout(() => navigate(`/trip/${tripId}`), 1200);
    } catch(e) { console.error(e); }
  };

  const updateTripDetails = async (formData: Record<string, any>) => {
    try {
      await api.put(`/trips/${tripId}`, formData);
      await loadTrip();
      setInfoExpanded(false);
    } catch(e) { console.error(e); }
  };

  const generateItinerary = () => send('Generate my itinerary');

  const isGathering = trip?.conversationPhase === 'gathering';
  const isReady     = trip && trip.conversationPhase !== 'gathering';
  const canGenerate = isReady && !itinerary && !sending;

  const quickReplies = !itinerary
    ? ['Add more adventure activities', 'Keep it budget-friendly', 'Make it family-friendly']
    : ['Add a spa day', 'Make day 1 more relaxed', 'Suggest cheaper accommodation', 'Add local food experiences'];

  return (
    <div className="chat-root">
      <header className="chat-header">
        <button className="btn-ghost back-btn" onClick={() => navigate('/')}><ArrowLeft size={16}/> Home</button>
        <div className="chat-trip-info">
          <div className="chat-trip-dot"/>
          <span>{trip?.destination || 'New Trip'}</span>
          {trip?.startDate && <span className="chat-trip-meta">· {new Date(trip.startDate).toLocaleDateString('en-IN',{month:'short',day:'numeric'})}</span>}
          {trip?.duration  && <span className="chat-trip-meta">· {trip.duration} days</span>}
        </div>
        {itinerary && !accepted && (
          <button className="btn-primary accept-btn" onClick={acceptItinerary}>
            <CheckCircle size={16}/> Accept Itinerary
          </button>
        )}
        {accepted && <div className="accepted-badge"><CheckCircle size={16}/> Saved!</div>}
      </header>

      <div className="chat-body">
        <div className="chat-panel">

          {/* Trip info strip — shown when trip has core details */}
          {isReady && (
            <div className="trip-info-strip">
              {infoExpanded ? (
                <TripEditForm trip={trip!} onSave={updateTripDetails} onCancel={() => setInfoExpanded(false)}/>
              ) : (
                <div className="trip-strip-compact">
                  {trip!.tripContext?.core_info?.start_location && <span className="strip-item"><MapPin size={12}/>{trip!.tripContext.core_info.start_location} →</span>}
                  {trip!.destination  && <span className="strip-item"><MapPin size={12}/>{trip!.destination}</span>}
                  {trip!.startDate    && <span className="strip-item"><Calendar size={12}/>{new Date(trip!.startDate).toLocaleDateString('en-IN',{month:'short',day:'numeric',year:'numeric'})}</span>}
                  {trip!.duration     && <span className="strip-item"><Clock size={12}/>{trip!.duration} days</span>}
                  {trip!.numberOfTravelers && <span className="strip-item"><Users size={12}/>{trip!.numberOfTravelers} traveler{trip!.numberOfTravelers !== 1 ? 's' : ''}</span>}
                  {trip!.totalBudget  && <span className="strip-item"><Wallet size={12}/>₹{Number(trip!.totalBudget).toLocaleString()}</span>}
                  <button className="strip-edit-btn" onClick={() => setInfoExpanded(true)}><Edit2 size={12}/> Edit</button>
                </div>
              )}
            </div>
          )}

          <div className="chat-messages">
            {/* Setup form — shown inline for new trips */}
            {isGathering && (
              <div className="msg msg-assistant setup-msg">
                <div className="msg-avatar"><Compass size={14}/></div>
                <TripSetupCard onSubmit={updateTripDetails}/>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={`msg msg-${m.role}`}>
                {m.role === 'assistant' && <div className="msg-avatar"><Compass size={14}/></div>}
                <div className="msg-bubble">
                  <ReactMarkdown>{_stripGraphInfo(m.content)}</ReactMarkdown>
                </div>
              </div>
            ))}

            {sending && (
              <div className="msg msg-assistant">
                <div className="msg-avatar"><Compass size={14}/></div>
                <div className="msg-bubble typing"><span/><span/><span/></div>
              </div>
            )}
            <div ref={bottomRef}/>
          </div>

          {/* Generate button — shown prominently when trip is ready but no itinerary */}
          {canGenerate && (
            <div className="generate-bar">
              <button className="generate-btn" onClick={generateItinerary}>
                <Zap size={16}/> Generate my Itinerary
              </button>
              <span className="generate-hint">All details set — ready to go!</span>
            </div>
          )}

          <div className="quick-replies">
            {quickReplies.map((q, i) => (
              <button key={i} className="quick-reply" onClick={() => send(q)} disabled={sending}>{q}</button>
            ))}
          </div>

          <div className="chat-input-row">
            <textarea ref={inputRef} value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={onKey} placeholder="Type a message… (Enter to send)"
              rows={1} disabled={sending}/>
            <button className="send-btn" onClick={() => send()} disabled={sending || !input.trim()}>
              <Send size={18}/>
            </button>
          </div>
        </div>

        <div className="itin-panel">
          {!itinerary ? (
            <div className="itin-empty">
              <div className="itin-empty-icon"><MapPin size={36} color="#6c8eff"/></div>
              <h3>Your itinerary will appear here</h3>
              <p>Fill in the trip details and click Generate to create your day-by-day plan</p>
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

/* ─── Trip Setup Card (inline form for new trips) ─────────────────────────── */
interface TripSetupCardProps { onSubmit: (data: Record<string,any>) => Promise<void>; }

function TripSetupCard({ onSubmit }: TripSetupCardProps) {
  const today = new Date().toISOString().split('T')[0];
  const [form, setForm] = useState({
    destination: '', startLocation: '', startDate: today, durationDays: 3,
    numberOfTravelers: 1, totalBudget: '', startTime: '09:00', travelMode: 'private',
  });
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<Record<string,string>>({});

  const set = (k: string, v: any) => setForm(f => ({ ...f, [k]: v }));

  const validate = () => {
    const e: Record<string,string> = {};
    if (!form.destination.trim()) e.destination = 'Required';
    if (!form.startDate)          e.startDate   = 'Required';
    if (!form.durationDays || form.durationDays < 1) e.durationDays = 'Must be ≥ 1';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const submit = async () => {
    if (!validate()) return;
    setSaving(true);
    await onSubmit({
      destination:       form.destination.trim(),
      startLocation:     form.startLocation.trim() || undefined,
      startDate:         form.startDate,
      durationDays:      Number(form.durationDays),
      numberOfTravelers: Number(form.numberOfTravelers),
      totalBudget:       form.totalBudget ? Number(form.totalBudget) : undefined,
      startTime:         form.startTime,
      travelMode:        form.travelMode,
    });
    setSaving(false);
  };

  return (
    <div className="setup-card">
      <div className="setup-card-header">
        <span className="setup-card-title">✈️ Plan your trip</span>
        <span className="setup-card-sub">Fill in the details below to get started</span>
      </div>

      <div className="setup-form">
        <div className="setup-field setup-field-full">
          <label>Destination <span className="req">*</span></label>
          <input type="text" placeholder="e.g. Goa, Rajasthan, Manali"
            value={form.destination} onChange={e => set('destination', e.target.value)}/>
          {errors.destination && <span className="field-err">{errors.destination}</span>}
        </div>

        <div className="setup-field setup-field-full">
          <label>Starting Location <span className="setup-opt">optional</span></label>
          <input type="text" placeholder="e.g. Mumbai, Delhi Airport"
            value={form.startLocation} onChange={e => set('startLocation', e.target.value)}/>
        </div>

        <div className="setup-field">
          <label>Start Date <span className="req">*</span></label>
          <input type="date" value={form.startDate} min={today}
            onChange={e => set('startDate', e.target.value)}/>
          {errors.startDate && <span className="field-err">{errors.startDate}</span>}
        </div>

        <div className="setup-field">
          <label>Number of Days <span className="req">*</span></label>
          <input type="number" min={1} max={30} value={form.durationDays}
            onChange={e => set('durationDays', e.target.value)}/>
          {errors.durationDays && <span className="field-err">{errors.durationDays}</span>}
        </div>

        <div className="setup-field">
          <label>Travelers</label>
          <input type="number" min={1} max={50} value={form.numberOfTravelers}
            onChange={e => set('numberOfTravelers', e.target.value)}/>
        </div>

        <div className="setup-field">
          <label>Daily Start Time</label>
          <input type="time" value={form.startTime}
            onChange={e => set('startTime', e.target.value)}/>
        </div>

        <div className="setup-field">
          <label>Travel Mode</label>
          <select value={form.travelMode} onChange={e => set('travelMode', e.target.value)}>
            <option value="private">🚗 Private (own car / bike)</option>
            <option value="public">🚌 Public (bus / train / flight)</option>
            <option value="mixed">🔀 Mixed</option>
          </select>
        </div>

        <div className="setup-field setup-field-full">
          <label>Total Budget (₹) <span className="setup-opt">optional</span></label>
          <input type="number" min={0} placeholder="e.g. 25000"
            value={form.totalBudget} onChange={e => set('totalBudget', e.target.value)}/>
        </div>
      </div>

      <button className="setup-submit-btn" onClick={submit} disabled={saving}>
        {saving ? 'Saving…' : <><Zap size={15}/> Confirm Details</>}
      </button>
    </div>
  );
}

/* ─── Trip Edit Form (for editing existing trip details) ──────────────────── */
interface TripEditFormProps {
  trip: Trip;
  onSave: (data: Record<string,any>) => Promise<void>;
  onCancel: () => void;
}
function TripEditForm({ trip, onSave, onCancel }: TripEditFormProps) {
  const ci = trip.tripContext?.core_info || {};
  const [form, setForm] = useState({
    destination:       trip.destination || '',
    startLocation:     ci.start_location || '',
    startDate:         trip.startDate ? trip.startDate.split('T')[0] : '',
    durationDays:      trip.duration || 3,
    numberOfTravelers: trip.numberOfTravelers || 1,
    totalBudget:       trip.totalBudget || '',
    startTime:         ci.start_time || '09:00',
    travelMode:        ci.travel_mode || 'private',
  });
  const [saving, setSaving] = useState(false);
  const set = (k: string, v: any) => setForm(f => ({ ...f, [k]: v }));

  const save = async () => {
    setSaving(true);
    await onSave({
      destination:       form.destination.trim(),
      startLocation:     form.startLocation.trim() || undefined,
      startDate:         form.startDate,
      durationDays:      Number(form.durationDays),
      numberOfTravelers: Number(form.numberOfTravelers),
      totalBudget:       form.totalBudget ? Number(form.totalBudget) : undefined,
      startTime:         form.startTime,
      travelMode:        form.travelMode,
    });
    setSaving(false);
  };

  return (
    <div className="trip-edit-form">
      <div className="trip-edit-header">
        <span>Edit Trip Details</span>
        <button className="btn-ghost icon-btn" onClick={onCancel}><X size={14}/></button>
      </div>
      <div className="setup-form">
        <div className="setup-field setup-field-full">
          <label>Destination</label>
          <input type="text" value={form.destination} onChange={e => set('destination', e.target.value)}/>
        </div>
        <div className="setup-field setup-field-full">
          <label>Starting Location</label>
          <input type="text" placeholder="e.g. Mumbai, Delhi Airport" value={form.startLocation} onChange={e => set('startLocation', e.target.value)}/>
        </div>
        <div className="setup-field">
          <label>Start Date</label>
          <input type="date" value={form.startDate} onChange={e => set('startDate', e.target.value)}/>
        </div>
        <div className="setup-field">
          <label>Days</label>
          <input type="number" min={1} max={30} value={form.durationDays} onChange={e => set('durationDays', e.target.value)}/>
        </div>
        <div className="setup-field">
          <label>Travelers</label>
          <input type="number" min={1} max={50} value={form.numberOfTravelers} onChange={e => set('numberOfTravelers', e.target.value)}/>
        </div>
        <div className="setup-field">
          <label>Start Time</label>
          <input type="time" value={form.startTime} onChange={e => set('startTime', e.target.value)}/>
        </div>
        <div className="setup-field">
          <label>Travel Mode</label>
          <select value={form.travelMode} onChange={e => set('travelMode', e.target.value)}>
            <option value="private">🚗 Private</option>
            <option value="public">🚌 Public</option>
            <option value="mixed">🔀 Mixed</option>
          </select>
        </div>
        <div className="setup-field">
          <label>Budget (₹)</label>
          <input type="number" min={0} value={form.totalBudget} onChange={e => set('totalBudget', e.target.value)}/>
        </div>
      </div>
      <div className="trip-edit-footer">
        <button className="btn-primary" onClick={save} disabled={saving}>{saving ? 'Saving…' : 'Save Changes'}</button>
        <button className="btn-ghost" onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}

/* ─── Itinerary View ──────────────────────────────────────────────────────── */
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
  if (plans.length > 0) {
    plans.forEach(dp => { dayMap[dp.day_number] = { activities:dp.activities, stay:dp.stay_name, theme:dp.theme }; });
  } else {
    (itinerary.activities || []).forEach(a => {
      if (!dayMap[a.dayNumber]) dayMap[a.dayNumber] = { activities:[] };
      dayMap[a.dayNumber].activities.push({ name:a.title, details:a.description, start_time:a.startTime, end_time:a.endTime, estimatedCost:a.estimatedCost, type:a.type });
    });
  }
  const days = Object.keys(dayMap).map(Number).sort((a,b) => a - b);

  return (
    <div className="itin-view">
      <div className="itin-header">
        <div>
          <h2>{trip?.destination || 'Your Trip'}</h2>
          {trip?.startDate && <p>{new Date(trip.startDate).toLocaleDateString('en-IN',{month:'long',day:'numeric',year:'numeric'})}</p>}
        </div>
        {!accepted
          ? <button className="btn-primary accept-btn-sm" onClick={onAccept}><CheckCircle size={14}/> Accept</button>
          : <div className="accepted-sm"><CheckCircle size={14}/> Saved</div>}
      </div>

      <div className="itin-summary">
        {trip?.duration          && <div className="sum-item"><Calendar size={14}/><span>{trip.duration} days</span></div>}
        {trip?.numberOfTravelers && <div className="sum-item"><Users size={14}/><span>{trip.numberOfTravelers} travelers</span></div>}
        {trip?.totalBudget       && <div className="sum-item"><Wallet size={14}/><span>₹{Number(trip.totalBudget).toLocaleString()}</span></div>}
      </div>

      <div className="day-tabs">
        {days.map(d => (
          <button key={d} className={`day-tab${openDay === d ? ' active' : ''}`} onClick={() => setOpenDay(d)}>Day {d}</button>
        ))}
      </div>

      {days.map(d => {
        if (openDay !== d) return null;
        const dp   = dayMap[d];
        const acts = dp?.activities || [];
        const stay = dp?.stay;
        const stayAct = acts.find(a => (a.type||'').includes('ACCOMMODATION') || (a.name||'').toLowerCase().match(/hotel|resort|stay/));
        return (
          <div key={d} className="day-content">
            {dp?.theme && <div className="day-theme">✨ {dp.theme}</div>}
            {(stay || stayAct) && (
              <div className="stay-card">
                <Hotel size={16} color="#6c8eff"/>
                <div>
                  <div className="stay-label">Tonight's Stay</div>
                  <div className="stay-name">{stay || stayAct?.name}</div>
                  {stayAct?.estimatedCost && <div className="stay-cost">≈ ₹{Number(stayAct.estimatedCost).toLocaleString()} / night</div>}
                </div>
              </div>
            )}
            <div className="activities-list">
              {acts.map((act, i) => {
                const title = act.name || act.title || 'Activity';
                const desc  = act.details || act.description || '';
                const emoji = ACT_ICON[act.type || 'OTHER'] || '📍';
                return (
                  <div key={i} className="act-item">
                    <div className="act-time">{act.start_time || act.startTime || act.time || ''}</div>
                    <div className="act-line">
                      <div className="act-dot">{emoji}</div>
                      {i < acts.length - 1 && <div className="act-connector"/>}
                    </div>
                    <div className="act-body">
                      <div className="act-title">{title}</div>
                      {desc && <div className="act-desc">{desc}</div>}
                      <div className="act-meta">
                        {(act.end_time || act.endTime) && <span className="act-duration"><Clock size={11}/> until {act.end_time || act.endTime}</span>}
                        {act.travel_from_previous      && <span className="act-travel"><Car size={11}/> {act.travel_from_previous}</span>}
                        {act.estimatedCost ? <span className="act-cost">₹{Number(act.estimatedCost).toLocaleString()}</span> : null}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            {d < Math.max(...days) && (
              <div className="next-day-note"><Hotel size={13}/><span>Tomorrow starts from your stay — {stay || 'your hotel'}</span></div>
            )}
          </div>
        );
      })}

      {(notes.packing || notes.tips) && (
        <div className="itin-notes">
          {notes.tips   && <div className="note-item"><span>💡</span><p>{notes.tips}</p></div>}
          {notes.packing && <div className="note-item"><span>🎒</span><p>{notes.packing}</p></div>}
        </div>
      )}
    </div>
  );
}
