import React, { useState, useEffect, ChangeEvent } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../api/client';
import { Trip, Itinerary, Activity, RawActivity, DayPlan } from '../types';
import { ArrowLeft, Calendar, Users, Wallet, Hotel,
         Clock, Car, CheckCircle, Edit3, ChevronRight, ChevronDown,
         Compass, Sun, Sunset, Moon, AlertCircle } from 'lucide-react';
import { format, parseISO, isToday, isTomorrow, isPast, addDays } from 'date-fns';
import './TripPage.css';
import DayMap from './DayMap';

const ACT_ICON: Record<string,string> = {
  FOOD:'🍽️', ACCOMMODATION:'🏨', TRANSPORT:'🚗',
  SIGHTSEEING:'🏛️', ACTIVITY:'🎯', SHOPPING:'🛍️',
  RELAXATION:'🌅', NIGHTLIFE:'🌃', OTHER:'📍',
};

type TOD = 'morning'|'afternoon'|'evening'|'night';
function timeOfDay(t?: string): TOD {
  const h = parseInt((t||'').split(':')[0]||'9');
  if (h<12) return 'morning';
  if (h<17) return 'afternoon';
  if (h<20) return 'evening';
  return 'night';
}
const TOD_ICON: Record<TOD, React.ReactNode> = {
  morning:   <Sun size={14}/>,
  afternoon: <Sun size={14}/>,
  evening:   <Sunset size={14}/>,
  night:     <Moon size={14}/>,
};

interface DayData {
  activities: (Activity | RawActivity)[];
  stay?: string;
  theme?: string;
}

interface EditValues {
  title?: string;
  description?: string;
  startTime?: string;
  endTime?: string;
  estimatedCost?: string;
  notes?: string;
}

// ── Budget Split ───────────────────────────────────────────────
const CAT_COLOR: Record<string, string> = {
  ACCOMMODATION: '#6c8eff',
  TRANSPORT:     '#34d399',
  FOOD:          '#fbbf24',
  SIGHTSEEING:   '#a78bfa',
  ACTIVITY:      '#f97316',
  SHOPPING:      '#ec4899',
  RELAXATION:    '#2dd4bf',
  NIGHTLIFE:     '#818cf8',
  OTHER:         '#6b7280',
};
const CAT_LABEL: Record<string, string> = {
  ACCOMMODATION: 'Stay',
  TRANSPORT:     'Transport',
  FOOD:          'Food & Dining',
  SIGHTSEEING:   'Sightseeing',
  ACTIVITY:      'Activities',
  SHOPPING:      'Shopping',
  RELAXATION:    'Relaxation',
  NIGHTLIFE:     'Nightlife',
  OTHER:         'Other',
};

function fmtK(n: number): string {
  return n >= 1000 ? `₹${(n / 1000).toFixed(1)}k` : `₹${n}`;
}

interface BudgetSplitProps {
  trip: Trip;
  plans: DayPlan[];
  acts: Activity[];
}
function BudgetSplit({ trip, plans, acts }: BudgetSplitProps) {
  const [open, setOpen] = useState(true);

  // Aggregate costs — prefer fullItinerary daily_plans when available
  const items: { type: string; cost: number; day: number }[] = [];
  if (plans.length > 0) {
    plans.forEach(dp => {
      (dp.activities || []).forEach((a: RawActivity) => {
        const cost = Number(a.estimatedCost) || 0;
        if (cost > 0) items.push({ type: (a.type || 'OTHER').toUpperCase(), cost, day: dp.day_number });
      });
    });
  } else {
    acts.forEach(a => {
      const cost = a.estimatedCost || 0;
      if (cost > 0) items.push({ type: (a.type || 'OTHER').toUpperCase(), cost, day: a.dayNumber });
    });
  }

  const totalEst   = items.reduce((s, i) => s + i.cost, 0);
  const totalBudget = trip.totalBudget || 0;
  if (totalEst === 0) return null;

  // By category
  const byCat: Record<string, number> = {};
  items.forEach(({ type, cost }) => { byCat[type] = (byCat[type] || 0) + cost; });
  const sortedCats = Object.entries(byCat).sort((a, b) => b[1] - a[1]);

  // By day
  const byDay: Record<number, number> = {};
  items.forEach(({ day, cost }) => { byDay[day] = (byDay[day] || 0) + cost; });
  const dayEntries = Object.entries(byDay).sort((a, b) => Number(a[0]) - Number(b[0]));
  const maxDay     = Math.max(...Object.values(byDay), 1);

  const overBudget = totalBudget > 0 && totalEst > totalBudget;
  const fillPct    = totalBudget > 0 ? Math.min((totalEst / totalBudget) * 100, 100) : 100;

  return (
    <div className="budget-split-wrap">
      <button className="budget-split-toggle" onClick={() => setOpen(o => !o)}>
        <Wallet size={13} />
        <span>Budget Overview</span>
        <span className={`bs-toggle-total${overBudget ? ' over' : ''}`}>
          ₹{totalEst.toLocaleString()}
          {totalBudget > 0 && <span className="bs-toggle-of"> / ₹{totalBudget.toLocaleString()}</span>}
        </span>
        <ChevronDown size={13} className={`bs-chevron${open ? ' open' : ''}`} />
      </button>

      {open && (
        <div className="budget-split-body">

          {/* Overall progress bar */}
          {totalBudget > 0 && (
            <div className="bs-progress-section">
              <div className="bs-progress-track">
                <div
                  className={`bs-progress-fill${overBudget ? ' over' : ''}`}
                  style={{ width: `${fillPct}%` }}
                />
              </div>
              <span className={`bs-progress-note${overBudget ? ' over' : ''}`}>
                {overBudget
                  ? `⚠ ₹${(totalEst - totalBudget).toLocaleString()} over budget`
                  : `₹${(totalBudget - totalEst).toLocaleString()} remaining`}
              </span>
            </div>
          )}

          {/* Category rows */}
          <div className="bs-cats">
            {sortedCats.map(([cat, amt]) => (
              <div key={cat} className="bs-cat-row">
                <div className="bs-cat-label">
                  <span className="bs-cat-dot" style={{ background: CAT_COLOR[cat] || '#6b7280' }} />
                  <span className="bs-cat-name">{CAT_LABEL[cat] || cat}</span>
                </div>
                <div className="bs-cat-right">
                  <div className="bs-cat-track">
                    <div
                      className="bs-cat-fill"
                      style={{ width: `${(amt / totalEst) * 100}%`, background: CAT_COLOR[cat] || '#6b7280' }}
                    />
                  </div>
                  <span className="bs-cat-amt">₹{amt.toLocaleString()}</span>
                  <span className="bs-cat-pct">{Math.round((amt / totalEst) * 100)}%</span>
                </div>
              </div>
            ))}
          </div>

          {/* Per-day mini bar chart */}
          {dayEntries.length > 1 && (
            <div className="bs-days-section">
              <div className="bs-days-title">Per Day</div>
              <div className="bs-day-bars">
                {dayEntries.map(([day, cost]) => (
                  <div key={day} className="bs-day-col" title={`Day ${day}: ₹${cost.toLocaleString()}`}>
                    <div className="bs-day-bar-wrap">
                      <div className="bs-day-bar" style={{ height: `${Math.round((cost / maxDay) * 44)}px` }} />
                    </div>
                    <span className="bs-day-label">D{day}</span>
                    <span className="bs-day-amt">{fmtK(Math.round(cost))}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  );
}

export default function TripPage() {
  const { tripId }  = useParams<{ tripId:string }>();
  const navigate    = useNavigate();
  const [trip,      setTrip]      = useState<Trip|null>(null);
  const [itinerary, setItinerary] = useState<Itinerary|null>(null);
  const [openDay,   setOpenDay]   = useState<number|null>(null);
  const [loading,   setLoading]   = useState<boolean>(true);
  const [editing,   setEditing]   = useState<string|null>(null);
  const [editVal,   setEditVal]   = useState<EditValues>({});

  useEffect(()=>{ loadTrip(); },[tripId]);

  const loadTrip = async () => {
    try {
      const { data } = await api.get<Trip>(`/trips/${tripId}`);
      setTrip(data);
      if (data.itinerary) {
        setItinerary(data.itinerary);
        const acts  = data.itinerary.activities||[];
        const days  = [...new Set(acts.map(a=>a.dayNumber))].sort((a,b)=>a-b);
        const start = data.startDate ? parseISO(data.startDate) : null;
        if (start) {
          const diff = Math.floor((Date.now()-start.getTime())/86400000)+1;
          setOpenDay(days.find(d=>d===diff)||days[0]||1);
        } else setOpenDay(days[0]||1);
      }
    } catch(e){ console.error(e); }
    finally { setLoading(false); }
  };

  const saveActivity = async (actId: string) => {
    try { await api.patch(`/itinerary/${tripId}/activity/${actId}`,editVal); setEditing(null); loadTrip(); }
    catch(e){ console.error(e); }
  };

  if (loading) return (
    <div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'100vh'}}>
      <div className="spinner-lg"/>
    </div>
  );
  if (!trip||!itinerary) return (
    <div style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',height:'100vh',gap:16}}>
      <AlertCircle size={40} color="#f87171"/>
      <p>No itinerary found.</p>
      <button className="btn-primary" onClick={()=>navigate(`/chat/${tripId}`)}>Go to Chat</button>
    </div>
  );

  const plans = itinerary.fullItinerary?.daily_plans||[];
  const acts  = itinerary.activities||[];
  const dayMap: Record<number,DayData> = {};
  if (plans.length>0) {
    plans.forEach(dp=>{ dayMap[dp.day_number]={ activities:dp.activities, stay:dp.stay_name, theme:dp.theme }; });
  } else {
    acts.forEach(a=>{
      if (!dayMap[a.dayNumber]) dayMap[a.dayNumber]={ activities:[] };
      dayMap[a.dayNumber].activities.push(a);
    });
  }
  const days = Object.keys(dayMap).map(Number).sort((a,b)=>a-b);
  const startDate = trip.startDate ? parseISO(trip.startDate) : null;

  const getDayDate  = (d:number) => startDate ? addDays(startDate,d-1) : null;
  const getDayLabel = (d:number) => {
    const dt = getDayDate(d);
    if (!dt) return `Day ${d}`;
    if (isToday(dt))    return `Day ${d} · Today`;
    if (isTomorrow(dt)) return `Day ${d} · Tomorrow`;
    return `Day ${d} · ${format(dt,'EEE, MMM d')}`;
  };
  const isCurrent = (d:number) => { const dt=getDayDate(d); return dt?isToday(dt):false; };
  const isPastDay = (d:number) => { const dt=getDayDate(d); return dt?isPast(addDays(dt,1)):false; };

  return (
    <div className="trip-root">
      <aside className="trip-sidebar">
        <div className="trip-sidebar-header">
          <button className="btn-ghost back-btn" onClick={()=>navigate('/')}><ArrowLeft size={14}/> Home</button>
          <h2>{trip.destination}</h2>
          {trip.startDate&&<p>{format(parseISO(trip.startDate),'MMM d')}{trip.endDate?` – ${format(parseISO(trip.endDate),'MMM d, yyyy')}`:''}</p>}
          <div className="trip-sb-meta">
            {trip.duration&&<span><Calendar size={12}/> {trip.duration}d</span>}
            {trip.numberOfTravelers&&<span><Users size={12}/> {trip.numberOfTravelers}</span>}
            {trip.totalBudget&&<span><Wallet size={12}/> ₹{Number(trip.totalBudget).toLocaleString()}</span>}
          </div>
        </div>
        <div className="day-list">
          {days.map(d=>{
            const dp=dayMap[d];
            return (
              <button key={d} className={`day-list-item${openDay===d?' active':''}${isPastDay(d)?' past':''}${isCurrent(d)?' current':''}`}
                onClick={()=>setOpenDay(d)}>
                <div className="day-list-num">Day {d}</div>
                <div className="day-list-date">{getDayLabel(d).split('·')[1]?.trim()||''}</div>
                {dp?.theme&&<div className="day-list-theme">{dp.theme}</div>}
                {dp?.stay&&<div className="day-list-stay"><Hotel size={10}/> {dp.stay}</div>}
                {isCurrent(d)&&<div className="today-dot"/>}
              </button>
            );
          })}
        </div>
        <div className="trip-sb-actions">
          <button className="btn-ghost" style={{width:'100%',display:'flex',justifyContent:'center',gap:8}}
            onClick={()=>navigate(`/chat/${tripId}`)}>
            <Edit3 size={14}/> Edit with AI
          </button>
        </div>
      </aside>

      <main className="trip-main">
        <div className="trip-main-inner">
          <BudgetSplit trip={trip} plans={plans} acts={acts} />
          {openDay&&dayMap[openDay]&&(
            <DayDetail
              day={openDay} dp={dayMap[openDay]}
              label={getDayLabel(openDay)}
              nextStay={dayMap[openDay+1]?.stay}
              isCurrent={isCurrent(openDay)}
              editing={editing} editVal={editVal}
              setEditing={setEditing}
              setEditVal={(v:EditValues)=>setEditVal(v)}
              onSave={saveActivity}
            />
          )}
        </div>
      </main>
    </div>
  );
}

interface DayDetailProps {
  day: number;
  dp: DayData;
  label: string;
  nextStay?: string;
  isCurrent: boolean;
  editing: string|null;
  editVal: EditValues;
  setEditing: (id: string|null) => void;
  setEditVal: (v: EditValues) => void;
  onSave: (id: string) => void;
}
function DayDetail({ day, dp, label, nextStay, isCurrent, editing, editVal, setEditing, setEditVal, onSave }: DayDetailProps) {
  const acts = dp.activities||[];
  const stay = dp.stay;
  const stayAct = acts.find(a=>{
    const title = ('title' in a ? a.title : a.name)||'';
    const type  = ('type'  in a ? a.type  : '')||'';
    return type.includes('ACCOMMODATION')||title.toLowerCase().match(/hotel|resort|hostel|stay/);
  });
  const stayCost = stayAct && 'estimatedCost' in stayAct ? stayAct.estimatedCost : undefined;

  return (
    <div className="day-detail">
      <div className="day-detail-header">
        <div>
          <h2>{label}</h2>
          {dp.theme&&<p className="day-detail-theme">✨ {dp.theme}</p>}
        </div>
        {isCurrent&&<div className="today-badge"><Clock size={14}/> Today</div>}
      </div>

      {stay&&(
        <div className="stay-banner">
          <div className="stay-banner-left">
            <Hotel size={20} color="#6c8eff"/>
            <div>
              <div className="stay-banner-label">Tonight's Stay</div>
              <div className="stay-banner-name">{stay}</div>
              {stayCost&&<div className="stay-banner-cost">Estimated: ₹{Number(stayCost).toLocaleString()} / night <span className="stay-change-hint">(change in chat if needed)</span></div>}
            </div>
          </div>
          {nextStay&&nextStay!==stay&&(
            <div className="stay-banner-next"><ChevronRight size={14}/><span>Next: {nextStay}</span></div>
          )}
        </div>
      )}

      <DayMap activities={acts} dayNumber={day} dayLabel={label} />

      <div className="act-timeline">
        {acts.map((act,i)=>{
          const isActivity = 'id' in act;
          const title  = isActivity ? (act as Activity).title  : ((act as RawActivity).name||(act as RawActivity).title||'Activity');
          const desc   = isActivity ? (act as Activity).description||'' : ((act as RawActivity).details||(act as RawActivity).description||'');
          const type   = (isActivity ? (act as Activity).type : (act as RawActivity).type)||'OTHER';
          const st     = isActivity ? (act as Activity).startTime : ((act as RawActivity).start_time||(act as RawActivity).startTime||(act as RawActivity).time);
          const et     = isActivity ? (act as Activity).endTime   : ((act as RawActivity).end_time  ||(act as RawActivity).endTime);
          const travel = isActivity ? undefined : (act as RawActivity).travel_from_previous;
          const cost   = isActivity ? (act as Activity).estimatedCost : (act as RawActivity).estimatedCost;
          const notes  = isActivity ? (act as Activity).notes : undefined;
          const actId  = isActivity ? (act as Activity).id : String(i);
          const emoji  = ACT_ICON[type]||'📍';
          const tod    = timeOfDay(st);
          const isEdit = editing===actId;

          return (
            <div key={actId} className={`tl-item ${type.toLowerCase()}`}>
              <div className="tl-time-col">
                <div className="tl-tod">{TOD_ICON[tod]}</div>
                <div className="tl-time">{st||''}</div>
              </div>
              <div className="tl-line">
                <div className="tl-dot">{emoji}</div>
                {i<acts.length-1&&<div className="tl-conn"/>}
              </div>
              <div className="tl-body card">
                {isEdit ? (
                  <div className="act-edit">
                    <input defaultValue={title} onChange={(e:ChangeEvent<HTMLInputElement>)=>setEditVal({...editVal,title:e.target.value})} placeholder="Title"/>
                    <textarea defaultValue={desc} onChange={(e:ChangeEvent<HTMLTextAreaElement>)=>setEditVal({...editVal,description:e.target.value})} placeholder="Description" rows={3}/>
                    <div className="act-edit-row">
                      <input defaultValue={st||''} onChange={(e:ChangeEvent<HTMLInputElement>)=>setEditVal({...editVal,startTime:e.target.value})} placeholder="Start time"/>
                      <input defaultValue={et||''} onChange={(e:ChangeEvent<HTMLInputElement>)=>setEditVal({...editVal,endTime:e.target.value})} placeholder="End time"/>
                    </div>
                    <input type="number" defaultValue={cost||''} onChange={(e:ChangeEvent<HTMLInputElement>)=>setEditVal({...editVal,estimatedCost:e.target.value})} placeholder="Estimated cost (₹)"/>
                    <textarea defaultValue={notes||''} onChange={(e:ChangeEvent<HTMLTextAreaElement>)=>setEditVal({...editVal,notes:e.target.value})} placeholder="Notes" rows={2}/>
                    <div className="act-edit-btns">
                      <button className="btn-primary" onClick={()=>isActivity&&onSave(actId)}>Save</button>
                      <button className="btn-ghost" onClick={()=>setEditing(null)}>Cancel</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="tl-body-top">
                      <div>
                        <div className="tl-title">{title}</div>
                        {desc&&<div className="tl-desc">{desc}</div>}
                      </div>
                      {isActivity&&<button className="edit-act-btn" onClick={()=>{ setEditing(actId); setEditVal({ title, description:desc }); }}><Edit3 size={13}/></button>}
                    </div>
                    <div className="tl-meta">
                      {st&&et&&<span className="tl-meta-item"><Clock size={11}/> {st} – {et}</span>}
                      {travel&&<span className="tl-meta-item"><Car size={11}/> {travel}</span>}
                      {(cost||0)>0&&<span className="tl-cost">₹{Number(cost).toLocaleString()}</span>}
                      {notes&&<div className="tl-notes">📝 {notes}</div>}
                    </div>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {nextStay&&(
        <div className="next-day-banner">
          <Hotel size={16}/>
          <div>
            <div className="next-day-title">Day {day+1} begins at your next stay</div>
            <div className="next-day-stay">{nextStay}</div>
          </div>
          <ChevronRight size={16}/>
        </div>
      )}
    </div>
  );
}
