import React, { useState, useEffect, MouseEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../api/client';
import { Trip } from '../types';
import { Plus, MapPin, Calendar, Users, Wallet, Compass, LogOut,
         ChevronRight, Clock, CheckCircle, Trash2, AlertCircle } from 'lucide-react';
import { format, parseISO } from 'date-fns';
import './HomePage.css';

interface StatusMeta { label: string; cls: string; icon: React.ReactNode; }
const STATUS_META: Record<string, StatusMeta> = {
  PLANNING: { label:'Planning',  cls:'badge-blue',   icon:<Clock size={11}/> },
  PLANNED:  { label:'Planned',   cls:'badge-green',  icon:<CheckCircle size={11}/> },
  ONGOING:  { label:'Ongoing',   cls:'badge-yellow', icon:<MapPin size={11}/> },
  FINISHED: { label:'Finished',  cls:'badge-red',    icon:<CheckCircle size={11}/> },
};

const COVERS: string[] = [
  'https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?w=600&q=80',
  'https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=600&q=80',
  'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=600&q=80',
  'https://images.unsplash.com/photo-1501854140801-50d01698950b?w=600&q=80',
  'https://images.unsplash.com/photo-1488085061387-422e29b40080?w=600&q=80',
];

export default function HomePage() {
  const { user, logout }  = useAuth();
  const navigate          = useNavigate();
  const [trips,    setTrips]    = useState<Trip[]>([]);
  const [loading,  setLoading]  = useState<boolean>(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [creating, setCreating] = useState<boolean>(false);

  useEffect(() => { loadTrips(); }, []);

  const loadTrips = async () => {
    try { const { data } = await api.get('/trips'); setTrips(data); }
    catch(e) { console.error(e); }
    finally  { setLoading(false); }
  };

  const newTrip = async () => {
    setCreating(true);
    try {
      const { data } = await api.post('/trips', { destination:'New Trip' });
      navigate(`/chat/${data.id}`);
    } catch(e) { console.error(e); setCreating(false); }
  };

  const deleteTrip = async (e: MouseEvent, id: string) => {
    e.stopPropagation();
    if (!window.confirm('Delete this trip?')) return;
    setDeleting(id);
    try { await api.delete(`/trips/${id}`); setTrips(t=>t.filter(x=>x.id!==id)); }
    catch(e) { console.error(e); }
    finally  { setDeleting(null); }
  };

  const openTrip = (trip: Trip) => {
    if (trip.hasItinerary) navigate(`/trip/${trip.id}`);
    else                   navigate(`/chat/${trip.id}`);
  };

  const planned  = trips.filter(t=>['PLANNED','ONGOING'].includes(t.status||''));
  const planning = trips.filter(t=>t.status==='PLANNING');
  const finished = trips.filter(t=>t.status==='FINISHED');

  return (
    <div className="home-root">
      <aside className="home-sidebar">
        <div className="sidebar-brand"><Compass size={26} color="#6c8eff"/><span>TravelCopilot</span></div>
        <button className="new-trip-btn" onClick={newTrip} disabled={creating}>
          <Plus size={18}/>{creating?'Creating…':'New Trip'}
        </button>
        <nav className="sidebar-nav">
          <div className="nav-section">All Trips</div>
          {trips.slice(0,8).map(t=>(
            <div key={t.id} className={`nav-item${t.hasItinerary?' has-itin':''}`} onClick={()=>openTrip(t)}>
              <MapPin size={14}/><span>{t.destination||'New Trip'}</span>
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="user-info">
            <div className="user-avatar">{user?.firstName?.[0]?.toUpperCase()||'?'}</div>
            <div>
              <div className="user-name">{user?.firstName} {user?.lastName}</div>
              <div className="user-email">{user?.email}</div>
            </div>
          </div>
          <button className="btn-ghost logout-btn" onClick={logout}><LogOut size={16}/></button>
        </div>
      </aside>

      <main className="home-main">
        <header className="home-header">
          <div>
            <h1>Welcome back, {user?.firstName||'Traveller'} 👋</h1>
            <p>Ready for your next adventure?</p>
          </div>
          <button className="btn-primary" onClick={newTrip} disabled={creating}>
            <Plus size={16}/>{creating?'Creating…':'Plan a new trip'}
          </button>
        </header>

        {loading ? (
          <div className="home-loading"><div className="spinner-lg"/><p>Loading your trips…</p></div>
        ) : trips.length===0 ? (
          <div className="home-empty">
            <div className="empty-icon"><Compass size={48} color="#6c8eff"/></div>
            <h2>No trips yet</h2>
            <p>Start planning your first adventure with AI assistance</p>
            <button className="btn-primary" onClick={newTrip} disabled={creating}><Plus size={16}/>Plan your first trip</button>
          </div>
        ) : (
          <>
            {planned.length>0  && <Section title={<><CheckCircle size={18} color="#34d399"/> Planned Trips</>}  trips={planned}  offset={0} onOpen={openTrip} onDelete={deleteTrip} deleting={deleting}/>}
            {planning.length>0 && <Section title={<><Clock size={18} color="#6c8eff"/> In Planning</>}          trips={planning} offset={2} onOpen={openTrip} onDelete={deleteTrip} deleting={deleting}/>}
            {finished.length>0 && <Section title={<><AlertCircle size={18} color="#8b95b0"/> Past Trips</>}     trips={finished} offset={4} onOpen={openTrip} onDelete={deleteTrip} deleting={deleting}/>}
          </>
        )}
      </main>
    </div>
  );
}

interface SectionProps {
  title: React.ReactNode;
  trips: Trip[];
  offset: number;
  onOpen: (t: Trip) => void;
  onDelete: (e: MouseEvent, id: string) => void;
  deleting: string | null;
}
function Section({ title, trips, offset, onOpen, onDelete, deleting }: SectionProps) {
  return (
    <section className="trip-section">
      <h2 className="section-title">{title}</h2>
      <div className="trip-grid">
        {trips.map((t,i)=>(
          <TripCard key={t.id} trip={t} cover={COVERS[(i+offset)%COVERS.length]}
            onOpen={onOpen} onDelete={onDelete} deleting={deleting===t.id}/>
        ))}
      </div>
    </section>
  );
}

interface TripCardProps {
  trip: Trip;
  cover: string;
  onOpen: (t: Trip) => void;
  onDelete: (e: MouseEvent, id: string) => void;
  deleting: boolean;
}
function TripCard({ trip, cover, onOpen, onDelete, deleting }: TripCardProps) {
  const meta = STATUS_META[trip.status||'PLANNING'] || STATUS_META.PLANNING;
  return (
    <div className="trip-card" onClick={()=>onOpen(trip)}>
      <div className="trip-cover" style={{backgroundImage:`url(${cover})`}}>
        <div className="trip-cover-overlay"/>
        <div className="trip-cover-top">
          <span className={`badge ${meta.cls}`}>{meta.icon} {meta.label}</span>
          <button className="delete-btn" onClick={e=>onDelete(e,trip.id)} disabled={deleting}>
            {deleting?'…':<Trash2 size={14}/>}
          </button>
        </div>
        <div className="trip-cover-bottom">
          <h3>{trip.destination||'New Trip'}</h3>
          {trip.title&&trip.title!==`Trip to ${trip.destination}`&&<p>{trip.title}</p>}
        </div>
      </div>
      <div className="trip-info">
        {trip.startDate&&<div className="trip-meta-item"><Calendar size={13}/><span>{format(parseISO(trip.startDate),'MMM d, yyyy')}</span></div>}
        {(trip.numberOfTravelers||0)>0&&<div className="trip-meta-item"><Users size={13}/><span>{trip.numberOfTravelers} traveler{trip.numberOfTravelers!==1?'s':''}</span></div>}
        {trip.totalBudget&&<div className="trip-meta-item"><Wallet size={13}/><span>₹{Number(trip.totalBudget).toLocaleString()}</span></div>}
        <div className="trip-card-footer">
          <span className="open-label">{trip.hasItinerary?'View itinerary':'Continue planning'}</span>
          <ChevronRight size={15}/>
        </div>
      </div>
    </div>
  );
}
