export interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
}

export interface Trip {
  id: string;
  destination: string;
  title?: string;
  description?: string;
  startDate?: string;
  endDate?: string;
  duration?: number;
  status?: 'PLANNING' | 'PLANNED' | 'ONGOING' | 'FINISHED';
  totalBudget?: number;
  numberOfTravelers?: number;
  currency?: string;
  conversationPhase?: string;
  hasItinerary?: boolean;
  coverImage?: string;
  tripContext?: any;
  itinerary?: Itinerary;
}

export interface Itinerary {
  id: string;
  summary?: string;
  highlights?: string[];
  fullItinerary?: FullItinerary;
  activities: Activity[];
}

export interface FullItinerary {
  destination?: string;
  start_date?: string;
  duration_days?: number;
  daily_plans: DayPlan[];
  notes?: { packing?: string; tips?: string; [key: string]: string | undefined };
}

export interface DayPlan {
  day_number: number;
  date?: string;
  theme?: string;
  stay_name?: string;
  activities: RawActivity[];
}

export interface RawActivity {
  name?: string;
  title?: string;
  details?: string;
  description?: string;
  time?: string;
  start_time?: string;
  end_time?: string;
  startTime?: string;
  endTime?: string;
  location?: { lat: number; lon: number } | string;
  travel_from_previous?: string;
  estimatedCost?: number;
  type?: string;
}

export interface Activity {
  id: string;
  dayNumber: number;
  date?: string;
  title: string;
  description?: string;
  type?: string;
  status?: string;
  location?: string;
  latitude?: number;
  longitude?: number;
  startTime?: string;
  endTime?: string;
  estimatedCost?: number;
  orderIndex: number;
  notes?: string;
}

export interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt?: string;
}

export interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<any>;
  register: (email: string, password: string, firstName: string, lastName: string) => Promise<any>;
  logout: () => void;
}
