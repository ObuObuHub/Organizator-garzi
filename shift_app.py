from __future__ import annotations
import calendar
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from io import BytesIO
import random
from collections import defaultdict
import yaml
from pathlib import Path

import json
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from gspread import utils
import locale

# Set locale for Romanian weekday names
try:
    locale.setlocale(locale.LC_TIME, 'ro_RO.UTF-8')
except locale.Error:
    st.warning("Could not set locale to 'ro_RO.UTF-8'. Weekday names might not be in Romanian.")

# ————————————————————————————————————
# 1. Model de ture
# ————————————————————————————————————
@dataclass
class Doctor:
    name: str
    monthly_target_hours: int = 168
    assigned_hours: int = 0
    preferences: Dict[str, any] = field(default_factory=dict)

@dataclass
class ShiftType:
    label: str
    start_hour: int
    end_hour: int
    duration: int

DAY_8_20 = ShiftType("Zi 8-20", 8, 20, 12)
NIGHT_20_8 = ShiftType("Noapte 20-8", 20, 8, 12)
FULL_24H   = ShiftType("Garda 24 h 8-8", 8, 8, 24) # Keep definition, but not used in _generate

SHIFT_TYPES = [
    ("Zi 08-20", "08:00", "20:00"),
    ("Noapte 20-08", "20:00", "08:00")
]

@dataclass
class Shift:
    date: dt.date
    stype: ShiftType
    personal_garda: Optional[str] = None

    @property
    def weekday_name(self):
        # Map English weekday names to Romanian for display
        weekday_map = {
            "Monday": "Luni",
            "Tuesday": "Marți",
            "Wednesday": "Miercuri",
            "Thursday": "Joi",
            "Friday": "Vineri",
            "Saturday": "Sâmbătă",
            "Sunday": "Duminică"
        }
        english_weekday = self.date.strftime("%A")
        return weekday_map.get(english_weekday, english_weekday) # Fallback to English if not found

def penalty(doctor: Doctor, shift: Shift) -> int:
    # Placeholder for penalty logic based on doctor preferences and shift type
    # For example:
    if doctor.preferences.get("avoid_weekends") and shift.date.weekday() in [5, 6]: # 5=Saturday, 6=Sunday
        return 1
    if doctor.preferences.get("prefers_night") and shift.stype != NIGHT_20_8:
        return 1
    return 0

from ortools.sat.python import cp_model

def get_week_of_month(date: dt.date) -> int:
    """Calculates the 7-day period week number within the month (1-indexed)."""
    return (date.day - 1) // 7 + 1

def is_night_shift(shift: Shift) -> bool:
    return shift.stype == NIGHT_20_8

def is_weekend_day_shift(shift: Shift) -> bool:
    return shift.date.weekday() in [5, 6] and shift.stype == DAY_8_20 # 5=Saturday, 6=Sunday

def optimize_schedule(shifts: list[Shift], doctors: list[Doctor]) -> None:
    model = cp_model.CpModel()

    # Create variables
    # x[i, j] is 1 if shift i is assigned to doctor j, 0 otherwise
    x = {}
    for i, shift in enumerate(shifts):
        for j, doctor in enumerate(doctors):
            x[(i, j)] = model.NewBoolVar(f"x_shift{i}_doctor{j}")

    # 1. Constraint: Each shift has exactly one doctor
    for i, shift in enumerate(shifts):
        model.Add(sum(x[(i, j)] for j, _ in enumerate(doctors)) == 1)

    # 2. Hard rules (simplified based on user's clarification)
    #    - No 'hemato' group specific rules.
    #    - Weekly constraints for night shifts and weekend day shifts.
    #    NOTE: The original prompt had specific hard rules for 'hemato' group.
    #    Based on user's clarification "Don't group, disregard" for 'hemato',
    #    these specific hard rules are not applied to all doctors as they would be too restrictive.
    #    Instead, the objective function focuses on minimizing penalties based on preferences.

    # Indisponibil (Unavailable) - Assuming this is handled via preferences/penalties for now.
    # If a doctor is explicitly unavailable for a shift, their penalty for that shift would be very high.
    # The current `penalty` function doesn't have a direct 'unavailable' flag.
    # If `pref[i,j] == 'X'` means a hard unavailability, it needs to be explicitly added.
    # For now, we assume unavailability is implicitly handled by a high penalty.

    # 3. Objective: Maximize preference score – penalties
    # The `penalty` function returns 0 for preferred, 1 otherwise.
    # To maximize score, we want to minimize total penalties.
    total_penalty = model.NewIntVar(0, len(shifts) * len(doctors), 'total_penalty')
    penalties = []
    for i, shift in enumerate(shifts):
        for j, doctor in enumerate(doctors):
            # If x[(i,j)] is true, add penalty(doctor, shift) to total
            penalties.append(x[(i,j)] * penalty(doctor, shift))
    model.Add(total_penalty == sum(penalties))
    model.Minimize(total_penalty)

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for i, shift in enumerate(shifts):
            for j, doctor in enumerate(doctors):
                if solver.Value(x[(i, j)]) == 1:
                    shift.personal_garda = doctor.name # Renamed from doctor
                    break # Each shift has one doctor
    else:
        st.error("No solution found for the optimal schedule. Falling back to random assignment. Try adjusting preferences or constraints.")
        # Fallback to random assignment if no solution is found
        auto_assign(shifts, doctors) # Keep the existing auto_assign as a fallback
        
        # Write the randomly assigned DataFrame to Google Sheet
        sch = Scheduler(shifts[0].date.year, shifts[0].date.month) # Re-create scheduler to get dataframe
        sch.shifts = shifts # Assign the updated shifts
        assigned_df = sch.to_dataframe()
        
        ws = get_worksheet()
        ws.clear()
        ws.update([assigned_df.columns.values.tolist()] + assigned_df.values.tolist())
        ws.freeze(rows=1)

def auto_assign(shifts: list[Shift], doctors: list[Doctor]) -> None:
    hours = defaultdict(int)               # doctor → assigned hours
    rnd = random.Random(42)                # reproducible

    for shift in sorted(shifts, key=lambda s: (s.date, s.stype.start_hour)):
        # pick doctors sorted by (hours so far, preference penalty, random jitter)
        ranked = sorted(
            doctors,
            key=lambda d: (
                hours[d.name],
                penalty(d, shift),         # 0 for preferred slots, 1 otherwise
                rnd.random()
            )
        )
        chosen = ranked[0]
        shift.personal_garda = chosen.name # Renamed from doctor
        hours[chosen.name] += shift.stype.duration

class Scheduler:
    """Generează turele unei luni"""
    def __init__(self, year: int, month: int):
        self.year = year
        self.month = month
        self.shifts: List[Shift] = []
        self._generate()

    def _generate(self):
        days = calendar.monthrange(self.year, self.month)[1]
        for d in range(1, days + 1):
            date = dt.date(self.year, self.month, d)
            for label, start, end in SHIFT_TYPES:
                self.shifts.append(
                    Shift(date, ShiftType(label, int(start[:2]), int(end[:2]), 12))
                )

    def to_dataframe(self):
        rows = []
        for idx, s in enumerate(self.shifts, 1):
            rows.append({
                "id": idx,
                "date": s.date.strftime("%d.%m.%Y"),
                "weekday": s.weekday_name,
                "label": s.stype.label,
                "start": s.stype.start_hour,
                "end": s.stype.end_hour,
                "duration": s.stype.duration,
                "personal_garda": s.personal_garda, # Renamed from doctor
            })
        return pd.DataFrame(rows)

# ————————————————————————————————————
# 2. Google Sheets helpers
# ————————————————————————————————————
SHEET_NAME = "Garzi_Laborator_2025"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]
CRED_FILE = "gcreds.json"  # puneți fișierul service-account aici

@st.cache_resource(show_spinner=False)
def get_gsheet_client():
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]
    # Dacă am pus secretul în Streamlit Cloud, îl luăm de acolo
    if "gcp_service_account" in st.secrets:
        import json
        info = json.loads(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    else:
        # fallback local, dacă vrei să testezi cu gcreds.json
        creds = Credentials.from_service_account_file(CRED_FILE, scopes=SCOPES)
    return gspread.authorize(creds)


@st.cache_resource(show_spinner=False)
def get_worksheet(year: int, month_num: int): # Added year and month_num parameters
    gc = get_gsheet_client()
    try:
        sh = gc.open(SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(SHEET_NAME)
    
    worksheet_name = f"{calendar.month_name[month_num]}_{year}" # Dynamic worksheet name
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows="60", cols="10")
        sch = Scheduler(year, month_num) # Use dynamic year and month
        df_init = sch.to_dataframe()
        ws.update([df_init.columns.values.tolist()] + df_init.values.tolist())
        ws.freeze(rows=1)
    return ws

import locale # For Romanian weekday names

@st.cache_data(show_spinner=False, ttl=60) # Live refresh every 60s
def load_df(year: int, month_num: int): # Added year and month_num parameters
    ws = get_worksheet(year, month_num) # Pass year and month_num
    data = ws.get_all_records()
    return pd.DataFrame(data)

def reserve_shift(shift_id: int, doctor_name: str, year: int, month_num: int) -> bool: # Added year and month_num parameters
    ws = get_worksheet(year, month_num) # Pass year and month_num
    df = load_df(year, month_num) # Pass year and month_num
    row_idx_list = df.index[df["id"] == shift_id]
    if row_idx_list.empty:
        return False
    row_idx = int(row_idx_list[0]) + 2  # header offset
    personal_garda_col = df.columns.get_loc("personal_garda") + 1
    current_val = ws.cell(row_idx, personal_garda_col).value
    if current_val:
        return False
    col_letter = utils.col_to_url_param(personal_garda_col)
    ws.update(f"{col_letter}{row_idx}", [[doctor_name]], value_input_option="USER_ENTERED", include_values_in_response=False)
    return True

# ————————————————————————————————————
# 3. Streamlit UI
# ————————————————————————————————————
@st.cache_data(show_spinner=False)
def load_doctors() -> List[Doctor]:
    with open(Path(__file__).parent / "doctors.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [Doctor(**d) for d in data]

@st.cache_data(show_spinner=False)
def load_holidays() -> list[dt.date]:
    """Returnează lista de sărbători (sau goală), indiferent de cum sunt datele în YAML."""
    with open("holidays.yaml", "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    holidays: list[dt.date] = []
    for item in raw:
        # Dacă e deja un obiect date, îl folosim direct
        if isinstance(item, dt.date):
            holidays.append(item)
        else:
            # În orice alt caz, încercăm să-l convertim din string
            try:
                holidays.append(dt.date.fromisoformat(str(item)))
            except Exception:
                # dacă nu merge, îl sărim
                continue
    return holidays


def main_ui():
    st.title("Organizator gărzi") # Changed title as per sketch

    # Year and Month selection - MOVED TO TOP
    current_year = dt.date.today().year
    current_month = dt.date.today().month
    
    year = st.selectbox("An", options=list(range(current_year, current_year + 3)), index=0)
    month_names = [calendar.month_name[i] for i in range(1, 13)]
    month = st.selectbox("Lună", options=month_names, index=current_month - 1)
    month_num = month_names.index(month) + 1 # Convert month name back to number

    doctors = load_doctors()
    holidays = load_holidays()

    default_names = [d.name for d in doctors]
    df = load_df(year, month_num) # Pass year and month_num
    extra = sorted(set(df["personal_garda"].dropna()) - set(default_names))
    names = ["— alege —"] + sorted(set(default_names + extra))

    user = st.selectbox("Eu sunt…", names)
    if user == "— alege —":
        st.info("Alege-ți numele din listă.")
        st.stop()

    tab_initial, tab_preferences, tab_final = st.tabs(["Initial", "Preferences", "Final"])

    with tab_initial:
        st.markdown("### Generează rota inițială")
        if st.button("Generează rota random"):
            with st.spinner("Se generează rota…"):
                sch = Scheduler(year, month_num)
                shifts_to_assign = sch.shifts
                auto_assign(shifts_to_assign, doctors)
                assigned_df = sch.to_dataframe()
                
                ws = get_worksheet(year, month_num) # Pass year and month_num
                ws.clear()
                ws.update([assigned_df.columns.values.tolist()] + assigned_df.values.tolist())
                ws.freeze(rows=1)
            st.success("Rota generată și salvată!")
            st.cache_data.clear() # Clear cache
            st.experimental_rerun()

    with tab_preferences:
        st.info("Completează preferinţele direct în Google Sheets, apoi apasă Optimizează.")
        if st.button("Optimizează"):
            with st.spinner("Se optimizează rota…"):
                sch = Scheduler(year, month_num) # Re-generate shifts for the month
                shifts_to_assign = sch.shifts
                
                # Load current assignments from Google Sheet to preserve manual changes
                current_df = load_df(year, month_num) # Pass year and month_num
                for shift_obj in shifts_to_assign:
                    # Find the corresponding row in the current_df
                    matching_rows = current_df[(current_df['date'] == shift_obj.date.strftime("%d.%m.%Y")) & 
                                               (current_df['label'] == shift_obj.stype.label)]
                    if not matching_rows.empty:
                        assigned_personal_garda = matching_rows.iloc[0]['personal_garda']
                        if assigned_personal_garda:
                            shift_obj.personal_garda = assigned_personal_garda

                optimize_schedule(shifts_to_assign, doctors) # Use the OR-Tools optimizer

                optimized_df = sch.to_dataframe()
                
                ws = get_worksheet(year, month_num) # Pass year and month_num
                ws.clear()
                ws.update([optimized_df.columns.values.tolist()] + optimized_df.values.tolist())
                ws.freeze(rows=1)
            st.success("Rota optimizată și salvată!")
            st.cache_data.clear() # Clear cache
            st.experimental_rerun()

    with tab_final:
        st.markdown("### Rota finală")
        df = load_df(year, month_num) # Reload df for final tab after potential changes
        st.dataframe(df)

    # The following sections remain outside the tabs as they are general UI elements
    st.markdown("### Ture libere")
    free_df = df[df["personal_garda"] == ""]
    if free_df.empty:
        st.info("Nu mai există ture libere!")
    else:
        for _, r in free_df.iterrows():
            col1, col2, col3, col4 = st.columns([2,2,1,1])
            col1.write(f"**{r['date']}** ({r['weekday']})")
            shift_time_display = f"{r['start']:02d}:00 – {r['end']:02d}:00"
            if r['label'] == NIGHT_20_8.label:
                shift_time_display += " (+1)"
            col2.write(f"{r['label']}: {shift_time_display}")
            col3.write(f"{r['duration']} h")
            if col4.button("Rezervă tură", key=int(r['id'])): # Changed button text
                with st.spinner("Se rezervă…"):
                    ok = reserve_shift(int(r['id']), user, year, month_num)
                if ok:
                    st.success("Gata, tură rezervată!")
                else:
                    st.error("Ups – altcineva a luat-o înainte.")
                st.experimental_rerun()

    st.markdown("### Turele mele")
    mine = df[df["personal_garda"] == user]
    st.dataframe(mine[[
        'date','weekday','label','start','end','duration'
    ]])

    st.markdown("### Export Excel complet")
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Garzi")
        writer.sheets['Garzi'].set_column('A:G', 14)
    st.download_button("Descarcă .xlsx", data=bio.getvalue(),
                       file_name="garzi_iun2025.xlsx")

if __name__ == "__main__":
    main_ui()
