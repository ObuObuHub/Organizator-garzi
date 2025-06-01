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

import os
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

@dataclass(frozen=True)
class ShiftType:
    label: str
    start_hour: int
    end_hour: int
    duration: int

# We define two ShiftType constants and then build SHIFT_TYPES from them,
# so that comparisons (e.g. shift.stype == NIGHT_20_8) will always work.
DAY_8_20 = ShiftType("Zi 08-20", 8, 20, 12)
NIGHT_20_08 = ShiftType("Noapte 20-08", 20, 8, 12)
FULL_24H   = ShiftType("Garda 24 h 08-08", 8, 8, 24)  # kept for completeness, not used in _generate

# Now SHIFT_TYPES is a list of the same instances:
SHIFT_TYPES: List[ShiftType] = [DAY_8_20, NIGHT_20_08]

@dataclass
class Shift:
    date: dt.date
    stype: ShiftType
    personal_garda: Optional[str] = None

    @property
    def weekday_name(self) -> str:
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
        return weekday_map.get(english_weekday, english_weekday)  # fallback to English if not found

def penalty(doctor: Doctor, shift: Shift) -> int:
    """
    Return 1 if this (doctor, shift) pair violates a soft preference,
    0 otherwise. Higher penalty → less desirable assignment.
    """
    # If a doctor prefers to avoid weekends, penalize any weekend day shift:
    if doctor.preferences.get("avoid_weekends", False) and shift.date.weekday() in (5, 6):
        return 1
    # If a doctor prefers nights but this is not a night shift, penalize:
    if doctor.preferences.get("prefers_night", False) and shift.stype != NIGHT_20_08:
        return 1
    # If a doctor *cannot* do nights (e.g. an “unavailable” marker), you could return a large penalty here.
    # For now, just return 0 if none of those soft‐preferences apply.
    return 0

from ortools.sat.python import cp_model

def get_week_of_month(date: dt.date) -> int:
    """Calculates the 7-day period week number within the month (1-indexed)."""
    return (date.day - 1) // 7 + 1

def is_night_shift(shift: Shift) -> bool:
    return shift.stype == NIGHT_20_08

def is_weekend_day_shift(shift: Shift) -> bool:
    return shift.date.weekday() in (5, 6) and shift.stype == DAY_8_20

def optimize_schedule(shifts: List[Shift], doctors: List[Doctor]) -> None:
    """
    Build and solve a CP‐SAT model where each shift is assigned exactly one doctor,
    minimizing the total penalty(distance to preferences).
    On failure, fallback to auto_assign() but do NOT attempt to write to Google Sheets here.
    """
    model = cp_model.CpModel()

    # Create boolean assignment variables:
    #   x[(i,j)] = 1 if shift i is assigned to doctor j.
    x: Dict[tuple[int,int], cp_model.IntVar] = {}
    for i, shift in enumerate(shifts):
        for j, doctor in enumerate(doctors):
            x[(i, j)] = model.NewBoolVar(f"x_shift{i}_doctor{j}")

    # 1. Constraint: Each shift has exactly one doctor
    for i in range(len(shifts)):
        model.Add(sum(x[(i, j)] for j in range(len(doctors))) == 1)

    # 2. (Optional) You could add “hard” constraints here (max hours/week, etc.)
    #    For now, we only enforce “every shift must be covered exactly once.”

    # 3. Objective: Minimize total penalty
    #    If x[(i,j)] == 1, add penalty(doctors[j], shifts[i]).
    penalty_terms = []
    for i, shift in enumerate(shifts):
        for j, doctor in enumerate(doctors):
            c = penalty(doctor, shift)
            if c != 0:
                # If penalty is 0, adding 0*x is pointless; skip it.
                penalty_terms.append(x[(i, j)] * c)

    total_penalty = model.NewIntVar(0, len(shifts) * len(doctors), "total_penalty")
    model.Add(total_penalty == sum(penalty_terms))
    model.Minimize(total_penalty)

    # Solve:
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, shift in enumerate(shifts):
            for j, doctor in enumerate(doctors):
                if solver.Value(x[(i, j)]) == 1:
                    shift.personal_garda = doctor.name
                    break
    else:
        st.error("Nu s-a găsit o soluție optimă. Se reîncercă cu alocare aleatoare.")
        auto_assign(shifts, doctors)
        # **Do not write to Google Sheets here**; the Streamlit UI calling function
        # is responsible for pushing the result back into Sheets.

def auto_assign(shifts: List[Shift], doctors: List[Doctor]) -> None:
    """
    A simple fallback: iterate through shifts (sorted by date & start hour),
    assign the doctor with the fewest total hours so far (plus penalty & random jitter).
    """
    hours = defaultdict(int)  # name → total assigned hours
    rnd = random.Random(42)   # reproducibility

    for shift in sorted(shifts, key=lambda s: (s.date, s.stype.start_hour)):
        ranked = sorted(
            doctors,
            key=lambda d: (
                hours[d.name],
                penalty(d, shift),
                rnd.random()
            )
        )
        chosen = ranked[0]
        shift.personal_garda = chosen.name
        hours[chosen.name] += shift.stype.duration

class Scheduler:
    """Generează turele pentru o lună dată."""
    def __init__(self, year: int, month: int):
        self.year = year
        self.month = month
        self.shifts: List[Shift] = []
        self._generate()

    def _generate(self) -> None:
        days_in_month = calendar.monthrange(self.year, self.month)[1]
        for day in range(1, days_in_month + 1):
            date = dt.date(self.year, self.month, day)
            for stype in SHIFT_TYPES:
                self.shifts.append(Shift(date=date, stype=stype))

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for idx, s in enumerate(self.shifts, start=1):
            rows.append({
                "id": idx,
                "date": s.date.strftime("%d.%m.%Y"),
                "weekday": s.weekday_name,
                "label": s.stype.label,
                "start": s.stype.start_hour,
                "end": s.stype.end_hour,
                "duration": s.stype.duration,
                "personal_garda": s.personal_garda or "",
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
@st.cache_resource(show_spinner=False)
def get_gsheet_client():
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]
    # Directly access the credentials dictionary from st.secrets.
    # The previous steps ensured 'gcp_service_account' is now a dict.
    info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

@st.cache_resource(show_spinner=False)
def get_worksheet(year: int, month_num: int):
    """
    Deschide (sau creează) spreadsheet-ul și worksheet-ul potrivit pentru an/ lună.
    Inițializează cu o rota goală, dacă nu există deja.
    """
    gc = get_gsheet_client()
    try:
        sh = gc.open(SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(SHEET_NAME)

    ws_title = f"{calendar.month_name[month_num]}_{year}"
    try:
        ws = sh.worksheet(ws_title)
    except gspread.WorksheetNotFound:
        # Dacă nu există foaia de lucru, o creăm și populăm cu rota inițială
        ws = sh.add_worksheet(title=ws_title, rows="60", cols="10")
        sch = Scheduler(year, month_num)
        df_init = sch.to_dataframe()
        ws.update([df_init.columns.values.tolist()] + df_init.values.tolist())
        ws.freeze(rows=1)
    return ws

@st.cache_data(show_spinner=False, ttl=60)
def load_df(year: int, month_num: int) -> pd.DataFrame:
    """
    Încarcă rotă din Google Sheets într-un DataFrame.
    """
    ws = get_worksheet(year, month_num)
    data = ws.get_all_records()
    return pd.DataFrame(data)

def reserve_shift(shift_id: int, doctor_name: str, year: int, month_num: int) -> bool:
    """
    Încearcă să reserve un loc pe rândul „shift_id” cu numele „doctor_name”.
    Returnează True dacă succes, False altfel.
    """
    ws = get_worksheet(year, month_num)
    df = load_df(year, month_num)
    matching = df.index[df["id"] == shift_id]
    if matching.empty:
        return False
    row_idx = int(matching[0]) + 2  # offset din cauza antetului (rândul 1)
    col_idx = df.columns.get_loc("personal_garda") + 1
    current_val = ws.cell(row_idx, col_idx).value
    if current_val:  # dacă există deja cineva
        return False

    # Construim notarea A1 pentru (row_idx, col_idx):
    a1_cell = utils.rowcol_to_a1(row_idx, col_idx)
    ws.update(a1_cell, doctor_name, value_input_option="USER_ENTERED")
    return True

# ————————————————————————————————————
# 3. Streamlit UI
# ————————————————————————————————————
@st.cache_data(show_spinner=False)
def load_doctors() -> List[Doctor]:
    yaml_path = Path(__file__).parent / "doctors.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [Doctor(**d) for d in data]

@st.cache_data(show_spinner=False)
def load_holidays() -> List[dt.date]:
    """Returnează lista de sărbători (ou goală)."""
    path = Path(__file__).parent / "holidays.yaml"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    holidays: List[dt.date] = []
    for item in raw:
        if isinstance(item, dt.date):
            holidays.append(item)
        else:
            try:
                holidays.append(dt.date.fromisoformat(str(item)))
            except Exception:
                continue
    return holidays

def main_ui():
    st.title("Organizator gărzi")

    # === DEBUG: afișează directorul curent și listarea lui ===
    st.write("🛠️ [DEBUG] current working directory (os.getcwd()):", os.getcwd())
    st.write("🛠️ [DEBUG] conținut director curent (os.listdir):", os.listdir(os.getcwd()))

    # === DEBUG: verifică dacă există subfolder .streamlit și ce fișiere are ===
    if os.path.isdir(".streamlit"):
        st.write("✅ [DEBUG] există folder .streamlit, conținut:", os.listdir(".streamlit"))
    else:
        st.write("❌ [DEBUG] NU există folder .streamlit în acest director!")

    # === DEBUG: afișează conținutul curent al st.secrets ===
    st.write("🔑 [DEBUG] st.secrets:", st.secrets)

    # Oprește restul aplicației, ca să vezi doar datele de debug
    st.stop()

    # Selectăm anul și luna (implicit: luna curentă)
    today = dt.date.today()
    current_year = today.year
    current_month = today.month

    year = st.selectbox("An", options=list(range(current_year, current_year + 3)), index=0)
    month_names = [calendar.month_name[i] for i in range(1, 13)]
    month_selected = st.selectbox("Lună", options=month_names, index=current_month - 1)
    month_num = month_names.index(month_selected) + 1

    doctors = load_doctors()
    holidays = load_holidays()

    default_names = [d.name for d in doctors]
    df = load_df(year, month_num)
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

                ws = get_worksheet(year, month_num)
                ws.clear()
                ws.update([assigned_df.columns.values.tolist()] + assigned_df.values.tolist())
                ws.freeze(rows=1)

            st.success("Rota generată și salvată!")
            st.cache_data.clear()
            st.experimental_rerun()

    with tab_preferences:
        st.info("Completează preferinţele direct în Google Sheets, apoi apasă Optimizează.")
        if st.button("Optimizează"):
            with st.spinner("Se optimizează rota…"):
                sch = Scheduler(year, month_num)
                shifts_to_assign = sch.shifts

                # Preluăm asignările curente (dacă s-au rezervat manual)
                current_df = load_df(year, month_num)
                for shift_obj in shifts_to_assign:
                    match = current_df[
                        (current_df['date'] == shift_obj.date.strftime("%d.%m.%Y")) &
                        (current_df['label'] == shift_obj.stype.label)
                    ]
                    if not match.empty:
                        assigned_personal_garda = match.iloc[0]['personal_garda']
                        if assigned_personal_garda:
                            shift_obj.personal_garda = assigned_personal_garda

                optimize_schedule(shifts_to_assign, doctors)

                optimized_df = sch.to_dataframe()
                ws = get_worksheet(year, month_num)
                ws.clear()
                ws.update([optimized_df.columns.values.tolist()] + optimized_df.values.tolist())
                ws.freeze(rows=1)

            st.success("Rota optimizată și salvată!")
            st.cache_data.clear()
            st.experimental_rerun()

    with tab_final:
        st.markdown("### Rota finală")
        df = load_df(year, month_num)
        st.dataframe(df)

    # Secțiunea „Ture libere”
    st.markdown("### Ture libere")
    free_df = df[df["personal_garda"] == ""]
    if free_df.empty:
        st.info("Nu mai există ture libere!")
    else:
        for _, r in free_df.iterrows():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            col1.write(f"**{r['date']}** ({r['weekday']})")
            shift_time_display = f"{int(r['start']):02d}:00 – {int(r['end']):02d}:00"
            if r['label'] == NIGHT_20_08.label:
                shift_time_display += " (+1)"
            col2.write(f"{r['label']}: {shift_time_display}")
            col3.write(f"{r['duration']} h")
            if col4.button("Rezervă tură", key=int(r['id'])):
                with st.spinner("Se rezervă…"):
                    ok = reserve_shift(int(r['id']), user, year, month_num)
                if ok:
                    st.success("Gata, tură rezervată!")
                else:
                    st.error("Ups – altcineva a luat-o înainte.")
                st.experimental_rerun()

    # Secțiunea „Turele mele”
    st.markdown("### Turele mele")
    mine = df[df["personal_garda"] == user]
    st.dataframe(mine[["date", "weekday", "label", "start", "end", "duration"]])

    # Export to Excel
    st.markdown("### Export Excel complet")
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Garzi")
        writer.sheets['Garzi'].set_column('A:G', 14)
    st.download_button(
        "Descarcă .xlsx",
        data=bio.getvalue(),
        file_name=f"garzi_{month_names[month_num-1].lower()}{year}.xlsx"
    )

if __name__ == "__main__":
    main_ui()
