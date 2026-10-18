"""
Stage 1 (Bonus Architecture): Dynamic SQL Database Store (`sql_store.py`).

Educational Concept:
--------------------
Why do we store Parking Availability, Prices, and Working Hours in SQLite
instead of Pinecone?
1. Vector databases (like Pinecone) store static text embeddings. If 10 cars
   enter the garage in 2 minutes, re-embedding and updating vectors is slow
   and prone to LLM hallucination on arithmetic.
2. Relational SQL databases (SQLite / PostgreSQL) give 100% exact, real-time
   numbers with ACID transactions when spaces are reserved or released.
3. Notice how we use **Pydantic Models (`BaseModel`)** (`ParkingZoneStatus`,
   `OperatingHours`, `DynamicParkingSnapshot`) to validate and structure every
   row coming out of SQLite!
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from src.config import SQLITE_DB_PATH


# ============================================================================
# 1. PYDANTIC MODELS (The "Strict Forms" for our SQL Data!)
# ============================================================================
class ParkingZoneStatus(BaseModel):
    """Represents real-time availability and pricing for a single parking zone."""

    zone_code: str = Field(description="Short zone identifier, e.g., 'ZONE_A'")
    zone_name: str = Field(description="Human-readable zone name")
    total_spaces: int = Field(ge=0, description="Total physical stalls in this zone")
    available_spaces: int = Field(ge=0, description="Currently free spaces right now")
    hourly_rate_usd: float = Field(ge=0.0, description="Price per hour in USD")
    daily_max_usd: float = Field(ge=0.0, description="Maximum price per 24 hours in USD")
    vehicle_type: str = Field(description="Supported vehicles (Standard, EV, VIP/Oversized)")


class OperatingHours(BaseModel):
    """Represents working hours for a specific day category."""

    day_category: str = Field(description="E.g., 'Monday - Friday', 'Saturday - Sunday', 'Holidays'")
    open_time: str = Field(description="Opening time in HH:MM format")
    close_time: str = Field(description="Closing time in HH:MM format")
    notes: str = Field(description="Special operational notes for these hours")


class DynamicParkingSnapshot(BaseModel):
    """Complete real-time snapshot of all dynamic parking data."""

    facility_name: str = "AeroPark Smart Garage"
    total_available_spaces: int
    zones: List[ParkingZoneStatus]
    operating_hours: List[OperatingHours]

    def to_context_string(self) -> str:
        """
        Convert this validated Pydantic model into a clean, readable text block
        that we can inject directly into the LLM's prompt during RAG!
        """
        lines = [
            f"=== LIVE SQL DATABASE SNAPSHOT: {self.facility_name} ===",
            f"Total Available Spaces Across All Zones: {self.total_available_spaces}",
            "",
            "--- Live Zone Availability & Pricing ---",
        ]
        for z in self.zones:
            lines.append(
                f"* {z.zone_name} ({z.zone_code}): "
                f"{z.available_spaces}/{z.total_spaces} spaces available | "
                f"${z.hourly_rate_usd:.2f}/hour (Daily Max: ${z.daily_max_usd:.2f}) | "
                f"For: {z.vehicle_type}"
            )

        lines.append("")
        lines.append("--- Current Working Hours ---")
        for h in self.operating_hours:
            lines.append(
                f"* {h.day_category}: {h.open_time} - {h.close_time} ({h.notes})"
            )
        return "\n".join(lines)


# ============================================================================
# 2. SQLITE DATABASE MANAGER
# ============================================================================
class DynamicParkingSQLStore:
    """Manages the SQLite database for dynamic parking availability, prices, and hours."""

    def __init__(self, db_path: Path | str = SQLITE_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_and_seed_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_and_seed_db(self) -> None:
        """Create tables if they don't exist and seed initial realistic data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS parking_zones (
                    zone_code TEXT PRIMARY KEY,
                    zone_name TEXT NOT NULL,
                    total_spaces INTEGER NOT NULL,
                    available_spaces INTEGER NOT NULL,
                    hourly_rate_usd REAL NOT NULL,
                    daily_max_usd REAL NOT NULL,
                    vehicle_type TEXT NOT NULL
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS operating_hours (
                    day_category TEXT PRIMARY KEY,
                    open_time TEXT NOT NULL,
                    close_time TEXT NOT NULL,
                    notes TEXT NOT NULL
                )
                """
            )

            # Seed default zones if table is empty
            cursor.execute("SELECT COUNT(*) FROM parking_zones")
            if cursor.fetchone()[0] == 0:
                cursor.executemany(
                    """
                    INSERT INTO parking_zones (
                        zone_code, zone_name, total_spaces, available_spaces,
                        hourly_rate_usd, daily_max_usd, vehicle_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "ZONE_A",
                            "Zone A - Standard Covered (Levels P1 & P2)",
                            120,
                            42,
                            4.50,
                            28.00,
                            "Standard Sedans, Hatchbacks & Compact SUVs (Max Height 2.10m)",
                        ),
                        (
                            "ZONE_B",
                            "Zone B - EV Charging & Green Zone (Level P1 East)",
                            30,
                            11,
                            6.00,
                            38.00,
                            "Electric Vehicles using 22kW Level 2 Chargers",
                        ),
                        (
                            "ZONE_C",
                            "Zone C - VIP & Oversized Executive Zone (Ground Level)",
                            20,
                            5,
                            9.50,
                            55.00,
                            "VIP Guests, Luxury Vehicles & Tall Vans (Max Height 2.65m)",
                        ),
                    ],
                )

            # Seed default operating hours if table is empty
            cursor.execute("SELECT COUNT(*) FROM operating_hours")
            if cursor.fetchone()[0] == 0:
                cursor.executemany(
                    """
                    INSERT INTO operating_hours (
                        day_category, open_time, close_time, notes
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            "Monday - Friday",
                            "06:00",
                            "23:59",
                            "Automated 24/7 exit for pre-booked vehicles; attendant on duty 06:00-23:59",
                        ),
                        (
                            "Saturday - Sunday",
                            "07:00",
                            "23:59",
                            "Weekend flat rate discount applies after 4 hours; 24/7 LPR exit",
                        ),
                        (
                            "Public Holidays",
                            "08:00",
                            "22:00",
                            "Gate B automated LPR entry remains active for confirmed reservations",
                        ),
                    ],
                )
            conn.commit()

    def get_zones(self) -> List[ParkingZoneStatus]:
        """Fetch all parking zones and validate each row with our Pydantic Model."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM parking_zones ORDER BY zone_code"
            ).fetchall()
            # Notice `.model_validate(dict(row))` from our Pydantic lesson!
            return [ParkingZoneStatus.model_validate(dict(row)) for row in rows]

    def get_operating_hours(self) -> List[OperatingHours]:
        """Fetch operating hours and validate each row with our Pydantic Model."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM operating_hours").fetchall()
            return [OperatingHours.model_validate(dict(row)) for row in rows]

    def get_live_snapshot(self) -> DynamicParkingSnapshot:
        """Return a complete validated Pydantic snapshot of all dynamic parking data."""
        zones = self.get_zones()
        hours = self.get_operating_hours()
        total_avail = sum(z.available_spaces for z in zones)
        return DynamicParkingSnapshot(
            total_available_spaces=total_avail,
            zones=zones,
            operating_hours=hours,
        )

    def reserve_space(self, zone_code: str = "ZONE_A") -> bool:
        """
        Decrement available_spaces by 1 for the specified zone once a reservation
        is confirmed. Returns True if successful, False if the zone is full.
        """
        normalized = zone_code.strip().upper()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE parking_zones
                SET available_spaces = available_spaces - 1
                WHERE zone_code = ? AND available_spaces > 0
                """,
                (normalized,),
            )
            conn.commit()
            return cursor.rowcount > 0

