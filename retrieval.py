"""Retrieval of verified customer records, policies, and conflict detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

from .config import settings


@dataclass
class RetrievalBundle:
    known_customer: bool
    customer_profile: dict[str, Any]
    verified_records: list[dict[str, Any]]
    unverified_record_ids: list[str]
    active_policies: list[dict[str, Any]]
    expired_policies: list[dict[str, Any]]
    data_conflicts: list[str]
    has_unresolved_conflict: bool


class DataPackRepository:
    """Loads customers, interaction records, and policy documents."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path or settings.data_pack_path)
        payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        self.customers: dict[str, dict[str, Any]] = {
            c["customer_id"]: c for c in payload.get("customers", [])
        }
        self.records: list[dict[str, Any]] = payload.get("records", [])
        self.policies: list[dict[str, Any]] = payload.get("policies", [])

    def retrieve(self, customer_id: str, session_date: str) -> RetrievalBundle:
        """Retrieve verified records, active/expired policies, and conflicts."""
        cid = customer_id.strip().upper()
        customer = self.customers.get(cid)
        if not customer:
            return RetrievalBundle(
                known_customer=False,
                customer_profile={},
                verified_records=[],
                unverified_record_ids=[],
                active_policies=[],
                expired_policies=[],
                data_conflicts=[],
                has_unresolved_conflict=False,
            )

        cust_records = [r for r in self.records if r.get("customer_id") == cid]
        verified_records = sorted(
            [r for r in cust_records if r.get("verified") is True],
            key=lambda r: r.get("date", ""),
        )
        unverified_ids = [
            r["record_id"] for r in cust_records if r.get("verified") is not True
        ]

        # Detect conflicting verified records on the same dispute_group
        conflicts: list[str] = []
        has_unresolved = False
        by_group: dict[str, list[dict[str, Any]]] = {}
        for rec in verified_records:
            grp = rec.get("dispute_group")
            if grp:
                by_group.setdefault(grp, []).append(rec)

        for grp, items in by_group.items():
            if len(items) < 2:
                continue
            statuses = {it.get("status") for it in items}
            if "resolved" in statuses and "open_dispute" in statuses:
                earlier = items[0]
                latest = items[-1]
                conflicts.append(
                    f"Conflict on {grp}: {earlier['record_id']} ({earlier['date']}) marked "
                    f"'{earlier['status']}', but later {latest['record_id']} ({latest['date']}) "
                    f"reports '{latest['status']}'. Latest verified record ({latest['record_id']}) used."
                )
                # If two contradictory records have the exact same date, cannot resolve by recency
                if earlier.get("date") == latest.get("date"):
                    has_unresolved = True

        # Evaluate policy validity against session_date
        sess_dt = date.fromisoformat(session_date)
        active_policies: list[dict[str, Any]] = []
        expired_policies: list[dict[str, Any]] = []
        for pol in self.policies:
            if pol.get("verified") is not True:
                continue
            v_from = date.fromisoformat(pol["valid_from"])
            v_to = date.fromisoformat(pol["valid_to"])
            if v_from <= sess_dt <= v_to:
                active_policies.append(pol)
            else:
                expired_policies.append(pol)

        return RetrievalBundle(
            known_customer=True,
            customer_profile=customer,
            verified_records=verified_records,
            unverified_record_ids=unverified_ids,
            active_policies=active_policies,
            expired_policies=expired_policies,
            data_conflicts=conflicts,
            has_unresolved_conflict=has_unresolved,
        )
