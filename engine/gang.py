from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GangKind(str, Enum):
    EXPOSED = "exposed"
    CONCEALED = "concealed"
    ADDED = "added"


@dataclass(frozen=True)
class GangRecord:
    gang_id: int
    gang_player: int
    gang_type: GangKind
    payments: tuple[tuple[int, int], ...]
    transferred_to: tuple[int, ...] = ()
    transfer_count: int = 0
    refunded: bool = False

    @property
    def total_amount(self) -> int:
        return sum(amount for _, amount in self.payments)

