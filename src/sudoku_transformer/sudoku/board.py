"""Bitmask representation of a 9x9 sudoku grid."""

from __future__ import annotations

N = 9
BOX = 3
NUM_CELLS = N * N
ALL_DIGITS = (1 << N) - 1
BIT: tuple[int, ...] = (0,) + tuple(1 << (d - 1) for d in range(1, N + 1))
EMPTY_CHARS = "0."


def row_of(cell: int) -> int:
    return cell // N


def col_of(cell: int) -> int:
    return cell % N


def box_of(cell: int) -> int:
    return (cell // N // BOX) * BOX + (cell % N) // BOX


def _build_units() -> tuple[tuple[int, ...], ...]:
    rows = [tuple(r * N + c for c in range(N)) for r in range(N)]
    cols = [tuple(r * N + c for r in range(N)) for c in range(N)]
    boxes = [
        tuple(
            (br * BOX + dr) * N + (bc * BOX + dc)
            for dr in range(BOX)
            for dc in range(BOX)
        )
        for br in range(BOX)
        for bc in range(BOX)
    ]
    return tuple(rows + cols + boxes)


UNITS: tuple[tuple[int, ...], ...] = _build_units()

UNITS_OF: tuple[tuple[int, ...], ...] = tuple(
    tuple(u for u, cells in enumerate(UNITS) if cell in cells)
    for cell in range(NUM_CELLS)
)

PEERS: tuple[tuple[int, ...], ...] = tuple(
    tuple(sorted({p for u in UNITS_OF[cell] for p in UNITS[u]} - {cell}))
    for cell in range(NUM_CELLS)
)


class Board:
    """A sudoku position.

    ``grid[cell]`` holds the placed digit or 0 while empty. ``cands[cell]`` is a
    9-bit mask where bit ``d - 1`` marks digit ``d`` as still legal; filled cells
    carry an empty mask. Cells are indexed 0..80 in row-major order.
    """

    __slots__ = ("grid", "cands", "n_empty")

    def __init__(self, grid: list[int], cands: list[int], n_empty: int) -> None:
        self.grid = grid
        self.cands = cands
        self.n_empty = n_empty

    @classmethod
    def empty(cls) -> Board:
        return cls([0] * NUM_CELLS, [ALL_DIGITS] * NUM_CELLS, NUM_CELLS)

    @classmethod
    def from_string(cls, puzzle: str) -> Board:
        """Build from an 81-character string, '.' or '0' marking empty cells."""
        if len(puzzle) != NUM_CELLS:
            raise ValueError(f"expected {NUM_CELLS} characters, got {len(puzzle)}")
        board = cls.empty()
        for cell, char in enumerate(puzzle):
            if char in EMPTY_CHARS:
                continue
            if not char.isdigit():
                raise ValueError(f"unexpected character {char!r} at index {cell}")
            digit = int(char)
            if not board.place(cell, digit):
                raise ValueError(
                    f"clue {digit} at cell {cell} contradicts an earlier clue"
                )
        return board

    def copy(self) -> Board:
        return Board(self.grid.copy(), self.cands.copy(), self.n_empty)

    def restore(self, snapshot: Board) -> None:
        """Rewind to a snapshot in place, without rebinding the caller's reference."""
        self.grid[:] = snapshot.grid
        self.cands[:] = snapshot.cands
        self.n_empty = snapshot.n_empty

    @property
    def is_solved(self) -> bool:
        return self.n_empty == 0

    def has_candidate(self, cell: int, digit: int) -> bool:
        return bool(self.cands[cell] & BIT[digit])

    def candidates(self, cell: int) -> tuple[int, ...]:
        mask = self.cands[cell]
        return tuple(d for d in range(1, N + 1) if mask & BIT[d])

    def n_candidates(self, cell: int) -> int:
        return self.cands[cell].bit_count()

    def empty_cells(self) -> list[int]:
        return [cell for cell in range(NUM_CELLS) if self.grid[cell] == 0]

    def is_contradictory(self) -> bool:
        """True if the position cannot be completed.

        Checks both an empty cell with no candidates and a unit with no home for
        some digit. The second is what Norvig's eliminate catches, and it prunes
        branches several placements earlier than the first.
        """
        grid, cands = self.grid, self.cands
        for cell in range(NUM_CELLS):
            if grid[cell] == 0 and cands[cell] == 0:
                return True
        for unit in UNITS:
            placed = 0
            available = 0
            for cell in unit:
                if grid[cell]:
                    placed |= BIT[grid[cell]]
                else:
                    available |= cands[cell]
            if (placed | available) != ALL_DIGITS:
                return True
        return False

    def place(self, cell: int, digit: int) -> bool:
        """Place a digit and eliminate it from the cell's peers.

        Returns False if the placement was illegal or starved a peer of every
        candidate. The board is mutated either way, so a caller that may need to
        undo must snapshot with copy() first.
        """
        if self.grid[cell] != 0 or not (self.cands[cell] & BIT[digit]):
            return False

        self.grid[cell] = digit
        self.cands[cell] = 0
        self.n_empty -= 1

        bit = BIT[digit]
        grid, cands = self.grid, self.cands
        ok = True
        for peer in PEERS[cell]:
            if cands[peer] & bit:
                cands[peer] &= ~bit
                if cands[peer] == 0 and grid[peer] == 0:
                    ok = False
        return ok

    def to_string(self, empty: str = ".") -> str:
        return "".join(str(d) if d else empty for d in self.grid)

    def __str__(self) -> str:
        rows = []
        for r in range(N):
            cells = [str(self.grid[r * N + c] or ".") for c in range(N)]
            body = " ".join(cells[0:3]), " ".join(cells[3:6]), " ".join(cells[6:9])
            rows.append(" | ".join(body))
            if r in (2, 5):
                rows.append("-" * 21)
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"Board({self.to_string()!r})"
