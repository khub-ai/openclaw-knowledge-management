import argparse
import json
import re
import textwrap
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Dict, List, Optional


PALETTE = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#AAAAAA",
    5: "#FFDC00",
    6: "#AA00FF",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#F012BE",
    10: "#7B7B7B",
    11: "#85144B",
    12: "#39CCCC",
}

DEFAULT_ROOT = Path(r"C:\_backup\github\khub-knowledge-fabric\tests\arc-agi-3\playlogs")


def load_steps(playlog_dir: Path) -> List[Dict]:
    step_files = sorted(playlog_dir.glob("[0-9][0-9][0-9]-*.json"))
    steps = []
    for path in step_files:
        payload = json.loads(path.read_text())
        frame = payload.get("returned", {}).get("frame")
        payload["_frame"] = decode_frame(frame)
        payload["_path"] = path
        steps.append(payload)
    if not steps:
        raise FileNotFoundError(f"No step JSON files found in {playlog_dir}")
    return steps


def latest_playlog(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No playlog folders found under {root}")
    return sorted(candidates)[-1]


def decode_frame(frame_value) -> Optional[List[List[int]]]:
    if not frame_value:
        return None

    if isinstance(frame_value, list):
        first = frame_value[0]
        if isinstance(first, list):
            return first
        if isinstance(first, str):
            return parse_frame_string(first)

    if isinstance(frame_value, str):
        return parse_frame_string(frame_value)

    return None


def parse_frame_string(frame_text: str) -> List[List[int]]:
    rows = []
    for raw_line in frame_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        numbers = [int(token) for token in re.findall(r"-?\d+", line)]
        if numbers:
            rows.append(numbers)
    return rows


class PlaylogViewer:
    def __init__(self, root: tk.Tk, playlog_dir: Path, cell_size: int, autoplay_ms: int):
        self.root = root
        self.playlog_dir = playlog_dir
        self.steps = load_steps(playlog_dir)
        self.index = 0
        self.cell_size = cell_size
        self.autoplay_ms = autoplay_ms
        self.autoplay = False
        self.after_id = None

        first_frame = self.steps[0]["_frame"]
        if not first_frame:
            raise ValueError("First step does not contain frame data")
        self.grid_height = len(first_frame)
        self.grid_width = len(first_frame[0])
        self.cell_size = self.fit_cell_size(cell_size)

        self.root.title(f"ARC Playlog Viewer - {playlog_dir.name}")
        self.build_ui()
        self.bind_keys()
        self.show_step(0)

    def fit_cell_size(self, requested: int) -> int:
        screen_w = max(self.root.winfo_screenwidth(), 800)
        screen_h = max(self.root.winfo_screenheight(), 600)

        max_canvas_w = max(320, min(int(screen_w * 0.55), screen_w - 520))
        max_canvas_h = max(320, int(screen_h * 0.72))

        cell_w = max_canvas_w // self.grid_width
        cell_h = max_canvas_h // self.grid_height
        fitted = max(2, min(requested, cell_w, cell_h))
        return fitted

    def build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=12)
        left.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self.canvas_width = self.grid_width * self.cell_size
        self.canvas_height = self.grid_height * self.cell_size

        canvas_frame = ttk.Frame(left)
        canvas_frame.grid(row=0, column=0, columnspan=4, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            background="black",
            highlightthickness=1,
            highlightbackground="#666666",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.canvas.configure(scrollregion=(0, 0, self.canvas_width, self.canvas_height))

        self.status_var = tk.StringVar()
        ttk.Label(left, textvariable=self.status_var, padding=(0, 8, 0, 8)).grid(
            row=1, column=0, columnspan=4, sticky="w"
        )

        ttk.Button(left, text="Prev", command=self.prev_step).grid(row=2, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(left, text="Next", command=self.next_step).grid(row=2, column=1, sticky="ew", padx=(0, 6))
        self.play_button = ttk.Button(left, text="Play", command=self.toggle_autoplay)
        self.play_button.grid(row=2, column=2, sticky="ew", padx=(0, 6))
        ttk.Button(left, text="Open...", command=self.choose_folder).grid(row=2, column=3, sticky="ew")

        ttk.Button(left, text="Zoom +", command=lambda: self.adjust_zoom(1)).grid(row=3, column=0, sticky="ew", pady=(6, 0), padx=(0, 6))
        ttk.Button(left, text="Zoom -", command=lambda: self.adjust_zoom(-1)).grid(row=3, column=1, sticky="ew", pady=(6, 0), padx=(0, 6))
        ttk.Button(left, text="Fit", command=self.fit_zoom).grid(row=3, column=2, sticky="ew", pady=(6, 0), padx=(0, 6))
        ttk.Label(left, text="Space=next, +=zoom, -=zoom", padding=(0, 8, 0, 0)).grid(row=3, column=3, sticky="w")

        meta = ttk.LabelFrame(right, text="Step Details", padding=10)
        meta.grid(row=0, column=0, sticky="ew")
        meta.columnconfigure(1, weight=1)

        self.meta_vars = {
            "step": tk.StringVar(),
            "action": tk.StringVar(),
            "state": tk.StringVar(),
            "levels": tk.StringVar(),
            "bbox": tk.StringVar(),
            "diff_count": tk.StringVar(),
        }
        row = 0
        for label, key in (
            ("Step", "step"),
            ("Action", "action"),
            ("State", "state"),
            ("Levels", "levels"),
            ("Change Box", "bbox"),
            ("Diff Count", "diff_count"),
        ):
            ttk.Label(meta, text=f"{label}:").grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=2)
            ttk.Label(meta, textvariable=self.meta_vars[key], wraplength=420, justify="left").grid(
                row=row, column=1, sticky="nw", pady=2
            )
            row += 1

        notes = ttk.LabelFrame(right, text="Player Notes", padding=10)
        notes.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        notes.rowconfigure(0, weight=1)
        notes.columnconfigure(0, weight=1)

        self.notes_text = tk.Text(notes, wrap="word", width=60, height=20)
        self.notes_text.grid(row=0, column=0, sticky="nsew")
        self.notes_text.configure(state="disabled")

        help_text = (
            "Controls: Space / Right Arrow = next step, Left Arrow = previous step, "
            "P = play or pause autoplay, Home = first step, End = last step."
        )
        ttk.Label(right, text=help_text, wraplength=460, justify="left", padding=(0, 10, 0, 0)).grid(
            row=2, column=0, sticky="w"
        )

    def bind_keys(self) -> None:
        self.root.bind("<space>", lambda event: self.next_step())
        self.root.bind("<Right>", lambda event: self.next_step())
        self.root.bind("<Left>", lambda event: self.prev_step())
        self.root.bind("<Home>", lambda event: self.show_step(0))
        self.root.bind("<End>", lambda event: self.show_step(len(self.steps) - 1))
        self.root.bind("p", lambda event: self.toggle_autoplay())
        self.root.bind("P", lambda event: self.toggle_autoplay())
        self.root.bind("+", lambda event: self.adjust_zoom(1))
        self.root.bind("-", lambda event: self.adjust_zoom(-1))
        self.root.bind("=", lambda event: self.adjust_zoom(1))
        self.root.bind("f", lambda event: self.fit_zoom())
        self.root.bind("F", lambda event: self.fit_zoom())

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose playlog directory",
            initialdir=str(self.playlog_dir.parent),
            mustexist=True,
        )
        if not selected:
            return
        self.stop_autoplay()
        self.playlog_dir = Path(selected)
        self.steps = load_steps(self.playlog_dir)
        self.index = 0
        self.root.title(f"ARC Playlog Viewer - {self.playlog_dir.name}")
        self.show_step(0)

    def show_step(self, index: int) -> None:
        self.index = max(0, min(index, len(self.steps) - 1))
        step = self.steps[self.index]
        self.render_frame(step["_frame"])
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)
        self.update_side_panel(step)
        self.status_var.set(
            f"{self.playlog_dir.name}  |  frame {self.index + 1}/{len(self.steps)}  |  {step.get('action_name', 'UNKNOWN')}"
        )

    def render_frame(self, frame: Optional[List[List[int]]]) -> None:
        self.canvas.delete("all")
        if not frame:
            return
        self.canvas_width = self.grid_width * self.cell_size
        self.canvas_height = self.grid_height * self.cell_size
        self.canvas.configure(width=self.canvas_width, height=self.canvas_height)
        self.canvas.configure(scrollregion=(0, 0, self.canvas_width, self.canvas_height))
        for y, row in enumerate(frame):
            y0 = y * self.cell_size
            y1 = y0 + self.cell_size
            for x, value in enumerate(row):
                x0 = x * self.cell_size
                x1 = x0 + self.cell_size
                color = PALETTE.get(value, "#FFFFFF")
                self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, fill=color)

    def adjust_zoom(self, delta: int) -> None:
        self.cell_size = max(1, min(16, self.cell_size + delta))
        self.show_step(self.index)

    def fit_zoom(self) -> None:
        self.cell_size = self.fit_cell_size(1)
        self.show_step(self.index)

    def update_side_panel(self, step: dict) -> None:
        change_summary = step.get("change_summary", {})
        bbox = change_summary.get("bbox")
        bbox_text = "none" if not bbox else f"x={bbox['x_min']}..{bbox['x_max']}, y={bbox['y_min']}..{bbox['y_max']}"

        self.meta_vars["step"].set(str(step.get("step_number", "?")))
        self.meta_vars["action"].set(step.get("action_name", "UNKNOWN"))
        self.meta_vars["state"].set(str(step.get("observation_state", "")))
        self.meta_vars["levels"].set(f"{step.get('levels_completed', '?')}/{step.get('win_levels', '?')}")
        self.meta_vars["bbox"].set(bbox_text)
        self.meta_vars["diff_count"].set(str(change_summary.get("diff_count", 0)))

        lines = [
            f"Decision note:\n{textwrap.fill(step.get('decision_note', ''), width=62)}",
            "",
            f"File:\n{step.get('_path', '')}",
            "",
            f"Change types:\n{textwrap.fill(str(change_summary.get('change_types', {})), width=62)}",
        ]
        samples = change_summary.get("samples") or []
        if samples:
            lines.extend(
                [
                    "",
                    "Sample cell diffs:",
                    textwrap.fill(", ".join(str(sample) for sample in samples[:10]), width=62),
                ]
            )

        self.notes_text.configure(state="normal")
        self.notes_text.delete("1.0", "end")
        self.notes_text.insert("1.0", "\n".join(lines))
        self.notes_text.configure(state="disabled")

    def next_step(self) -> None:
        if self.index < len(self.steps) - 1:
            self.show_step(self.index + 1)
        else:
            self.stop_autoplay()

    def prev_step(self) -> None:
        self.show_step(self.index - 1)

    def toggle_autoplay(self) -> None:
        self.autoplay = not self.autoplay
        self.play_button.configure(text="Pause" if self.autoplay else "Play")
        if self.autoplay:
            self.schedule_next()
        else:
            self.stop_autoplay()

    def schedule_next(self) -> None:
        self.after_id = self.root.after(self.autoplay_ms, self.autoplay_tick)

    def autoplay_tick(self) -> None:
        if not self.autoplay:
            return
        if self.index >= len(self.steps) - 1:
            self.stop_autoplay()
            return
        self.next_step()
        self.schedule_next()

    def stop_autoplay(self) -> None:
        self.autoplay = False
        self.play_button.configure(text="Play")
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ARC playlogs with side commentary.")
    parser.add_argument(
        "--playlog-dir",
        type=Path,
        default=None,
        help="Specific playlog directory to open. Defaults to the latest under tests/arc-agi-3/playlogs.",
    )
    parser.add_argument("--cell-size", type=int, default=1, help="Rendered pixel size for each frame cell.")
    parser.add_argument("--autoplay-ms", type=int, default=600, help="Autoplay delay in milliseconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    playlog_dir = args.playlog_dir or latest_playlog(DEFAULT_ROOT)

    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    viewer = PlaylogViewer(root, playlog_dir, cell_size=args.cell_size, autoplay_ms=args.autoplay_ms)
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    width = min(1080, max(900, screen_w - 80))
    height = min(760, max(620, screen_h - 80))
    root.geometry(f"{width}x{height}+0+0")
    root.minsize(820, 560)
    root.mainloop()


if __name__ == "__main__":
    main()
