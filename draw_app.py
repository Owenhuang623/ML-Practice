"""Draw a digit and see what the CNN in CNN.py predicts.

Run:  python draw_app.py
Requires CNN_MNIST.keras in the same directory (produced by running CNN.py).
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CNN_MNIST.keras")
CANVAS_SIZE = 280          # 10x the MNIST resolution
BRUSH_RADIUS = 12          # roughly MNIST stroke width after downscaling
PREVIEW_SIZE = 112         # the 28x28 model input, shown 4x


def preprocess(img: Image.Image):
    """Convert a white-on-black drawing into a (1, 28, 28, 1) float array.

    Mirrors how MNIST was built: crop to the digit, scale it to fit a 20x20
    box while keeping aspect ratio, then place it on a 28x28 canvas so its
    centre of mass sits in the middle.  Returns (array, preview_image) or
    (None, None) if the canvas is blank.
    """
    arr = np.asarray(img)
    ys, xs = np.nonzero(arr > 0)
    if xs.size == 0:
        return None, None

    crop = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = crop.size
    scale = 20 / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    crop = crop.resize(new_size, Image.LANCZOS)

    small = np.asarray(crop, dtype=np.float64)
    total = small.sum()
    grid_y, grid_x = np.indices(small.shape)
    cy = (grid_y * small).sum() / total
    cx = (grid_x * small).sum() / total
    offset = (int(round(14 - cx)), int(round(14 - cy)))

    out = Image.new("L", (28, 28), 0)
    out.paste(crop, offset)

    x = np.asarray(out, dtype=np.float32) / 255.0
    return x.reshape(1, 28, 28, 1), out


class DigitApp:
    def __init__(self, root: tk.Tk, model):
        self.root = root
        self.model = model
        root.title("MNIST CNN - draw a digit")
        root.resizable(False, False)

        main = ttk.Frame(root, padding=12)
        main.grid()

        # Drawing surface. The PIL image mirrors the canvas so we can read pixels.
        self.canvas = tk.Canvas(
            main, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="crosshair", highlightthickness=1,
        )
        self.canvas.grid(row=0, column=0, rowspan=2, padx=(0, 16))
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last = None
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Right-hand panel: result, preview, probability bars.
        side = ttk.Frame(main)
        side.grid(row=0, column=1, sticky="n")

        self.result_var = tk.StringVar(value="Draw a digit")
        ttk.Label(side, textvariable=self.result_var, font=("Helvetica", 28, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 8)
        )

        ttk.Label(side, text="Model input (28x28)").grid(row=1, column=0, columnspan=2)
        self.preview_label = ttk.Label(side)
        self.preview_label.grid(row=2, column=0, columnspan=2, pady=(2, 10))
        self._set_preview(Image.new("L", (28, 28), 0))

        self.bars = []
        self.bar_labels = []
        for d in range(10):
            ttk.Label(side, text=str(d), width=2).grid(row=3 + d, column=0, sticky="e")
            bar = ttk.Progressbar(side, length=160, maximum=100)
            bar.grid(row=3 + d, column=1, sticky="w", padx=(4, 0))
            lbl = ttk.Label(side, text="", width=6)
            lbl.grid(row=3 + d, column=2, sticky="w")
            self.bars.append(bar)
            self.bar_labels.append(lbl)

        buttons = ttk.Frame(main)
        buttons.grid(row=1, column=1, sticky="sew", pady=(12, 0))
        ttk.Button(buttons, text="Predict", command=self.predict).pack(side="left", expand=True, fill="x")
        ttk.Button(buttons, text="Clear", command=self.clear).pack(side="left", expand=True, fill="x", padx=(8, 0))

        self.live_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(buttons, text="Predict on release", variable=self.live_var).pack(side="left", padx=(8, 0))

    # --- drawing -----------------------------------------------------------
    def on_press(self, event):
        self.last = (event.x, event.y)
        self._dot(event.x, event.y)

    def on_move(self, event):
        if self.last is not None:
            x0, y0 = self.last
            self.canvas.create_line(
                x0, y0, event.x, event.y,
                fill="white", width=BRUSH_RADIUS * 2, capstyle=tk.ROUND, smooth=True,
            )
            self.draw.line([x0, y0, event.x, event.y], fill=255, width=BRUSH_RADIUS * 2)
        self._dot(event.x, event.y)
        self.last = (event.x, event.y)

    def on_release(self, _event):
        self.last = None
        if self.live_var.get():
            self.predict()

    def _dot(self, x, y):
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_var.set("Draw a digit")
        self._set_preview(Image.new("L", (28, 28), 0))
        for bar, lbl in zip(self.bars, self.bar_labels):
            bar["value"] = 0
            lbl["text"] = ""

    # --- inference ---------------------------------------------------------
    def predict(self):
        x, preview = preprocess(self.image)
        if x is None:
            self.result_var.set("Draw a digit")
            return
        probs = self.model.predict(x, verbose=0)[0]
        digit = int(np.argmax(probs))
        self.result_var.set(f"{digit}  ({probs[digit] * 100:.1f}%)")
        self._set_preview(preview)
        for d, (bar, lbl) in enumerate(zip(self.bars, self.bar_labels)):
            bar["value"] = float(probs[d] * 100)
            lbl["text"] = f"{probs[d] * 100:.1f}%"

    def _set_preview(self, img: Image.Image):
        big = img.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(big)  # keep a reference or Tk drops it
        self.preview_label["image"] = self._preview_photo


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}. Run CNN.py first to train and save it.")
    import tensorflow as tf  # imported late so the window appears quickly on error paths
    model = tf.keras.models.load_model(MODEL_PATH)

    root = tk.Tk()
    DigitApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
