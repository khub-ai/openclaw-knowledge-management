const PALETTE = {
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
};

export function renderArcGrid({ ctx, canvas, step, zoom, showDiff }) {
  const frame = step.artifacts?.primary?.data;
  if (!frame || !frame.length) {
    canvas.width = 32;
    canvas.height = 32;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const height = frame.length;
  const width = frame[0].length;
  canvas.width = width * zoom;
  canvas.height = height * zoom;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      ctx.fillStyle = PALETTE[frame[y][x]] || "#FFFFFF";
      ctx.fillRect(x * zoom, y * zoom, zoom, zoom);
    }
  }

  if (!showDiff) return;
  const diff = step.transition?.diff;
  if (!diff) return;

  for (const [x, y] of diff.cells || []) {
    ctx.lineWidth = Math.max(1, Math.floor(zoom / 3));
    ctx.strokeStyle = "rgba(255,255,255,0.92)";
    ctx.strokeRect(x * zoom + 0.5, y * zoom + 0.5, zoom - 1, zoom - 1);
    ctx.strokeStyle = "rgba(11,95,255,0.95)";
    ctx.strokeRect(x * zoom + 1.5, y * zoom + 1.5, Math.max(zoom - 3, 1), Math.max(zoom - 3, 1));
  }

  for (const component of diff.components || []) {
    const x = component.x_min * zoom;
    const y = component.y_min * zoom;
    const boxWidth = (component.x_max - component.x_min + 1) * zoom;
    const boxHeight = (component.y_max - component.y_min + 1) * zoom;
    ctx.lineWidth = Math.max(2, Math.floor(zoom / 2));
    ctx.strokeStyle = "rgba(255,133,27,0.95)";
    ctx.strokeRect(x + 0.5, y + 0.5, boxWidth - 1, boxHeight - 1);
  }
}

export function summarizeArcComponents(components) {
  if (!components || !components.length) return "0";
  return components.map((component, index) => {
    const width = component.x_max - component.x_min + 1;
    const height = component.y_max - component.y_min + 1;
    return `#${index + 1} ${width}x${height}`;
  }).join(", ");
}
