/* Display geometry only. Training runs on Jetson in cuda_training.py. */
(function (root) {
  'use strict';
  const W = 1100, H = 700;
  class Track {
    constructor(kind = 'flow') {
      this.kind = kind; this.width = kind === 'technical' ? 64 : 78;
      this.points = []; this.length = 0;
      for (let i = 0; i < 420; i++) {
        const a = i / 420 * Math.PI * 2;
        const r = kind === 'technical' ? 1 + .17 * Math.sin(3 * a) + .065 * Math.cos(5 * a) : 1 + .055 * Math.sin(3 * a);
        this.points.push({ x: 550 + 414 * r * Math.cos(a), y: 350 + 231 * r * Math.sin(a) });
      }
      for (let i = 0; i < 420; i++) {
        const p = this.points[i], q = this.points[(i + 1) % 420];
        this.length += Math.hypot(q.x - p.x, q.y - p.y);
      }
      this.spacing = this.length / 420;
      // A small occupancy grid keeps seven range sensors inexpensive on Jetson.
      this.cell = 3; this.cols = Math.ceil(W / this.cell); this.rows = Math.ceil(H / this.cell);
      this.mask = new Uint8Array(this.cols * this.rows);
      const radius = this.width / 2 - 7;
      for (const p of this.points) {
        const x0 = Math.max(0, Math.floor((p.x - radius) / 3));
        const x1 = Math.min(this.cols - 1, Math.ceil((p.x + radius) / 3));
        const y0 = Math.max(0, Math.floor((p.y - radius) / 3));
        const y1 = Math.min(this.rows - 1, Math.ceil((p.y + radius) / 3));
        for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
          if ((x * 3 + 1.5 - p.x) ** 2 + (y * 3 + 1.5 - p.y) ** 2 < radius ** 2) this.mask[y * this.cols + x] = 1;
        }
      }
    }
    inside(x, y) {
      return x >= 0 && y >= 0 && x < W && y < H && !!this.mask[Math.floor(y / 3) * this.cols + Math.floor(x / 3)];
    }
    nearest(x, y, hint) {
      let best = 0, distance = Infinity;
      const start = hint === undefined ? 0 : hint - 15, end = hint === undefined ? 420 : hint + 16;
      for (let j = start; j < end; j++) {
        const i = (j + 420) % 420, p = this.points[i], d = (x - p.x) ** 2 + (y - p.y) ** 2;
        if (d < distance) { best = i; distance = d; }
      }
      return best;
    }
    ray(x, y, angle) {
      const dx = Math.cos(angle), dy = Math.sin(angle);
      for (let d = 0; d < 150; d += 4) if (!this.inside(x + dx * d, y + dy * d)) return d;
      return 150;
    }
  }
  root.EvoRacer = { Track, W, H, ANGLES: [-1.35, -.9, -.45, 0, .45, .9, 1.35] };
})(globalThis);
