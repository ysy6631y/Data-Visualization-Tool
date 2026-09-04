import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import Papa from "papaparse";
import * as d3 from "d3";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ChevronDown, ChevronRight, Download, FileJson, Maximize2, RotateCcw, Upload } from "lucide-react";
import "./styles.css";

type ColumnType = "number" | "string" | "date" | "boolean";
type Status = "available" | "unavailable" | "requires mapping";
type VizId =
  | "line"
  | "scatter2d"
  | "bar"
  | "heatmap"
  | "contour"
  | "network2d"
  | "xy"
  | "xz"
  | "yz"
  | "scatter3d"
  | "line3d"
  | "network3d"
  | "wireframe"
  | "surface"
  | "pointcloud";
type Projection = "3D" | "XY" | "XZ" | "YZ";
type Normalization = "none" | "minmax" | "zscore";
type CameraPreset = "iso" | "front" | "top" | "side";

interface ColumnSummary {
  name: string;
  type: ColumnType;
  missing: number;
  unique: number;
  min?: number | string;
  max?: number | string;
  mean?: number;
}

interface Dataset {
  filename: string;
  rows: Record<string, unknown>[];
  columns: ColumnSummary[];
  matrix?: number[][];
}

interface Mapping {
  x: string;
  y: string;
  z: string;
  value: string;
  group: string;
  time: string;
  source: string;
  target: string;
}

interface Capability {
  id: VizId;
  label: string;
  group: "2D" | "3D";
  status: Status;
  reason: string;
  method?: string;
}

interface TransformState {
  normalization: Normalization;
  xScale: number;
  yScale: number;
  zScale: number;
  noise: number;
  smoothing: number;
  thresholdMin: string;
  thresholdMax: string;
  pointSize: number;
  lineWidth: number;
  opacity: number;
}

interface DisplayState {
  showGrid: boolean;
  showBounds: boolean;
  showAxisLabels: boolean;
  cameraPreset: CameraPreset;
}

const emptyMapping: Mapping = { x: "", y: "", z: "", value: "", group: "", time: "", source: "", target: "" };
const defaultTransform: TransformState = {
  normalization: "none",
  xScale: 1,
  yScale: 1,
  zScale: 1,
  noise: 0,
  smoothing: 0,
  thresholdMin: "",
  thresholdMax: "",
  pointSize: 1.4,
  lineWidth: 1,
  opacity: 0.92,
};

const defaultDisplay: DisplayState = {
  showGrid: true,
  showBounds: true,
  showAxisLabels: true,
  cameraPreset: "iso",
};

const vizLabels: Record<VizId, string> = {
  line: "Line chart",
  scatter2d: "Scatter plot",
  bar: "Bar chart",
  heatmap: "Heatmap",
  contour: "Contour",
  network2d: "2D Network",
  xy: "XY projection",
  xz: "XZ projection",
  yz: "YZ projection",
  scatter3d: "3D Scatter",
  line3d: "3D Line",
  network3d: "3D Network",
  wireframe: "Wireframe",
  surface: "Surface",
  pointcloud: "Point Cloud",
};

const demoRows = {
  a: Array.from({ length: 70 }, (_, i) => ({
    x: Number((i / 6).toFixed(3)),
    y: Number((Math.sin(i / 6) + i / 60).toFixed(3)),
    series: i % 2 ? "B" : "A",
  })),
  b: Array.from({ length: 180 }, (_, i) => {
    const t = i / 14;
    return {
      x: Number((Math.cos(t) * (1 + i / 180)).toFixed(3)),
      y: Number((Math.sin(t) * (1 + i / 180)).toFixed(3)),
      z: Number((i / 18).toFixed(3)),
      time: i,
    };
  }),
  c: Array.from({ length: 15 * 15 }, (_, i) => {
    const gx = (i % 15) - 7;
    const gy = Math.floor(i / 15) - 7;
    return { x: gx, y: gy, z: Number((Math.sin(gx / 2) * Math.cos(gy / 2)).toFixed(3)) };
  }),
  d: [
    ["A", "B", 4],
    ["A", "C", 2],
    ["B", "D", 5],
    ["C", "D", 1],
    ["C", "E", 3],
    ["E", "F", 2],
    ["D", "F", 4],
  ].map(([source, target, weight]) => ({ source, target, weight })),
};

function isMissing(value: unknown) {
  return value === null || value === undefined || value === "";
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function inferColumns(rows: Record<string, unknown>[]): ColumnSummary[] {
  const names = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  return names.map((name) => {
    const values = rows.map((row) => row[name]);
    const present = values.filter((v) => !isMissing(v));
    const unique = new Set(present.map((v) => String(v))).size;
    const numericValues = present.map(toNumber).filter((v): v is number => v !== null);
    const boolValues = present.filter((v) => ["true", "false", true, false].includes(v as string | boolean));
    const dateValues = present.map((v) => (typeof v === "string" || typeof v === "number" ? Date.parse(String(v)) : NaN)).filter(Number.isFinite);
    let type: ColumnType = "string";
    if (present.length && numericValues.length / present.length >= 0.9) type = "number";
    else if (present.length && boolValues.length / present.length >= 0.9) type = "boolean";
    else if (present.length && dateValues.length / present.length >= 0.9) type = "date";
    const summary: ColumnSummary = { name, type, missing: values.length - present.length, unique };
    if (type === "number" && numericValues.length) {
      summary.min = d3.min(numericValues);
      summary.max = d3.max(numericValues);
      summary.mean = d3.mean(numericValues);
    }
    if (type === "date" && dateValues.length) {
      summary.min = new Date(d3.min(dateValues) ?? 0).toISOString().slice(0, 10);
      summary.max = new Date(d3.max(dateValues) ?? 0).toISOString().slice(0, 10);
    }
    return summary;
  });
}

function analyzeDataset(filename: string, rows: Record<string, unknown>[]): Dataset {
  return { filename, rows: rows.filter((row) => Object.keys(row).length), columns: inferColumns(rows) };
}

function autoMap(dataset: Dataset): Mapping {
  const numbers = dataset.columns.filter((c) => c.type === "number").map((c) => c.name);
  const strings = dataset.columns.filter((c) => c.type === "string").map((c) => c.name);
  const dates = dataset.columns.filter((c) => c.type === "date").map((c) => c.name);
  const byName = (names: string[], candidates: string[]) => candidates.find((c) => names.includes(c.toLowerCase())) ?? "";
  const source = byName(["source", "src", "from"], strings) || byName(["source", "src", "from"], dataset.columns.map((c) => c.name));
  const target = byName(["target", "dst", "to"], strings) || byName(["target", "dst", "to"], dataset.columns.map((c) => c.name));
  return {
    x: byName(["x", "lon", "longitude", "time", "year"], numbers) || dates[0] || numbers[0] || "",
    y: byName(["y", "lat", "latitude", "value"], numbers) || numbers.find((n) => n !== numbers[0]) || "",
    z: byName(["z", "elevation", "height", "depth"], numbers) || numbers.find((n) => ![numbers[0], numbers[1]].includes(n)) || "",
    value: byName(["value", "z", "temperature", "weight"], numbers) || numbers[2] || numbers[1] || "",
    group: strings.find((s) => s !== source && s !== target) || "",
    time: dates[0] || byName(["time", "date", "year"], numbers) || "",
    source,
    target,
  };
}

function column(dataset: Dataset | null, name: string) {
  return dataset?.columns.find((c) => c.name === name);
}

function isNumeric(dataset: Dataset | null, name: string) {
  return column(dataset, name)?.type === "number";
}

function isNumericOrTime(dataset: Dataset | null, name: string) {
  const t = column(dataset, name)?.type;
  return t === "number" || t === "date";
}

function isCategorical(dataset: Dataset | null, name: string) {
  const t = column(dataset, name)?.type;
  return t === "string" || t === "boolean";
}

function hasNetwork(mapping: Mapping) {
  return Boolean(mapping.source && mapping.target);
}

function gridInfo(dataset: Dataset | null, mapping: Mapping) {
  if (!dataset || !isNumeric(dataset, mapping.x) || !isNumeric(dataset, mapping.y) || !isNumeric(dataset, mapping.z || mapping.value)) {
    return { regular: false, irregular: false, points: 0, reason: "X/Y/Z numeric mapping is required." };
  }
  const z = mapping.z || mapping.value;
  const pairs = new Set<string>();
  const xs = new Set<string>();
  const ys = new Set<string>();
  for (const row of dataset.rows) {
    const x = toNumber(row[mapping.x]);
    const y = toNumber(row[mapping.y]);
    const zv = toNumber(row[z]);
    if (x === null || y === null || zv === null) continue;
    xs.add(String(x));
    ys.add(String(y));
    pairs.add(`${x}|${y}`);
  }
  const expected = xs.size * ys.size;
  const regular = xs.size > 1 && ys.size > 1 && pairs.size === expected && pairs.size >= 9;
  const irregular = !regular && pairs.size >= 12 && xs.size > 2 && ys.size > 2;
  return { regular, irregular, points: pairs.size, reason: regular ? "" : "X/Y grid structure was not detected." };
}

function detectCapabilities(dataset: Dataset | null, mapping: Mapping): Capability[] {
  if (!dataset) return [];
  const numericCols = dataset.columns.filter((c) => c.type === "number").map((c) => c.name);
  const categoricalCols = dataset.columns.filter((c) => c.type === "string" || c.type === "boolean").map((c) => c.name);
  const grid = gridInfo(dataset, mapping);
  const req = (label: string) => `${label} mapping is required. Current numeric columns: ${numericCols.join(", ") || "none"}.`;
  const caps: Capability[] = [
    {
      id: "line",
      label: vizLabels.line,
      group: "2D",
      status: isNumericOrTime(dataset, mapping.x) && isNumeric(dataset, mapping.y) ? "available" : numericCols.length >= 1 ? "requires mapping" : "unavailable",
      reason: "Line chart requires a numeric or time X column and a numeric Y column.",
    },
    {
      id: "scatter2d",
      label: vizLabels.scatter2d,
      group: "2D",
      status: isNumeric(dataset, mapping.x) && isNumeric(dataset, mapping.y) ? "available" : numericCols.length >= 2 ? "requires mapping" : "unavailable",
      reason: "Scatter plot requires numeric X and Y columns.",
    },
    {
      id: "bar",
      label: vizLabels.bar,
      group: "2D",
      status: isCategorical(dataset, mapping.x) && isNumeric(dataset, mapping.y) ? "available" : categoricalCols.length && numericCols.length ? "requires mapping" : "unavailable",
      reason: "Bar chart requires one categorical column and one numeric column.",
    },
    {
      id: "heatmap",
      label: vizLabels.heatmap,
      group: "2D",
      status: (mapping.x && mapping.y && isNumeric(dataset, mapping.value || mapping.z)) || grid.regular ? "available" : numericCols.length >= 3 ? "requires mapping" : "unavailable",
      reason: "Heatmap requires X/Y dimensions plus a numeric Value column, or matrix-like data.",
    },
    {
      id: "contour",
      label: vizLabels.contour,
      group: "2D",
      status: isNumeric(dataset, mapping.x) && isNumeric(dataset, mapping.y) && isNumeric(dataset, mapping.value || mapping.z) ? "available" : numericCols.length >= 3 ? "requires mapping" : "unavailable",
      reason: "Contour requires numeric X/Y and numeric Z or Value. Irregular data uses interpolation and reports it.",
      method: grid.regular ? "Regular grid contours" : "Interpolated from irregular samples",
    },
    {
      id: "network2d",
      label: vizLabels.network2d,
      group: "2D",
      status: hasNetwork(mapping) ? "available" : dataset.columns.length >= 2 ? "requires mapping" : "unavailable",
      reason: "Network requires source and target relationship columns. Relationships are never inferred from unrelated fields.",
    },
    ...(["xy", "xz", "yz"] as VizId[]).map((id) => ({
      id,
      label: vizLabels[id],
      group: "2D" as const,
      status: isNumeric(dataset, mapping.x) && isNumeric(dataset, mapping.y) && isNumeric(dataset, mapping.z) ? "available" as const : numericCols.length >= 3 ? "requires mapping" as const : "unavailable" as const,
      reason: req("X/Y/Z numeric"),
    })),
    {
      id: "scatter3d",
      label: vizLabels.scatter3d,
      group: "3D",
      status: isNumeric(dataset, mapping.x) && isNumeric(dataset, mapping.y) && isNumeric(dataset, mapping.z) ? "available" : numericCols.length >= 3 ? "requires mapping" : "unavailable",
      reason: req("3D spatial X/Y/Z"),
    },
    {
      id: "line3d",
      label: vizLabels.line3d,
      group: "3D",
      status: isNumeric(dataset, mapping.x) && isNumeric(dataset, mapping.y) && isNumeric(dataset, mapping.z) ? "available" : numericCols.length >= 3 ? "requires mapping" : "unavailable",
      reason: "3D Line requires numeric X/Y/Z and uses file row order or Time mapping for sequence order.",
    },
    {
      id: "network3d",
      label: vizLabels.network3d,
      group: "3D",
      status: hasNetwork(mapping) && isNumeric(dataset, mapping.x) && isNumeric(dataset, mapping.y) && isNumeric(dataset, mapping.z) ? "available" : hasNetwork(mapping) ? "requires mapping" : "unavailable",
      reason: "3D Network requires source/target columns and explicit numeric X/Y/Z coordinates. Node positions are not invented.",
    },
    {
      id: "wireframe",
      label: vizLabels.wireframe,
      group: "3D",
      status: grid.regular ? "available" : grid.irregular ? "requires mapping" : "unavailable",
      reason: grid.irregular ? "Irregular samples require Delaunay triangulation before showing wireframe." : "Wireframe requires a regular grid or explicit connectivity.",
      method: grid.regular ? "Regular X/Y grid" : "Delaunay triangulation candidate",
    },
    {
      id: "surface",
      label: vizLabels.surface,
      group: "3D",
      status: grid.regular ? "available" : grid.irregular ? "requires mapping" : "unavailable",
      reason: grid.irregular ? "Surface can be generated only after acknowledging Delaunay triangulation over irregular points." : "Surface requires an X/Y grid with a Z value for each X-Y location.",
      method: grid.regular ? "Regular X/Y grid" : "Delaunay triangulation candidate",
    },
    {
      id: "pointcloud",
      label: vizLabels.pointcloud,
      group: "3D",
      status: isNumeric(dataset, mapping.x) && isNumeric(dataset, mapping.y) && isNumeric(dataset, mapping.z) ? "available" : numericCols.length >= 3 ? "requires mapping" : "unavailable",
      reason: req("Point cloud X/Y/Z"),
    },
  ];
  return caps;
}

function numericExtent(rows: Record<string, unknown>[], key: string): [number, number] {
  const values = rows.map((row) => toNumber(row[key])).filter((v): v is number => v !== null);
  const min = d3.min(values) ?? 0;
  const max = d3.max(values) ?? 1;
  return min === max ? [min - 1, max + 1] : [min, max];
}

function transformedRows(dataset: Dataset | null, mapping: Mapping, transform: TransformState) {
  if (!dataset) return [];
  const keys = [mapping.x, mapping.y, mapping.z, mapping.value].filter(Boolean);
  const stats = Object.fromEntries(
    keys.map((key) => {
      const values = dataset.rows.map((row) => toNumber(row[key])).filter((v): v is number => v !== null);
      return [key, { min: d3.min(values) ?? 0, max: d3.max(values) ?? 1, mean: d3.mean(values) ?? 0, dev: d3.deviation(values) || 1 }];
    }),
  );
  const tMin = transform.thresholdMin === "" ? -Infinity : Number(transform.thresholdMin);
  const tMax = transform.thresholdMax === "" ? Infinity : Number(transform.thresholdMax);
  const mapped = dataset.rows
    .map((row, index) => {
      const out: Record<string, unknown> = { ...row, __index: index };
      for (const key of keys) {
        const n = toNumber(row[key]);
        if (n === null) continue;
        const stat = stats[key];
        let v = n;
        if (transform.normalization === "minmax") v = (n - stat.min) / (stat.max - stat.min || 1);
        if (transform.normalization === "zscore") v = (n - stat.mean) / stat.dev;
        if (key === mapping.x) v *= transform.xScale;
        if (key === mapping.y) v *= transform.yScale;
        if (key === mapping.z) v *= transform.zScale;
        if (transform.noise) v += (Math.sin(index * 12.9898) * 43758.5453 % 1) * transform.noise;
        out[`__${key}`] = v;
      }
      return out;
    });
  if (transform.smoothing > 0) {
    const radius = Math.max(1, Math.round(transform.smoothing * 8));
    for (const key of keys) {
      const smoothValues = mapped.map((row, index) => {
        const values: number[] = [];
        for (let i = Math.max(0, index - radius); i <= Math.min(mapped.length - 1, index + radius); i++) {
          const n = toNumber(mapped[i][`__${key}`]);
          if (n !== null) values.push(n);
        }
        const current = toNumber(row[`__${key}`]);
        return current === null ? null : d3.mean(values) ?? current;
      });
      smoothValues.forEach((value, index) => {
        if (value !== null) mapped[index][`__${key}`] = value;
      });
    }
  }
  return mapped.filter((row) => {
      const key = mapping.value || mapping.z || mapping.y;
      const v = toNumber(row[`__${key}`] ?? row[key]);
      return v === null || (v >= tMin && v <= tMax);
    });
}

function valueOf(row: Record<string, unknown>, key: string) {
  return toNumber(row[`__${key}`] ?? row[key]);
}

function chooseDefaultViz(dataset: Dataset, mapping: Mapping) {
  const caps = detectCapabilities(dataset, mapping);
  const grid = gridInfo(dataset, mapping);
  if (grid.regular && caps.find((c) => c.id === "wireframe")?.status === "available") return "wireframe" as VizId;
  if (caps.find((c) => c.id === "scatter3d")?.status === "available") return "scatter3d" as VizId;
  if (caps.find((c) => c.id === "network2d")?.status === "available") return "network2d" as VizId;
  return caps.find((c) => c.status === "available")?.id ?? "scatter2d";
}

function pointTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, 64, 64);
  ctx.beginPath();
  ctx.arc(32, 32, 19, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function labelSprite(text: string) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "24px Inter, Arial, sans-serif";
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.82, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(14, 3.5, 1);
  return sprite;
}

function addAxisLabels(scene: THREE.Scene, mapping: Mapping, bounds: THREE.Box3, extents: Record<"x" | "y" | "z", [number, number]>, projection: Projection) {
  const center = bounds.getCenter(new THREE.Vector3());
  const axisNames = projection === "XY" ? [mapping.x, mapping.y, ""] : projection === "XZ" ? [mapping.x, mapping.z, ""] : projection === "YZ" ? [mapping.y, mapping.z, ""] : [mapping.x, mapping.z, mapping.y];
  const minX = labelSprite(`${axisNames[0]} ${extents.x[0].toFixed(2)}`);
  minX.position.set(bounds.min.x, bounds.min.y - 5, center.z);
  const maxX = labelSprite(`${extents.x[1].toFixed(2)}`);
  maxX.position.set(bounds.max.x, bounds.min.y - 5, center.z);
  const minY = labelSprite(`${axisNames[1]} ${extents.z[0].toFixed(2)}`);
  minY.position.set(bounds.min.x - 6, bounds.min.y, bounds.min.z);
  const maxY = labelSprite(`${extents.z[1].toFixed(2)}`);
  maxY.position.set(bounds.min.x - 6, bounds.max.y, bounds.min.z);
  scene.add(minX, maxX, minY, maxY);
  if (projection === "3D" && axisNames[2]) {
    const minZ = labelSprite(`${axisNames[2]} ${extents.y[0].toFixed(2)}`);
    minZ.position.set(center.x, bounds.min.y - 5, bounds.min.z);
    const maxZ = labelSprite(`${extents.y[1].toFixed(2)}`);
    maxZ.position.set(center.x, bounds.min.y - 5, bounds.max.z);
    scene.add(minZ, maxZ);
  }
}

function Section({ title, children, initial = true }: { title: string; children: React.ReactNode; initial?: boolean }) {
  const [open, setOpen] = useState(initial);
  return (
    <section className="section">
      <button className="sectionHeader" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span>{title}</span>
      </button>
      {open && <div className="sectionBody">{children}</div>}
    </section>
  );
}

function SelectField({ label, value, onChange, columns, allowAny = false }: { label: string; value: string; onChange: (v: string) => void; columns: ColumnSummary[]; allowAny?: boolean }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">None</option>
        {columns.filter((c) => allowAny || c.type === "number" || c.type === "date").map((c) => (
          <option value={c.name} key={c.name}>{c.name} · {c.type}</option>
        ))}
      </select>
    </label>
  );
}

function TwoDView({ rows, mapping, viz, onSelect, transform }: { rows: Record<string, unknown>[]; mapping: Mapping; viz: VizId; onSelect: (row: Record<string, unknown>) => void; transform: TransformState }) {
  const ref = useRef<SVGSVGElement | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const width = ref.current?.clientWidth || 800;
    const height = ref.current?.clientHeight || 560;
    const pad = 42;
    const root = svg.append("g").attr("class", "plotRoot");
    svg.call(d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.5, 18]).on("zoom", (event) => root.attr("transform", event.transform.toString())));
    if (viz === "network2d") {
      const links = rows.filter((r) => r[mapping.source] && r[mapping.target]).map((r) => ({ source: String(r[mapping.source]), target: String(r[mapping.target]), row: r }));
      const nodes = Array.from(new Set(links.flatMap((l) => [l.source, l.target]))).map((id) => ({ id }));
      const simulation = d3.forceSimulation(nodes as d3.SimulationNodeDatum[]).force("link", d3.forceLink(links).id((d: any) => d.id).distance(90)).force("charge", d3.forceManyBody().strength(-220)).force("center", d3.forceCenter(width / 2, height / 2)).stop();
      for (let i = 0; i < 180; i++) simulation.tick();
      root.selectAll("line.link").data(links).join("line").attr("x1", (d: any) => d.source.x).attr("y1", (d: any) => d.source.y).attr("x2", (d: any) => d.target.x).attr("y2", (d: any) => d.target.y).attr("stroke", "#ffffff").attr("stroke-width", transform.lineWidth).attr("opacity", 0.62);
      root.selectAll("circle.node").data(nodes as any).join("circle").attr("cx", (d: any) => d.x).attr("cy", (d: any) => d.y).attr("r", transform.pointSize + 2).attr("fill", "#ffffff").attr("opacity", transform.opacity);
      root.selectAll("text.node").data(nodes as any).join("text").attr("x", (d: any) => d.x + 8).attr("y", (d: any) => d.y + 3).attr("fill", "#d9dde3").attr("font-size", 10).text((d: any) => d.id);
      return;
    }
    if (viz === "bar") {
      const agg = d3.rollups(rows, (v) => d3.mean(v, (d) => valueOf(d, mapping.y) ?? 0) ?? 0, (d) => String(d[mapping.x])).slice(0, 40);
      const xb = d3.scaleBand().domain(agg.map((d) => d[0])).range([pad, width - pad]).padding(0.25);
      const yb = d3.scaleLinear().domain([0, d3.max(agg, (d) => d[1]) || 1]).nice().range([height - pad, pad]);
      root.append("g").attr("transform", `translate(0,${height - pad})`).call(d3.axisBottom(xb).tickValues(agg.slice(0, 12).map((d) => d[0]))).attr("color", "#9da3ad");
      root.append("g").attr("transform", `translate(${pad},0)`).call(d3.axisLeft(yb).ticks(6)).attr("color", "#9da3ad");
      root.selectAll("rect").data(agg).join("rect").attr("x", (d) => xb(d[0]) ?? 0).attr("y", (d) => yb(d[1])).attr("width", xb.bandwidth()).attr("height", (d) => height - pad - yb(d[1])).attr("fill", "#ffffff").attr("opacity", transform.opacity * 0.8);
      return;
    }
    const projection = viz === "xz" ? [mapping.x, mapping.z] : viz === "yz" ? [mapping.y, mapping.z] : [mapping.x, mapping.y];
    const [xKey, yKey] = projection;
    const valid = rows.filter((row) => valueOf(row, xKey) !== null && valueOf(row, yKey) !== null);
    const x = d3.scaleLinear().domain(d3.extent(valid, (d) => valueOf(d, xKey) ?? 0) as [number, number]).nice().range([pad, width - pad]);
    const y = d3.scaleLinear().domain(d3.extent(valid, (d) => valueOf(d, yKey) ?? 0) as [number, number]).nice().range([height - pad, pad]);
    root.append("g").attr("transform", `translate(0,${height - pad})`).call(d3.axisBottom(x).ticks(6)).attr("color", "#9da3ad");
    root.append("g").attr("transform", `translate(${pad},0)`).call(d3.axisLeft(y).ticks(6)).attr("color", "#9da3ad");
    root.append("text").attr("x", width - pad).attr("y", height - 10).attr("text-anchor", "end").attr("fill", "#d9dde3").attr("font-size", 10).text(xKey);
    root.append("text").attr("x", 10).attr("y", pad - 14).attr("fill", "#d9dde3").attr("font-size", 10).text(yKey);
    if (viz === "line") {
      const line = d3.line<Record<string, unknown>>().x((d) => x(valueOf(d, xKey) ?? 0)).y((d) => y(valueOf(d, yKey) ?? 0));
      root.append("path").datum(valid).attr("d", line).attr("fill", "none").attr("stroke", "#ffffff").attr("stroke-width", transform.lineWidth).attr("opacity", transform.opacity);
    } else if (viz === "heatmap" || viz === "contour") {
      const vKey = mapping.value || mapping.z;
      const values = valid.map((d) => valueOf(d, vKey) ?? 0);
      const color = d3.scaleSequential(d3.interpolateGreys).domain([d3.max(values) || 1, d3.min(values) || 0]);
      root.selectAll("rect.cell").data(valid).join("rect").attr("class", "cell").attr("x", (d) => x(valueOf(d, xKey) ?? 0) - 4).attr("y", (d) => y(valueOf(d, yKey) ?? 0) - 4).attr("width", 8).attr("height", 8).attr("fill", (d) => color(valueOf(d, vKey) ?? 0)).attr("opacity", transform.opacity);
      if (viz === "contour") root.append("text").attr("x", pad).attr("y", 24).attr("fill", "#c1c5cc").attr("font-size", 10).text("Contour preview uses sampled value cells; interpolation is reported in Method.");
    } else {
      root.selectAll("circle").data(valid).join("circle").attr("cx", (d) => x(valueOf(d, xKey) ?? 0)).attr("cy", (d) => y(valueOf(d, yKey) ?? 0)).attr("r", transform.pointSize).attr("fill", "#ffffff").attr("opacity", transform.opacity).on("click", (_, d) => onSelect(d));
    }
  }, [rows, mapping, viz, onSelect, transform]);
  return <svg className="vizSvg" ref={ref} />;
}

function ThreeDView({ rows, mapping, viz, projection, onSelect, transform, display, showSurface, resetSignal }: { rows: Record<string, unknown>[]; mapping: Mapping; viz: VizId; projection: Projection; onSelect: (row: Record<string, unknown>) => void; transform: TransformState; display: DisplayState; showSurface: boolean; resetSignal: number }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    const host = ref.current;
    host.innerHTML = "";
    const width = host.clientWidth || 900;
    const height = host.clientHeight || 620;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#000000");
    const camera = projection === "3D" ? new THREE.PerspectiveCamera(42, width / height, 0.1, 2000) : new THREE.OrthographicCamera(-62, 62, 62 / (width / height), -62 / (width / height), 0.1, 2000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    host.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.panSpeed = 0.7;
    controls.zoomSpeed = 0.75;
    const valid = rows.filter((row) => valueOf(row, mapping.x) !== null && valueOf(row, mapping.y) !== null && valueOf(row, mapping.z) !== null);
    const ext = { x: numericExtent(valid, `__${mapping.x}`), y: numericExtent(valid, `__${mapping.y}`), z: numericExtent(valid, `__${mapping.z}`) };
    const norm = (v: number, [min, max]: [number, number]) => ((v - min) / (max - min || 1) - 0.5) * 74;
    const vectors = valid.map((row) => {
      const vx = norm(valueOf(row, mapping.x) ?? 0, ext.x);
      const vy = norm(valueOf(row, mapping.y) ?? 0, ext.y);
      const vz = norm(valueOf(row, mapping.z) ?? 0, ext.z);
      if (projection === "XY") return new THREE.Vector3(vx, vy, 0);
      if (projection === "XZ") return new THREE.Vector3(vx, vz, 0);
      if (projection === "YZ") return new THREE.Vector3(vy, vz, 0);
      return new THREE.Vector3(vx, vz, vy);
    });
    const bounds = new THREE.Box3().setFromPoints(vectors.length ? vectors : [new THREE.Vector3()]);
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(36, size.length() * 0.56);
    if (display.showGrid) {
      const grid = new THREE.GridHelper(radius * 1.9, 18, "#404040", "#181818");
      grid.position.y = bounds.min.y - 2;
      scene.add(grid);
    }
    if (display.showBounds) {
      const boundsFrame = new THREE.Box3Helper(bounds, "#3f3f3f");
      scene.add(boundsFrame);
    }
    if (display.showAxisLabels) addAxisLabels(scene, mapping, bounds, ext, projection);
    controls.target.copy(center);
    if (projection === "3D") {
      if (display.cameraPreset === "front") camera.position.set(center.x, center.y, center.z + radius * 2.1);
      else if (display.cameraPreset === "top") camera.position.set(center.x, center.y + radius * 2.1, center.z + 0.01);
      else if (display.cameraPreset === "side") camera.position.set(center.x + radius * 2.1, center.y, center.z);
      else camera.position.set(center.x + radius * 1.18, center.y + radius * 0.82, center.z + radius * 1.08);
    } else camera.position.set(center.x, center.y, center.z + radius * 1.9);
    camera.near = Math.max(0.1, radius / 100);
    camera.far = radius * 12;
    camera.updateProjectionMatrix();
    const positions = new Float32Array(vectors.flatMap((v) => [v.x, v.y, v.z]));
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const pointMaterial = new THREE.PointsMaterial({ color: "#ffffff", size: transform.pointSize, opacity: transform.opacity, transparent: true, map: pointTexture(), alphaTest: 0.08, sizeAttenuation: false });
    scene.add(new THREE.Points(pointGeometry, pointMaterial));
    if (viz === "line3d") {
      scene.add(new THREE.Line(pointGeometry, new THREE.LineBasicMaterial({ color: "#ffffff", linewidth: transform.lineWidth, opacity: transform.opacity, transparent: true })));
    }
    if ((viz === "wireframe" || viz === "surface") && showSurface) {
      const byY = d3.group(valid, (r) => String(valueOf(r, mapping.y)));
      const sortedRows = Array.from(byY.values()).map((list) => list.sort((a, b) => (valueOf(a, mapping.x) ?? 0) - (valueOf(b, mapping.x) ?? 0))).sort((a, b) => (valueOf(a[0], mapping.y) ?? 0) - (valueOf(b[0], mapping.y) ?? 0));
      const w = sortedRows[0]?.length ?? 0;
      const h = sortedRows.length;
      const indices: number[] = [];
      if (w > 1 && h > 1 && sortedRows.every((r) => r.length === w)) {
        for (let yy = 0; yy < h - 1; yy++) {
          for (let xx = 0; xx < w - 1; xx++) {
            const a = yy * w + xx, b = a + 1, c = a + w, d = c + 1;
            indices.push(a, c, b, b, c, d);
          }
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geo.setIndex(indices);
        const material = new THREE.MeshBasicMaterial({ color: "#ffffff", wireframe: viz === "wireframe", opacity: viz === "surface" ? 0.18 : 0.88, transparent: true, side: THREE.DoubleSide });
        const mesh = new THREE.Mesh(geo, material);
        scene.add(mesh);
      }
    }
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const click = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(scene.children);
      const idx = hits[0]?.index;
      if (idx !== undefined && valid[idx]) onSelect(valid[idx]);
    };
    renderer.domElement.addEventListener("click", click);
    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    return () => {
      cancelAnimationFrame(frame);
      renderer.domElement.removeEventListener("click", click);
      renderer.dispose();
      controls.dispose();
    };
  }, [rows, mapping, viz, projection, onSelect, transform, display, showSurface, resetSignal]);
  return <div className="threeHost" ref={ref} />;
}

function App() {
  const [dataset, setDataset] = useState<Dataset | null>(() => analyzeDataset("Demo B · 3D coordinates", demoRows.b));
  const [mapping, setMapping] = useState<Mapping>(() => autoMap(analyzeDataset("Demo B · 3D coordinates", demoRows.b)));
  const [viz, setViz] = useState<VizId>("scatter3d");
  const [projection, setProjection] = useState<Projection>("3D");
  const [transform, setTransform] = useState<TransformState>(defaultTransform);
  const [selected, setSelected] = useState<Record<string, unknown> | null>(null);
  const [notice, setNotice] = useState("3D spatial data detected. Projection modes are available because X/Y/Z numeric mappings are present.");
  const [showSurface, setShowSurface] = useState(false);
  const [resetSignal, setResetSignal] = useState(0);
  const [display, setDisplay] = useState<DisplayState>(defaultDisplay);
  const caps = useMemo(() => detectCapabilities(dataset, mapping), [dataset, mapping]);
  const activeCap = caps.find((c) => c.id === viz);
  const rows = useMemo(() => transformedRows(dataset, mapping, transform), [dataset, mapping, transform]);
  const grid = useMemo(() => gridInfo(dataset, mapping), [dataset, mapping]);
  const is3d = ["scatter3d", "line3d", "pointcloud", "wireframe", "surface", "network3d"].includes(viz) && projection === "3D";

  const loadDataset = (next: Dataset) => {
    setDataset(next);
    const nextMapping = autoMap(next);
    setMapping(nextMapping);
    const first = chooseDefaultViz(next, nextMapping);
    setViz(first);
    setProjection(["scatter3d", "line3d", "pointcloud", "wireframe", "surface"].includes(first) ? "3D" : "XY");
    setSelected(null);
    setShowSurface(gridInfo(next, nextMapping).regular);
    setResetSignal((n) => n + 1);
    setNotice(`Loaded ${next.rows.length} rows and ${next.columns.length} columns. Visualization capabilities were recalculated from detected schema.`);
  };

  const onUpload = async (file: File) => {
    const text = await file.text();
    if (file.name.toLowerCase().endsWith(".json")) {
      const parsed = JSON.parse(text);
      const rows = Array.isArray(parsed) ? parsed : Array.isArray(parsed.data) ? parsed.data : [parsed];
      loadDataset(analyzeDataset(file.name, rows));
      return;
    }
    Papa.parse<Record<string, unknown>>(text, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (result) => loadDataset(analyzeDataset(file.name, result.data)),
    });
  };

  const chooseViz = (cap: Capability) => {
    if (cap.status === "available") {
      setViz(cap.id);
      if (cap.id === "xy") setProjection("XY");
      else if (cap.id === "xz") setProjection("XZ");
      else if (cap.id === "yz") setProjection("YZ");
      else if (cap.group === "2D") setProjection("XY");
      else setProjection("3D");
      setNotice(`${cap.label} selected. ${cap.method ? `Method: ${cap.method}.` : cap.reason}`);
      return;
    }
    setNotice(`${cap.label} unavailable. ${cap.reason}`);
  };

  const exportSettings = () => {
    const blob = new Blob([JSON.stringify({ source: dataset?.filename, mappings: mapping, visualizationType: viz, transform, projection, interpolationMethod: activeCap?.method ?? "None" }, null, 2)], { type: "application/json" });
    downloadBlob(blob, "visualization-settings.json");
  };
  const exportCsv = () => {
    downloadBlob(new Blob([Papa.unparse(rows)], { type: "text/csv" }), "transformed-values.csv");
  };
  const exportPng = () => {
    const canvas = document.querySelector("canvas");
    if (canvas) canvas.toBlob((blob) => blob && downloadBlob(blob, "visualization.png"));
    else {
      const svg = document.querySelector(".vizSvg") as SVGSVGElement | null;
      if (!svg) return;
      const data = new XMLSerializer().serializeToString(svg);
      const image = new Image();
      image.onload = () => {
        const c = document.createElement("canvas");
        c.width = svg.clientWidth;
        c.height = svg.clientHeight;
        const ctx = c.getContext("2d");
        if (!ctx) return;
        ctx.fillStyle = "#0a0b0d";
        ctx.fillRect(0, 0, c.width, c.height);
        ctx.drawImage(image, 0, 0);
        c.toBlob((blob) => blob && downloadBlob(blob, "visualization.png"));
      };
      image.src = `data:image/svg+xml;base64,${btoa(data)}`;
    }
  };
  const exportSvg = () => {
    const svg = document.querySelector(".vizSvg") as SVGSVGElement | null;
    if (svg) downloadBlob(new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml" }), "visualization.svg");
    else setNotice("SVG export is available only for 2D visualizations.");
  };

  const numericCount = dataset?.columns.filter((c) => c.type === "number").length ?? 0;

  return (
    <main className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="mark">DV</span>
          <div>
            <strong>Data Workbench</strong>
            <small>schema-gated 2D / 3D visualization</small>
          </div>
        </div>
        <Section title="Data">
          <label className="upload">
            <Upload size={15} />
            Upload CSV / JSON
            <input type="file" accept=".csv,.json,application/json,text/csv" onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
          </label>
          <div className="demoGrid">
            <button onClick={() => loadDataset(analyzeDataset("Demo A · x/y only", demoRows.a))}>Demo A</button>
            <button onClick={() => loadDataset(analyzeDataset("Demo B · x/y/z", demoRows.b))}>Demo B</button>
            <button onClick={() => loadDataset(analyzeDataset("Demo C · x/y grid + z", demoRows.c))}>Demo C</button>
            <button onClick={() => loadDataset(analyzeDataset("Demo D · source/target", demoRows.d))}>Demo D</button>
          </div>
          <div className="summary">
            <span>{dataset?.filename ?? "No data"}</span>
            <b>{dataset?.rows.length ?? 0} rows</b>
          </div>
          <div className="columns">
            {dataset?.columns.map((c) => (
              <div className="colRow" key={c.name} title={`missing: ${c.missing}, unique: ${c.unique}`}>
                <span>{c.name}</span>
                <em>{c.type}</em>
                {c.type === "number" && <small>{Number(c.min).toFixed(2)}..{Number(c.max).toFixed(2)}</small>}
              </div>
            ))}
          </div>
        </Section>
        <Section title="Mapping">
          {(["x", "y", "z", "value", "group", "time"] as (keyof Mapping)[]).map((key) => (
            <SelectField key={key} label={key === "x" ? "X Axis" : key === "y" ? "Y Axis" : key === "z" ? "Z Axis" : key[0].toUpperCase() + key.slice(1)} value={mapping[key]} columns={dataset?.columns ?? []} allowAny={key === "group" || key === "time"} onChange={(v) => setMapping({ ...mapping, [key]: v })} />
          ))}
          <SelectField label="Source" value={mapping.source} columns={dataset?.columns ?? []} allowAny onChange={(v) => setMapping({ ...mapping, source: v })} />
          <SelectField label="Target" value={mapping.target} columns={dataset?.columns ?? []} allowAny onChange={(v) => setMapping({ ...mapping, target: v })} />
        </Section>
        <Section title="Visualization">
          <div className="vizList">
            {caps.map((cap) => (
              <button key={cap.id} className={cap.id === viz ? "active" : ""} onClick={() => chooseViz(cap)} title={cap.reason}>
                <span className={`dot ${cap.status.replace(" ", "")}`}>{cap.status === "available" ? "●" : cap.status === "requires mapping" ? "○" : "×"}</span>
                <span>{cap.label}</span>
                <small>{cap.status === "available" ? "Available" : cap.status === "requires mapping" ? "Requires mapping" : "Not supported"}</small>
              </button>
            ))}
          </div>
          <div className="segmented">
            {(["3D", "XY", "XZ", "YZ"] as Projection[]).map((p) => (
              <button key={p} disabled={!isNumeric(dataset, mapping.x) || !isNumeric(dataset, mapping.y) || !isNumeric(dataset, mapping.z)} className={projection === p ? "on" : ""} onClick={() => setProjection(p)}>{p}</button>
            ))}
          </div>
          {projection !== "3D" && <p className="hint">{projection} Projection. {projection === "XY" ? "Z" : projection === "XZ" ? "Y" : "X"} axis is omitted. Original values are preserved.</p>}
          {(viz === "surface" || viz === "wireframe") && grid.irregular && (
            <div className="warning">
              <strong>Surface method: Delaunay triangulation</strong>
              <span>This can add faces that do not exist in the source data.</span>
              <button onClick={() => setShowSurface(true)}>Show surface</button>
              <button onClick={() => setShowSurface(false)}>Keep points only</button>
            </div>
          )}
        </Section>
        <Section title="Transform" initial={false}>
          <label className="field"><span>Normalize</span><select value={transform.normalization} onChange={(e) => setTransform({ ...transform, normalization: e.target.value as Normalization })}><option value="none">None</option><option value="minmax">Min-Max</option><option value="zscore">Z-score</option></select></label>
          {(["xScale", "yScale", "zScale", "noise", "smoothing", "pointSize", "lineWidth", "opacity"] as (keyof TransformState)[]).map((key) => (
            <label className="field sliderField" key={key}>
              <span>{String(key)}</span>
              <input type="range" min={key === "opacity" ? 0.1 : 0} max={key.includes("Scale") ? 5 : key === "pointSize" ? 8 : key === "lineWidth" ? 4 : 1} step={key.includes("Scale") || key === "pointSize" || key === "lineWidth" ? 0.1 : 0.01} value={Number(transform[key])} onChange={(e) => setTransform({ ...transform, [key]: Number(e.target.value) })} />
              <output>{Number(transform[key]).toFixed(key.includes("Scale") || key === "pointSize" || key === "lineWidth" ? 1 : 2)}</output>
            </label>
          ))}
          <div className="splitFields">
            <input placeholder="threshold min" value={transform.thresholdMin} onChange={(e) => setTransform({ ...transform, thresholdMin: e.target.value })} />
            <input placeholder="threshold max" value={transform.thresholdMax} onChange={(e) => setTransform({ ...transform, thresholdMax: e.target.value })} />
          </div>
          <button className="toolBtn" onClick={() => setTransform(defaultTransform)}><RotateCcw size={14} /> Reset Transform</button>
        </Section>
        <Section title="Display" initial={false}>
          <div className="kv"><span>Geometry</span><b>{vizLabels[viz]}</b></div>
          <div className="kv"><span>Projection</span><b>{projection}</b></div>
          <div className="kv"><span>Representation</span><b>{transform.normalization === "none" ? "Raw" : "Normalized"}</b></div>
          <label className="checkField"><input type="checkbox" checked={display.showGrid} onChange={(e) => setDisplay({ ...display, showGrid: e.target.checked })} /> Grid</label>
          <label className="checkField"><input type="checkbox" checked={display.showBounds} onChange={(e) => setDisplay({ ...display, showBounds: e.target.checked })} /> Bounding box</label>
          <label className="checkField"><input type="checkbox" checked={display.showAxisLabels} onChange={(e) => setDisplay({ ...display, showAxisLabels: e.target.checked })} /> Axis numeric labels</label>
          <div className="presetGrid" aria-label="3D camera presets">
            {(["iso", "front", "top", "side"] as CameraPreset[]).map((preset) => (
              <button key={preset} className={display.cameraPreset === preset ? "on" : ""} onClick={() => { setDisplay({ ...display, cameraPreset: preset }); setProjection("3D"); setResetSignal((n) => n + 1); }}>
                {preset.toUpperCase()}
              </button>
            ))}
          </div>
        </Section>
        <Section title="Export" initial={false}>
          <button className="toolBtn" onClick={exportPng}><Download size={14} /> PNG</button>
          <button className="toolBtn" onClick={exportSvg}><Download size={14} /> SVG 2D</button>
          <button className="toolBtn" onClick={exportSettings}><FileJson size={14} /> JSON settings</button>
          <button className="toolBtn" onClick={exportCsv}><Download size={14} /> CSV transformed values</button>
        </Section>
      </aside>
      <section className="workspace">
        <header className="topbar">
          <div>
            <span>{dataset?.filename}</span>
            <b>{vizLabels[viz]}</b>
            <em>{projection} · {rows.length} visual rows · {numericCount} numeric columns</em>
          </div>
          <div className="viewStrip" aria-label="View controls">
            {(["3D", "XY", "XZ", "YZ"] as Projection[]).map((p) => (
              <button key={p} disabled={!isNumeric(dataset, mapping.x) || !isNumeric(dataset, mapping.y) || !isNumeric(dataset, mapping.z)} className={projection === p ? "on" : ""} onClick={() => setProjection(p)}>{p}</button>
            ))}
            <button onClick={() => setResetSignal((n) => n + 1)}><Maximize2 size={12} /> Fit</button>
          </div>
          <p>{notice}</p>
        </header>
        <div className="canvas">
          {!dataset || numericCount === 0 && !hasNetwork(mapping) ? (
            <div className="empty">No valid numeric visualization available. Detected columns: {dataset?.columns.map((c) => `${c.name}: ${c.type}`).join(", ") || "none"}.</div>
          ) : is3d || ["scatter3d", "line3d", "pointcloud", "wireframe", "surface"].includes(viz) ? (
            <ThreeDView rows={rows} mapping={mapping} viz={viz} projection={projection} onSelect={setSelected} transform={transform} display={display} showSurface={showSurface || grid.regular} resetSignal={resetSignal} />
          ) : (
            <TwoDView rows={rows} mapping={mapping} viz={viz} onSelect={setSelected} transform={transform} />
          )}
          {selected && (
            <div className="inspector">
              <strong>Selected Point</strong>
              {Object.entries(selected).filter(([k]) => !k.startsWith("__")).map(([k, v]) => <div key={k}><span>{k}</span><b>{String(v)}</b></div>)}
            </div>
          )}
        </div>
        <footer className="method">
          <div><span>Visualization</span><b>{vizLabels[viz]}</b></div>
          <div><span>Mapping</span><b>X={mapping.x || "none"} · Y={mapping.y || "none"} · Z={mapping.z || "none"} · Value={mapping.value || "none"}</b></div>
          <div><span>Transform</span><b>{transform.normalization === "none" ? "Normalization off" : transform.normalization} · smoothing {transform.smoothing} · noise {transform.noise}</b></div>
          <div><span>Generated geometry</span><b>{rows.length} points · {(viz === "surface" || viz === "wireframe") && (showSurface || grid.regular) ? Math.max(0, (new Set(rows.map((r) => String(valueOf(r, mapping.x)))).size - 1) * (new Set(rows.map((r) => String(valueOf(r, mapping.y)))).size - 1) * 2) : 0} generated surfaces · View: {display.cameraPreset.toUpperCase()}</b></div>
        </footer>
      </section>
    </main>
  );
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

createRoot(document.getElementById("root")!).render(<App />);
