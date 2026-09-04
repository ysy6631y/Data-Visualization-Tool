import Papa from "papaparse";

const csv3d = `x,y,z,time
0,0,0,0
1,0,0.4,1
1,1,0.8,2
0,1,0.2,3`;

const jsonNetwork = [
  { source: "A", target: "B", weight: 2 },
  { source: "B", target: "C", weight: 4 },
];

const grid = [];
for (let x = 0; x < 3; x++) {
  for (let y = 0; y < 3; y++) grid.push({ x, y, z: Math.sin(x) + Math.cos(y) });
}

function toNumber(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function inferColumns(rows) {
  const names = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  return names.map((name) => {
    const values = rows.map((row) => row[name]).filter((v) => v !== null && v !== undefined && v !== "");
    const numeric = values.map(toNumber).filter((v) => v !== null);
    return { name, type: values.length && numeric.length / values.length >= 0.9 ? "number" : "string" };
  });
}

function analyze(rows) {
  const columns = inferColumns(rows);
  const numeric = columns.filter((c) => c.type === "number").map((c) => c.name);
  const source = columns.find((c) => c.name === "source")?.name ?? "";
  const target = columns.find((c) => c.name === "target")?.name ?? "";
  const mapping = { x: numeric[0] ?? "", y: numeric[1] ?? "", z: numeric[2] ?? "", source, target };
  const pairs = new Set(rows.map((r) => `${r.x}|${r.y}`));
  const xs = new Set(rows.map((r) => r.x));
  const ys = new Set(rows.map((r) => r.y));
  return {
    numeric,
    mapping,
    scatter2d: numeric.length >= 2,
    scatter3d: numeric.length >= 3,
    network: Boolean(source && target),
    regularGrid: xs.size > 1 && ys.size > 1 && pairs.size === xs.size * ys.size && pairs.size >= 9,
  };
}

const parsedCsv = Papa.parse(csv3d, { header: true, dynamicTyping: true, skipEmptyLines: true }).data;
const csvResult = analyze(parsedCsv);
const jsonResult = analyze(jsonNetwork);
const gridResult = analyze(grid);

const checks = [
  ["CSV numeric columns inferred", csvResult.numeric.join(",") === "x,y,z,time"],
  ["CSV enables 3D scatter", csvResult.scatter3d],
  ["JSON source/target enables network", jsonResult.network],
  ["Grid enables surface/wireframe condition", gridResult.regularGrid],
];

for (const [label, pass] of checks) {
  console.log(`${pass ? "PASS" : "FAIL"} ${label}`);
  if (!pass) process.exitCode = 1;
}
