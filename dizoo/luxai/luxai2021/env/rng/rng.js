const seedrandom = require("./seedrandom");
const seed = parseInt(process.argv[2]);
const N = parseInt(process.argv[3]);
const rng = seedrandom(`gen_${seed}`);
const vals = [];
for (let i = 0; i < N; i++) {
  vals.push(rng());
}
process.stdout.write(vals.join(","));