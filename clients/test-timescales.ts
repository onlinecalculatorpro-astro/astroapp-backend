import { echoTimescales } from "./timescales-client";

async function main() {
  const base = "https://astroapp-backend.onrender.com";
  const r = await echoTimescales(base, {
    date: "2016-12-31",
    time: "23:59:60",
    place_tz: "UTC",
    dut1: 0.0,
  });
  console.log(r);
}
main().catch(err => { console.error(err); process.exit(1); });
