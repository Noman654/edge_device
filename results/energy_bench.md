# Energy per token: NOT measurable on the QDC QRD board (2026-09-07)
Battery node: current_now = -2.4 to -4.3 mA (0.02-0.03 W) both idle and under full decode load; status "Discharging", capacity 100%.
USB node: online=0, current 0. Wireless: offline. The SoC is fed by an external supply that sysfs does not expose, so
neither current*voltage sampling (scripts/energy_bench.sh) nor batterystats can attribute power to inference.
Consequence for the report: the polling optimization (OPPOLL=1) keeps one CPU core spinning during NPU batches; its
energy cost is real but unmeasured here. On a phone it should be re-measured with the battery counters (the script
is ready) before OPPOLL is adopted for battery-sensitive use. Recommendation stands for plugged-in / latency-first use.
