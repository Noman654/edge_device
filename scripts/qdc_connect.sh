#!/usr/bin/env bash
# Usage: scripts/qdc_connect.sh <key.pem> <sa-HOST>
# Opens the QDC ADB tunnel in the foreground. Run `adb devices` in another terminal.
set -euo pipefail
KEY="$1"; HOST="$2"
chmod 600 "$KEY"
adb kill-server || true
exec ssh -i "$KEY" -L 5037:"$HOST":5037 -N sshtunnel@ssh.qdc.qualcomm.com
