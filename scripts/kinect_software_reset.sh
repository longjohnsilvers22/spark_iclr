#!/bin/bash
# Software power-cycle of the Azure Kinect DKs via uhubctl.
#
# Replaces the physical 12V-barrel unplug. The Azure Kinect's internal
# USB hubs (045e:097a SS, 045e:097b HS) report per-port power switching
# (ppps), so Vbus to the depth and RGB cameras can be cut without
# unplugging anything. After this script runs, both Kinects are
# re-enumerated and the SDK can open them on the next attempt.
#
# Limitation: this cuts only USB Vbus, not the Kinect's 12V barrel
# input. The Kinect's depth MCU draws most of its current from the
# 12V; cutting USB power is enough to reset the SDK-visible state in
# most cases but NOT to recover from a hard MCU lockup. If this
# script doesn't bring the Kinects back, fall back to the physical
# 12V unplug.
#
# Usage:
#   sudo ./scripts/kinect_software_reset.sh
# Or as a pipeline helper (no sudo needed if uhubctl udev rule is set up).

set -e

if ! command -v uhubctl >/dev/null 2>&1; then
    echo "uhubctl not installed; install with: sudo apt-get install uhubctl" >&2
    exit 1
fi

# Find all Kinect hubs by USB IDs. The internal hubs are 045e:097a
# (SuperSpeed, carries depth + 4K) and 045e:097b (High-Speed, carries
# microphone array). Each Kinect has one of each. uhubctl reports
# their bus locations (e.g. 2-9, 2-6, 1-7, 1-5).
hubs=$(uhubctl 2>/dev/null \
    | grep -E "Current status for hub .* \[045e:097[ab]" \
    | sed -E 's/Current status for hub ([0-9.-]+).*/\1/')

if [ -z "$hubs" ]; then
    echo "No Azure Kinect hubs found via uhubctl. Are the Kinects plugged in?" >&2
    exit 2
fi

echo "Found Kinect hubs:"
for hub in $hubs; do echo "  $hub"; done
echo ""

echo "Cutting power to all ports on each hub..."
for hub in $hubs; do
    uhubctl -l "$hub" -a off >/dev/null 2>&1 || true
done

sleep 5

echo "Restoring power..."
for hub in $hubs; do
    uhubctl -l "$hub" -a on >/dev/null 2>&1 || true
done

echo "Waiting 8s for re-enumeration..."
sleep 8

echo ""
echo "Post-reset Kinect devices:"
lsusb -d 045e: | head -10

echo ""
echo "Done. Restart the SPARK server (or call /api/kinect_reset once wired)"
echo "to have the SDK reopen them."
