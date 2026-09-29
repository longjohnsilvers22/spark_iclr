#!/bin/bash
# Sysfs `authorized` toggle recovery for Azure Kinects.
#
# Use this when `kinect_software_reset.sh` (uhubctl Vbus cycle) wasn't
# enough - e.g. `k4a_device_open` fails with `LIBUSB_ERROR_IO` on
# `libusb_get_bos_descriptor`. This is a kernel-level USB detach/reattach,
# deeper than uhubctl, and recovers cases where the kernel's USB
# enumeration has gone stale but the Kinect's depth MCU is still alive.
#
# Recovers MOST Kinect dropouts. The case it does NOT fix is when the
# Kinect's depth MCU is itself hung: `lsusb -d 045e:097c` is missing
# (no depth camera enumerated) AND dmesg shows
# `device not accepting address ... error -71` retrying forever on the
# relevant SS hub port. The depth MCU's USB processor is unresponsive
# at the hardware level - only a 12V barrel power-cycle clears it.
# This script reports that case in its exit code so callers can escalate.
#
# Usage:
#   sudo ./scripts/kinect_authorize_reset.sh
#
# Safe to re-run. Idempotent.
#
# DO NOT replace this with an `xhci_hcd unbind/bind` cycle. That has
# damaged a Kinect's color MCU on this host. Stick to per-hub sysfs
# `authorized` toggles only.
#
# DO NOT use `/sys/bus/usb/drivers/usb/unbind` on individual Kinect
# hubs either - it leaves the hub in a state where downstream children
# disappear and you have to re-auth the entire root hub to recover.

set -u

if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        exec sudo "$0" "$@"
    fi
    echo "must run as root (sysfs writes need it)" >&2
    exit 1
fi

discover_hubs() {
    ss_hubs=()
    hs_hubs=()
    for d in /sys/bus/usb/devices/*-*; do
        [ -e "$d/idVendor" ] || continue
        [ "$(cat "$d/idVendor")" = "045e" ] || continue
        pid=$(cat "$d/idProduct" 2>/dev/null || echo "")
        base=$(basename "$d")
        # Top-level hubs only (no `.` in name).
        case "$base" in *.*) continue;; esac
        case "$pid" in
            097a) ss_hubs+=("$base") ;;
            097b) hs_hubs+=("$base") ;;
        esac
    done
}

count_depth_cams() {
    # Count enumerated 097c depth cameras (one per healthy Kinect).
    local n=0
    for d in /sys/bus/usb/devices/*-*; do
        [ -e "$d/idProduct" ] || continue
        [ "$(cat "$d/idProduct" 2>/dev/null)" = "097c" ] || continue
        [ "$(cat "$d/idVendor" 2>/dev/null)" = "045e" ] || continue
        n=$((n + 1))
    done
    echo "$n"
}

discover_hubs
echo "Found Kinect hubs:"
echo "  HS (097b): ${hs_hubs[*]:-none}"
echo "  SS (097a): ${ss_hubs[*]:-none}"

if [ "${#hs_hubs[@]}" -eq 0 ] && [ "${#ss_hubs[@]}" -eq 0 ]; then
    echo "no Kinect hubs enumerated - both bricked or unplugged" >&2
    echo "physical 12V barrel cycle required" >&2
    exit 2
fi

depth_before=$(count_depth_cams)
echo "Depth cameras enumerated before: $depth_before / ${#ss_hubs[@]} expected"

# Phase 1: per-hub authorize toggle. Cheapest and safest; clears most
# stale enumeration states.
all_hubs=("${hs_hubs[@]}" "${ss_hubs[@]}")
echo ""
echo "Phase 1: per-hub sysfs authorize toggle"
for h in "${all_hubs[@]}"; do
    echo 0 > "/sys/bus/usb/devices/$h/authorized" 2>/dev/null || true
done
sleep 2
for h in "${all_hubs[@]}"; do
    echo 1 > "/sys/bus/usb/devices/$h/authorized" 2>/dev/null || true
done
sleep 6

depth_after=$(count_depth_cams)
echo "Depth cameras enumerated after phase 1: $depth_after / ${#ss_hubs[@]}"

# Phase 2: if still missing depth cams, escalate to a root-hub
# authorize toggle. This re-enumerates the entire usb2 bus and brings
# back devices that got stranded after sub-hub manipulation. Safe (no
# driver unbind, no power-cycle of the controller).
if [ "$depth_after" -lt "${#ss_hubs[@]}" ]; then
    echo ""
    echo "Phase 2: usb2 root-hub authorize toggle (broader re-enumeration)"
    echo 0 > /sys/bus/usb/devices/usb2/authorized 2>/dev/null || true
    sleep 2
    echo 1 > /sys/bus/usb/devices/usb2/authorized 2>/dev/null || true
    sleep 8
    depth_after=$(count_depth_cams)
    echo "Depth cameras enumerated after phase 2: $depth_after / ${#ss_hubs[@]}"
fi

echo ""
echo "Post-reset 045e devices:"
lsusb -d 045e: | grep -v "Audio Device" | head -20

echo ""
if [ "$depth_after" -ge "${#ss_hubs[@]}" ] && [ "$depth_after" -ge 2 ]; then
    echo "OK: all expected Kinects have depth enumerated."
    echo "Restart the SPARK server (SIGTERM) - SDK should reopen both."
    exit 0
fi

# Find which Kinect(s) are missing depth. Match SS hub -> child 097c.
echo "WARNING: $((${#ss_hubs[@]} - depth_after)) Kinect(s) missing depth (097c)."
echo ""
for ss in "${ss_hubs[@]}"; do
    has_097c=0
    for d in /sys/bus/usb/devices/${ss}.*; do
        [ -e "$d/idProduct" ] || continue
        [ "$(cat "$d/idProduct")" = "097c" ] || continue
        has_097c=1
        break
    done
    if [ "$has_097c" -eq 1 ]; then
        echo "  SS hub $ss: depth OK"
    else
        echo "  SS hub $ss: depth MISSING - depth MCU hung, 12V cycle required for this Kinect"
        # Look for the giveaway dmesg signature on the matching port.
        recent=$(dmesg --since "5 minutes ago" 2>/dev/null \
            | grep -E "usb $ss\.[0-9]+: device not accepting address" \
            | tail -3)
        [ -n "$recent" ] && echo "    dmesg confirms (error -71 on address-set):" && echo "$recent" | sed 's/^/      /'
    fi
done

echo ""
echo "The OTHER Kinect is recovered - the SPARK pipeline runs on one camera."
echo "If you need both, physically unplug the 12V barrel on the bad Kinect"
echo "for ~5s and plug back in, then re-run this script."
exit 3
