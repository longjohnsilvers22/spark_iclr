#!/usr/bin/env bash
# DEPRECATED: librealsense is built with FORCE_RSUSB_BACKEND=ON (installed to
# /usr/local; pyrealsense2 in the sam3/spark_conda/pi05 envs matches). The
# RealSense works over libusb with NO kernel uvcvideo driver, so this script's
# modprobe+unbind dance is unnecessary and `enable` reintroduces the exact
# crash risk the blacklist prevents. Kept only for reference; `status` is
# still harmless.
#
# Toggle the wrist RealSense (D435i) on/off without endangering the Kinects.
#
# WHY THIS EXISTS
#   uvcvideo is blacklisted on this host (modprobe.d/blacklist-uvcvideo.conf)
#   because uvcvideo + libusb racing for the Azure Kinect 4K color interface
#   (045e:097d) corrupts the xHCI TRB ring and crashes the host under
#   sustained streaming. But this librealsense build needs the KERNEL uvcvideo
#   backend (its libusb backend enumerates 0 devices), so the RealSense can't
#   produce frames until uvcvideo is loaded.
#
#   The safe path (this script): load uvcvideo, then immediately detach it from
#   every Kinect color interface (the 99-k4a-unbind-uvcvideo.rules udev rule
#   does this on the bind event; this script ALSO sweeps explicitly),
#   leaving uvcvideo bound ONLY to the RealSense. Kinect depth/color keep using
#   libk4a's libusb path untouched.
#
#   VLA needs only RGB from every camera, so the heavy Kinect depth
#   stream need not run alongside this -- lower contention, lower risk.
#
# USAGE
#   scripts/wrist_realsense.sh enable    # load uvcvideo, bind RealSense only
#   scripts/wrist_realsense.sh disable   # remove uvcvideo (safest for SPARK)
#   scripts/wrist_realsense.sh status    # show what's bound where
#
# Run `enable` while the Kinects are IDLE (no spark server streaming) for the
# lowest risk. Re-run `disable` before heavy Kinect depth work / SPARK.

set -uo pipefail

KINECT_COLOR_VID="045e"
KINECT_COLOR_PID="097d"
REALSENSE_VID="8086"
REALSENSE_PID="0b3a"
PY="${SPARK_PY:-$HOME/miniconda3/envs/spark_conda/bin/python}"

# Resolve sudo: passwordless if available, else plain sudo (will prompt).
if sudo -n true 2>/dev/null; then SUDO="sudo"; else SUDO="sudo"; fi

unbind_uvcvideo_from_kinects() {
    # Detach uvcvideo from every 045e:097d interface it may have claimed.
    local detached=0
    for dev in /sys/bus/usb/devices/*; do
        [ -f "$dev/idVendor" ] || continue
        [ -f "$dev/idProduct" ] || continue
        [ "$(cat "$dev/idVendor")" = "$KINECT_COLOR_VID" ] || continue
        [ "$(cat "$dev/idProduct")" = "$KINECT_COLOR_PID" ] || continue
        for intf in "$dev":*; do
            [ -d "$intf/driver" ] || continue
            local drv; drv=$(basename "$(readlink "$intf/driver")")
            if [ "$drv" = "uvcvideo" ]; then
                echo "  detaching uvcvideo from Kinect color $(basename "$intf")"
                echo "$(basename "$intf")" | $SUDO tee /sys/bus/usb/drivers/uvcvideo/unbind >/dev/null 2>&1 \
                    && detached=$((detached+1))
            fi
        done
    done
    return 0
}

driver_of() {  # $1 = interface sysfs path
    [ -L "$1/driver" ] && basename "$(readlink "$1/driver")" || echo none
}

show_bindings() {
    echo "uvcvideo loaded : $([ -d /sys/module/uvcvideo ] && echo yes || echo no)"
    echo "/dev/video*     : $(ls /dev/video* 2>/dev/null | tr '\n' ' ' || echo none)"
    for want in "$KINECT_COLOR_VID:$KINECT_COLOR_PID:Kinect-color" "$REALSENSE_VID:$REALSENSE_PID:RealSense"; do
        IFS=: read -r vid pid label <<<"$want"
        for dev in /sys/bus/usb/devices/*; do
            [ -f "$dev/idVendor" ] && [ "$(cat "$dev/idVendor")" = "$vid" ] || continue
            [ "$(cat "$dev/idProduct")" = "$pid" ] || continue
            local drivers=""
            for intf in "$dev":*; do [ -d "$intf" ] && drivers+="$(driver_of "$intf") "; done
            echo "$label ($vid:$pid @ $(basename "$dev")): $drivers"
        done
    done
}

case "${1:-status}" in
enable)
    echo "[wrist] DEPRECATED: the RSUSB librealsense build no longer needs uvcvideo."
    echo "        Loading it only adds Kinect crash risk. Re-run with FORCE=1 if you"
    echo "        really need the kernel V4L2 path (e.g. testing the apt viewer)."
    [ "${FORCE:-0}" = "1" ] || exit 1
    echo "[wrist] enabling RealSense (loading uvcvideo, protecting Kinects)..."
    if pgrep -f "spark_real.server" >/dev/null 2>&1; then
        echo "  WARNING: spark_real.server is running (Kinects may be streaming)."
        echo "           Loading uvcvideo now is the documented crash scenario."
        echo "           Stop the server first, or re-run with FORCE=1 to override."
        [ "${FORCE:-0}" = "1" ] || exit 1
    fi
    $SUDO udevadm control --reload-rules 2>/dev/null
    $SUDO modprobe uvcvideo
    sleep 0.3
    unbind_uvcvideo_from_kinects        # belt-and-braces beyond the udev rule
    sleep 0.5
    echo "[wrist] post-enable state:"
    show_bindings
    echo "[wrist] librealsense check:"
    "$PY" - <<'PYEOF'
import pyrealsense2 as rs
devs = rs.context().query_devices()
print(f"  RealSense devices: {len(devs)}")
for d in devs:
    print("   -", d.get_info(rs.camera_info.name), "SN", d.get_info(rs.camera_info.serial_number))
PYEOF
    # Final guard: assert no Kinect color interface is on uvcvideo.
    if grep -ql uvcvideo /sys/bus/usb/devices/*"$KINECT_COLOR_PID"*/*/driver 2>/dev/null; then
        echo "  ERROR: a Kinect color interface is STILL on uvcvideo -- detaching again"
        unbind_uvcvideo_from_kinects
    else
        echo "  OK: no Kinect color interface bound to uvcvideo"
    fi
    ;;
disable)
    echo "[wrist] disabling RealSense (removing uvcvideo -> safest for SPARK)..."
    $SUDO modprobe -r uvcvideo 2>&1 | sed 's/^/  /' || {
        echo "  modprobe -r failed (module busy?). Stop any RealSense user first."
        exit 1
    }
    sleep 0.3
    show_bindings
    ;;
status)
    show_bindings
    ;;
*)
    echo "usage: $0 {enable|disable|status}" >&2
    exit 2
    ;;
esac
