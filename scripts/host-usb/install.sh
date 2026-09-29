#!/usr/bin/env bash
# Install SPARK host USB rules for Azure Kinect and Intel RealSense over USB 3.0.
# Copies the udev rules and the uvcvideo blacklist into /etc, reloads udev, and
# unloads uvcvideo. Run with sudo. Re-run after a kernel or distro upgrade.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

install -m 0644 "$HERE"/rules.d/*.rules /etc/udev/rules.d/
install -m 0644 "$HERE"/modprobe.d/*.conf /etc/modprobe.d/
echo "Installed udev rules and modprobe blacklist."

udevadm control --reload-rules
udevadm trigger
echo "Reloaded udev."

# Drop uvcvideo now if it is loaded; the blacklist keeps it out on reboot.
if lsmod | grep -q '^uvcvideo'; then
    modprobe -r uvcvideo 2>/dev/null || \
        echo "Could not unload uvcvideo now (in use); it stays out after reboot."
fi

echo "Done. A reboot or camera replug is recommended so all rules apply cleanly."
