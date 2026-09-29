# Host USB rules (Azure Kinect + Intel RealSense)

Host-level (`/etc`) fixes for running Azure Kinect and Intel RealSense cameras
over USB 3.0 reliably. These are not Python and are not applied by the SPARK
server; install them once per machine.

## Why

Both libk4a (Azure Kinect) and the pyrealsense2 rsusb backend claim their USB
interfaces through libusb. If the kernel `uvcvideo` driver also binds those
interfaces, the two race under sustained isochronous streaming and can corrupt
the xHCI TRB ring, which shows up as USB reprobing storms and silent kernel
hangs. USB autosuspend on an active depth stream causes similar event cascades.

## What gets installed

- `modprobe.d/blacklist-uvcvideo.conf`: keep the kernel UVC driver from loading
  at all (the load-bearing fix).
- `rules.d/99-k4a.rules`: plugdev device-node permissions for the Kinect.
- `rules.d/99-k4a-disable-autosuspend.rules`: pin Kinect USB power on.
- `rules.d/99-k4a-unbind-uvcvideo.rules`: detach uvcvideo from the 4K color
  camera (redundant when uvcvideo is blacklisted; kept for mixed-camera hosts).
- `rules.d/99-realsense-no-autosuspend.rules`: pin RealSense D400 power on.

RealSense device-node permissions come from librealsense itself
(`99-realsense-libusb.rules`); install librealsense to get those.

## Install

```bash
sudo scripts/host-usb/install.sh
```

Re-run after a kernel or distro upgrade. A reboot is recommended so the
uvcvideo blacklist takes effect everywhere.
