#!/bin/bash
# xHCI / IOMMU health sampler.
#
# Samples the counts of four kernel signatures on a fixed cadence and appends
# one JSON line per sample, so a soak that ends in a hard power-off still
# leaves a monotonic record on disk of what the kernel was seeing.
#
#   not part of TD     xHCI transfer-ring desync. The controller's event
#                      pointer no longer lies inside the driver's ring. This
#                      is the corruption signature.
#   Set TR Deq Ptr     endpoint-context / teardown-race signature.
#   CLEAR_HALT         usbfs CLEAR_HALT issued on an ACTIVE endpoint, which
#                      rewrites the hardware dequeue pointer under queued
#                      transfers.
#   DMAR               IOMMU fault. Zero while the IOMMU is in lazy
#                      invalidation mode, where a stray DMA into a freed page
#                      succeeds silently. With iommu.strict=1 on the kernel
#                      cmdline the same write is blocked and logged here, so
#                      a DMAR fault appearing means the corruption was real
#                      and is now contained.
#
# Reads only. Touches no USB device, opens no camera.
#
# Usage:
#   scripts/xhci_watch.sh [interval_seconds] [output_path]
#   scripts/xhci_watch.sh 30 logs/xhci_watch.jsonl

set -u

INTERVAL="${1:-30}"
OUT="${2:-logs/xhci_watch.jsonl}"
mkdir -p "$(dirname "$OUT")"

count() { journalctl -k -b 2>/dev/null | grep -c -- "$1" || true; }

echo "sampling every ${INTERVAL}s -> ${OUT} (Ctrl-C to stop)" >&2
while true; do
    ts=$(date -Is)
    td=$(count "not part of TD")
    deq=$(count "Set TR Deq Ptr")
    halt=$(count "CLEAR_HALT")
    dmar=$(count "DMAR:")
    unclaimed=$(count "did not claim interface")
    printf '{"ts":"%s","not_part_of_td":%s,"set_tr_deq":%s,"clear_halt":%s,"dmar":%s,"usbfs_unclaimed":%s}\n' \
        "$ts" "$td" "$deq" "$halt" "$dmar" "$unclaimed" >> "$OUT"
    sync "$OUT" 2>/dev/null || true
    sleep "$INTERVAL"
done
