"""
fNIRS Walking Trigger Sender
Alternates standing/walking blocks with random durations.
Plays audio cues and sends LSL triggers simultaneously on walking events.
"""

import random
import threading
import time
from datetime import datetime

import winsound
from pylsl import StreamInfo, StreamOutlet, local_clock

# Trigger values
TRIGGERS = {
    "standing_start": 1,
    "standing_end": 2,
    "walking_start": 3,
    "walking_end": 4,
}

# Duration ranges (min, max) in seconds
DURATIONS = {
    "standing": (12, 35),
    "walking": (4, 28),
}

# Beep params: (frequency Hz, duration ms)
# High tone = start walking, low tone = stop walking
BEEP_WALKING_START = (1200, 400)
BEEP_WALKING_END = (500, 400)

NUM_BLOCKS = 10
STARTUP_DELAY = 5  # seconds before first block


def _log(log_file, timestamp, lsl_time, trigger_val, event_name):
    with open(log_file, "a") as f:
        f.write(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')},{lsl_time:.6f},{trigger_val},{event_name}\n")
    print(f"  [{timestamp.strftime('%H:%M:%S.%f')[:-3]}] Trigger {trigger_val} — {event_name}")


def send_trigger(outlet, trigger_val, log_file):
    """Send LSL trigger with no audio cue."""
    lsl_time = local_clock()
    outlet.push_sample([trigger_val])
    timestamp = datetime.now()
    event_name = next(k for k, v in TRIGGERS.items() if v == trigger_val)
    _log(log_file, timestamp, lsl_time, trigger_val, event_name)


def send_trigger_with_beep(outlet, trigger_val, beep_params, log_file):
    """Send LSL trigger and play beep simultaneously via threading."""
    beep_thread = threading.Thread(target=winsound.Beep, args=beep_params, daemon=True)
    beep_thread.start()
    lsl_time = local_clock()
    outlet.push_sample([trigger_val])
    timestamp = datetime.now()
    beep_thread.join()

    event_name = next(k for k, v in TRIGGERS.items() if v == trigger_val)
    _log(log_file, timestamp, lsl_time, trigger_val, event_name)


def main():
    log_filename = f"trigger_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Logging to: {log_filename}")

    with open(log_filename, "w") as f:
        f.write("Timestamp,LSL_Time,Trigger,Event_Name\n")

    print("Creating LSL stream 'Trigger'...")
    info = StreamInfo(
        name="Trigger",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="int32",
        source_id="trigger_sender_001",
    )
    outlet = StreamOutlet(info)
    print("LSL stream 'Trigger' created and ready.")
    print(f"\nStarting in {STARTUP_DELAY}s — {NUM_BLOCKS} blocks")
    print("HIGH beep = start walking | LOW beep = stop walking\n")
    time.sleep(STARTUP_DELAY)

    try:
        for block in range(1, NUM_BLOCKS + 1):
            stand_dur = random.randint(*DURATIONS["standing"])
            walk_dur = random.randint(*DURATIONS["walking"])

            print(f"\n--- Block {block}/{NUM_BLOCKS} ---")

            print(f"STAND ({stand_dur}s)")
            send_trigger(outlet, TRIGGERS["standing_start"], log_filename)
            time.sleep(stand_dur)
            send_trigger(outlet, TRIGGERS["standing_end"], log_filename)

            print(f"WALK  ({walk_dur}s)")
            send_trigger_with_beep(outlet, TRIGGERS["walking_start"], BEEP_WALKING_START, log_filename)
            time.sleep(walk_dur)
            send_trigger_with_beep(outlet, TRIGGERS["walking_end"], BEEP_WALKING_END, log_filename)

        print("\nProtocol complete.")

    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
