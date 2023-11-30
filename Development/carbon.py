from codecarbon import OfflineEmissionsTracker
import time

tracker = OfflineEmissionsTracker(
    country_iso_code="ITA",
    measure_power_secs=30,
    tracking_mode='process',
    save_to_file=False,
    log_level='critical')
tracker.start_task("load dataset")

time.sleep(7)

t1 = tracker.stop_task()
# GPU intensive training code

print(dict(t1.values))

tracker.start_task("load eeee")

# time.sleep(5)
#
# t2 = tracker.stop_task()
#
# print(t2.toJSON())

# print(tracker.final_emissions_data.toJSON())
