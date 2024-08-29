from codecarbon import OfflineEmissionsTracker
import time
tracker = OfflineEmissionsTracker(country_iso_code="ITA")
tracker.start()
time.sleep(3)
tracker.stop()

print(tracker.final_emissions_data)