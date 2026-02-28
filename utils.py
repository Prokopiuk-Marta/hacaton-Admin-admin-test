import sys
import time
import threading

class SpinnerTimer:
    def __enter__(self):
        self.stop_event = threading.Event()
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join()
        sys.stdout.write("\rGeneration completed. Check dataset file.             \n")
        sys.stdout.flush()

    def _animate(self):
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        while not self.stop_event.is_set():
            elapsed = int(time.time() - self.start_time)
            mins, secs = divmod(elapsed, 60)

            spin = spinner_chars[elapsed % len(spinner_chars)]

            sys.stdout.write(f"\r{spin} Generating dataset... [{mins:02d}:{secs:02d}]")
            sys.stdout.flush()
            time.sleep(0.1)