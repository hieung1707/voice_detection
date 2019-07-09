import threading


class SoundLoaderThread(threading.Thread):
    def __init__(self, agi):
        super(SoundLoaderThread, self).__init__()
        self.agi = agi
        self._stop_event = threading.Event

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def playAudio(self, file_path):
        self.agi.stream_file(file_path)
        self.start()
