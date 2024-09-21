class Event:
    def __init__(self, event_time, event_type, flow):
        self.event_time = event_time
        self.event_type = event_type
        self.associated_flow = flow
