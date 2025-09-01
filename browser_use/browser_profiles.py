class BrowserProfile:
    def __init__(self, profile_name: str = "Default"):
        self.profile_name = profile_name

    def __repr__(self):
        return f"<BrowserProfile name={self.profile_name}>"
