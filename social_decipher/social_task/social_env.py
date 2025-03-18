from openai import swarm

class SocialTaskEnvironment(swarm.Environment):
    def __init__(self):
        super().__init__()
        self.state = "Start"
        self.goal = "Complete  Social Task"
        self.turns = 10

    def update_state(self, new_state: str):
        self.state = new_state
        self.broadcast({"state": self.state})

    def check_completion(self):
        return self.state == self.goal
