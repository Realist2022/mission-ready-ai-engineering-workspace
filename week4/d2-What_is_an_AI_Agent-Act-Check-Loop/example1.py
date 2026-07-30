class AgentState:
    def __init__(self):
        self.memory = {}

    def remember(self, key, value):
        self.memory[key] = value

    def recall(self, key):
        return self.memory.get(key)

state = AgentState()
state.remember("preferred_city", "Auckland")
print(state.recall("yes"))