
## 由于工具类无法传参，所以使用全局变量来传递模型和对应的知识库介绍
class ModelContainer:
    def __init__(self):
        self.MODEL = None
        self.LLM_MODEL_NAME = None
        self.DATABASE = None

model_container = ModelContainer()
