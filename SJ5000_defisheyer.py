from defisheyer import BaseDeFisheyer

class SJ5000_DeFisheyer(BaseDeFisheyer):
    def __init__(self,
                 width: int,
                 height: int,
                 rotation: int = 0):
        super().__init__(width, height, 3, 0.08, -1.3, 1.25, rotation) #__init__(width, height, 3, 0.08, -1.3, 1.25, rotation)