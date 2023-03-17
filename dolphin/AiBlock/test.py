

class parentA:
    def __init__(self):
        self.public = "public"
        self._protected = "protected"
        self.__private = "private"

    def something(self):
        print(f"parentA.something() : {self.childstuff}")


class Child(parentA):
    def __init__(self, pu, pro, pri):
        super().__init__()
        self.public = pu
        self._protected = pro
        self.__private = pri


c = Child('child_public', 'child_protected', 'child_private')
c.public
#c.__private
c._protected
