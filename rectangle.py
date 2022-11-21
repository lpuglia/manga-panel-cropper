from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Rectangle():
    #  x,y---------
    #  |          |
    #  |          |
    #  ---------w,z

    x : float
    y : float
    width : float
    height : float
    
    @property
    def w(self):
        return self.x+self.width
    
    @property
    def z(self):
        return self.y+self.height

    @property
    def center_x(self):
        return int(self.width/2+self.x)

    @property
    def center_y(self):
        return int(self.height/2+self.y)
    
    @property
    def area(self):
        return self.width * self.height
        
    def __add__(self, r):
        x = min(self.x, r.x)
        y = min(self.y, r.y)
        w = max(self.w, r.w)
        z = max(self.z, r.z)
        return Rectangle(x, y, w-x, z-y)
   
    def __sub__(self, r):
        x = max(self.x, r.x)
        y = max(self.y, r.y)
        w = min(self.w, r.w)
        z = min(self.z, r.z)
        if x>w: raise ValueError()
        if y>z: raise ValueError()
        return Rectangle(x, y, w-x, z-y)
    
    def intersection_area(self, r):
        try:
            return (self-r).area
        except:
            return 0
        
    #              before area
    # -----x,y---------
    #      |          |
    #      |          |
    #      ---------w,z------
    #  after area
    
    def intersection_before_area(self, r):
        r1 = Rectangle(0,0,self.w,self.y)
        if r.w>self.w:
            r2 = Rectangle(self.w,0,r.w-self.w,self.z)
        else:
            r2 = None
        return r.intersection_area(r1) + (r.intersection_area(r2) if r2 else 0)
